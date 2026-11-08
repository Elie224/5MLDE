"""Registre MLflow : challenger si le naïf est battu, champion si MAE ≤ champion, rollback."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from contextlib import suppress
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import get_model_info
from mlflow.tracking import MlflowClient

from retail.common.config import AppConfig
from retail.common.logging import get_logger
from retail.common.mlflow_uri import tracking_uri
from retail.features.panel import TOP_SKUS_FILENAME
from retail.registry.errors import RegistryError
from retail.training.metrics import mae, wape

log = get_logger(__name__)

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
META_FILENAME = "meta.json"
TAG_PREVIOUS = "previous_champion"
TAG_DECISION = "decision"
INCOMPATIBLE_DECISION = "promoted: champion incompatible (feature schema changed)"
NO_SIGNATURE_DECISION = "promoted: champion without signature"
TAG_PIPELINE = "pipeline_status"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETE = "complete"


def _client() -> MlflowClient:
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def _model_name(app: AppConfig) -> str:
    return app.promotion.registered_model_name


def _champion_dir(app: AppConfig) -> Path:
    return app.paths.models_dir / "champion"


def _alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except MlflowException:
        return None


def _ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)


def _version_tag(client: MlflowClient, name: str, version: str, key: str) -> str | None:
    mv = client.get_model_version(name, version)
    raw = (mv.tags or {}).get(key)
    return str(raw) if raw else None


def _set_tags(client: MlflowClient, name: str, version: str, tags: dict) -> None:
    for key, value in tags.items():
        if value is None or value == "":
            continue
        client.set_model_version_tag(name, version, key, str(value))


def _score(model_uri: str, x_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    model = mlflow.sklearn.load_model(model_uri)
    pred = np.clip(np.asarray(model.predict(x_test), dtype="float64"), 0, None)
    y = np.asarray(y_test, dtype="float64")
    return {"mae": mae(y, pred), "wape": wape(y, pred)}


def _signature_input_names(model_uri: str) -> set[str] | None:
    info = get_model_info(model_uri)
    signature = info.signature
    if signature is None or signature.inputs is None:
        return None
    names = {item.name for item in signature.inputs if getattr(item, "name", None)}
    return names or None


def _champion_schema_compatible(model_uri: str, x_test: pd.DataFrame) -> bool | None:
    """True / False si une signature existe, None sinon."""
    expected = _signature_input_names(model_uri)
    if not expected:
        return None
    return expected.issubset(set(x_test.columns))


def find_week_version(app: AppConfig, week: str | None):
    """Dernière version **non terminée** pour ``until_week`` (relance après crash)."""
    if not week:
        return None
    client = _client()
    name = _model_name(app)
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except MlflowException:
        return None
    pending = [
        item
        for item in versions
        if (item.tags or {}).get("until_week") == str(week)
        and (item.tags or {}).get(TAG_PIPELINE) == STATUS_IN_PROGRESS
    ]
    if not pending:
        return None
    return max(pending, key=lambda item: int(item.version))


def find_complete_week_version(app: AppConfig, week: str | None):
    """Dernière version **terminée** entraînée jusqu'à ``until_week`` (évite les doublons)."""
    if not week:
        return None
    client = _client()
    name = _model_name(app)
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except MlflowException:
        return None
    done = [
        item
        for item in versions
        if (item.tags or {}).get("until_week") == str(week)
        and (item.tags or {}).get(TAG_PIPELINE) == STATUS_COMPLETE
    ]
    if not done:
        return None
    return max(done, key=lambda item: int(item.version))


def _should_export_on_resume(client: MlflowClient, name: str, version: str, tags: dict) -> bool:
    decision = str(tags.get(TAG_DECISION) or "")
    champ = _alias_version(client, name, CHAMPION_ALIAS)
    if champ is None or champ == str(version):
        return True
    return decision.startswith("promoted")


def resume_week_version(app: AppConfig, version: str) -> dict:
    """Termine une version ``in_progress`` sans réentraîner (relance Prefect)."""
    client = _client()
    name = _model_name(app)
    mv = client.get_model_version(name, version)
    tags = mv.tags or {}
    decision = str(tags.get(TAG_DECISION) or "")
    exported = False
    if _should_export_on_resume(client, name, version, tags):
        export_version(app, version)
        exported = True
        if _alias_version(client, name, CHALLENGER_ALIAS) is None:
            client.set_registered_model_alias(name, CHALLENGER_ALIAS, version)
        if _alias_version(client, name, CHAMPION_ALIAS) is None:
            client.set_registered_model_alias(name, CHAMPION_ALIAS, version)
    elif not decision:
        from retail.features.panel import FEATURE_COLUMNS, build_features
        from retail.training.train import temporal_split

        until = tags.get("until_week")
        featured, _skus = build_features(app, until_week=until)
        _train, test = temporal_split(featured, app.training.test_weeks)
        return _finish_versus_champion(
            app,
            client,
            name,
            str(version),
            mv.run_id,
            test[FEATURE_COLUMNS],
            test["y"].to_numpy(dtype="float64"),
        )
    _set_tags(client, name, version, {TAG_PIPELINE: STATUS_COMPLETE})
    return {
        "trained": False,
        "resumed": True,
        "exported": exported,
        "model_version": str(version),
        "registry": {
            "status": "resumed",
            "model_version": str(version),
            "exported": exported,
            "decision": decision or None,
        },
    }


def load_run_skus(run_id: str) -> set[str]:
    """SKU figés du run champion (artefact ``top_skus.json``), jamais ``data/interim``."""
    scratch = Path(tempfile.mkdtemp(prefix="retail-skus-"))
    try:
        _download_top_skus(run_id, scratch)
        path = scratch / TOP_SKUS_FILENAME
        if not path.is_file():
            return set()
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("skus", [])
        return {str(item) for item in items}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _find_mlmodel(root: Path) -> Path:
    direct = root / "MLmodel"
    if direct.is_file():
        return root
    nested = root / "model" / "MLmodel"
    if nested.is_file():
        return nested.parent
    found = list(root.rglob("MLmodel"))
    if not found:
        raise RegistryError("Artefact modèle introuvable dans le run MLflow.")
    return found[0].parent


def _download_model(run_id: str, dest_model: Path) -> None:
    scratch = dest_model.parent / "_artifact_dl"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    try:
        local = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/model", dst_path=str(scratch)
            )
        )
        src = _find_mlmodel(local)
        if dest_model.exists():
            shutil.rmtree(dest_model)
        shutil.copytree(src, dest_model)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _download_top_skus(run_id: str, dest_dir: Path) -> None:
    scratch = dest_dir / "_skus_dl"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    try:
        local = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/{TOP_SKUS_FILENAME}",
                dst_path=str(scratch),
            )
        )
        src = local if local.is_file() else scratch / TOP_SKUS_FILENAME
        if not src.is_file():
            matches = list(scratch.rglob(TOP_SKUS_FILENAME))
            if not matches:
                return
            src = matches[0]
        shutil.copy2(src, dest_dir / TOP_SKUS_FILENAME)
    except MlflowException:
        log.warning("top_skus.json absent des artefacts du run", extra={"run_id": run_id})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _rmtree_retry(path: Path, *, attempts: int = 8, delay: float = 0.15) -> None:
    last: OSError | None = None
    for _ in range(attempts):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except OSError as exc:
            last = exc
            time.sleep(delay)
    if last is not None:
        shutil.rmtree(path, ignore_errors=True)


def _rename_retry(src: Path, dest: Path, *, attempts: int = 8, delay: float = 0.15) -> None:
    last: OSError | None = None
    for _ in range(attempts):
        try:
            src.rename(dest)
            return
        except OSError as exc:
            last = exc
            time.sleep(delay)
    if last is not None:
        raise last


def _copy_replace_contents(src: Path, dest: Path) -> None:
    """Windows : l'API peut verrouiller ``models/champion`` (WinError 5)."""
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {item.name for item in src.iterdir()}
    for item in list(dest.iterdir()):
        if item.name in wanted:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            with suppress(OSError):
                item.unlink()
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _replace_dir(tmp: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    backup = dest.parent / f".{dest.name}.bak"
    if backup.exists():
        _rmtree_retry(backup)
    moved = False
    try:
        if dest.exists():
            try:
                _rename_retry(dest, backup)
                moved = True
            except OSError:
                _copy_replace_contents(tmp, dest)
                _rmtree_retry(tmp)
                return
        try:
            _rename_retry(tmp, dest)
        except OSError:
            _copy_replace_contents(tmp, dest)
            _rmtree_retry(tmp)
    except OSError:
        if moved and backup.exists() and not dest.exists():
            with suppress(OSError):
                backup.rename(dest)
        raise
    if backup.exists():
        shutil.rmtree(backup, ignore_errors=True)


def export_version(app: AppConfig, version: str, previous: str | None = None) -> Path:
    """Copie atomique des artefacts du run vers ``models/champion/`` (sans toucher aux alias)."""
    client = _client()
    name = _model_name(app)
    mv = client.get_model_version(name, version)
    dest = _champion_dir(app)
    tmp = dest.parent / f".champion.{version}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        _download_model(mv.run_id, tmp / "model")
        _download_top_skus(mv.run_id, tmp)
        run = client.get_run(mv.run_id)
        mae_run = run.data.metrics.get("mae_model")
        test_weeks = run.data.tags.get("test_weeks")
        prev = (
            previous
            if previous is not None
            else _version_tag(client, name, str(version), TAG_PREVIOUS)
        )
        meta = {
            "registered_model": name,
            "model_version": str(version),
            "run_id": mv.run_id,
            "mae_own_test": float(mae_run) if mae_run is not None else None,
            "test_weeks": test_weeks.split(",") if test_weeks else None,
            "alias": CHAMPION_ALIAS,
            "previous_champion": prev,
        }
        (tmp / META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _replace_dir(tmp, dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log.info("Champion exporté", extra={"dir": str(dest), "version": str(version)})
    return dest


def export_champion(app: AppConfig) -> Path:
    """Réexporte le champion actuellement aliasé (mode dégradé)."""
    client = _client()
    version = _alias_version(client, _model_name(app), CHAMPION_ALIAS)
    if version is None:
        raise RegistryError("Aucun champion à exporter.")
    return export_version(app, version)


def _promote(
    client: MlflowClient, app: AppConfig, name: str, version: str, previous: str | None
) -> None:
    export_version(app, version, previous=previous)
    client.set_registered_model_alias(name, CHAMPION_ALIAS, version)
    if previous:
        _set_tags(client, name, version, {TAG_PREVIOUS: previous})


def _promote_incompatible(
    client: MlflowClient,
    app: AppConfig,
    name: str,
    version: str,
    previous: str,
    result: dict,
    decision: str,
) -> dict:
    _promote(client, app, name, version, previous=previous)
    _set_tags(client, name, version, {TAG_DECISION: decision, TAG_PIPELINE: STATUS_COMPLETE})
    result.update({"status": "champion", "promoted": True, "decision": decision})
    log.warning("Champion remplacé", extra={"version": version, "decision": decision})
    return result


def register_run(
    app: AppConfig,
    metrics: dict,
    *,
    x_test: pd.DataFrame | None = None,
    y_test: np.ndarray | None = None,
) -> dict:
    """Enregistre le run en challenger s'il bat le naïf ; promeut si MAE ≤ champion (même test)."""
    if app.promotion.must_beat_baseline and not metrics.get("beats_baseline"):
        log.info("Run non enregistré : n'a pas battu la baseline naïve")
        return {"status": "skipped", "reason": "baseline"}
    run_id = metrics.get("mlflow_run_id")
    if not run_id:
        raise RegistryError("mlflow_run_id manquant : relancer `retail train`.")

    client = _client()
    name = _model_name(app)
    _ensure_registered_model(client, name)
    mv = mlflow.register_model(f"runs:/{run_id}/model", name)
    version = str(mv.version)
    client.set_registered_model_alias(name, CHALLENGER_ALIAS, version)
    _set_tags(
        client,
        name,
        version,
        {
            "data_hash": metrics.get("data_hash"),
            "features_hash": metrics.get("features_hash"),
            "until_week": metrics.get("until_week"),
            "beats_baseline": metrics.get("beats_baseline"),
            "run_id": run_id,
            TAG_PIPELINE: STATUS_IN_PROGRESS,
        },
    )
    result: dict = {"status": "challenger", "model_version": version, "run_id": run_id}
    return _finish_versus_champion(app, client, name, version, run_id, x_test, y_test, result)


def _finish_versus_champion(
    app: AppConfig,
    client: MlflowClient,
    name: str,
    version: str,
    run_id: str,
    x_test: pd.DataFrame | None,
    y_test: np.ndarray | None,
    result: dict | None = None,
) -> dict:
    result = result or {"status": "challenger", "model_version": version, "run_id": run_id}
    champ = _alias_version(client, name, CHAMPION_ALIAS)
    if champ is None:
        _promote(client, app, name, version, previous=None)
        decision = "promoted: first champion"
        _set_tags(client, name, version, {TAG_DECISION: decision, TAG_PIPELINE: STATUS_COMPLETE})
        result.update({"status": "champion", "promoted": True, "decision": decision})
        log.info("Premier champion", extra={"version": version})
        return result

    if x_test is None or y_test is None:
        raise RegistryError(
            "Jeu de test requis pour comparer le champion et le challenger sur les mêmes semaines."
        )
    new_scores = _score(f"runs:/{run_id}/model", x_test, y_test)
    result["mae_challenger_same_test"] = new_scores["mae"]
    result["wape_challenger_same_test"] = new_scores["wape"]
    result["champion_version"] = champ
    champ_uri = f"models:/{name}/{champ}"
    compatible = _champion_schema_compatible(champ_uri, x_test)
    if compatible is False:
        return _promote_incompatible(
            client, app, name, version, champ, result, INCOMPATIBLE_DECISION
        )
    if compatible is None:
        try:
            champ_scores = _score(champ_uri, x_test, y_test)
        except (ValueError, KeyError, MlflowException):
            return _promote_incompatible(
                client, app, name, version, champ, result, NO_SIGNATURE_DECISION
            )
    else:
        champ_scores = _score(champ_uri, x_test, y_test)
    result["mae_champion_same_test"] = champ_scores["mae"]

    if new_scores["mae"] <= champ_scores["mae"]:
        _promote(client, app, name, version, previous=champ)
        decision = (
            f"promoted: mae {new_scores['mae']:.4f} <= champion {champ} "
            f"mae {champ_scores['mae']:.4f} on same test"
        )
        _set_tags(client, name, version, {TAG_DECISION: decision, TAG_PIPELINE: STATUS_COMPLETE})
        result.update({"status": "champion", "promoted": True, "decision": decision})
        log.info("Champion promu", extra={"version": version, "previous": champ})
        return result

    decision = (
        f"challenger: mae {new_scores['mae']:.4f} > champion {champ} "
        f"mae {champ_scores['mae']:.4f} on same test"
    )
    _set_tags(client, name, version, {TAG_DECISION: decision, TAG_PIPELINE: STATUS_COMPLETE})
    result.update({"promoted": False, "decision": decision})
    log.info("Reste challenger", extra={"version": version, "champion": champ})
    return result


def rollback_champion(app: AppConfig, to_version: str | None = None) -> dict:
    """Ré-alias une version précédente (tag MLflow ``previous_champion`` ou ``--to-version``)."""
    client = _client()
    name = _model_name(app)
    current = _alias_version(client, name, CHAMPION_ALIAS)
    if current is None:
        raise RegistryError("Aucun champion : rien à restaurer.")
    if to_version:
        target = str(to_version)
        try:
            client.get_model_version(name, target)
        except MlflowException as exc:
            raise RegistryError(f"Version {target} introuvable dans le registre.") from exc
    else:
        target = _version_tag(client, name, current, TAG_PREVIOUS)
        if not target:
            raise RegistryError("Pas de version précédente (premier champion).")
    if target == current:
        raise RegistryError(f"Le champion est déjà la version {current}.")

    export_version(app, target)
    client.set_registered_model_alias(name, CHAMPION_ALIAS, target)
    log.info("Rollback champion", extra={"from": current, "to": target})
    return {"status": "rolled_back", "model_version": target, "replaced": current}
