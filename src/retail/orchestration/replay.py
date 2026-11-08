"""Un lot par déclenchement : qualité de la semaine, puis features/train bornés à cette semaine."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any

import requests

from retail.api.features import clear_feature_cache
from retail.common.config import AppConfig, get_settings
from retail.common.logging import get_logger
from retail.data.make_batches import load_manifest
from retail.db.auto_feedback import auto_feedback_for_week
from retail.db.repo import persist_alert, persist_run
from retail.orchestration.errors import ReplayDone, ReplayError
from retail.orchestration.lock import replay_lock
from retail.quality.errors import QualityError
from retail.quality.validate import validate_all
from retail.training.errors import TrainError
from retail.training.train import train_and_evaluate

log = get_logger(__name__)

CURSOR_FILENAME = "cursor.json"
RUNS_FILENAME = "runs.jsonl"
ALERTS_FILENAME = "alerts.jsonl"
LOCK_FILENAME = "replay.lock"
BLOCKING_STATUSES = frozenset({"quality_failed", "technical_failed"})


def _orch_dir(app: AppConfig):
    path = app.paths.reports_dir / "orchestration"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cursor_path(app: AppConfig):
    return _orch_dir(app) / CURSOR_FILENAME


def load_cursor(app: AppConfig) -> dict:
    path = cursor_path(app)
    if not path.is_file():
        return {"last_week": None, "status": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cursor(app: AppConfig, week: str, status: str, *, force: bool = False) -> None:
    """N'écrit pas une semaine **antérieure** en `ok` (évite de reculer après `--week`).

    ``force=True`` : ``retail demo --reset-cursor``. Un échec qualité/technique
    pose toujours le curseur sur la semaine en cause.
    """
    if not force and status == "ok":
        current = load_cursor(app)
        last = current.get("last_week")
        prev = current.get("status")
        if last and prev not in BLOCKING_STATUSES and str(week) < str(last):
            log.info("Curseur conservé", extra={"last_week": last, "replayed": week})
            return
    cursor_path(app).write_text(
        json.dumps(
            {
                "last_week": week,
                "status": status,
                "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def append_jsonl(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_run(app: AppConfig, payload: dict) -> None:
    record = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), **payload}
    append_jsonl(_orch_dir(app) / RUNS_FILENAME, record)
    persist_run(record)


def emit_alert(app: AppConfig, payload: dict) -> None:
    """SQL synchrone, puis POST vers l'API. Copie locale si le webhook est absent ou en échec."""
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        **payload,
    }
    persist_alert({**record, "origin": "pipeline"})
    settings = get_settings()
    url = settings.alert_webhook_url
    if url:
        try:
            response = requests.post(url, json=record, timeout=settings.alert_timeout_seconds)
            response.raise_for_status()
            return
        except requests.RequestException as exc:
            log.warning(
                "Webhook d'alerte injoignable (copie locale écrite)", extra={"error": str(exc)}
            )
    append_jsonl(app.paths.reports_dir / ALERTS_FILENAME, record)


def manifest_weeks(app: AppConfig) -> list[str]:
    path = app.paths.batches_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} introuvable : lancer d'abord `retail data`.")
    return [info.week for info in load_manifest(app.paths.batches_dir)]


def next_week(app: AppConfig, requested: str | None = None) -> str:
    weeks = manifest_weeks(app)
    if not weeks:
        raise ReplayError("Manifeste sans lots.")
    if requested:
        if requested not in weeks:
            raise ReplayError(f"Semaine {requested} absente du manifeste.")
        return requested
    cursor = load_cursor(app)
    last = cursor.get("last_week")
    status = cursor.get("status")
    if last and status in BLOCKING_STATUSES:
        return last
    if last is None:
        return weeks[0]
    if last not in weeks:
        return weeks[0]
    index = weeks.index(last) + 1
    if index >= len(weeks):
        raise ReplayDone("Tous les lots du manifeste ont déjà été rejoués.")
    return weeks[index]


def validate_lot(app: AppConfig, week: str) -> None:
    validate_all(app, week=week)


def _record_failure(
    app: AppConfig, week: str, status: str, exc: BaseException, started: float
) -> None:
    kind = "quality" if status == "quality_failed" else "technical"
    emit_alert(app, {"kind": kind, "week": week, "message": str(exc)})
    save_cursor(app, week, status)
    append_run(
        app,
        {
            "week": week,
            "status": status,
            "duration_s": round(time.monotonic() - started, 3),
            "error": str(exc),
        },
    )


def validate_step(app: AppConfig, week: str | None, started: float) -> str:
    chosen = next_week(app, week)
    try:
        validate_lot(app, chosen)
    except QualityError as exc:
        _record_failure(app, chosen, "quality_failed", exc, started)
        raise
    clear_feature_cache()
    auto_feedback_for_week(app, chosen)
    return chosen


def train_lot(app: AppConfig, week: str) -> dict[str, Any]:
    """Entraîne, ou reprend seulement une version encore ``in_progress`` pour cette semaine."""
    from retail.registry.promote import (
        find_complete_week_version,
        find_week_version,
        resume_week_version,
    )

    existing = find_week_version(app, week)
    if existing is not None:
        log.info("Reprise sans réentraînement", extra={"week": week, "version": existing.version})
        return resume_week_version(app, str(existing.version))
    done = find_complete_week_version(app, week)
    if done is not None:
        # Même lot rejoué (démo, relance manuelle) : mêmes données -> même modèle.
        # Réentraîner créerait une version identique (cas v3/v5) : on la réutilise.
        log.info(
            "Lot déjà entraîné, pas de nouvelle version",
            extra={"week": week, "version": done.version},
        )
        return {
            "trained": False,
            "train_skipped": f"version {done.version} déjà entraînée jusqu'à {week}",
            "model_version": str(done.version),
        }
    try:
        metrics = train_and_evaluate(app, until_week=week)
    except TrainError as exc:
        log.info("Entraînement reporté", extra={"week": week, "reason": str(exc)})
        return {"trained": False, "train_skipped": str(exc)}
    registry = metrics.get("registry") or {}
    return {
        "trained": True,
        "mae_model": metrics.get("mae_model"),
        "beats_baseline": metrics.get("beats_baseline"),
        "model_version": registry.get("model_version"),
        "registry": registry,
    }


def train_step(app: AppConfig, week: str, started: float) -> dict[str, Any]:
    result: dict[str, Any] = {"week": week, "quality": "ok", "trained": False}
    try:
        result.update(train_lot(app, week))
    except QualityError as exc:
        _record_failure(app, week, "quality_failed", exc, started)
        raise
    except Exception as exc:
        _record_failure(app, week, "technical_failed", exc, started)
        raise
    save_cursor(app, week, "ok")
    result.update(forecast_next_week(app, week))
    append_run(
        app,
        {
            "week": week,
            "status": "ok",
            "duration_s": round(time.monotonic() - started, 3),
            "trained": result.get("trained"),
            "model_version": result.get("model_version") or _served_version(app),
            "forecast_week": result.get("forecast_week"),
            "forecast_n": result.get("forecast_n"),
        },
    )
    return result


def forecast_next_week(app: AppConfig, week: str) -> dict[str, Any]:
    """Prévoit N+1 via l'API (POST /predict/batch) pour que le lot suivant ferme la boucle.

    Best effort : une API arrêtée ne fait pas échouer le lot (la donnée est déjà validée).
    """
    from retail.api.features import next_iso_week

    target = next_iso_week(week)
    settings = get_settings()
    url = settings.api_url.rstrip("/") + "/predict/batch"
    try:
        response = requests.post(
            url, json={"week": target}, timeout=settings.forecast_timeout_seconds
        )
        response.raise_for_status()
        body = response.json()
    except (requests.RequestException, ValueError) as exc:
        log.warning("Prévision N+1 non produite", extra={"week": target, "error": str(exc)})
        return {"forecast_week": target, "forecast_n": 0, "forecast_error": str(exc)}
    n = len(body.get("predictions") or [])
    log.info("Prévision N+1 enregistrée", extra={"week": target, "n": n})
    return {
        "forecast_week": target,
        "forecast_n": n,
        "forecast_model_version": body.get("model_version"),
    }


def _served_version(app: AppConfig) -> str | None:
    """Version du champion exporté (``models/champion/meta.json``), pour tracer les runs."""
    meta = app.paths.models_dir / "champion" / "meta.json"
    try:
        value = json.loads(meta.read_text(encoding="utf-8")).get("model_version")
    except (OSError, ValueError):
        return None
    return str(value) if value is not None else None


def run_tick(app: AppConfig, week: str | None = None, *, train: bool = True) -> dict:
    """Valide **un** lot. Un échec qualité bloque sur cette semaine jusqu'à correction."""
    started = time.monotonic()
    with replay_lock(_orch_dir(app) / LOCK_FILENAME):
        chosen = validate_step(app, week, started)
        if not train:
            save_cursor(app, chosen, "ok")
            forecast = forecast_next_week(app, chosen)
            append_run(
                app,
                {
                    "week": chosen,
                    "status": "ok",
                    "duration_s": round(time.monotonic() - started, 3),
                    "trained": False,
                    "model_version": _served_version(app),
                    **forecast,
                },
            )
            return {"week": chosen, "quality": "ok", "trained": False, **forecast}
        result = train_step(app, chosen, started)
        log.info("Lot rejoué", extra={"week": chosen, "trained": result.get("trained")})
        return result
