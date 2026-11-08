"""Démonstration de 10 min : nominal, lot invalide, Postgres coupé, rollback."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from retail.common.config import get_settings, load_app_config
from retail.common.logging import get_logger
from retail.orchestration.errors import ReplayError
from retail.orchestration.replay import (
    emit_alert,
    load_cursor,
    manifest_weeks,
    run_tick,
    save_cursor,
)
from retail.quality.validate import INVALID_DEMO_WEEK, run_invalid_demo
from retail.registry.promote import (
    TAG_DECISION,
    TAG_PREVIOUS,
    _alias_version,
    _client,
    _model_name,
    _promote,
    _version_tag,
    rollback_champion,
)

log = get_logger("retail.demo")

POSTGRES_CONTAINER = "retail-postgres"
PROM_CONTAINER = "retail-prometheus"
LOAD_TEST_WEEK = "2011-W48"
DEMO_WEEK_N = "2011-W40"
DEMO_WEEK_N1 = "2011-W41"
DEMO_CURSOR_RESET = "2011-W40"
DB_DOWN_ALERT = "RetailDbDown"
PROM_ALERT_WAIT_S = 45.0


class DemoError(ReplayError):
    """Etape de demonstration en echec."""


def pick_demo_weeks(
    weeks: list[str],
    *,
    load_week: str = LOAD_TEST_WEEK,
    last_week: str | None = None,
    preferred: tuple[str, str] = (DEMO_WEEK_N, DEMO_WEEK_N1),
) -> tuple[str, str]:
    """N = curseur (ou W40 au premier run), N+1 = semaine suivante hors Locust."""
    usable = [week for week in weeks if week != load_week]
    if len(usable) < 2:
        raise DemoError(f"Pas assez de semaines hors charge Locust ({load_week}) pour la demo.")
    if last_week:
        after = [week for week in usable if week > last_week]
        if last_week in usable and after:
            return last_week, after[0]
        if len(after) >= 2:
            return after[0], after[1]
        if len(after) == 1 and last_week in usable:
            raise DemoError(f"Pas de semaine apres {last_week} (eviter {load_week}).")
        raise DemoError(f"Pas assez de semaines apres le curseur {last_week} (eviter {load_week}).")
    if preferred[0] in usable and preferred[1] in usable:
        return preferred
    return usable[0], usable[1]


def _api_base() -> str:
    port = os.environ.get("API_PORT") or "8001"
    return f"http://127.0.0.1:{port}"


def _request(
    method: str,
    path: str,
    *,
    payload: dict | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    url = f"{_api_base().rstrip('/')}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as extra:
        body = extra.read().decode("utf-8", errors="replace")
        raise DemoError(f"{method} {path} HTTP {extra.code}: {body[:500]}") from extra
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as extra:
        refused = "10061" in str(extra) or "refused" in str(extra).lower()
        hint = (
            " Demarre l'API dans un autre terminal : retail api " "(pas le conteneur Locust)."
            if refused
            else ""
        )
        raise DemoError(f"{method} {path} injoignable ({_api_base()}) : {extra}.{hint}") from extra


def _get(path: str, **kwargs) -> dict[str, Any]:
    return _request("GET", path, **kwargs)


def _post(path: str, payload: dict | None = None, **kwargs) -> dict[str, Any]:
    return _request("POST", path, payload=payload, **kwargs)


def _banner(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def _ok(message: str) -> None:
    print(f"OK  {message}", flush=True)


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise DemoError(message)
    _ok(message)


def _probe(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _prom_base() -> str:
    port = os.environ.get("PROMETHEUS_PORT") or "9091"
    return f"http://127.0.0.1:{port}"


def _prom_json(path: str, *, method: str = "GET", timeout: float = 5.0) -> dict[str, Any]:
    url = f"{_prom_base().rstrip('/')}{path}"
    req = urllib.request.Request(
        url,
        data=b"" if method != "GET" else None,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ) as extra:
        raise DemoError(f"Prometheus {method} {path} : {extra}") from extra


def reload_prometheus() -> None:
    try:
        _prom_json("/-/reload", method="POST")
        return
    except DemoError:
        pass
    result = _docker("kill", "-s", "HUP", PROM_CONTAINER, check=False)
    if result.returncode != 0:
        raise DemoError(
            f"Impossible de recharger Prometheus ({PROM_CONTAINER}) : "
            f"{(result.stderr or result.stdout or '').strip()}"
        )


def prometheus_alert_names() -> set[str]:
    payload = _prom_json("/api/v1/rules")
    names: set[str] = set()
    for group in (payload.get("data") or {}).get("groups") or []:
        for rule in group.get("rules") or []:
            name = rule.get("name") or rule.get("alert")
            if name:
                names.add(str(name))
    return names


def prometheus_alert_firing(name: str) -> bool:
    payload = _prom_json("/api/v1/alerts")
    for alert in (payload.get("data") or {}).get("alerts") or []:
        labels = alert.get("labels") or {}
        if labels.get("alertname") == name and alert.get("state") == "firing":
            return True
    return False


def step_checks(app, *, skip_docker: bool = False) -> dict[str, Any]:
    _banner("0. Verifications")
    health = _get("/health")
    _require(health.get("status") == "ok", f"/health status={health.get('status')}")
    _require(health.get("degraded") is False, f"degraded={health.get('degraded')}")
    _require(bool(health.get("model_version")), f"model_version={health.get('model_version')}")
    backend = health.get("db_backend")
    _require(backend == "postgresql", f"db_backend={backend} (attendu postgresql)")
    _require(health.get("db_ok") is True, f"db_ok={health.get('db_ok')}")
    started_as = str(health.get("model_version"))
    print(
        f"    source={health.get('source')}  version={started_as}",
        flush=True,
    )
    settings = get_settings()
    mlflow = (settings.mlflow_tracking_uri or "").rstrip("/")
    if mlflow.startswith("http"):
        ok = _probe(f"{mlflow}/health") or _probe(mlflow)
        print(f"    MLflow {mlflow} : {'joignable' if ok else 'absent (optionnel)'}", flush=True)
    legal = prepare_rollback_chain(app)
    health = _get("/health")
    meta_path = app.paths.models_dir / "champion" / "meta.json"
    _require(meta_path.is_file(), f"champion disque {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    previous = meta.get("previous_champion")
    _require(
        bool(previous),
        "Toujours une seule version champion apres preparation du rollback.",
    )
    print(f"    previous_champion={previous}", flush=True)
    if not skip_docker:
        reload_prometheus()
        time.sleep(1)
        names = prometheus_alert_names()
        _require(
            DB_DOWN_ALERT in names,
            f"Prometheus charge {sorted(names)} (attendu {DB_DOWN_ALERT})",
        )
    health["_started_as"] = str(legal)
    return health


def _ensure_mae_champion(app) -> str:
    """Si l'alias champion a perdu la MAE, revenir au ``previous_champion``."""
    client = _client()
    name = _model_name(app)
    current = _alias_version(client, name, "champion")
    if not current:
        raise DemoError("Aucun champion dans le registre.")
    decision = str(_version_tag(client, name, current, TAG_DECISION) or "")
    if not decision.startswith("challenger"):
        return str(current)
    previous = _version_tag(client, name, current, TAG_PREVIOUS)
    if not previous:
        raise DemoError(
            f"Champion {current} a perdu la MAE sans previous_champion : "
            "entrainer deux horizons (2011-W38 puis 2011-W40)."
        )
    print(
        f"    champion {current} a perdu la MAE, rollback vers {previous}",
        flush=True,
    )
    rollback_champion(app, to_version=str(previous))
    _post("/model/reload")
    return str(previous)


def prepare_rollback_chain(app) -> str:
    """Assure un champion MAE-valide, puis une chaine ``previous_champion`` pour le rollback.

    Le challenger temporaire peut etre moins bon : c'est volontaire pour la scene 4.
    ``finally`` restaure le champion legal (valeur de retour), pas le temporaire.
    """
    legal = _ensure_mae_champion(app)
    client = _client()
    name = _model_name(app)
    current = _alias_version(client, name, "champion") or legal
    if current and _version_tag(client, name, current, TAG_PREVIOUS):
        print(f"    chaine rollback deja en place (champion {current})", flush=True)
        return legal
    versions = [
        int(mv.version)
        for mv in client.search_model_versions(f"name='{name}'")
        if str(mv.version) != str(current)
    ]
    if not current or not versions:
        raise DemoError(
            "Pas de challenger a promouvoir pour le rollback. "
            "Entrainer deux horizons (ex. 2011-W38 puis 2011-W40)."
        )
    target = str(max(versions))
    _promote(client, app, name, target, previous=str(current))
    _post("/model/reload")
    print(
        f"    challenger {target} promu champion (MAE n'avait pas battu {current}) "
        f"pour demontrer le rollback vers {current}",
        flush=True,
    )
    return legal


def step_nominal(app) -> dict[str, Any]:
    _banner("1. Nominal (rejeu N -> predire N+1 -> rejeu N+1)")
    weeks = manifest_weeks(app)
    cursor = load_cursor(app)
    last = cursor.get("last_week")
    week_n, week_n1 = pick_demo_weeks(weeks, last_week=last)
    print(f"    curseur={last}  N={week_n}  N+1={week_n1}", flush=True)
    if last is None or str(last) < week_n:
        print(f"    rejeu {week_n} (qualite, sans train)...", flush=True)
        run_tick(app, week=week_n, train=False)
    batch = _post("/predict/batch", {"week": week_n1}, timeout=120.0)
    preds = batch.get("predictions") or []
    _require(len(preds) >= 1, f"/predict/batch {week_n1} : {len(preds)} predictions")
    _require(
        all(item.get("model_version") for item in preds),
        f"model_version={batch.get('model_version')} source={batch.get('source')}",
    )
    time.sleep(2)
    print(f"    rejeu {week_n1} (feedback auto, sans train)...", flush=True)
    run_tick(app, week=week_n1, train=False)
    wape = None
    model: dict = {}
    for _ in range(12):
        model = _get("/model")
        wape = model.get("production_wape")
        if wape is not None:
            break
        time.sleep(1)
    _require(wape is not None, f"GET /model production_wape={wape}")
    print(f"    WAPE reelle={wape:.4f}  n_skus={model.get('n_skus')}", flush=True)
    return {"week_n": week_n, "week_n1": week_n1, "wape": wape, "n_pred": len(preds)}


def step_invalid(app) -> None:
    _banner("2. Incident lot invalide")
    run_invalid_demo(app)
    report = app.paths.reports_dir / "invalid_demo" / f"{INVALID_DEMO_WEEK}.json"
    _require(report.is_file(), f"rapport {report}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    _require(payload.get("status") == "failed", f"status={payload.get('status')}")
    quality_copy = app.paths.reports_dir / "quality" / f"{INVALID_DEMO_WEEK}.json"
    _require(not quality_copy.exists(), "aucun rapport qualite 2099-W01 (pas d'entrainement)")
    emit_alert(
        app,
        {
            "kind": "quality",
            "week": INVALID_DEMO_WEEK,
            "message": "Lot invalide de demonstration rejete (porte qualite).",
        },
    )
    alerts = _get("/alerts")
    found = any(
        INVALID_DEMO_WEEK in json.dumps(item, default=str) for item in (alerts.get("alerts") or [])
    )
    _require(found, "GET /alerts contient le lot 2099-W01")


def stop_postgres() -> None:
    stop = _docker("stop", POSTGRES_CONTAINER, check=False)
    if stop.returncode != 0:
        raise DemoError(
            f"docker stop {POSTGRES_CONTAINER} a echoue : "
            f"{(stop.stderr or stop.stdout or '').strip()}"
        )
    _ok(f"conteneur {POSTGRES_CONTAINER} arrete")


def step_postgres_down(week: str, sku: str) -> None:
    grafana = os.environ.get("GRAFANA_PORT") or "3001"
    print(
        f"    Grafana http://127.0.0.1:{grafana} : retail_db_ok et ALERTS "
        f"{DB_DOWN_ALERT} (laisser Postgres coupe > 15 s)",
        flush=True,
    )
    predict = _post(
        "/predict",
        {"StockCode": sku, "week": week},
        timeout=30.0,
    )
    _require("prediction" in predict, "/predict repond toujours sans Postgres")
    health = _get("/health", timeout=15.0)
    _require(health.get("db_ok") is False, f"db_ok={health.get('db_ok')} (attendu false)")
    print(
        f"    API ok  version={health.get('model_version')}  db_ok=false  "
        "RetailApiDown ne s'applique pas",
        flush=True,
    )
    deadline = time.monotonic() + PROM_ALERT_WAIT_S
    while time.monotonic() < deadline:
        if prometheus_alert_firing(DB_DOWN_ALERT):
            _ok(f"Prometheus GET /api/v1/alerts : {DB_DOWN_ALERT} firing")
            return
        time.sleep(2)
    raise DemoError(
        f"{DB_DOWN_ALERT} n'est pas firing apres {int(PROM_ALERT_WAIT_S)} s "
        f"(GET {_prom_base()}/api/v1/alerts)."
    )


def _wait_db_ok(timeout_s: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        try:
            health = _get("/health", timeout=5.0)
            last = str(health)
            if health.get("db_ok") is True:
                return health
        except DemoError as extra:
            last = str(extra)
        time.sleep(2)
    raise DemoError(f"Postgres n'est pas revenu (db_ok) : {last}")


def step_restore_postgres() -> None:
    start = _docker("start", POSTGRES_CONTAINER, check=False)
    if start.returncode != 0:
        raise DemoError(
            f"docker start {POSTGRES_CONTAINER} a echoue : "
            f"{(start.stderr or start.stdout or '').strip()}"
        )
    health = _wait_db_ok()
    _ok(f"Postgres revenu  db_ok={health.get('db_ok')}  backend={health.get('db_backend')}")


def step_rollback(app, original_version: str) -> str:
    _banner("4. Retour arriere")
    result = rollback_champion(app)
    new_version = str(result.get("model_version"))
    _require(
        new_version != original_version,
        f"champion {original_version} -> {new_version}",
    )
    _post("/model/reload")
    health = _get("/health")
    _require(
        str(health.get("model_version")) == new_version,
        f"/health model_version={health.get('model_version')}",
    )
    meta = app.paths.models_dir / "champion" / "meta.json"
    if meta.is_file():
        exported = json.loads(meta.read_text(encoding="utf-8"))
        print(f"    models/champion/meta.json version={exported.get('model_version')}", flush=True)
    return new_version


def restore_champion(app, original_version: str, rolled_to: str | None) -> None:
    if not rolled_to or rolled_to == original_version:
        return
    rollback_champion(app, to_version=original_version)
    _post("/model/reload")
    health = _get("/health")
    _require(
        str(health.get("model_version")) == str(original_version),
        f"champion restaure {health.get('model_version')}",
    )


def step_bilan() -> None:
    _banner("5. Bilan -- ou regarder")
    grafana = os.environ.get("GRAFANA_PORT") or "3001"
    prometheus = os.environ.get("PROMETHEUS_PORT") or "9091"
    mlflow = os.environ.get("MLFLOW_PORT") or "5001"
    prefect = os.environ.get("PREFECT_PORT") or "4201"
    print(
        "\n".join(
            [
                f"    Grafana     http://127.0.0.1:{grafana}  (dashboard retail-mlops, WAPE, runs)",
                f"    Prometheus  http://127.0.0.1:{prometheus}/alerts",
                f"    MLflow      http://127.0.0.1:{mlflow}  (alias champion / challenger)",
                f"    Prefect     http://127.0.0.1:{prefect}  (flow weekly_lot)",
                "    API         GET /health  GET /model  GET /alerts  GET /metrics",
            ]
        ),
        flush=True,
    )


def reset_demo_cursor(app) -> None:
    """Remet le curseur a W40 pour un cycle N/N+1 = W40/W41 (avant soutenance)."""
    save_cursor(app, DEMO_CURSOR_RESET, "ok", force=True)
    print(f"    curseur remis a {DEMO_CURSOR_RESET}", flush=True)


def run_demo(*, skip_docker: bool = False, reset_cursor: bool = False) -> int:
    app = load_app_config()
    original = None
    rolled = None
    postgres_stopped = False
    try:
        if reset_cursor:
            reset_demo_cursor(app)
        health0 = step_checks(app, skip_docker=skip_docker)
        original = str(health0.get("_started_as") or health0.get("model_version"))
        served = str(health0.get("model_version"))
        nominal = step_nominal(app)
        step_invalid(app)
        if skip_docker:
            print("\n=== 3. Incident Postgres : ignore (--skip-docker) ===", flush=True)
        else:
            _banner("3. Incident Postgres coupe")
            sku = str((_get("/model").get("skus") or ["85123A"])[0])
            stop_postgres()
            postgres_stopped = True
            step_postgres_down(nominal["week_n1"], sku)
            step_restore_postgres()
            postgres_stopped = False
        rolled = step_rollback(app, served)
        step_bilan()
        print("\nDemo terminee.", flush=True)
        return 0
    finally:
        if postgres_stopped:
            try:
                step_restore_postgres()
            except Exception as extra:
                log.warning("Postgres non restaure", extra={"error": str(extra)})
        if original and rolled:
            try:
                restore_champion(app, original, rolled)
            except Exception as extra:
                log.warning("Champion non restaure", extra={"error": str(extra)})
