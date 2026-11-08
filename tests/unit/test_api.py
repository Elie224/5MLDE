import json
import shutil

import pandas as pd
from fastapi.testclient import TestClient

from retail.api.app import create_app
from retail.api.store import ModelStore


def _features(sku: str = "AAA") -> dict:
    return {
        "StockCode": sku,
        "lag_1": 10.0,
        "lag_2": 8.0,
        "lag_4": 6.0,
        "ma_4": 8.0,
        "price_lag_1": 1.5,
        "week_of_year": 10,
        "month": 3,
    }


def _write_panels(app, skus: list[str], weeks: list[str]) -> None:
    panel_dir = app.paths.interim_dir / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for index, week in enumerate(weeks):
        frame = pd.DataFrame(
            {
                "StockCode": skus,
                "iso_week": week,
                "quantity": [10 + index, 4 + index],
                "unit_price": [1.5, 2.0],
            }
        )
        frame.to_parquet(panel_dir / f"{week}.parquet", index=False)


def test_predict_from_week_health_alerts_and_errors(tmp_configs, register_dummy):
    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    _write_panels(
        app_cfg,
        ["AAA", "BBB"],
        [f"2011-W{week:02d}" for week in range(10, 21)],
    )
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_version"]
    assert health.json()["degraded"] is False
    assert health.headers.get("x-model-version") or health.headers.get("X-Model-Version")

    load = client.post(
        "/predict",
        json={"StockCode": "AAA", "week": "2011-W20"},
        headers={"X-Load-Test": "1"},
    )
    assert load.status_code == 200
    assert load.json()["source"] == "loadtest"

    ok = client.post("/predict", json={"StockCode": "AAA", "week": "2011-W20"})
    assert ok.status_code == 200
    body = ok.json()
    assert "prediction" in body
    assert body["prediction_id"]
    assert body["week"] == "2011-W20"
    assert body["StockCode"] == "AAA"
    assert body["model_version"]

    missing = client.post("/predict", json={"StockCode": "UNKNOWN", "week": "2011-W20"})
    assert missing.status_code == 404

    invalid = client.post("/predict", json={"StockCode": "AAA"})
    assert invalid.status_code == 422

    batch = client.post("/predict/batch", json={"week": "2011-W20"})
    assert batch.status_code == 200
    assert len(batch.json()["predictions"]) == 2
    assert all(row["prediction_id"] for row in batch.json()["predictions"])

    raw = client.post("/predict/features", json=_features())
    assert raw.status_code == 200

    fb = client.post(
        "/feedback",
        json={
            "StockCode": "AAA",
            "iso_week": "2011-W20",
            "actual_qty": 12.0,
            "prediction_id": body["prediction_id"],
        },
    )
    assert fb.status_code == 202

    bad_week = client.post(
        "/feedback",
        json={"StockCode": "AAA", "iso_week": "bad", "actual_qty": 1},
    )
    assert bad_week.status_code == 422
    negative = client.post(
        "/feedback",
        json={"StockCode": "AAA", "iso_week": "2011-W20", "actual_qty": -1},
    )
    assert negative.status_code == 422
    missing_id = client.post(
        "/feedback",
        json={"StockCode": "AAA", "iso_week": "2011-W20", "actual_qty": 1},
    )
    assert missing_id.status_code == 422
    unknown_fb = client.post(
        "/feedback",
        json={
            "StockCode": "UNKNOWN",
            "iso_week": "2011-W20",
            "actual_qty": 1,
            "prediction_id": body["prediction_id"],
        },
    )
    assert unknown_fb.status_code == 404
    ghost_fb = client.post(
        "/feedback",
        json={
            "StockCode": "AAA",
            "iso_week": "2011-W20",
            "actual_qty": 1,
            "prediction_id": "00000000-0000-0000-0000-000000000000",
        },
    )
    assert ghost_fb.status_code == 404

    alert = client.post("/alerts", json={"kind": "quality", "week": "2010-W49", "message": "demo"})
    assert alert.status_code == 202
    listed = client.get("/alerts")
    assert listed.status_code == 200
    assert listed.json()["alerts"]
    received = app_cfg.paths.reports_dir / "alerts_received.jsonl"
    assert received.is_file()

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    text = metrics.text
    assert "retail_http_requests_total" in text
    assert "retail_model_degraded" in text
    assert 'le="0.001"' in text
    assert 'path="/predict"' in text or 'path="/predict",' in text
    assert "422" in text

    info = client.get("/model")
    assert info.status_code == 200
    reloaded = client.post("/model/reload")
    assert reloaded.status_code == 200


def test_api_falls_back_to_champion_bak(tmp_configs, register_dummy, monkeypatch):
    from mlflow.exceptions import MlflowException

    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    dest = app_cfg.paths.models_dir / "champion"
    bak = app_cfg.paths.models_dir / ".champion.bak"
    shutil.move(str(dest), str(bak))

    class _Down:
        def __init__(self, *args, **kwargs):
            pass

        def get_model_version_by_alias(self, *args, **kwargs):
            raise MlflowException("MLflow indisponible")

    monkeypatch.setattr("retail.api.store.MlflowClient", _Down)
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["source"] == "bak"
    assert health.json()["degraded"] is True
    assert client.post("/predict/features", json=_features()).status_code == 200


def test_health_503_without_model(tmp_configs):
    client = TestClient(create_app())
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json={"StockCode": "AAA", "week": "2011-W20"}).status_code == 503


def test_health_degraded_after_registry_drop(tmp_configs, register_dummy, monkeypatch):
    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    monkeypatch.setattr(ModelStore, "_probe_registry", lambda self: False)
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["degraded"] is True
    assert health.json()["registry_ok"] is False


def test_reload_loads_model_by_version_number(tmp_configs, register_dummy, monkeypatch):
    app_cfg, _ = tmp_configs
    result = register_dummy(app_cfg, constant=8.0)
    seen: list[str] = []
    real = __import__("mlflow.sklearn", fromlist=["load_model"]).load_model

    def _load(uri, *args, **kwargs):
        seen.append(str(uri))
        return real(uri, *args, **kwargs)

    monkeypatch.setattr("retail.api.store.mlflow.sklearn.load_model", _load)
    client = TestClient(create_app())
    assert client.post("/model/reload").status_code == 200
    assert any(f"/{result['model_version']}" in uri and "@" not in uri for uri in seen)


def test_mlflow_skus_not_from_interim(tmp_configs, register_dummy):
    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    interim = app_cfg.paths.interim_dir / "top_skus.json"
    interim.write_text(json.dumps(["INTERIM_ONLY"]), encoding="utf-8")
    client = TestClient(create_app())
    info = client.get("/model").json()
    assert info["n_skus"] == 2
    assert set(info["skus"]) == {"AAA", "BBB"}
    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"


def test_invalid_iso_week_and_horizon_are_422(tmp_configs, register_dummy):
    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    _write_panels(app_cfg, ["AAA", "BBB"], [f"2011-W{week:02d}" for week in range(10, 21)])
    client = TestClient(create_app())
    impossible = client.post("/predict", json={"StockCode": "AAA", "week": "2011-W60"})
    assert impossible.status_code == 422
    far = client.post("/predict", json={"StockCode": "AAA", "week": "2013-W30"})
    assert far.status_code == 422
    assert "horizon dépassé" in far.json()["detail"]
    next_ok = client.post("/predict", json={"StockCode": "AAA", "week": "2011-W21"})
    assert next_ok.status_code == 200
    batch_far = client.post("/predict/batch", json={"week": "2013-W30"})
    assert batch_far.status_code == 422


def test_feature_cache_avoids_rereading_panel(tmp_configs, register_dummy, monkeypatch):
    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    _write_panels(app_cfg, ["AAA", "BBB"], [f"2011-W{week:02d}" for week in range(10, 21)])
    import retail.api.features as feat_mod

    calls = {"n": 0}
    real = feat_mod._load_history

    def _counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(feat_mod, "_load_history", _counted)
    client = TestClient(create_app())
    payload = {"StockCode": "AAA", "week": "2011-W20"}
    assert client.post("/predict", json=payload).status_code == 200
    after_first = calls["n"]
    assert after_first >= 1
    assert client.post("/predict", json=payload).status_code == 200
    assert client.post("/predict/batch", json={"week": "2011-W20"}).status_code == 200
    assert calls["n"] == after_first


def test_metrics_unmatched_path_and_500(tmp_configs, register_dummy):
    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    api = create_app()

    @api.get("/__boom")
    def _boom():
        raise RuntimeError("boom")

    client = TestClient(api, raise_server_exceptions=False)
    assert client.get("/nimporte/quoi/123").status_code == 404
    assert client.get("/__boom").status_code == 500
    text = client.get("/metrics").text
    assert 'path="unmatched"' in text
    assert "/nimporte" not in text
    assert 'status="500"' in text
    assert 'path="/__boom"' in text


def test_predict_persists_to_sqlite(tmp_configs, register_dummy):
    from retail.db.models import Prediction
    from retail.db.repo import count_rows

    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    _write_panels(app_cfg, ["AAA", "BBB"], [f"2011-W{week:02d}" for week in range(10, 21)])
    client = TestClient(create_app())
    assert client.post("/predict", json={"StockCode": "AAA", "week": "2011-W20"}).status_code == 200
    assert count_rows(Prediction) >= 1
    health = client.get("/health").json()
    assert health["db_ok"] is True
    assert health["db_backend"] == "sqlite"


def test_predict_errors_are_persisted(tmp_configs, register_dummy):
    from sqlalchemy import select

    from retail.db.models import Prediction
    from retail.db.session import session_scope

    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    _write_panels(app_cfg, ["AAA", "BBB"], [f"2011-W{week:02d}" for week in range(10, 21)])
    client = TestClient(create_app())
    assert (
        client.post("/predict", json={"StockCode": "UNKNOWN", "week": "2011-W20"}).status_code
        == 404
    )
    assert client.post("/predict", json={"StockCode": "AAA", "week": "2011-W60"}).status_code == 422
    with session_scope() as session:
        errors = session.scalars(select(Prediction).where(Prediction.status == "error")).all()
    assert len(errors) >= 2
    assert any("404" in (row.error or "") for row in errors)
    assert any("422" in (row.error or "") for row in errors)


def test_predict_still_works_when_database_is_down(tmp_configs, register_dummy, monkeypatch):
    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    _write_panels(app_cfg, ["AAA", "BBB"], [f"2011-W{week:02d}" for week in range(10, 21)])
    monkeypatch.setattr(
        "retail.db.session.get_engine",
        lambda: (_ for _ in ()).throw(OSError("postgres down")),
    )
    monkeypatch.setattr("retail.api.app.db_ping", lambda: False)
    client = TestClient(create_app())
    ok = client.post("/predict", json={"StockCode": "AAA", "week": "2011-W20"})
    assert ok.status_code == 200
    assert ok.json()["prediction_id"]
    health = client.get("/health").json()
    assert health["db_ok"] is False


def test_health_reloads_when_registry_returns(tmp_configs, register_dummy, monkeypatch):
    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    probes = {"ok": False}
    monkeypatch.setattr(ModelStore, "_probe_registry", lambda self: probes["ok"])
    client = TestClient(create_app())
    down = client.get("/health").json()
    assert down["degraded"] is True
    assert down["registry_ok"] is False
    probes["ok"] = True
    up = client.get("/health").json()
    assert up["registry_ok"] is True
    assert up["source"] == "mlflow"
    assert up["degraded"] is False


def test_skip_registry_probe_serves_disk_without_degraded(tmp_configs, register_dummy, monkeypatch):
    monkeypatch.setenv("RETAIL_SKIP_REGISTRY_PROBE", "1")
    app_cfg, _ = tmp_configs
    register_dummy(app_cfg, constant=8.0)
    store = ModelStore(app_cfg)
    store.reload()
    assert store.source == "disk"
    assert store.degraded is False
    store.refresh_registry_status()
    assert store.source == "disk"


def test_store_reloads_when_champion_alias_changes(tmp_configs, register_dummy):
    app_cfg, _ = tmp_configs
    first = register_dummy(app_cfg, constant=0.0)
    store = ModelStore(app_cfg)
    store.reload()
    assert store.model_version == first["model_version"]
    second = register_dummy(app_cfg, constant=8.0)
    assert second["status"] == "champion"
    store.refresh_registry_status()
    assert store.model_version == second["model_version"]


def test_set_version_gauge_clears_previous_without_multiprocess():
    from prometheus_client import CollectorRegistry, Gauge, generate_latest

    from retail.api.app import set_version_gauge

    registry = CollectorRegistry()
    gauge = Gauge(
        "retail_test_version_single",
        "x",
        ["model_version", "source"],
        registry=registry,
    )
    set_version_gauge(gauge, None, ("1", "mlflow"), multiprocess=False)
    set_version_gauge(gauge, ("1", "mlflow"), ("2", "mlflow"), multiprocess=False)
    text = generate_latest(registry).decode("utf-8")
    assert 'model_version="1"' not in text
    assert 'model_version="2"' in text


def test_set_version_gauge_zeros_previous_when_multiprocess_flag():
    from prometheus_client import CollectorRegistry, Gauge, generate_latest

    from retail.api.app import set_version_gauge

    registry = CollectorRegistry()
    gauge = Gauge(
        "retail_test_version_multi",
        "x",
        ["model_version", "source"],
        registry=registry,
    )
    previous = set_version_gauge(gauge, None, ("1", "mlflow"), multiprocess=True)
    set_version_gauge(gauge, previous, ("2", "mlflow"), multiprocess=True)
    text = generate_latest(registry).decode("utf-8")
    lines = {
        line.rsplit("{", 1)[-1]
        for line in text.splitlines()
        if line.startswith("retail_test_version_multi{")
    }
    assert any(line.startswith('model_version="1"') and line.endswith(" 0.0") for line in lines)
    assert any(line.startswith('model_version="2"') and line.endswith(" 1.0") for line in lines)
