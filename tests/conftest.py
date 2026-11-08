"""Fixtures partagées : un mini-dataset au format UCI et une configuration isolée."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.models import infer_signature
from sklearn.dummy import DummyRegressor

from retail.common.config import AppConfig, DataConfig, load_app_config, load_data_config
from retail.registry.promote import register_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_runtime_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Les tests ne dépendent pas du `.env` de la machine (webhook, sondage, cache Settings)."""
    monkeypatch.setenv("RETAIL_CHAMPION_POLL_SECONDS", "0")
    monkeypatch.setenv("RETAIL_ALERT_WEBHOOK_URL", "http://127.0.0.1:8001/alerts")
    from retail.common.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_fake_retail(
    start: str = "2010-12-01", end: str = "2011-01-21", seed: int = 0
) -> pd.DataFrame:
    """Factures synthétiques au schéma UCI, avec quelques cas réels : annulations,
    quantités négatives, prix nuls, codes non-produits, client manquant."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end, freq="D")
    rows = []
    inv = 536365
    for day in days:
        for _ in range(3):
            inv += 1
            ts = day + pd.Timedelta(
                hours=int(rng.integers(8, 18)), minutes=int(rng.integers(0, 60))
            )
            for code in rng.choice(
                ["85123A", "71053", "84406B", "22423", "POST"], size=3, replace=False
            ):
                rows.append(
                    {
                        "InvoiceNo": str(inv),
                        "StockCode": str(code),
                        "Description": f"ITEM {code}",
                        "Quantity": int(rng.integers(1, 24)),
                        "InvoiceDate": ts,
                        "UnitPrice": float(rng.choice([1.25, 2.55, 3.39, 0.0])),
                        "CustomerID": float(rng.choice([17850.0, 13047.0, np.nan])),
                        "Country": "United Kingdom",
                    }
                )
    df = pd.DataFrame(rows)
    df.loc[0, ["InvoiceNo", "Quantity"]] = ["C536366", -2]
    return df


@pytest.fixture
def fake_retail() -> pd.DataFrame:
    return make_fake_retail()


@pytest.fixture
def fake_retail_long() -> pd.DataFrame:
    return make_fake_retail(end="2011-04-01")


@pytest.fixture
def tmp_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[AppConfig, DataConfig]:
    """Copie des configs réelles avec des chemins redirigés vers un répertoire temporaire."""
    cfg_dir = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", cfg_dir)
    app = yaml.safe_load((cfg_dir / "config.yaml").read_text(encoding="utf-8"))
    app["paths"] = {k: str(tmp_path / v) for k, v in app["paths"].items()}
    app["quality"]["min_rows_per_batch"] = 30
    app["quality"]["min_distinct_skus"] = 3
    app["quality"]["max_share_non_positive_price"] = 0.4
    (cfg_dir / "config.yaml").write_text(yaml.safe_dump(app), encoding="utf-8")
    data = yaml.safe_load((cfg_dir / "data.yaml").read_text(encoding="utf-8"))
    data["source"]["sha256"] = ""
    (cfg_dir / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setenv("RETAIL_CONFIG", str(cfg_dir / "config.yaml"))
    monkeypatch.setenv("RETAIL_DATA_CONFIG", str(cfg_dir / "data.yaml"))
    mlruns = tmp_path / "mlruns"
    mlruns.mkdir()
    monkeypatch.setenv("RETAIL_MLFLOW_TRACKING_URI", mlruns.resolve().as_uri())
    db_path = (tmp_path / "retail.sqlite").resolve().as_posix()
    monkeypatch.setenv("RETAIL_DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("RETAIL_DATABASE_SYNC", "true")
    # Prévision N+1 : jamais vers l'API locale réelle pendant les tests (port 9 = discard).
    monkeypatch.setenv("RETAIL_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("RETAIL_FORECAST_TIMEOUT_SECONDS", "0.5")
    from retail.common import config as config_module
    from retail.db.session import reset_engine

    config_module.get_settings.cache_clear()
    reset_engine()
    yield load_app_config(), load_data_config()
    reset_engine()
    config_module.get_settings.cache_clear()


def registry_xy() -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame(
        {
            "StockCode": ["AAA", "BBB", "AAA", "BBB"],
            "lag_1": [10.0, 4.0, 12.0, 5.0],
            "lag_2": [8.0, 3.0, 10.0, 4.0],
            "lag_4": [6.0, 2.0, 8.0, 3.0],
            "ma_4": [8.0, 3.0, 10.0, 4.0],
            "price_lag_1": [1.5, 2.0, 1.5, 2.0],
            "week_of_year": [10, 10, 11, 11],
            "month": [3, 3, 3, 3],
        }
    )
    y = np.array([10.0, 4.0, 12.0, 5.0])
    return frame, y


def registry_metrics(run_id: str, **extra) -> dict:
    payload = {
        "mlflow_run_id": run_id,
        "mae_model": 1.0,
        "mae_naive": 50.0,
        "beats_baseline": True,
        "features_hash": "abc",
        "data_hash": "def",
        "until_week": "2011-W20",
    }
    payload.update(extra)
    return payload


@pytest.fixture
def log_dummy_model():
    def _log(
        app,
        constant: float,
        skus: list[str] | None = None,
        *,
        model=None,
        x_frame=None,
        with_signature: bool = True,
    ) -> str:
        x_test, y = registry_xy()
        frame = x_test if x_frame is None else x_frame
        estimator = model
        if estimator is None:
            estimator = DummyRegressor(strategy="constant", constant=constant)
            estimator.fit(np.zeros((len(y), 1)), y)
        pred = estimator.predict(frame)
        mlflow.set_tracking_uri(os.environ["RETAIL_MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(app.project.name)
        app.paths.interim_dir.mkdir(parents=True, exist_ok=True)
        sku_path = app.paths.interim_dir / "top_skus.json"
        sku_path.write_text(json.dumps(skus or ["AAA", "BBB"]), encoding="utf-8")
        with mlflow.start_run() as run:
            mlflow.log_metrics({"mae_model": float(np.mean(np.abs(y - pred))), "mae_naive": 50.0})
            mlflow.set_tags({"test_weeks": "2011-W19,2011-W20"})
            mlflow.log_artifact(str(sku_path))
            extra = {"pip_requirements": ["scikit-learn"]}
            if with_signature:
                extra["signature"] = infer_signature(frame, pred)
                extra["input_example"] = frame.head(2)
            mlflow.sklearn.log_model(estimator, artifact_path="model", **extra)
            return run.info.run_id

    return _log


@pytest.fixture
def registry_frame() -> tuple[pd.DataFrame, np.ndarray]:
    return registry_xy()


@pytest.fixture
def make_registry_metrics():
    return registry_metrics


@pytest.fixture
def register_dummy(log_dummy_model):
    def _register(app, constant: float, skus: list[str] | None = None) -> dict:
        x_test, y_test = registry_xy()
        return register_run(
            app,
            registry_metrics(log_dummy_model(app, constant, skus=skus)),
            x_test=x_test,
            y_test=y_test,
        )

    return _register
