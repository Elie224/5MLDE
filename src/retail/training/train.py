"""Split temporel, baseline naïve, pipeline sklearn, run MLflow local."""

from __future__ import annotations

import json
import subprocess

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retail.common.config import REPO_ROOT, AppConfig
from retail.common.logging import get_logger
from retail.common.mlflow_uri import tracking_uri
from retail.data.download import content_hash, sha256_file
from retail.features.panel import (
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TOP_SKUS_FILENAME,
    build_features,
    ordered_weeks,
)
from retail.training.errors import TrainError
from retail.training.metrics import mae, wape

log = get_logger(__name__)

METRICS_FILENAME = "metrics.json"


def temporal_split(frame: pd.DataFrame, test_weeks: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    weeks = ordered_weeks(frame["iso_week"].unique().tolist())
    if len(weeks) <= test_weeks:
        raise TrainError(
            f"Pas assez de semaines d'origine ({len(weeks)}) pour test_weeks={test_weeks}."
        )
    holdout = set(weeks[-test_weeks:])
    train = frame.loc[~frame["iso_week"].isin(holdout)].copy()
    test = frame.loc[frame["iso_week"].isin(holdout)].copy()
    if train.empty or test.empty:
        raise TrainError("Split temporel vide : réduire test_weeks ou warmup_weeks.")
    if max(ordered_weeks(train["iso_week"].unique().tolist())) >= min(
        ordered_weeks(test["iso_week"].unique().tolist())
    ):
        raise TrainError("Fuite temporelle : le train chevauche le test.")
    return train, test


def make_pipeline() -> Pipeline:
    transform = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("sku", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["StockCode"]),
        ]
    )
    return Pipeline(
        [
            ("prep", transform),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _data_hash(app: AppConfig) -> str | None:
    """Empreinte de l'archive / parquet UCI **complet**, pas de la tranche ``until_week``.

    La semaine d'entraînement est le tag ``until_week`` ; le panel utilisé est
    ``features_hash``.
    """
    ingest = app.paths.interim_dir / "ingestion_manifest.json"
    if ingest.is_file():
        payload = json.loads(ingest.read_text(encoding="utf-8"))
        return payload.get("parquet_content_hash") or payload.get("archive_sha256")
    zip_path = app.paths.raw_dir / "online_retail.zip"
    if zip_path.is_file():
        return sha256_file(zip_path)
    return None


def train_and_evaluate(
    app: AppConfig, until_week: str | None = None, *, register: bool = True
) -> dict:
    if register and until_week:
        from retail.registry.promote import find_complete_week_version

        done = find_complete_week_version(app, until_week)
        if done is not None:
            log.info(
                "Entraînement ignoré : version déjà complète",
                extra={"week": until_week, "version": done.version},
            )
            return {
                "trained": False,
                "train_skipped": (f"version {done.version} déjà entraînée jusqu'à {until_week}"),
                "until_week": until_week,
                "mlflow_run_id": None,
                "registry": {
                    "status": "reused",
                    "model_version": str(done.version),
                },
                "model_version": str(done.version),
            }
    featured, skus = build_features(app, until_week=until_week)
    train, test = temporal_split(featured, app.training.test_weeks)
    y_train = train["y"].to_numpy(dtype="float64")
    y_test = test["y"].to_numpy(dtype="float64")
    x_train = train[FEATURE_COLUMNS]
    x_test = test[FEATURE_COLUMNS]

    pipe = make_pipeline()
    pipe.fit(x_train, y_train)
    pred = np.clip(pipe.predict(x_test), 0, None)
    naive = np.clip(test["lag_1"].to_numpy(dtype="float64"), 0, None)
    example = x_train.head(3)

    metrics = {
        "mae_model": mae(y_test, pred),
        "wape_model": wape(y_test, pred),
        "mae_naive": mae(y_test, naive),
        "wape_naive": wape(y_test, naive),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_skus": len(skus),
        "until_week": until_week,
        "test_weeks": ordered_weeks(test["iso_week"].unique().tolist()),
        "train_weeks": ordered_weeks(train["iso_week"].unique().tolist()),
        "beats_baseline": mae(y_test, pred) < mae(y_test, naive),
        "features_hash": content_hash(featured[FEATURE_COLUMNS + ["y"]]),
        "data_hash": _data_hash(app),
        "git_commit": _git_commit(),
        "top_skus": skus,
    }

    reports_dir = app.paths.reports_dir / "training"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / METRICS_FILENAME
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(app.project.name)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "algorithm": "ridge",
                "alpha": 1.0,
                "top_n": app.panel.top_n_products,
                "warmup_weeks": app.panel.warmup_weeks,
                "test_weeks": app.training.test_weeks,
                "lags": ",".join(str(x) for x in app.training.lags),
                "until_week": until_week or "",
            }
        )
        mlflow.log_metrics(
            {
                "mae_model": metrics["mae_model"],
                "wape_model": metrics["wape_model"],
                "mae_naive": metrics["mae_naive"],
                "wape_naive": metrics["wape_naive"],
            }
        )
        mlflow.set_tags(
            {
                "features_hash": metrics["features_hash"],
                "beats_baseline": str(metrics["beats_baseline"]).lower(),
                "train_weeks": ",".join(metrics["train_weeks"]),
                "test_weeks": ",".join(metrics["test_weeks"]),
                **({"data_hash": metrics["data_hash"]} if metrics["data_hash"] else {}),
                **({"git_commit": metrics["git_commit"]} if metrics["git_commit"] else {}),
            }
        )
        top_skus_path = app.paths.interim_dir / TOP_SKUS_FILENAME
        if top_skus_path.is_file():
            mlflow.log_artifact(str(top_skus_path))
        mlflow.log_artifact(str(metrics_path))
        signature = infer_signature(example, pipe.predict(example))
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=signature,
            input_example=example,
            pip_requirements=["scikit-learn"],
        )
        metrics["mlflow_run_id"] = run.info.run_id

    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if register:
        from mlflow.tracking import MlflowClient

        from retail.registry.promote import register_run

        metrics["registry"] = register_run(app, metrics, x_test=x_test, y_test=y_test)
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        client = MlflowClient(tracking_uri=tracking_uri())
        run_id = metrics["mlflow_run_id"]
        registry = metrics["registry"]
        client.set_tag(run_id, "registry_status", registry["status"])
        if registry.get("decision"):
            client.set_tag(run_id, "registry_decision", registry["decision"])
        for key in ("mae_challenger_same_test", "mae_champion_same_test"):
            if key in registry:
                client.log_metric(run_id, key, float(registry[key]))
        client.log_artifact(run_id, str(metrics_path))
    log.info(
        "Entraînement terminé",
        extra={
            "mae_model": round(metrics["mae_model"], 3),
            "mae_naive": round(metrics["mae_naive"], 3),
            "wape_model": round(metrics["wape_model"], 4),
            "beats_baseline": metrics["beats_baseline"],
            "until_week": until_week,
            "mlflow_run_id": metrics["mlflow_run_id"],
        },
    )
    return metrics
