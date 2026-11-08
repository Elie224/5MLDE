"""Adresse de suivi MLflow : variable d'environnement, sinon ``mlruns/`` à la racine."""

from __future__ import annotations

from retail.common.config import REPO_ROOT, get_settings


def tracking_uri() -> str:
    settings = get_settings()
    if settings.mlflow_tracking_uri:
        return settings.mlflow_tracking_uri
    return (REPO_ROOT / "mlruns").resolve().as_uri()
