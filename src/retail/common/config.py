"""Chargement et validation de la configuration.

Deux sources, volontairement séparées :
- les fichiers YAML de ``configs/`` (règles métier, chemins, source des données), versionnés ;
- les variables d'environnement / ``.env`` (ports, identifiants), jamais versionnées.

Les chemins relatifs sont résolus par rapport à la racine du dépôt, pas au CWD.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/retail/common/config.py → parents[3] = racine du dépôt
REPO_ROOT = Path(__file__).resolve().parents[3]


class ConfigError(RuntimeError):
    """Configuration absente ou invalide."""


class ProjectCfg(BaseModel):
    name: str
    random_seed: int = 42


class PathsCfg(BaseModel):
    raw_dir: Path
    interim_dir: Path
    batches_dir: Path
    reports_dir: Path
    models_dir: Path

    @model_validator(mode="after")
    def _resolve_against_repo(self) -> PathsCfg:
        for name in ("raw_dir", "interim_dir", "batches_dir", "reports_dir", "models_dir"):
            path = getattr(self, name)
            if not path.is_absolute():
                object.__setattr__(self, name, REPO_ROOT / path)
        return self


class BatchingCfg(BaseModel):
    frequency: str = Field(pattern="^weekly$")
    drop_partial_weeks: bool = True


class PanelCfg(BaseModel):
    top_n_products: int = Field(gt=0)
    warmup_weeks: int = Field(ge=5)


class QualityCfg(BaseModel):
    min_rows_per_batch: int = Field(gt=0)
    max_share_non_positive_price: float = Field(ge=0, le=1)
    country: str = "United Kingdom"
    max_abs_quantity: int = Field(gt=0)
    max_unit_price: float = Field(gt=0)
    min_distinct_skus: int = Field(gt=0)
    non_product_codes: list[str] = Field(min_length=1)


class TrainingCfg(BaseModel):
    test_weeks: int = Field(ge=1)
    lags: list[int] = Field(min_length=1)
    rolling_weeks: int = Field(ge=2)


class PromotionCfg(BaseModel):
    metric: str = Field(pattern="^(mae|wape)$")
    must_beat_baseline: bool = True
    registered_model_name: str = "retail-weekly-qty"


class OrchestrationCfg(BaseModel):
    interval_seconds: int = Field(default=600, ge=30)
    train_retries: int = Field(default=2, ge=0)
    train_retry_delay_seconds: int = Field(default=10, ge=1)


class AppConfig(BaseModel):
    project: ProjectCfg
    paths: PathsCfg
    batching: BatchingCfg
    panel: PanelCfg
    quality: QualityCfg
    training: TrainingCfg
    promotion: PromotionCfg
    orchestration: OrchestrationCfg = OrchestrationCfg()


class SourceCfg(BaseModel):
    name: str
    url: HttpUrl
    license: str
    doi: str
    archive_filename: str
    excel_filename: str
    sha256: str = ""


class DownloadCfg(BaseModel):
    timeout_seconds: int = Field(gt=0)
    max_attempts: int = Field(ge=1, le=5)


class DataConfig(BaseModel):
    source: SourceCfg
    download: DownloadCfg


class Settings(BaseSettings):
    """Paramètres d'exécution lus dans l'environnement (préfixe RETAIL_) ou .env."""

    model_config = SettingsConfigDict(
        env_prefix="RETAIL_", env_file=REPO_ROOT / ".env", extra="ignore"
    )

    config: Path = Path("configs/config.yaml")
    data_config: Path = Path("configs/data.yaml")
    log_level: str = "INFO"
    mlflow_tracking_uri: str = ""
    alert_webhook_url: str = "http://127.0.0.1:8001/alerts"
    api_url: str = "http://127.0.0.1:8001"
    forecast_timeout_seconds: float = 30.0
    alert_timeout_seconds: float = 3.0
    database_url: str = ""
    database_sync: bool = False


def _read_yaml(path: Path) -> dict:
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise ConfigError(f"Fichier de configuration introuvable : {path}")
    with path.open(encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    if not isinstance(content, dict):
        raise ConfigError(f"Configuration vide ou mal formée : {path}")
    return content


def load_app_config(path: Path | None = None) -> AppConfig:
    path = path or get_settings().config
    try:
        return AppConfig.model_validate(_read_yaml(path))
    except ValueError as exc:
        raise ConfigError(f"Configuration invalide ({path}) :\n{exc}") from exc


def load_data_config(path: Path | None = None) -> DataConfig:
    path = path or get_settings().data_config
    try:
        return DataConfig.model_validate(_read_yaml(path))
    except ValueError as exc:
        raise ConfigError(f"Configuration invalide ({path}) :\n{exc}") from exc


@lru_cache
def get_settings() -> Settings:
    return Settings()
