from datetime import date
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from retail.features.panel import NUMERIC_FEATURES

ISO_WEEK = r"^\d{4}-W\d{2}$"


def _calendar_iso_week(week: str) -> str:
    year_s, week_s = week.split("-W", 1)
    date.fromisocalendar(int(year_s), int(week_s), 1)
    return week


IsoWeek = Annotated[str, Field(pattern=ISO_WEEK), AfterValidator(_calendar_iso_week)]


class PredictRequest(BaseModel):
    """Prédiction métier : le serveur calcule les lags depuis le panel."""

    model_config = ConfigDict(extra="forbid")

    StockCode: str = Field(min_length=1)
    week: IsoWeek


class FeaturePredictRequest(BaseModel):
    """Vecteur déjà calculé (debug / rétrocompatibilité)."""

    model_config = ConfigDict(extra="forbid")

    StockCode: str = Field(min_length=1)
    lag_1: float
    lag_2: float
    lag_4: float
    ma_4: float
    price_lag_1: float
    week_of_year: int = Field(ge=1, le=53)
    month: int = Field(ge=1, le=12)


class PredictResponse(BaseModel):
    StockCode: str
    prediction: float
    model_version: str
    source: str
    prediction_id: str
    week: str | None = None


class BatchPredictRequest(BaseModel):
    week: IsoWeek


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str
    source: str
    week: str | None = None


class FeatureBatchPredictRequest(BaseModel):
    items: list[FeaturePredictRequest] = Field(min_length=1)


class FeedbackRequest(BaseModel):
    StockCode: str = Field(min_length=1)
    iso_week: IsoWeek
    actual_qty: float = Field(ge=0)
    predicted_qty: float | None = Field(default=None, ge=0)
    model_version: str | None = None
    prediction_id: UUID


class AlertIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    kind: str = Field(min_length=1)
    message: str = Field(min_length=1)
    week: str | None = None
    flow_run: str | None = None


class ModelInfo(BaseModel):
    model_version: str
    source: str
    degraded: bool
    registry_ok: bool = True
    registered_model: str | None = None
    n_skus: int
    skus: list[str] = Field(default_factory=list)
    features: list[str] = NUMERIC_FEATURES
    production_wape: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    source: str
    degraded: bool
    registry_ok: bool = True
    db_ok: bool = True
    db_backend: str = "sqlite"
