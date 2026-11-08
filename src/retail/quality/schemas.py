"""Deux schémas Pandera : factures brutes d'un lot, puis panel nettoyé."""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import INT64, DataFrameModel, Field, dataframe_check
from pandera.typing import Series

from retail.common.config import QualityCfg
from retail.quality.errors import QualityError


def iso_week_bounds(week: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Lundi 00:00 inclus, lundi suivant exclus pour une étiquette ISO ``AAAA-Www``."""
    try:
        year_s, week_s = week.split("-W")
        start = pd.Timestamp.fromisocalendar(int(year_s), int(week_s), 1)
    except (ValueError, AttributeError) as exc:
        raise QualityError(f"Étiquette de semaine invalide : {week!r}") from exc
    return start, start + pd.Timedelta(days=7)


def invoice_schema(week: str, quality: QualityCfg) -> type[DataFrameModel]:
    week_start, week_end = iso_week_bounds(week)
    min_rows = quality.min_rows_per_batch
    max_share = quality.max_share_non_positive_price

    class InvoiceSchema(DataFrameModel):
        """Factures brutes d'une semaine ISO, avant nettoyage métier."""

        InvoiceNo: Series[str] = Field(str_length={"min_value": 1})
        StockCode: Series[str] = Field(str_length={"min_value": 1})
        Description: Series[str] = Field(nullable=True)
        Quantity: Series[int]
        InvoiceDate: Series[pd.Timestamp]
        UnitPrice: Series[float]
        CustomerID: Series[INT64] = Field(nullable=True)
        Country: Series[str] = Field(nullable=True)

        class Config:
            strict = True
            coerce = True

        @dataframe_check
        def volume_minimum(cls, df: pd.DataFrame) -> bool:
            return len(df) >= min_rows

        @dataframe_check
        def dates_dans_la_semaine(cls, df: pd.DataFrame) -> Series[bool]:
            return (df["InvoiceDate"] >= week_start) & (df["InvoiceDate"] < week_end)

        @dataframe_check
        def part_prix_non_positifs(cls, df: pd.DataFrame) -> bool:
            if df.empty:
                return False
            return float((df["UnitPrice"] <= 0).mean()) <= max_share

    InvoiceSchema.__name__ = "InvoiceSchema"
    return InvoiceSchema


def panel_schema(week: str) -> type[DataFrameModel]:
    class PanelSchema(DataFrameModel):
        """Une ligne par SKU pour la semaine, après nettoyage : quantités ≥ 0."""

        StockCode: Series[str] = Field(str_length={"min_value": 1})
        iso_week: Series[str] = Field(isin=[week])
        quantity: Series[int] = Field(ge=0)
        unit_price: Series[float] = Field(ge=0)
        revenue: Series[float] = Field(ge=0)
        n_lines: Series[int] = Field(gt=0)
        n_invoices: Series[int] = Field(gt=0)

        class Config:
            strict = True
            coerce = True
            unique = ["StockCode", "iso_week"]

        @dataframe_check
        def au_moins_un_sku(cls, df: pd.DataFrame) -> bool:
            return len(df) >= 1

    PanelSchema.__name__ = "PanelSchema"
    return PanelSchema


def run_schema(model: type[DataFrameModel], df: pd.DataFrame, label: str) -> None:
    try:
        model.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise QualityError(_format_schema_errors(label, exc)) from exc
    except pa.errors.SchemaError as exc:
        raise QualityError(f"{label} : {exc}") from exc


def _format_schema_errors(label: str, exc: pa.errors.SchemaErrors) -> str:
    failure = exc.failure_cases
    if failure is None or failure.empty:
        return f"{label} : schéma invalide"
    checks = sorted({str(c) for c in failure["check"].dropna().unique()})
    n_rows = int(failure["index"].nunique()) if "index" in failure.columns else len(failure)
    n_checks = len(checks)
    details = ", ".join(checks) or "schéma"
    return f"{label} : {n_checks} contrôle(s) en échec ({details}) ; {n_rows} ligne(s) concernées"
