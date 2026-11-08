"""Nettoyage métier tracé : chaque ligne écartée a une raison unique (premier motif)."""

from __future__ import annotations

import pandas as pd

from retail.common.config import QualityCfg

DROP_REASONS = (
    "not_united_kingdom",
    "cancellation",
    "non_positive_quantity",
    "non_positive_price",
    "non_product_code",
    "outlier",
)


def _non_product_mask(codes: pd.Series, listed: list[str]) -> pd.Series:
    listed_set = {c.strip().upper() for c in listed}
    upper = codes.astype(str).str.strip().str.upper()
    return upper.isin(listed_set) | upper.str.startswith("GIFT")


def drop_reasons(df: pd.DataFrame, quality: QualityCfg) -> pd.Series:
    """Motif d'exclusion, ou chaîne vide si la ligne est conservée."""
    if df.empty:
        return pd.Series(dtype="string")
    country = quality.country
    not_uk = df["Country"].astype("string").str.strip() != country
    masks = [
        not_uk.fillna(True),
        df["InvoiceNo"].astype(str).str.upper().str.startswith("C").fillna(False),
        (df["Quantity"] <= 0).fillna(False),
        (df["UnitPrice"] <= 0).fillna(False),
        _non_product_mask(df["StockCode"], quality.non_product_codes).fillna(False),
        (
            (df["Quantity"].abs() > quality.max_abs_quantity)
            | (df["UnitPrice"] > quality.max_unit_price)
        ).fillna(False),
    ]
    reason = pd.Series("", index=df.index, dtype="object")
    for name, mask in zip(DROP_REASONS, masks, strict=True):
        reason = reason.where(~(mask & (reason == "")), name)
    return reason.astype("string")


def clean_invoices(df: pd.DataFrame, quality: QualityCfg) -> tuple[pd.DataFrame, dict[str, int]]:
    reasons = drop_reasons(df, quality)
    counts = {name: int((reasons == name).sum()) for name in DROP_REASONS}
    kept = df.loc[reasons == ""].copy().reset_index(drop=True)
    return kept, counts


def to_week_panel(cleaned: pd.DataFrame, week: str) -> pd.DataFrame:
    """Agrège les factures nettoyées : 1 ligne / SKU / semaine, quantités ≥ 0."""
    if cleaned.empty:
        return pd.DataFrame(
            columns=[
                "StockCode",
                "iso_week",
                "quantity",
                "unit_price",
                "revenue",
                "n_lines",
                "n_invoices",
            ]
        )
    work = cleaned.copy()
    work["line_revenue"] = work["Quantity"].astype("int64") * work["UnitPrice"].astype("float64")
    grouped = work.groupby("StockCode", sort=True, as_index=False)
    panel = grouped.agg(
        quantity=("Quantity", "sum"),
        revenue=("line_revenue", "sum"),
        n_lines=("InvoiceNo", "count"),
        n_invoices=("InvoiceNo", "nunique"),
    )
    panel["iso_week"] = week
    panel["unit_price"] = (panel["revenue"] / panel["quantity"].replace(0, pd.NA)).astype("float64")
    panel["unit_price"] = panel["unit_price"].fillna(0.0)
    panel["quantity"] = panel["quantity"].astype("int64")
    panel["n_lines"] = panel["n_lines"].astype("int64")
    panel["n_invoices"] = panel["n_invoices"].astype("int64")
    return panel[
        ["StockCode", "iso_week", "quantity", "unit_price", "revenue", "n_lines", "n_invoices"]
    ]
