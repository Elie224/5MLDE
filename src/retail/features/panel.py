"""Panel dense, top 50 figé sur le warmup, variables de retard sans fuite."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from retail.common.config import AppConfig, TrainingCfg
from retail.common.logging import get_logger
from retail.data.make_batches import load_manifest
from retail.quality.errors import QualityError
from retail.quality.schemas import iso_week_bounds
from retail.training.errors import TrainError

log = get_logger(__name__)

TOP_SKUS_FILENAME = "top_skus.json"
FEATURES_FILENAME = "features.parquet"
NUMERIC_FEATURES = ["lag_1", "lag_2", "lag_4", "ma_4", "price_lag_1", "week_of_year", "month"]
FEATURE_COLUMNS = ["StockCode", *NUMERIC_FEATURES]


def ordered_weeks(weeks: list[str]) -> list[str]:
    return sorted(weeks)


def weeks_until(weeks: list[str], until_week: str | None) -> list[str]:
    ordered = ordered_weeks(weeks)
    if until_week is None:
        return ordered
    kept = [week for week in ordered if week <= until_week]
    if not kept:
        raise TrainError(f"Aucune semaine du manifeste ≤ {until_week}.")
    return kept


def calendar_span(weeks: list[str]) -> list[str]:
    """Toutes les semaines ISO du calendrier entre la première et la dernière (trous à 0)."""
    ordered = ordered_weeks(weeks)
    if not ordered:
        return []
    start, end = iso_week_bounds(ordered[0])[0], iso_week_bounds(ordered[-1])[0]
    labels: list[str] = []
    cursor = start
    while cursor <= end:
        iso = cursor.isocalendar()
        labels.append(f"{iso.year}-W{iso.week:02d}")
        cursor += pd.Timedelta(days=7)
    return labels


def assert_quality_passed(app: AppConfig, until_week: str | None = None) -> list[str]:
    """Refuse l'entraînement si un lot du manifeste n'a pas un rapport qualité OK."""
    batches_dir = app.paths.batches_dir
    manifest_path = batches_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{manifest_path} introuvable : lancer `retail data` puis `retail quality`."
        )
    weeks = weeks_until([info.week for info in load_manifest(batches_dir)], until_week)
    reports = app.paths.reports_dir / "quality"
    for week in weeks:
        path = reports / f"{week}.json"
        if not path.is_file():
            raise QualityError(f"Rapport qualité manquant pour {week} : lancer `retail quality`.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            raise QualityError(
                f"Lot {week} rejeté par la qualité ({payload.get('error')}) : pas d'entraînement."
            )
    return weeks


def load_panels(panel_dir: Path, weeks: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing_files: list[str] = []
    for week in weeks:
        path = panel_dir / f"{week}.parquet"
        if not path.is_file():
            missing_files.append(week)
            continue
        frames.append(pd.read_parquet(path))
    if missing_files:
        raise FileNotFoundError(f"Panels absents pour {missing_files} : lancer `retail quality`.")
    if not frames:
        raise FileNotFoundError(f"Aucun panel dans {panel_dir} : lancer `retail quality`.")
    panel = pd.concat(frames, ignore_index=True)
    panel["StockCode"] = panel["StockCode"].astype(str)
    panel["iso_week"] = panel["iso_week"].astype(str)
    return panel.loc[panel["iso_week"].isin(weeks)].reset_index(drop=True)


def warmup_week_labels(weeks: list[str], n: int) -> list[str]:
    ordered = ordered_weeks(weeks)
    if len(ordered) < n:
        raise TrainError(f"Pas assez de semaines ({len(ordered)}) pour warmup_weeks={n}.")
    return ordered[:n]


def freeze_top_skus(panel: pd.DataFrame, warmup_weeks: list[str], top_n: int) -> list[str]:
    """Top N calculé uniquement sur les semaines de démarrage (anti-fuite)."""
    part = panel.loc[panel["iso_week"].isin(warmup_weeks)]
    if part.empty:
        raise TrainError("Aucune ligne de panel sur les semaines de warmup.")
    totals = (
        part.groupby("StockCode", as_index=False)["quantity"]
        .sum()
        .sort_values(["quantity", "StockCode"], ascending=[False, True])
    )
    skus = totals["StockCode"].head(top_n).tolist()
    if len(skus) < top_n:
        raise TrainError(f"Seulement {len(skus)} SKU sur le warmup (top_n={top_n}).")
    return skus


def densify(panel: pd.DataFrame, skus: list[str], weeks: list[str]) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product([skus, weeks], names=["StockCode", "iso_week"]).to_frame(
        index=False
    )
    cols = ["StockCode", "iso_week", "quantity", "unit_price"]
    merged = grid.merge(panel[cols], on=["StockCode", "iso_week"], how="left")
    merged["quantity"] = merged["quantity"].fillna(0).astype("int64")
    merged["unit_price"] = merged.groupby("StockCode", sort=False)["unit_price"].ffill()
    return merged


def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    starts = frame["iso_week"].map(lambda w: iso_week_bounds(w)[0])
    frame = frame.copy()
    frame["week_of_year"] = starts.map(lambda t: int(t.isocalendar().week)).astype("int64")
    frame["month"] = starts.map(lambda t: int(t.month)).astype("int64")
    return frame


def add_lags_and_target(dense: pd.DataFrame, training: TrainingCfg) -> pd.DataFrame:
    """Lags −1/−2/−4 et MA : uniquement le passé. ``y`` = quantité de la semaine de la ligne."""
    lags = training.lags
    window = training.rolling_weeks
    out = dense.sort_values(["StockCode", "iso_week"]).copy()
    qty = out.groupby("StockCode", sort=False)["quantity"]
    for lag in lags:
        out[f"lag_{lag}"] = qty.shift(lag)
    out["ma_4"] = qty.transform(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
    out["price_lag_1"] = out.groupby("StockCode", sort=False)["unit_price"].shift(1)
    out["y"] = out["quantity"].astype("float64")
    needed = [f"lag_{lag}" for lag in lags] + ["ma_4", "price_lag_1"]
    return out.dropna(subset=needed).reset_index(drop=True)


def build_feature_frame(
    panel: pd.DataFrame, skus: list[str], weeks: list[str], warmup: list[str], training: TrainingCfg
) -> pd.DataFrame:
    dense = add_calendar(densify(panel, skus, calendar_span(weeks)))
    featured = add_lags_and_target(dense, training)
    ordered = ordered_weeks(weeks)
    warmup_end = ordered.index(warmup[-1])
    if warmup_end + 1 >= len(ordered):
        raise TrainError("Aucune semaine après le warmup : impossible de constituer une cible.")
    keep_from = set(ordered[warmup_end + 1 :])
    featured = featured.loc[featured["iso_week"].isin(keep_from)].reset_index(drop=True)
    if featured.empty:
        raise TrainError(
            "Aucune ligne de features après warmup et lags : historique trop court "
            "(il faut au moins une semaine après le warmup)."
        )
    return featured


def write_top_skus(path: Path, skus: list[str], warmup: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"warmup_weeks": warmup, "skus": skus, "n": len(skus)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def build_features(app: AppConfig, until_week: str | None = None) -> tuple[pd.DataFrame, list[str]]:
    weeks = assert_quality_passed(app, until_week=until_week)
    panel = load_panels(app.paths.interim_dir / "panel", weeks)
    warmup = warmup_week_labels(weeks, app.panel.warmup_weeks)
    skus = freeze_top_skus(panel, warmup, app.panel.top_n_products)
    if len(skus) != app.panel.top_n_products:
        raise TrainError(
            f"Top produits : {len(skus)} SKU figés, {app.panel.top_n_products} attendus."
        )
    featured = build_feature_frame(panel, skus, weeks, warmup, app.training)
    n_featured = int(featured["StockCode"].nunique())
    if n_featured != app.panel.top_n_products:
        raise TrainError(
            f"Features : {n_featured} SKU dans le panel, {app.panel.top_n_products} attendus."
        )
    dest = app.paths.interim_dir / FEATURES_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(dest, index=False)
    write_top_skus(app.paths.interim_dir / TOP_SKUS_FILENAME, skus, warmup)
    log.info(
        "Features écrites",
        extra={
            "rows": len(featured),
            "skus": len(skus),
            "warmup_first": warmup[0],
            "warmup_last": warmup[-1],
            "path": str(dest),
        },
    )
    return featured, skus
