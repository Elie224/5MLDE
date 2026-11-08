"""Construction des variables de prédiction à partir du panel (même code que l'entraînement)."""

from __future__ import annotations

import threading
import time

import pandas as pd

from retail.common.config import AppConfig
from retail.features.panel import (
    FEATURE_COLUMNS,
    add_calendar,
    add_lags_and_target,
    calendar_span,
    densify,
    ordered_weeks,
)
from retail.quality.schemas import iso_week_bounds

_CACHE_LOCK = threading.Lock()
_FEATURE_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
_CACHE_MAX = 8
_PANEL_TTL_S = 5.0
_panel_listing_cache: tuple[float, str, tuple[str, ...]] | None = None


class FeatureLookupError(ValueError):
    """Historique insuffisant, semaine absente ou horizon dépassé."""


def _scan_panel_dir(app: AppConfig) -> tuple[str, tuple[str, ...]]:
    panel_dir = app.paths.interim_dir / "panel"
    if not panel_dir.is_dir():
        return "0", ()
    paths = list(panel_dir.glob("*.parquet"))
    mtimes = [path.stat().st_mtime_ns for path in paths]
    fingerprint = str(max(mtimes) if mtimes else 0)
    weeks = tuple(ordered_weeks([path.stem for path in paths]))
    return fingerprint, weeks


def _list_panel(app: AppConfig) -> tuple[str, list[str]]:
    """Liste du dossier panel, rafraîchie au plus toutes les 5 s (I/O bind Windows)."""
    global _panel_listing_cache
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _panel_listing_cache
        if cached is not None and now - cached[0] < _PANEL_TTL_S:
            return cached[1], list(cached[2])
    fingerprint, weeks = _scan_panel_dir(app)
    with _CACHE_LOCK:
        _panel_listing_cache = (now, fingerprint, weeks)
    return fingerprint, list(weeks)


def panel_weeks(app: AppConfig) -> list[str]:
    _fingerprint, weeks = _list_panel(app)
    if not weeks:
        raise FeatureLookupError("Aucun panel : lancer `retail quality`.")
    return weeks


def next_iso_week(week: str) -> str:
    nxt = iso_week_bounds(week)[0] + pd.Timedelta(days=7)
    iso = nxt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def max_predict_week(app: AppConfig) -> str:
    return next_iso_week(panel_weeks(app)[-1])


def assert_week_horizon(app: AppConfig, week: str) -> None:
    limit = max_predict_week(app)
    if week > limit:
        raise FeatureLookupError(f"horizon dépassé : {week} au-delà de {limit}")


def clear_feature_cache() -> None:
    global _panel_listing_cache
    with _CACHE_LOCK:
        _FEATURE_CACHE.clear()
        _panel_listing_cache = None


def panel_fingerprint(app: AppConfig) -> str:
    fingerprint, _weeks = _list_panel(app)
    return fingerprint


def _load_history(app: AppConfig, until_week: str) -> pd.DataFrame:
    panel_dir = app.paths.interim_dir / "panel"
    weeks = [week for week in panel_weeks(app) if week <= until_week]
    if not weeks:
        raise FeatureLookupError(f"Aucun panel ≤ {until_week}.")
    frames = [
        pd.read_parquet(panel_dir / f"{week}.parquet", engine="pyarrow", memory_map=False)
        for week in weeks
    ]
    panel = pd.concat(frames, ignore_index=True)
    panel["StockCode"] = panel["StockCode"].astype(str)
    panel["iso_week"] = panel["iso_week"].astype(str)
    return panel


def _span_until(history_weeks: list[str], week: str) -> list[str]:
    start = min(history_weeks + [week])
    labels: list[str] = []
    cursor = iso_week_bounds(start)[0]
    end = iso_week_bounds(week)[0]
    while cursor <= end:
        iso = cursor.isocalendar()
        labels.append(f"{iso.year}-W{iso.week:02d}")
        cursor += pd.Timedelta(days=7)
    if not labels:
        return calendar_span([week])
    return labels


def features_for_week(
    app: AppConfig,
    week: str,
    skus: list[str],
    *,
    model_version: str | None = None,
) -> pd.DataFrame:
    """Une ligne par SKU pour la semaine demandée, lags calculés côté serveur."""
    if not skus:
        raise FeatureLookupError("Liste de produits du champion vide.")
    assert_week_horizon(app, week)
    cache_key = (week, model_version or "", panel_fingerprint(app))
    if model_version:
        with _CACHE_LOCK:
            hit = _FEATURE_CACHE.get(cache_key)
        if hit is not None:
            return hit.copy()
    panel = _load_history(app, week)
    history_weeks = ordered_weeks(panel["iso_week"].unique().tolist())
    span = _span_until(history_weeks, week)
    dense = add_calendar(densify(panel, skus, span))
    featured = add_lags_and_target(dense, app.training)
    rows = featured.loc[featured["iso_week"] == week].copy()
    if rows.empty:
        raise FeatureLookupError(
            f"Historique insuffisant pour prédire {week} (lags / moyenne mobile)."
        )
    missing = [sku for sku in skus if sku not in set(rows["StockCode"].astype(str))]
    if missing:
        raise FeatureLookupError(f"Pas de features pour {missing} à la semaine {week}.")
    rows = rows.sort_values("StockCode").reset_index(drop=True)
    if model_version:
        with _CACHE_LOCK:
            _FEATURE_CACHE[cache_key] = rows
            while len(_FEATURE_CACHE) > _CACHE_MAX:
                oldest = next(iter(_FEATURE_CACHE))
                if oldest == cache_key:
                    break
                del _FEATURE_CACHE[oldest]
    return rows.copy() if model_version else rows


def feature_row(
    app: AppConfig,
    stock_code: str,
    week: str,
    skus: list[str],
    *,
    model_version: str | None = None,
) -> pd.DataFrame:
    rows = features_for_week(app, week, skus, model_version=model_version)
    part = rows.loc[rows["StockCode"].astype(str) == stock_code]
    if part.empty:
        raise FeatureLookupError(f"SKU {stock_code} absent du panel pour {week}.")
    return part[FEATURE_COLUMNS].reset_index(drop=True)
