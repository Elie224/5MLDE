"""Enchaînement bloquant : schéma factures → nettoyage → schéma panel → rapport JSON."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from retail.common.config import AppConfig
from retail.common.logging import get_logger
from retail.data.download import RAW_COLUMNS, content_hash
from retail.data.make_batches import load_manifest
from retail.quality.clean import clean_invoices, to_week_panel
from retail.quality.errors import QualityError, QualityGateBypassError
from retail.quality.schemas import invoice_schema, panel_schema, run_schema

log = get_logger(__name__)

INVALID_DEMO_WEEK = "2099-W01"


def _reports_dir(app: AppConfig) -> Path:
    path = app.paths.reports_dir / "quality"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _week_output_paths(app: AppConfig, week: str) -> tuple[Path, Path]:
    return (
        app.paths.interim_dir / "cleaned" / f"{week}.parquet",
        app.paths.interim_dir / "panel" / f"{week}.parquet",
    )


def _unlink_retry(path: Path, *, attempts: int = 10, delay: float = 0.2) -> None:
    """Windows : pyarrow/Docker peuvent garder le parquet ouvert (WinError 32)."""
    last: PermissionError | None = None
    for _ in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError as exc:
            last = exc
            time.sleep(delay)
    if last is not None:
        raise last


def _clear_week_outputs(app: AppConfig, week: str) -> None:
    for path in _week_output_paths(app, week):
        _unlink_retry(path)


def _batch_content_hash(app: AppConfig, week: str, invoices: pd.DataFrame) -> str:
    manifest_path = app.paths.batches_dir / "manifest.json"
    if manifest_path.is_file():
        for info in load_manifest(app.paths.batches_dir):
            if info.week == week:
                return info.content_hash
    return content_hash(invoices)


def validate_batch(
    app: AppConfig,
    week: str,
    invoices: pd.DataFrame | None = None,
    *,
    report_dir: Path | None = None,
) -> dict:
    """Valide un lot. Écrit le rapport même en échec, n'écrit panel/nettoyé qu'en succès."""
    quality = app.quality
    _clear_week_outputs(app, week)
    batch_path = app.paths.batches_dir / f"{week}.parquet"
    if invoices is None:
        if not batch_path.is_file():
            raise FileNotFoundError(f"Lot introuvable : {batch_path}")
        invoices = pd.read_parquet(batch_path)
    missing = [c for c in RAW_COLUMNS if c not in invoices.columns]
    report: dict = {
        "week": week,
        "status": "failed",
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "batch_content_hash": _batch_content_hash(app, week, invoices),
        "quality": quality.model_dump(mode="json"),
        "rows_in": int(len(invoices)),
        "invoice_schema": "pending",
        "dropped": {},
        "rows_out": 0,
        "panel_rows": 0,
        "error": None,
    }
    report_path = (report_dir or _reports_dir(app)) / f"{week}.json"
    try:
        if missing:
            raise QualityError(f"InvoiceSchema : colonnes manquantes {missing}")
        run_schema(invoice_schema(week, quality), invoices[RAW_COLUMNS], "InvoiceSchema")
        report["invoice_schema"] = "ok"
        cleaned, dropped = clean_invoices(invoices[RAW_COLUMNS], quality)
        report["dropped"] = dropped
        report["rows_out"] = int(len(cleaned))
        panel = to_week_panel(cleaned, week)
        run_schema(panel_schema(week), panel, "PanelSchema")
        report["panel_rows"] = int(len(panel))
        report["status"] = "ok"
        cleaned_path, panel_path = _week_output_paths(app, week)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(cleaned_path, index=False)
        panel.to_parquet(panel_path, index=False)
    except QualityError as exc:
        report["error"] = str(exc)
        _write_report(report_path, report)
        log.error("Lot rejeté", extra={"week": week, "error": str(exc)})
        raise
    _write_report(report_path, report)
    log.info(
        "Lot validé",
        extra={
            "week": week,
            "rows_in": report["rows_in"],
            "rows_out": report["rows_out"],
            "panel_rows": report["panel_rows"],
        },
    )
    return report


def validate_all(app: AppConfig, week: str | None = None) -> list[dict]:
    batches_dir = app.paths.batches_dir
    if week:
        return [validate_batch(app, week)]
    manifest_path = batches_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} introuvable : lancer d'abord `retail data`.")
    reports = [validate_batch(app, info.week) for info in load_manifest(batches_dir)]
    _assert_enough_skus(app)
    return reports


def _assert_enough_skus(app: AppConfig) -> None:
    panel_dir = app.paths.interim_dir / "panel"
    files = sorted(panel_dir.glob("*.parquet"))
    if not files:
        raise QualityError("PanelSchema : aucun panel écrit")
    skus: set[str] = set()
    for path in files:
        skus.update(pd.read_parquet(path, columns=["StockCode"])["StockCode"].astype(str))
    needed = app.quality.min_distinct_skus
    if len(skus) < needed:
        raise QualityError(
            f"PanelSchema : {len(skus)} SKU distincts après nettoyage (minimum {needed})"
        )
    log.info("Couverture SKU suffisante", extra={"n_skus": len(skus), "minimum": needed})


def generate_invalid_batch() -> pd.DataFrame:
    """Lot de démo : trop petit, dates hors semaine, tous les prix nuls."""
    n = 8
    return pd.DataFrame(
        {
            "InvoiceNo": [str(i) for i in range(n)],
            "StockCode": ["85123A"] * n,
            "Description": ["WHITE HANGING HEART"] * n,
            "Quantity": [1] * n,
            "InvoiceDate": pd.to_datetime(["2012-06-01"] * n),
            "UnitPrice": [0.0] * n,
            "CustomerID": pd.array([None] * n, dtype="Int64"),
            "Country": ["United Kingdom"] * n,
        }
    )


def run_invalid_demo(app: AppConfig) -> None:
    """Écrit un lot corrompu ; succès = rejet (code 0). Échec = porte ouverte (code 1)."""
    dest_dir = app.paths.reports_dir / "invalid_demo"
    dest_dir.mkdir(parents=True, exist_ok=True)
    week = INVALID_DEMO_WEEK
    path = dest_dir / f"{week}.parquet"
    generate_invalid_batch().to_parquet(path, index=False)
    log.info("Lot invalide de démo écrit", extra={"path": str(path)})
    try:
        validate_batch(app, week, invoices=pd.read_parquet(path), report_dir=dest_dir)
    except QualityError:
        log.info(
            "Lot invalide rejeté comme attendu (pas d'entraînement)",
            extra={"week": week, "report": str(dest_dir / f"{week}.json")},
        )
        return
    raise QualityGateBypassError(
        "Le lot invalide de démo a été accepté : la porte qualité ne bloque pas."
    )
