"""Rapproche les ventes du lot livré des prédictions de cette semaine (WAPE réelle)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import exists, insert, select

from retail.common.config import AppConfig
from retail.common.logging import get_logger
from retail.db.models import Feedback, Prediction
from retail.db.repo import LOADTEST_SOURCE
from retail.db.session import ensure_schema, session_scope
from retail.training.metrics import wape

log = get_logger(__name__)


def auto_feedback_for_week(app: AppConfig, week: str) -> dict:
    """Quand le lot ``week`` est validé, ses quantités ferment les prédictions de cette semaine."""
    panel_path = Path(app.paths.interim_dir) / "panel" / f"{week}.parquet"
    if not panel_path.is_file():
        return {"week": week, "matched": 0, "wape": None}
    try:
        ensure_schema()
        panel = pd.read_parquet(panel_path, engine="pyarrow", memory_map=False)
        actuals = {
            str(code): float(qty)
            for code, qty in zip(
                panel["StockCode"].astype(str), panel["quantity"].astype(float), strict=True
            )
        }
        unmatched = ~exists(
            select(Feedback.id).where(Feedback.prediction_id == Prediction.prediction_id)
        )
        with session_scope() as session:
            preds = session.scalars(
                select(Prediction).where(
                    Prediction.iso_week == week,
                    Prediction.status == "ok",
                    Prediction.source != LOADTEST_SOURCE,
                    unmatched,
                )
            ).all()
            now = datetime.now(UTC)
            rows: list[dict] = []
            actual_list: list[float] = []
            pred_list: list[float] = []
            for pred in preds:
                actual = actuals.get(pred.stock_code, 0.0)
                rows.append(
                    {
                        "prediction_id": pred.prediction_id,
                        "stock_code": pred.stock_code,
                        "iso_week": week,
                        "actual_qty": actual,
                        "predicted_qty": pred.predicted_qty,
                        "abs_error": abs(actual - pred.predicted_qty),
                        "model_version": pred.model_version,
                        "origin": "auto",
                        "created_at": now,
                    }
                )
                actual_list.append(actual)
                pred_list.append(pred.predicted_qty)
            if rows:
                session.execute(insert(Feedback), rows)
            matched = len(rows)
        score = None
        if actual_list:
            score = wape(np.array(actual_list, dtype=float), np.array(pred_list, dtype=float))
        log.info(
            "Feedback automatique",
            extra={"week": week, "matched": matched, "wape": score},
        )
        return {"week": week, "matched": matched, "wape": score}
    except Exception as exc:
        log.warning("Feedback automatique ignoré (base indisponible)", extra={"error": str(exc)})
        return {"week": week, "matched": 0, "wape": None, "error": str(exc)}
