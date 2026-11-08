"""Écritures résilientes : une file bornée, un worker, lots, jamais d'exception métier."""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import func, insert, select

from retail.common.config import get_settings
from retail.common.logging import get_logger
from retail.db.models import Alert, Feedback, PipelineRun, Prediction
from retail.db.session import ensure_schema, session_scope
from retail.training.metrics import wape

log = get_logger(__name__)

LOADTEST_SOURCE = "loadtest"
QUEUE_MAX = 256
BATCH_SIZE = 50
_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(maxsize=QUEUE_MAX)
_worker_lock = threading.Lock()
_worker_started = False
_dropped = 0
_drop_hook = None


def dropped_total() -> int:
    return _dropped


def set_drop_hook(hook) -> None:
    global _drop_hook
    _drop_hook = hook


def reset_worker() -> None:
    global _queue, _worker_started, _dropped
    with _worker_lock:
        _queue = queue.Queue(maxsize=QUEUE_MAX)
        _worker_started = False
        _dropped = 0


def _record_drop(n: int = 1) -> None:
    global _dropped
    if n <= 0:
        return
    _dropped += n
    if _drop_hook is not None:
        _drop_hook(n)
    log.warning("Écritures SQL abandonnées", extra={"n": n})


def _item_weight(kind: str, row: dict[str, Any]) -> int:
    if kind == "predictions":
        return max(len(row.get("rows") or []), 1)
    return 1


def _batch_weight(items: list[tuple[str, dict[str, Any]]]) -> int:
    return sum(_item_weight(kind, row) for kind, row in items)


def _prediction_values(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction_id": row["prediction_id"],
        "created_at": row.get("created_at") or datetime.now(UTC),
        "iso_week": row.get("iso_week"),
        "stock_code": str(row.get("stock_code") or "unknown"),
        "predicted_qty": float(row.get("predicted_qty") or 0.0),
        "model_version": str(row.get("model_version") or "unknown"),
        "source": str(row.get("source") or "unknown"),
        "latency_ms": row.get("latency_ms"),
        "status": str(row.get("status") or "ok"),
        "error": row.get("error"),
    }


def _add_feedback(session, row: dict[str, Any]) -> None:
    predicted = row.get("predicted_qty")
    actual = float(row["actual_qty"])
    abs_error = None if predicted is None else abs(actual - float(predicted))
    pred_id = row.get("prediction_id")
    if pred_id:
        existing = session.scalar(select(Feedback.id).where(Feedback.prediction_id == str(pred_id)))
        if existing is not None:
            return
    session.add(
        Feedback(
            prediction_id=str(pred_id) if pred_id else None,
            stock_code=str(row["stock_code"]),
            iso_week=str(row["iso_week"]),
            actual_qty=actual,
            predicted_qty=None if predicted is None else float(predicted),
            abs_error=abs_error,
            model_version=row.get("model_version"),
            origin=str(row.get("origin") or "http"),
            created_at=row.get("created_at") or datetime.now(UTC),
        )
    )


def _add_run(session, row: dict[str, Any]) -> None:
    session.add(
        PipelineRun(
            iso_week=str(row["week"]),
            status=str(row["status"]),
            duration_s=row.get("duration_s"),
            model_version=row.get("model_version"),
            error=row.get("error"),
            trained=row.get("trained"),
            created_at=datetime.now(UTC),
        )
    )


def _add_alert(session, row: dict[str, Any]) -> None:
    session.add(
        Alert(
            kind=str(row.get("kind") or "unknown"),
            iso_week=row.get("week") or row.get("iso_week"),
            message=str(row.get("message") or ""),
            origin=str(row.get("origin") or "http"),
            created_at=datetime.now(UTC),
            payload={
                key: value
                for key, value in row.items()
                if key not in {"kind", "week", "iso_week", "message", "origin", "created_at"}
            },
        )
    )


def _flush_batch(items: list[tuple[str, dict[str, Any]]]) -> None:
    ensure_schema()
    pred_rows: list[dict[str, Any]] = []
    others: list[tuple[str, dict[str, Any]]] = []
    for kind, row in items:
        if kind == "prediction":
            pred_rows.append(row)
        elif kind == "predictions":
            pred_rows.extend(row.get("rows") or [])
        else:
            others.append((kind, row))
    with session_scope() as session:
        if pred_rows:
            session.execute(insert(Prediction), [_prediction_values(row) for row in pred_rows])
        for kind, row in others:
            if kind == "feedback":
                _add_feedback(session, row)
            elif kind == "run":
                _add_run(session, row)
            elif kind == "alert":
                _add_alert(session, row)


def _worker_loop(work_queue: queue.Queue) -> None:
    while True:
        item = work_queue.get()
        batch = [item]
        deadline = time.monotonic() + 0.05
        while len(batch) < BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(work_queue.get(timeout=remaining))
            except queue.Empty:
                break
        try:
            _flush_batch(batch)
        except Exception as exc:
            _record_drop(_batch_weight(batch))
            log.warning("Persistance ignorée (l'API continue)", extra={"error": str(exc)})
        for _ in batch:
            work_queue.task_done()


def _ensure_worker() -> None:
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        threading.Thread(
            target=_worker_loop,
            args=(_queue,),
            name="retail-db",
            daemon=True,
        ).start()
        _worker_started = True


def enqueue(kind: str, row: dict[str, Any]) -> None:
    if get_settings().database_sync:
        try:
            _flush_batch([(kind, row)])
        except Exception as exc:
            _record_drop(_item_weight(kind, row))
            log.warning("Persistance ignorée (l'API continue)", extra={"error": str(exc)})
        return
    _ensure_worker()
    try:
        _queue.put_nowait((kind, row))
    except queue.Full:
        _record_drop(_item_weight(kind, row))


def insert_prediction(row: dict[str, Any]) -> None:
    _flush_batch([("prediction", row)])


def insert_feedback(row: dict[str, Any]) -> None:
    _flush_batch([("feedback", row)])


def insert_run(row: dict[str, Any]) -> None:
    _flush_batch([("run", row)])


def insert_alert(row: dict[str, Any]) -> None:
    _flush_batch([("alert", row)])


def persist_prediction(row: dict[str, Any]) -> None:
    enqueue("prediction", row)


def persist_predictions(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    enqueue("predictions", {"rows": rows})


def persist_feedback(row: dict[str, Any]) -> None:
    enqueue("feedback", row)


def persist_run(row: dict[str, Any]) -> None:
    """Sync : un `retail replay` / Prefect tuerait un worker daemon avant le flush."""
    try:
        insert_run(row)
    except Exception as exc:
        _record_drop(1)
        log.warning("Persistance ignorée (l'API continue)", extra={"error": str(exc)})


def persist_alert(row: dict[str, Any]) -> None:
    """Sync, comme ``persist_run`` : un process CLI court tuerait le worker daemon."""
    try:
        insert_alert(row)
    except Exception as exc:
        _record_drop(1)
        log.warning("Persistance ignorée (l'API continue)", extra={"error": str(exc)})


def get_prediction(prediction_id: str) -> dict[str, Any] | None:
    try:
        ensure_schema()
        with session_scope() as session:
            row = session.get(Prediction, str(prediction_id))
            if row is None:
                return None
            return {
                "prediction_id": row.prediction_id,
                "stock_code": row.stock_code,
                "iso_week": row.iso_week,
                "predicted_qty": row.predicted_qty,
                "model_version": row.model_version,
                "status": row.status,
            }
    except Exception as exc:
        log.warning("Lecture prédiction impossible", extra={"error": str(exc)})
        return None


def production_wape(model_version: str | None = None) -> float | None:
    try:
        ensure_schema()
        with session_scope() as session:
            ranked = (
                func.row_number()
                .over(
                    partition_by=(
                        Feedback.stock_code,
                        Feedback.iso_week,
                        Feedback.model_version,
                    ),
                    order_by=Feedback.created_at.desc(),
                )
                .label("rn")
            )
            inner = (
                select(
                    Feedback.actual_qty,
                    Feedback.predicted_qty,
                    ranked,
                )
                .outerjoin(Prediction, Prediction.prediction_id == Feedback.prediction_id)
                .where(
                    Feedback.predicted_qty.is_not(None),
                    (Prediction.source.is_(None)) | (Prediction.source != LOADTEST_SOURCE),
                )
            )
            if model_version:
                inner = inner.where(Feedback.model_version == model_version)
            sub = inner.subquery()
            rows = session.execute(
                select(sub.c.actual_qty, sub.c.predicted_qty).where(sub.c.rn == 1)
            ).all()
        if not rows:
            return None
        actual = np.array([float(a) for a, _p in rows], dtype=float)
        pred = np.array([float(p) for _a, p in rows], dtype=float)
        return wape(actual, pred)
    except Exception as exc:
        log.warning("WAPE production indisponible", extra={"error": str(exc)})
        return None


def count_rows(model) -> int:
    with session_scope() as session:
        return int(session.scalar(select(func.count()).select_from(model)) or 0)


def error_detail_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail[:2000]
    try:
        return json.dumps(detail, default=str)[:2000]
    except TypeError:
        return str(detail)[:2000]
