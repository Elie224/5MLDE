from datetime import UTC, datetime, timedelta

import pandas as pd

from retail.db.auto_feedback import auto_feedback_for_week
from retail.db.models import Feedback, PipelineRun, Prediction
from retail.db.repo import count_rows, insert_prediction, persist_predictions, persist_run
from retail.db.session import (
    _mark_legacy_loadtest,
    ensure_schema,
    get_engine,
    ping,
    ping_cached,
    session_scope,
)


def test_sqlite_schema_and_prediction_roundtrip(tmp_configs):
    ensure_schema()
    assert ping() is True
    insert_prediction(
        {
            "prediction_id": "11111111-1111-1111-1111-111111111111",
            "iso_week": "2011-W20",
            "stock_code": "AAA",
            "predicted_qty": 8.0,
            "model_version": "1",
            "source": "mlflow",
            "latency_ms": 1.2,
            "status": "ok",
        }
    )
    persist_run({"week": "2011-W20", "status": "ok", "trained": False, "duration_s": 0.1})
    assert count_rows(Prediction) == 1
    assert count_rows(PipelineRun) == 1


def test_auto_feedback_computes_wape(tmp_configs):
    app, _ = tmp_configs
    ensure_schema()
    insert_prediction(
        {
            "prediction_id": "22222222-2222-2222-2222-222222222222",
            "created_at": datetime.now(UTC),
            "iso_week": "2011-W20",
            "stock_code": "AAA",
            "predicted_qty": 8.0,
            "model_version": "1",
            "source": "disk",
            "status": "ok",
        }
    )
    panel_dir = app.paths.interim_dir / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "StockCode": ["AAA"],
            "iso_week": ["2011-W20"],
            "quantity": [10.0],
            "unit_price": [1.5],
        }
    ).to_parquet(panel_dir / "2011-W20.parquet", index=False)
    result = auto_feedback_for_week(app, "2011-W20")
    assert result["matched"] == 1
    assert result["wape"] == 0.2
    assert count_rows(Feedback) == 1
    again = auto_feedback_for_week(app, "2011-W20")
    assert again["matched"] == 0
    assert count_rows(Feedback) == 1


def test_auto_feedback_skips_loadtest_source(tmp_configs):
    from retail.db.repo import LOADTEST_SOURCE

    app, _ = tmp_configs
    ensure_schema()
    insert_prediction(
        {
            "prediction_id": "l1l1l1l1-l1l1-l1l1-l1l1-l1l1l1l1l1l1",
            "iso_week": "2011-W20",
            "stock_code": "AAA",
            "predicted_qty": 8.0,
            "model_version": "1",
            "source": LOADTEST_SOURCE,
            "status": "ok",
        }
    )
    panel_dir = app.paths.interim_dir / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "StockCode": ["AAA"],
            "iso_week": ["2011-W20"],
            "quantity": [10.0],
            "unit_price": [1.5],
        }
    ).to_parquet(panel_dir / "2011-W20.parquet", index=False)
    result = auto_feedback_for_week(app, "2011-W20")
    assert result["matched"] == 0
    assert count_rows(Feedback) == 0


def test_auto_feedback_zero_sales_counts_as_overstock(tmp_configs):
    app, _ = tmp_configs
    ensure_schema()
    insert_prediction(
        {
            "prediction_id": "44444444-4444-4444-4444-444444444444",
            "iso_week": "2011-W20",
            "stock_code": "AAA",
            "predicted_qty": 10.0,
            "model_version": "1",
            "source": "mlflow",
            "status": "ok",
        }
    )
    insert_prediction(
        {
            "prediction_id": "55555555-5555-5555-5555-555555555555",
            "iso_week": "2011-W20",
            "stock_code": "BBB",
            "predicted_qty": 30.0,
            "model_version": "1",
            "source": "mlflow",
            "status": "ok",
        }
    )
    panel_dir = app.paths.interim_dir / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "StockCode": ["AAA"],
            "iso_week": ["2011-W20"],
            "quantity": [10.0],
            "unit_price": [1.5],
        }
    ).to_parquet(panel_dir / "2011-W20.parquet", index=False)
    result = auto_feedback_for_week(app, "2011-W20")
    assert result["matched"] == 2
    assert result["wape"] == 3.0
    assert count_rows(Feedback) == 2


def test_persist_predictions_writes_all_rows(tmp_configs):
    ensure_schema()
    persist_predictions(
        [
            {
                "prediction_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "iso_week": "2011-W20",
                "stock_code": "AAA",
                "predicted_qty": 1.0,
                "model_version": "1",
                "source": "disk",
                "status": "ok",
            },
            {
                "prediction_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "iso_week": "2011-W20",
                "stock_code": "BBB",
                "predicted_qty": 2.0,
                "model_version": "1",
                "source": "disk",
                "status": "ok",
            },
        ]
    )
    assert count_rows(Prediction) == 2


def test_failed_flush_counts_dropped_rows(tmp_configs, monkeypatch):
    from retail.common import config as config_module
    from retail.db import repo

    monkeypatch.setenv("RETAIL_DATABASE_SYNC", "true")
    config_module.get_settings.cache_clear()
    repo.reset_worker()
    monkeypatch.setattr(
        repo,
        "_flush_batch",
        lambda items: (_ for _ in ()).throw(RuntimeError("db down")),
    )
    repo.persist_predictions(
        [
            {"prediction_id": "1", "stock_code": "A", "predicted_qty": 1.0, "model_version": "1"},
            {"prediction_id": "2", "stock_code": "B", "predicted_qty": 1.0, "model_version": "1"},
        ]
    )
    assert repo.dropped_total() >= 2


def test_full_queue_drops_writes(tmp_configs, monkeypatch):
    from retail.common import config as config_module
    from retail.db import repo

    monkeypatch.setenv("RETAIL_DATABASE_SYNC", "false")
    config_module.get_settings.cache_clear()
    repo.reset_worker()
    monkeypatch.setattr(repo, "_ensure_worker", lambda: None)
    for index in range(repo.QUEUE_MAX):
        repo._queue.put_nowait(("prediction", {"prediction_id": f"fill-{index}"}))
    repo.persist_prediction(
        {
            "prediction_id": "dropped",
            "stock_code": "AAA",
            "predicted_qty": 1.0,
            "model_version": "1",
        }
    )
    assert repo.dropped_total() >= 1


def test_worker_stays_on_its_queue_after_reset(tmp_configs, monkeypatch):
    import threading
    import time

    from retail.common import config as config_module
    from retail.db import repo

    monkeypatch.setenv("RETAIL_DATABASE_SYNC", "false")
    config_module.get_settings.cache_clear()
    repo.reset_worker()
    seen: list = []

    def _flush(batch):
        seen.append(len(batch))

    monkeypatch.setattr(repo, "_flush_batch", _flush)
    old = repo._queue
    threading.Thread(target=repo._worker_loop, args=(old,), daemon=True).start()
    repo.reset_worker()
    old.put(("prediction", {"prediction_id": "on-old-queue"}))
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and not seen:
        time.sleep(0.02)
    assert seen == [1]


def test_legacy_w48_predictions_marked_loadtest(tmp_configs):
    from retail.db.repo import LOADTEST_SOURCE

    ensure_schema()
    pid = "w48w48w4-w48w-w48w-w48w-w48w48w48w48"
    insert_prediction(
        {
            "prediction_id": pid,
            "iso_week": "2011-W48",
            "stock_code": "AAA",
            "predicted_qty": 1.0,
            "model_version": "1",
            "source": "disk",
            "status": "ok",
        }
    )
    assert _mark_legacy_loadtest(get_engine()) == 1
    with session_scope() as session:
        row = session.get(Prediction, pid)
        assert row is not None
        assert row.source == LOADTEST_SOURCE


def test_production_wape_keeps_latest_feedback_per_sku_week(tmp_configs):
    from retail.db.repo import production_wape

    ensure_schema()
    older = datetime.now(UTC) - timedelta(minutes=5)
    newer = datetime.now(UTC)
    with session_scope() as session:
        session.add(
            Feedback(
                stock_code="AAA",
                iso_week="2011-W20",
                model_version="1",
                actual_qty=10.0,
                predicted_qty=0.0,
                abs_error=10.0,
                origin="auto",
                created_at=older,
            )
        )
        session.add(
            Feedback(
                stock_code="AAA",
                iso_week="2011-W20",
                model_version="1",
                actual_qty=10.0,
                predicted_qty=10.0,
                abs_error=0.0,
                origin="auto",
                created_at=newer,
            )
        )
    assert production_wape("1") == 0.0


def test_ping_cached_reuses_last_result(tmp_configs, monkeypatch):
    from retail.db import session as db_session

    ensure_schema()
    assert ping() is True
    calls = {"n": 0}
    original = db_session.session_scope

    def counted():
        calls["n"] += 1
        return original()

    monkeypatch.setattr(db_session, "session_scope", counted)
    assert ping_cached(ttl_s=30.0) is True
    assert calls["n"] == 0
