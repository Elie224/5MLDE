import json
from unittest.mock import MagicMock

import pytest
from requests import RequestException

from retail.orchestration.errors import ReplayDone, ReplayError
from retail.orchestration.replay import next_week, run_tick, save_cursor
from retail.quality.errors import QualityError
from retail.training.errors import TrainError


def _write_manifest(app, weeks: list[str]) -> None:
    app.paths.batches_dir.mkdir(parents=True, exist_ok=True)
    batches = []
    for week in weeks:
        batches.append(
            {
                "week": week,
                "week_start": "2010-12-06",
                "week_end": "2010-12-12",
                "rows": 10,
                "invoices": 2,
                "content_hash": "abc",
                "file": f"{week}.parquet",
            }
        )
    payload = {"batches": batches, "missing_weeks": []}
    (app.paths.batches_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _cursor(app) -> dict:
    return json.loads(
        (app.paths.reports_dir / "orchestration" / "cursor.json").read_text(encoding="utf-8")
    )


def test_next_week_follows_cursor(tmp_configs):
    app, _ = tmp_configs
    _write_manifest(app, ["2010-W49", "2010-W50", "2010-W51"])
    assert next_week(app) == "2010-W49"
    save_cursor(app, "2010-W49", "ok")
    assert next_week(app) == "2010-W50"
    save_cursor(app, "2010-W51", "ok")
    with pytest.raises(ReplayDone):
        next_week(app)


def test_quality_failed_stays_on_same_week_next_tick(tmp_configs, monkeypatch):
    app, _ = tmp_configs
    _write_manifest(app, ["2010-W49", "2010-W50"])
    seen: list[str | None] = []

    def _validate(_app, week=None):
        seen.append(week)
        if week == "2010-W49":
            raise QualityError("lot hors schéma")
        return [{"week": week}]

    monkeypatch.setattr("retail.orchestration.replay.validate_all", _validate)
    monkeypatch.setattr(
        "retail.orchestration.replay.requests.post",
        MagicMock(side_effect=RequestException("down")),
    )
    train = MagicMock()
    monkeypatch.setattr("retail.orchestration.replay.train_and_evaluate", train)

    with pytest.raises(QualityError):
        run_tick(app)
    assert next_week(app) == "2010-W49"
    with pytest.raises(QualityError):
        run_tick(app)
    assert seen == ["2010-W49", "2010-W49"]
    train.assert_not_called()
    assert _cursor(app)["status"] == "quality_failed"
    alerts = (
        (app.paths.reports_dir / "alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )
    assert len(alerts) == 2


def test_quality_error_during_train_alerts_and_blocks(tmp_configs, monkeypatch):
    app, _ = tmp_configs
    _write_manifest(app, ["2010-W49", "2010-W50"])
    monkeypatch.setattr("retail.orchestration.replay.validate_all", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "retail.orchestration.replay.requests.post",
        MagicMock(side_effect=RequestException("down")),
    )

    def _train(_app, until_week=None):
        raise QualityError("Lot 2010-W49 rejeté par la qualité")

    monkeypatch.setattr("retail.orchestration.replay.train_and_evaluate", _train)
    with pytest.raises(QualityError):
        run_tick(app)
    assert next_week(app) == "2010-W49"
    assert _cursor(app)["status"] == "quality_failed"
    alerts = (app.paths.reports_dir / "alerts.jsonl").read_text(encoding="utf-8")
    assert "quality" in alerts


def test_run_tick_validates_only_the_delivered_week(tmp_configs, monkeypatch):
    app, _ = tmp_configs
    _write_manifest(app, ["2010-W49", "2010-W50"])
    seen: list[str | None] = []

    def _validate(_app, week=None):
        seen.append(week)
        return [{"week": week}]

    monkeypatch.setattr("retail.orchestration.replay.validate_all", _validate)
    monkeypatch.setattr(
        "retail.orchestration.replay.train_and_evaluate",
        lambda *_a, **_k: (_ for _ in ()).throw(TrainError("pas assez de semaines")),
    )
    result = run_tick(app)
    assert result["week"] == "2010-W49"
    assert seen == ["2010-W49"]
    assert result["trained"] is False
    assert _cursor(app)["last_week"] == "2010-W49"
    assert _cursor(app)["status"] == "ok"
    runs = (app.paths.reports_dir / "orchestration" / "runs.jsonl").read_text(encoding="utf-8")
    assert "2010-W49" in runs
    from retail.db.models import PipelineRun
    from retail.db.repo import count_rows
    from retail.db.session import ensure_schema

    ensure_schema()
    assert count_rows(PipelineRun) == 1


def test_quality_failure_writes_local_alert(tmp_configs, monkeypatch):
    app, _ = tmp_configs
    _write_manifest(app, ["2010-W49"])
    monkeypatch.setattr(
        "retail.orchestration.replay.validate_all",
        lambda *_a, **_k: (_ for _ in ()).throw(QualityError("lot hors schéma")),
    )
    monkeypatch.setattr(
        "retail.orchestration.replay.requests.post",
        MagicMock(side_effect=RequestException("down")),
    )
    with pytest.raises(QualityError, match="hors schéma"):
        run_tick(app)
    line = json.loads(
        (app.paths.reports_dir / "alerts.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert line["kind"] == "quality"
    assert line["week"] == "2010-W49"


def test_emit_alert_skips_local_file_when_webhook_ok(tmp_configs, monkeypatch):
    from retail.db.models import Alert
    from retail.db.repo import count_rows
    from retail.db.session import ensure_schema
    from retail.orchestration.replay import emit_alert

    app, _ = tmp_configs
    ensure_schema()
    posted = MagicMock()
    posted.return_value.raise_for_status = MagicMock()
    monkeypatch.setattr("retail.orchestration.replay.requests.post", posted)
    emit_alert(app, {"kind": "quality", "message": "x", "week": "2010-W49"})
    posted.assert_called_once()
    assert not (app.paths.reports_dir / "alerts.jsonl").is_file()
    assert count_rows(Alert) == 1


def test_successful_lot_triggers_auto_feedback(tmp_configs, monkeypatch):
    import pandas as pd

    from retail.db.auto_feedback import auto_feedback_for_week
    from retail.db.models import Feedback
    from retail.db.repo import count_rows, insert_prediction
    from retail.db.session import ensure_schema

    app, _ = tmp_configs
    _write_manifest(app, ["2011-W20"])
    panel_dir = app.paths.interim_dir / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "StockCode": ["AAA"],
            "iso_week": ["2011-W20"],
            "quantity": [12.0],
            "unit_price": [1.5],
        }
    ).to_parquet(panel_dir / "2011-W20.parquet", index=False)
    ensure_schema()
    insert_prediction(
        {
            "prediction_id": "33333333-3333-3333-3333-333333333333",
            "iso_week": "2011-W20",
            "stock_code": "AAA",
            "predicted_qty": 10.0,
            "model_version": "1",
            "source": "mlflow",
            "status": "ok",
        }
    )
    monkeypatch.setattr("retail.orchestration.replay.validate_all", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "retail.orchestration.replay.train_and_evaluate",
        lambda *_a, **_k: (_ for _ in ()).throw(TrainError("skip")),
    )
    run_tick(app, week="2011-W20")
    assert count_rows(Feedback) == 1
    assert auto_feedback_for_week(app, "2011-W20")["matched"] == 0


def test_lock_does_not_mask_inner_oserror(tmp_path):
    from retail.orchestration.lock import replay_lock

    with pytest.raises(PermissionError, match="WinError 32"), replay_lock(tmp_path / "replay.lock"):
        raise PermissionError(32, "WinError 32")


def test_overlapping_replay_is_rejected(tmp_configs, monkeypatch):
    app, _ = tmp_configs
    _write_manifest(app, ["2010-W49"])
    from retail.orchestration.lock import replay_lock
    from retail.orchestration.replay import LOCK_FILENAME, _orch_dir

    monkeypatch.setattr("retail.orchestration.replay.validate_all", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "retail.orchestration.replay.train_and_evaluate",
        lambda *_a, **_k: (_ for _ in ()).throw(TrainError("skip")),
    )
    with (
        replay_lock(_orch_dir(app) / LOCK_FILENAME),
        pytest.raises(ReplayError, match="déjà en cours"),
    ):
        run_tick(app)


def test_successful_lot_forecasts_next_week(tmp_configs, monkeypatch):
    app, _ = tmp_configs
    _write_manifest(app, ["2010-W49", "2010-W50"])
    monkeypatch.setattr("retail.orchestration.replay.validate_all", lambda *_a, **_k: [])
    monkeypatch.setattr("retail.orchestration.replay.auto_feedback_for_week", lambda *_a, **_k: {})
    posted = MagicMock()
    posted.return_value.raise_for_status = MagicMock()
    posted.return_value.json.return_value = {
        "predictions": [{"StockCode": "A"}, {"StockCode": "B"}],
        "model_version": "7",
    }
    monkeypatch.setattr("retail.orchestration.replay.requests.post", posted)
    result = run_tick(app, train=False)
    assert result["forecast_week"] == "2010-W50"
    assert result["forecast_n"] == 2
    assert posted.call_args.kwargs["json"] == {"week": "2010-W50"}
    run = json.loads(
        (app.paths.reports_dir / "orchestration" / "runs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )
    assert run["duration_s"] is not None
    assert run["forecast_n"] == 2


def test_forecast_failure_does_not_block_the_lot(tmp_configs, monkeypatch):
    app, _ = tmp_configs
    _write_manifest(app, ["2010-W49", "2010-W50"])
    monkeypatch.setattr("retail.orchestration.replay.validate_all", lambda *_a, **_k: [])
    monkeypatch.setattr("retail.orchestration.replay.auto_feedback_for_week", lambda *_a, **_k: {})
    monkeypatch.setattr(
        "retail.orchestration.replay.requests.post",
        MagicMock(side_effect=RequestException("api down")),
    )
    result = run_tick(app, train=False)
    assert result["forecast_n"] == 0
    assert _cursor(app)["status"] == "ok"


def test_already_trained_week_reuses_version(tmp_configs, monkeypatch):
    from retail.orchestration.replay import train_lot

    app, _ = tmp_configs
    done = MagicMock()
    done.version = "5"
    monkeypatch.setattr("retail.registry.promote.find_week_version", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "retail.registry.promote.find_complete_week_version", lambda *_a, **_k: done
    )
    train = MagicMock()
    monkeypatch.setattr("retail.orchestration.replay.train_and_evaluate", train)
    result = train_lot(app, "2011-W40")
    train.assert_not_called()
    assert result["trained"] is False
    assert result["model_version"] == "5"


def test_replay_past_week_does_not_recede_cursor(tmp_configs, monkeypatch):
    app, _ = tmp_configs
    _write_manifest(app, ["2011-W40", "2011-W41", "2011-W42"])
    save_cursor(app, "2011-W42", "ok")
    monkeypatch.setattr("retail.orchestration.replay.validate_all", lambda *_a, **_k: [])
    monkeypatch.setattr("retail.orchestration.replay.auto_feedback_for_week", lambda *_a, **_k: {})
    monkeypatch.setattr(
        "retail.orchestration.replay.requests.post",
        MagicMock(side_effect=RequestException("down")),
    )
    result = run_tick(app, week="2011-W41", train=False)
    assert result["week"] == "2011-W41"
    assert _cursor(app)["last_week"] == "2011-W42"
    with pytest.raises(ReplayDone):
        next_week(app)
