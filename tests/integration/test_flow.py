import pytest

from retail.quality.errors import QualityError
from retail.training.errors import TrainError


@pytest.mark.integration
def test_weekly_lot_validate_then_train(tmp_configs, monkeypatch):
    pytest.importorskip("prefect")
    from flows.weekly_lot import retry_if_technical, weekly_lot
    from prefect.testing.utilities import prefect_test_harness

    app, _ = tmp_configs
    app.paths.batches_dir.mkdir(parents=True, exist_ok=True)
    (app.paths.batches_dir / "manifest.json").write_text(
        '{"batches": [{"week": "2010-W49", "week_start": "2010-12-06", '
        '"week_end": "2010-12-12", "rows": 1, "invoices": 1, '
        '"content_hash": "x", "file": "2010-W49.parquet"}], "missing_weeks": []}',
        encoding="utf-8",
    )
    seen: list[str] = []

    def _validate(_app, week=None):
        seen.append(week or "")
        return []

    monkeypatch.setattr("retail.orchestration.replay.validate_all", _validate)
    monkeypatch.setattr(
        "retail.orchestration.replay.train_and_evaluate",
        lambda *_a, **_k: {
            "mae_model": 1.0,
            "beats_baseline": True,
            "registry": {"model_version": "1"},
        },
    )
    with prefect_test_harness():
        result = weekly_lot()
    assert seen == ["2010-W49"]
    assert result["week"] == "2010-W49"
    assert result["trained"] is True

    class _State:
        def result(self, raise_on_failure=True):
            raise QualityError("lot")

    assert retry_if_technical(None, None, _State()) is False

    class _Tech:
        def result(self, raise_on_failure=True):
            raise ConnectionError("mlflow down")

    assert retry_if_technical(None, None, _Tech()) is True

    class _Train:
        def result(self, raise_on_failure=True):
            raise TrainError("pas assez")

    assert retry_if_technical(None, None, _Train()) is False
    assert (
        retry_if_technical(
            None, None, type("Ok", (), {"result": lambda self, raise_on_failure=True: None})()
        )
        is False
    )
