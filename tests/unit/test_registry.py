import json
import os

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retail.registry.errors import RegistryError
from retail.registry.promote import INCOMPATIBLE_DECISION, register_run, rollback_champion


def test_register_skips_when_not_beating_baseline(tmp_configs):
    app, _ = tmp_configs
    result = register_run(app, {"beats_baseline": False, "mae_model": 10, "mae_naive": 5})
    assert result == {"status": "skipped", "reason": "baseline"}
    assert not (app.paths.models_dir / "champion").exists()


def test_first_challenger_becomes_champion_keeps_signature(tmp_configs, register_dummy):
    app, _ = tmp_configs
    result = register_dummy(app, constant=8.0)
    assert result["status"] == "champion"
    dest = app.paths.models_dir / "champion"
    mlmodel = (dest / "model" / "MLmodel").read_text(encoding="utf-8")
    assert "signature:" in mlmodel
    assert (dest / "top_skus.json").is_file()
    meta = json.loads((dest / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == result["model_version"]
    assert "mae_own_test" in meta
    assert meta["test_weeks"] == ["2011-W19", "2011-W20"]


def test_worse_model_on_same_test_stays_challenger(tmp_configs, register_dummy):
    app, _ = tmp_configs
    first = register_dummy(app, constant=8.0)
    second = register_dummy(app, constant=0.0)
    assert first["status"] == "champion"
    assert second["status"] == "challenger"
    assert second["mae_challenger_same_test"] > second["mae_champion_same_test"]
    with pytest.raises(RegistryError, match="premier champion"):
        rollback_champion(app)


def test_better_model_on_same_test_promotes(tmp_configs, register_dummy):
    app, _ = tmp_configs
    first = register_dummy(app, constant=0.0)
    second = register_dummy(app, constant=8.0)
    assert second["status"] == "champion"
    assert second["model_version"] != first["model_version"]
    assert second["mae_challenger_same_test"] <= second["mae_champion_same_test"]


def test_equal_mae_on_same_test_promotes(tmp_configs, register_dummy):
    app, _ = tmp_configs
    first = register_dummy(app, constant=8.0)
    second = register_dummy(app, constant=8.0)
    assert second["status"] == "champion"
    assert second["model_version"] != first["model_version"]


def test_rollback_walks_previous_champion_tags_not_ping_pong(tmp_configs, register_dummy):
    app, _ = tmp_configs
    v1 = register_dummy(app, constant=0.0)
    v2 = register_dummy(app, constant=4.0)
    v3 = register_dummy(app, constant=8.0)
    assert [v1["status"], v2["status"], v3["status"]] == ["champion"] * 3

    first = rollback_champion(app)
    assert first["model_version"] == v2["model_version"]
    second = rollback_champion(app)
    assert second["model_version"] == v1["model_version"]
    meta = json.loads((app.paths.models_dir / "champion" / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == v1["model_version"]
    with pytest.raises(RegistryError, match="premier champion"):
        rollback_champion(app)


def test_rollback_to_version(tmp_configs, register_dummy):
    app, _ = tmp_configs
    v1 = register_dummy(app, constant=0.0)
    register_dummy(app, constant=4.0)
    register_dummy(app, constant=8.0)
    rolled = rollback_champion(app, to_version=v1["model_version"])
    assert rolled["model_version"] == v1["model_version"]
    meta = json.loads((app.paths.models_dir / "champion" / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == v1["model_version"]


def test_rollback_without_champion_fails(tmp_configs):
    app, _ = tmp_configs
    with pytest.raises(RegistryError, match="Aucun champion"):
        rollback_champion(app)


def test_top_skus_come_from_champion_run_not_interim(tmp_configs, register_dummy):
    app, _ = tmp_configs
    register_dummy(app, constant=0.0, skus=["OLD"])
    register_dummy(app, constant=8.0, skus=["NEW"])
    rollback_champion(app)
    exported = app.paths.models_dir / "champion" / "top_skus.json"
    skus = json.loads(exported.read_text(encoding="utf-8"))
    assert skus == ["OLD"]


def test_failed_export_keeps_existing_champion_dir(
    tmp_configs, register_dummy, log_dummy_model, make_registry_metrics, registry_frame, monkeypatch
):
    app, _ = tmp_configs
    first = register_dummy(app, constant=0.0)
    dest = app.paths.models_dir / "champion"
    before = (dest / "meta.json").read_text(encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("MLflow indisponible")

    monkeypatch.setattr("retail.registry.promote._download_model", _boom)
    x_test, y_test = registry_frame
    with pytest.raises(RuntimeError, match="indisponible"):
        register_run(
            app,
            make_registry_metrics(log_dummy_model(app, 8.0)),
            x_test=x_test,
            y_test=y_test,
        )
    assert (dest / "meta.json").read_text(encoding="utf-8") == before
    meta = json.loads(before)
    assert meta["model_version"] == first["model_version"]


def test_replace_dir_copies_when_rename_denied(tmp_path, monkeypatch):
    from pathlib import Path

    from retail.registry.promote import _replace_dir

    dest = tmp_path / "champion"
    dest.mkdir()
    (dest / "stale.txt").write_text("old", encoding="utf-8")
    tmp = tmp_path / ".champion.1.tmp"
    tmp.mkdir()
    (tmp / "meta.json").write_text('{"model_version": "1"}', encoding="utf-8")
    (tmp / "model").mkdir()
    (tmp / "model" / "MLmodel").write_text("ok", encoding="utf-8")

    def denied(self, target):
        raise PermissionError(5, "Accès refusé")

    monkeypatch.setattr(Path, "rename", denied)
    _replace_dir(tmp, dest)
    assert (dest / "meta.json").read_text(encoding="utf-8") == '{"model_version": "1"}'
    assert (dest / "model" / "MLmodel").read_text(encoding="utf-8") == "ok"
    assert not (dest / "stale.txt").exists()


def test_logged_mae_on_other_weeks_is_ignored(
    tmp_configs, log_dummy_model, make_registry_metrics, registry_frame
):
    app, _ = tmp_configs
    x_test, y_test = registry_frame
    weak = log_dummy_model(app, 0.0)
    register_run(app, make_registry_metrics(weak, mae_model=0.01), x_test=x_test, y_test=y_test)
    strong = log_dummy_model(app, 8.0)
    result = register_run(
        app, make_registry_metrics(strong, mae_model=999.0), x_test=x_test, y_test=y_test
    )
    assert result["status"] == "champion"
    assert result["mae_challenger_same_test"] <= result["mae_champion_same_test"]


def test_incompatible_champion_schema_promotes_challenger(
    tmp_configs, log_dummy_model, register_dummy, make_registry_metrics
):
    app, _ = tmp_configs
    x_old = pd.DataFrame(
        {
            "StockCode": ["AAA", "BBB", "AAA", "BBB"],
            "unit_price": [1.5, 2.0, 1.5, 2.0],
            "lag_1": [10.0, 4.0, 12.0, 5.0],
        }
    )
    y = np.array([10.0, 4.0, 12.0, 5.0])
    pipe = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), ["unit_price", "lag_1"]),
                        (
                            "sku",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ["StockCode"],
                        ),
                    ]
                ),
            ),
            ("model", Ridge()),
        ]
    )
    pipe.fit(x_old, y)
    old_run = log_dummy_model(app, 0.0, model=pipe, x_frame=x_old)
    first = register_run(app, make_registry_metrics(old_run), x_test=x_old, y_test=y)
    assert first["status"] == "champion"

    result = register_dummy(app, constant=8.0)
    assert result["status"] == "champion"
    assert result["decision"] == INCOMPATIBLE_DECISION
    assert result["model_version"] != first["model_version"]
    client = MlflowClient(os.environ["RETAIL_MLFLOW_TRACKING_URI"])
    tags = client.get_model_version(
        app.promotion.registered_model_name, result["model_version"]
    ).tags
    assert tags["decision"] == INCOMPATIBLE_DECISION


def test_champion_without_signature_promotes_on_score_error(
    tmp_configs, log_dummy_model, register_dummy, make_registry_metrics
):
    from retail.registry.promote import NO_SIGNATURE_DECISION

    app, _ = tmp_configs
    x_old = pd.DataFrame(
        {
            "StockCode": ["AAA", "BBB", "AAA", "BBB"],
            "unit_price": [1.5, 2.0, 1.5, 2.0],
            "lag_1": [10.0, 4.0, 12.0, 5.0],
        }
    )
    y = np.array([10.0, 4.0, 12.0, 5.0])
    pipe = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), ["unit_price", "lag_1"]),
                        (
                            "sku",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ["StockCode"],
                        ),
                    ]
                ),
            ),
            ("model", Ridge()),
        ]
    )
    pipe.fit(x_old, y)
    old_run = log_dummy_model(app, 0.0, model=pipe, x_frame=x_old, with_signature=False)
    first = register_run(app, make_registry_metrics(old_run), x_test=x_old, y_test=y)
    assert first["status"] == "champion"
    result = register_dummy(app, constant=8.0)
    assert result["decision"] == NO_SIGNATURE_DECISION


def test_train_lot_resumes_instead_of_retraining(tmp_configs, register_dummy, monkeypatch):
    from unittest.mock import MagicMock

    from retail.orchestration.replay import train_lot
    from retail.registry.promote import STATUS_IN_PROGRESS, TAG_PIPELINE

    app, _ = tmp_configs
    first = register_dummy(app, constant=8.0)
    client = MlflowClient(os.environ["RETAIL_MLFLOW_TRACKING_URI"])
    client.set_model_version_tag(
        app.promotion.registered_model_name,
        first["model_version"],
        TAG_PIPELINE,
        STATUS_IN_PROGRESS,
    )
    train = MagicMock()
    monkeypatch.setattr("retail.orchestration.replay.train_and_evaluate", train)
    result = train_lot(app, "2011-W20")
    train.assert_not_called()
    assert result["resumed"] is True
    assert result["exported"] is True
    assert result["model_version"] == first["model_version"]


def test_resume_challenger_does_not_overwrite_champion_dir(tmp_configs, register_dummy):
    from retail.registry.promote import STATUS_IN_PROGRESS, TAG_PIPELINE, resume_week_version

    app, _ = tmp_configs
    first = register_dummy(app, constant=8.0)
    second = register_dummy(app, constant=0.0)
    assert second["status"] == "challenger"
    meta_before = json.loads(
        (app.paths.models_dir / "champion" / "meta.json").read_text(encoding="utf-8")
    )
    assert meta_before["model_version"] == first["model_version"]
    client = MlflowClient(os.environ["RETAIL_MLFLOW_TRACKING_URI"])
    client.set_model_version_tag(
        app.promotion.registered_model_name,
        second["model_version"],
        TAG_PIPELINE,
        STATUS_IN_PROGRESS,
    )
    result = resume_week_version(app, second["model_version"])
    assert result["exported"] is False
    meta_after = json.loads(
        (app.paths.models_dir / "champion" / "meta.json").read_text(encoding="utf-8")
    )
    assert meta_after["model_version"] == first["model_version"]


def test_complete_week_is_not_retrained(tmp_configs, register_dummy, monkeypatch):
    from unittest.mock import MagicMock

    from retail.orchestration.replay import train_lot

    app, _ = tmp_configs
    first = register_dummy(app, constant=8.0)
    train = MagicMock()
    monkeypatch.setattr("retail.orchestration.replay.train_and_evaluate", train)
    result = train_lot(app, "2011-W20")
    train.assert_not_called()
    assert result["trained"] is False
    assert result["model_version"] == first["model_version"]


def test_cli_train_skips_duplicate_until_week(tmp_configs, register_dummy, monkeypatch):
    from unittest.mock import MagicMock

    from retail.training.train import train_and_evaluate

    app, _ = tmp_configs
    first = register_dummy(app, constant=8.0)
    build = MagicMock()
    monkeypatch.setattr("retail.training.train.build_features", build)
    skipped = train_and_evaluate(app, until_week="2011-W20")
    build.assert_not_called()
    assert skipped["trained"] is False
    assert skipped["model_version"] == first["model_version"]
