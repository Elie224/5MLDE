import pandas as pd
import pytest

from retail.common.config import TrainingCfg
from retail.features.panel import (
    add_lags_and_target,
    build_feature_frame,
    calendar_span,
    densify,
    freeze_top_skus,
    load_panels,
    ordered_weeks,
    warmup_week_labels,
    weeks_until,
)
from retail.training.train import temporal_split


def _training() -> TrainingCfg:
    return TrainingCfg(test_weeks=3, lags=[1, 2, 4], rolling_weeks=4)


def _panel() -> pd.DataFrame:
    weeks = [f"2011-W{i:02d}" for i in range(1, 17)]
    rows = []
    for i, week in enumerate(weeks):
        rows.append(
            {
                "StockCode": "AAA",
                "iso_week": week,
                "quantity": 100 + i,
                "unit_price": 2.0,
                "revenue": 200.0,
                "n_lines": 1,
                "n_invoices": 1,
            }
        )
        rows.append(
            {
                "StockCode": "BBB",
                "iso_week": week,
                "quantity": 40,
                "unit_price": 1.0,
                "revenue": 40.0,
                "n_lines": 1,
                "n_invoices": 1,
            }
        )
        late = 5 if i < 8 else 900
        rows.append(
            {
                "StockCode": "CCC",
                "iso_week": week,
                "quantity": late,
                "unit_price": 3.0,
                "revenue": float(late),
                "n_lines": 1,
                "n_invoices": 1,
            }
        )
    return pd.DataFrame(rows)


def test_top_skus_ignore_sales_after_warmup():
    panel = _panel()
    warmup = warmup_week_labels(panel["iso_week"].unique().tolist(), 8)
    assert warmup[-1] == "2011-W08"
    skus = freeze_top_skus(panel, warmup, top_n=2)
    assert skus == ["AAA", "BBB"]
    assert "CCC" not in skus


def test_lags_are_strictly_past_of_the_target():
    panel = _panel()
    weeks = ordered_weeks(panel["iso_week"].unique().tolist())
    dense = densify(panel, ["AAA"], weeks)
    featured = add_lags_and_target(dense, _training())
    qty = dense.set_index("iso_week")["quantity"]
    row = featured.loc[featured["iso_week"] == "2011-W09"].iloc[0]
    assert row["y"] == qty.loc["2011-W09"]
    assert row["lag_1"] == qty.loc["2011-W08"]
    assert row["lag_2"] == qty.loc["2011-W07"]
    assert row["lag_4"] == qty.loc["2011-W05"]
    assert row["lag_1"] != row["y"]
    assert row["ma_4"] == pytest.approx(
        float(qty.loc[["2011-W05", "2011-W06", "2011-W07", "2011-W08"]].mean())
    )
    assert row["price_lag_1"] == dense.set_index("iso_week")["unit_price"].loc["2011-W08"]


def test_temporal_split_keeps_train_strictly_before_test():
    panel = _panel()
    weeks = ordered_weeks(panel["iso_week"].unique().tolist())
    warmup = weeks[:8]
    featured = build_feature_frame(panel, ["AAA", "BBB"], weeks, warmup, _training())
    train, test = temporal_split(featured, test_weeks=3)
    assert max(train["iso_week"]) < min(test["iso_week"])
    assert set(test["iso_week"]) == {"2011-W14", "2011-W15", "2011-W16"}


def test_calendar_span_inserts_missing_iso_week():
    assert calendar_span(["2010-W51", "2011-W01"]) == ["2010-W51", "2010-W52", "2011-W01"]


def test_lags_use_calendar_weeks_across_a_gap():
    rows = []
    for week, qty in [
        ("2010-W49", 10),
        ("2010-W50", 20),
        ("2010-W51", 30),
        ("2011-W01", 40),
        ("2011-W02", 50),
        ("2011-W03", 60),
        ("2011-W04", 70),
        ("2011-W05", 80),
    ]:
        rows.append(
            {
                "StockCode": "AAA",
                "iso_week": week,
                "quantity": qty,
                "unit_price": 2.0,
            }
        )
    panel = pd.DataFrame(rows)
    weeks = calendar_span(panel["iso_week"].unique().tolist())
    dense = densify(panel, ["AAA"], weeks)
    hole = dense.loc[dense["iso_week"] == "2010-W52"].iloc[0]
    assert hole["quantity"] == 0
    featured = add_lags_and_target(dense, _training())
    row = featured.loc[featured["iso_week"] == "2011-W01"].iloc[0]
    assert row["lag_1"] == 0
    assert row["lag_2"] == 30
    assert row["lag_4"] == 10


def test_price_is_not_backfilled_from_the_future():
    panel = pd.DataFrame(
        [
            {"StockCode": "AAA", "iso_week": "2011-W01", "quantity": 0, "unit_price": pd.NA},
            {"StockCode": "AAA", "iso_week": "2011-W02", "quantity": 5, "unit_price": 9.0},
        ]
    )
    dense = densify(panel, ["AAA"], ["2011-W01", "2011-W02"])
    assert pd.isna(dense.loc[dense["iso_week"] == "2011-W01", "unit_price"].iloc[0])
    assert dense.loc[dense["iso_week"] == "2011-W02", "unit_price"].iloc[0] == 9.0


def test_weeks_until_cuts_the_timeline():
    weeks = ["2010-W49", "2010-W50", "2011-W01"]
    assert weeks_until(weeks, "2010-W50") == ["2010-W49", "2010-W50"]


def test_load_panels_ignores_files_not_in_the_manifest(tmp_path):
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    good = pd.DataFrame(
        {
            "StockCode": ["AAA"],
            "iso_week": ["2011-W01"],
            "quantity": [1],
            "unit_price": [1.0],
            "revenue": [1.0],
            "n_lines": [1],
            "n_invoices": [1],
        }
    )
    junk = good.copy()
    junk["iso_week"] = "2099-W01"
    good.to_parquet(panel_dir / "2011-W01.parquet", index=False)
    junk.to_parquet(panel_dir / "2099-W01.parquet", index=False)
    loaded = load_panels(panel_dir, ["2011-W01"])
    assert set(loaded["iso_week"]) == {"2011-W01"}


def test_panel_listing_cached_for_five_seconds(tmp_configs, monkeypatch):
    from retail.api import features as feat

    app, _ = tmp_configs
    panel_dir = app.paths.interim_dir / "panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "StockCode": ["AAA"],
            "iso_week": ["2011-W20"],
            "quantity": [1.0],
            "unit_price": [1.0],
        }
    ).to_parquet(panel_dir / "2011-W20.parquet", index=False)
    feat.clear_feature_cache()
    calls = {"n": 0}
    original = feat._scan_panel_dir

    def wrapped(cfg):
        calls["n"] += 1
        return original(cfg)

    monkeypatch.setattr(feat, "_scan_panel_dir", wrapped)
    feat.panel_fingerprint(app)
    feat.panel_weeks(app)
    feat.panel_fingerprint(app)
    assert calls["n"] == 1
