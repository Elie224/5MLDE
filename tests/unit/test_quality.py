import json

import pandas as pd
import pytest

from retail.common.config import QualityCfg
from retail.data.download import RAW_COLUMNS, normalize_raw
from retail.data.make_batches import split_weekly, week_label
from retail.quality.clean import DROP_REASONS, clean_invoices, to_week_panel
from retail.quality.errors import QualityError
from retail.quality.schemas import invoice_schema, iso_week_bounds, panel_schema, run_schema
from retail.quality.validate import _unlink_retry, generate_invalid_batch, validate_batch


def _quality(**overrides) -> QualityCfg:
    base = dict(
        min_rows_per_batch=5,
        max_share_non_positive_price=0.5,
        country="United Kingdom",
        max_abs_quantity=10000,
        max_unit_price=1000.0,
        min_distinct_skus=1,
        non_product_codes=["POST", "D", "M", "DOT", "BANK CHARGES"],
    )
    base.update(overrides)
    return QualityCfg(**base)


def _invoices(fake_retail: pd.DataFrame) -> pd.DataFrame:
    return normalize_raw(fake_retail)


def test_iso_week_bounds_are_monday_to_next_monday():
    start, end = iso_week_bounds("2010-W49")
    assert start == pd.Timestamp("2010-12-06")
    assert end == pd.Timestamp("2010-12-13")
    assert start.dayofweek == 0


def test_invoice_schema_accepts_a_complete_week(fake_retail):
    df = _invoices(fake_retail)
    start = pd.Timestamp("2010-12-06")
    part = df[(df["InvoiceDate"] >= start) & (df["InvoiceDate"] < start + pd.Timedelta(days=7))]
    run_schema(invoice_schema("2010-W49", _quality()), part, "InvoiceSchema")


def test_invoice_schema_rejects_too_few_rows(fake_retail):
    df = _invoices(fake_retail).head(3)
    with pytest.raises(QualityError, match="InvoiceSchema"):
        run_schema(
            invoice_schema("2010-W49", _quality(min_rows_per_batch=500)),
            df,
            "InvoiceSchema",
        )


def test_invoice_schema_rejects_dates_outside_week(fake_retail):
    df = _invoices(fake_retail)
    with pytest.raises(QualityError, match="InvoiceSchema"):
        run_schema(invoice_schema("2010-W49", _quality()), df, "InvoiceSchema")


def test_invoice_schema_rejects_too_many_zero_prices():
    start = pd.Timestamp("2010-12-06")
    df = pd.DataFrame(
        {
            "InvoiceNo": ["1", "2", "3", "4"],
            "StockCode": ["85123A"] * 4,
            "Description": ["X"] * 4,
            "Quantity": [1] * 4,
            "InvoiceDate": [start] * 4,
            "UnitPrice": [0.0, 0.0, 0.0, 1.0],
            "CustomerID": [None] * 4,
            "Country": ["United Kingdom"] * 4,
        }
    )
    with pytest.raises(QualityError, match="InvoiceSchema"):
        run_schema(
            invoice_schema(
                "2010-W49",
                _quality(max_share_non_positive_price=0.05, min_rows_per_batch=4),
            ),
            df,
            "InvoiceSchema",
        )


def test_cleaning_traces_each_exclusion_once():
    start = pd.Timestamp("2010-12-06")
    df = pd.DataFrame(
        {
            "InvoiceNo": ["1", "C2", "3", "4", "5", "6", "7"],
            "StockCode": ["85123A", "85123A", "85123A", "85123A", "POST", "85123A", "85123A"],
            "Description": ["X"] * 7,
            "Quantity": [2, -2, 0, 2, 1, 20000, 3],
            "InvoiceDate": [start] * 7,
            "UnitPrice": [1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0],
            "CustomerID": [1] * 7,
            "Country": [
                "France",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
                "United Kingdom",
            ],
        }
    )
    kept, counts = clean_invoices(df, _quality())
    assert list(kept["InvoiceNo"]) == ["7"]
    assert counts["not_united_kingdom"] == 1
    assert counts["cancellation"] == 1
    assert counts["non_positive_quantity"] == 1
    assert counts["non_positive_price"] == 1
    assert counts["non_product_code"] == 1
    assert counts["outlier"] == 1
    assert set(counts) == set(DROP_REASONS)


def test_missing_country_is_dropped_as_not_uk():
    start = pd.Timestamp("2010-12-06")
    df = pd.DataFrame(
        {
            "InvoiceNo": ["1", "2"],
            "StockCode": ["85123A", "85123A"],
            "Description": ["X", "X"],
            "Quantity": [2, 2],
            "InvoiceDate": [start, start],
            "UnitPrice": [1.0, 1.0],
            "CustomerID": [1, 1],
            "Country": [pd.NA, "United Kingdom"],
        }
    )
    kept, counts = clean_invoices(df, _quality())
    assert counts["not_united_kingdom"] == 1
    assert list(kept["InvoiceNo"]) == ["2"]


def test_panel_is_one_row_per_sku_non_negative(fake_retail):
    cleaned, _ = clean_invoices(_invoices(fake_retail), _quality())
    week = "2010-W49"
    start = pd.Timestamp("2010-12-06")
    part = cleaned[
        (cleaned["InvoiceDate"] >= start) & (cleaned["InvoiceDate"] < start + pd.Timedelta(days=7))
    ]
    panel = to_week_panel(part, week)
    run_schema(panel_schema(week), panel, "PanelSchema")
    assert panel["StockCode"].is_unique
    assert (panel["quantity"] >= 0).all()
    assert set(panel["iso_week"].unique()) == {week}


def test_panel_schema_rejects_duplicates_and_negative_qty():
    week = "2010-W49"
    bad = pd.DataFrame(
        {
            "StockCode": ["A", "A"],
            "iso_week": [week, week],
            "quantity": [1, -4],
            "unit_price": [1.0, 1.0],
            "revenue": [1.0, 1.0],
            "n_lines": [1, 1],
            "n_invoices": [1, 1],
        }
    )
    with pytest.raises(QualityError, match="PanelSchema"):
        run_schema(panel_schema(week), bad, "PanelSchema")


def test_validate_batch_writes_report_and_stops_on_invalid(tmp_configs):
    app, _ = tmp_configs
    app.paths.batches_dir.mkdir(parents=True)
    generate_invalid_batch().to_parquet(app.paths.batches_dir / "2099-W01.parquet", index=False)
    with pytest.raises(QualityError, match="InvoiceSchema"):
        validate_batch(app, "2099-W01")
    report = app.paths.reports_dir / "quality" / "2099-W01.json"
    assert report.is_file()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["batch_content_hash"]
    assert payload["quality"]["min_rows_per_batch"] == app.quality.min_rows_per_batch
    assert "contrôle(s) en échec" in payload["error"]
    assert not (app.paths.interim_dir / "panel" / "2099-W01.parquet").exists()


def test_validate_batch_keeps_cleaned_outputs_on_success(tmp_configs, fake_retail):
    app, _ = tmp_configs
    df = _invoices(fake_retail)
    batches = split_weekly(df)
    start = min(batches)
    week = week_label(start)
    part = batches[start]
    app.paths.batches_dir.mkdir(parents=True)
    part.to_parquet(app.paths.batches_dir / f"{week}.parquet", index=False)
    report = validate_batch(app, week)
    assert report["status"] == "ok"
    assert report["rows_out"] < report["rows_in"]
    assert (app.paths.interim_dir / "cleaned" / f"{week}.parquet").is_file()
    assert (app.paths.interim_dir / "panel" / f"{week}.parquet").is_file()
    panel = pd.read_parquet(app.paths.interim_dir / "panel" / f"{week}.parquet")
    assert list(panel.columns) == [
        "StockCode",
        "iso_week",
        "quantity",
        "unit_price",
        "revenue",
        "n_lines",
        "n_invoices",
    ]
    assert set(RAW_COLUMNS) <= set(part.columns)
    assert report["batch_content_hash"]
    assert report["quality"]["country"] == app.quality.country


def test_rejected_week_removes_stale_panel(tmp_configs, fake_retail):
    app, _ = tmp_configs
    df = _invoices(fake_retail)
    batches = split_weekly(df)
    start = min(batches)
    week = week_label(start)
    part = batches[start]
    app.paths.batches_dir.mkdir(parents=True)
    batch_path = app.paths.batches_dir / f"{week}.parquet"
    part.to_parquet(batch_path, index=False)
    assert validate_batch(app, week)["status"] == "ok"
    assert (app.paths.interim_dir / "panel" / f"{week}.parquet").is_file()

    generate_invalid_batch().to_parquet(batch_path, index=False)
    with pytest.raises(QualityError, match="InvoiceSchema"):
        validate_batch(app, week)
    assert not (app.paths.interim_dir / "panel" / f"{week}.parquet").exists()
    assert not (app.paths.interim_dir / "cleaned" / f"{week}.parquet").exists()


def test_unlink_retry_succeeds_after_permission_error(tmp_path, monkeypatch):
    from pathlib import Path

    from retail.quality import validate as validate_mod

    target = tmp_path / "locked.parquet"
    target.write_bytes(b"x")
    hits = {"n": 0}
    original = Path.unlink

    def flaky(self, missing_ok=False):
        if self == target and hits["n"] < 2:
            hits["n"] += 1
            raise PermissionError(32, "locked")
        return original(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky)
    monkeypatch.setattr(validate_mod.time, "sleep", lambda *_a, **_k: None)
    _unlink_retry(target, attempts=5, delay=0)
    assert not target.exists()
    assert hits["n"] == 2
