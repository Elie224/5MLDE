import pandas as pd

from retail.data.download import normalize_raw
from retail.data.make_batches import (
    load_manifest,
    load_missing_weeks,
    split_weekly,
    week_label,
    write_batches,
)


def test_week_label_is_iso():
    assert week_label(pd.Timestamp("2011-01-03")) == "2011-W01"
    assert week_label(pd.Timestamp("2010-12-27")) == "2010-W52"


def test_partial_edge_weeks_are_dropped(fake_retail):
    batches = split_weekly(normalize_raw(fake_retail), drop_partial_weeks=True)
    starts = sorted(batches)
    assert starts[0] == pd.Timestamp("2010-12-06")
    assert starts[-1] == pd.Timestamp("2011-01-10")
    assert all(s.dayofweek == 0 for s in starts)


def test_every_invoice_line_lands_in_its_own_week(fake_retail):
    df = normalize_raw(fake_retail)
    batches = split_weekly(df, drop_partial_weeks=False)
    assert sum(len(b) for b in batches.values()) == len(df)
    for start, part in batches.items():
        assert part["InvoiceDate"].min() >= start
        assert part["InvoiceDate"].max() < start + pd.Timedelta(days=7)


def test_write_batches_is_idempotent_and_traced(tmp_path, fake_retail):
    batches = split_weekly(normalize_raw(fake_retail))
    (tmp_path / "2099-W01.parquet").write_bytes(b"ancien lot")
    first = write_batches(batches, tmp_path)
    second = write_batches(batches, tmp_path)

    assert [b.content_hash for b in first] == [b.content_hash for b in second]
    assert not (tmp_path / "2099-W01.parquet").exists()
    manifest = load_manifest(tmp_path)
    assert [m.week for m in manifest] == [b.week for b in first]
    assert all((tmp_path / m.file).is_file() for m in manifest)
    assert load_missing_weeks(tmp_path) == []


def test_manifest_records_internal_missing_weeks(tmp_path):
    w49 = pd.Timestamp("2010-12-06")
    w51 = pd.Timestamp("2010-12-20")
    frame = pd.DataFrame({"InvoiceNo": ["1", "2"], "x": [1, 1]})
    write_batches({w49: frame, w51: frame}, tmp_path)
    assert load_missing_weeks(tmp_path) == ["2010-W50"]
