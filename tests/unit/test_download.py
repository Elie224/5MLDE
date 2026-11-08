import json
import zipfile

import pandas as pd
import pytest
import requests

from retail.data import download as dl
from retail.data.download import (
    DataSourceError,
    content_hash,
    ingest,
    normalize_raw,
    sha256_file,
    verify_checksum,
)


def _make_archive(tmp_path, df, excel_name="Online Retail.xlsx"):
    xlsx = tmp_path / excel_name
    df.to_excel(xlsx, index=False)
    archive = tmp_path / "online_retail.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(xlsx, arcname=excel_name)
    xlsx.unlink()
    return archive


def test_verify_checksum_accepts_matching_hash(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"retail")
    assert verify_checksum(f, sha256_file(f)) == sha256_file(f)


def test_verify_checksum_blocks_on_mismatch(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"retail")
    with pytest.raises(DataSourceError, match="inattendue"):
        verify_checksum(f, "0" * 64)


def test_normalize_raw_types_and_rejects_missing_columns(fake_retail):
    out = normalize_raw(fake_retail)
    assert out["Quantity"].dtype == "int64"
    assert str(out["CustomerID"].dtype) == "Int64"
    assert out["InvoiceDate"].is_monotonic_increasing
    with pytest.raises(DataSourceError, match="Colonnes manquantes"):
        normalize_raw(fake_retail.drop(columns=["UnitPrice"]))


def test_content_hash_is_deterministic(fake_retail):
    df = normalize_raw(fake_retail)
    assert content_hash(df) == content_hash(df.copy())
    assert content_hash(df) != content_hash(df.iloc[1:])


def test_ingest_from_local_archive_writes_parquet_and_manifest(tmp_configs, fake_retail):
    app, data = tmp_configs
    app.paths.raw_dir.mkdir(parents=True)
    archive = _make_archive(app.paths.raw_dir, fake_retail)
    data.source.sha256 = sha256_file(archive)

    manifest = ingest(app, data)

    parquet = pd.read_parquet(manifest.parquet_path)
    assert len(parquet) == len(fake_retail) == manifest.rows
    saved = json.loads((app.paths.interim_dir / dl.MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert saved["archive_sha256"] == data.source.sha256


def test_ingest_stops_when_source_changed(tmp_configs, fake_retail):
    app, data = tmp_configs
    app.paths.raw_dir.mkdir(parents=True)
    _make_archive(app.paths.raw_dir, fake_retail)
    data.source.sha256 = "f" * 64
    with pytest.raises(DataSourceError):
        ingest(app, data)
    assert not (app.paths.interim_dir / dl.INTERIM_FILENAME).exists()


def test_download_gives_up_after_max_attempts(tmp_path, monkeypatch):
    calls = []

    def failing_get(*args, **kwargs):
        calls.append(1)
        raise requests.ConnectionError("réseau coupé")

    monkeypatch.setattr(dl.requests, "get", failing_get)
    monkeypatch.setattr(dl.time, "sleep", lambda _: None)
    with pytest.raises(DataSourceError, match="Repli manuel"):
        dl.download_archive(
            "https://example.org/x.zip", tmp_path / "x.zip", timeout=1, max_attempts=2
        )
    assert len(calls) == 2
    assert not (tmp_path / "x.zip.part").exists()
