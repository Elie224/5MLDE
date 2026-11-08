import zipfile

import pytest

from retail.cli import main
from retail.data.download import sha256_file
from retail.data.make_batches import load_manifest


@pytest.mark.integration
def test_cli_data_end_to_end(tmp_configs, fake_retail):
    app, _ = tmp_configs
    app.paths.raw_dir.mkdir(parents=True)
    xlsx = app.paths.raw_dir / "Online Retail.xlsx"
    fake_retail.to_excel(xlsx, index=False)
    with zipfile.ZipFile(app.paths.raw_dir / "online_retail.zip", "w") as zf:
        zf.write(xlsx, arcname=xlsx.name)
    xlsx.unlink()

    assert main(["data"]) == 0
    manifest = load_manifest(app.paths.batches_dir)
    assert [m.week for m in manifest][:2] == ["2010-W49", "2010-W50"]
    assert sha256_file(app.paths.raw_dir / "online_retail.zip")


@pytest.mark.integration
def test_cli_returns_error_code_when_source_unavailable(tmp_configs, monkeypatch):
    from retail.data import download as dl

    def no_network(*a, **k):
        raise dl.requests.ConnectionError("hors ligne")

    monkeypatch.setattr(dl.requests, "get", no_network)
    monkeypatch.setattr(dl.time, "sleep", lambda _: None)
    assert main(["download"]) == 2
