import zipfile

import pytest

from retail.cli import main
from retail.data.make_batches import load_manifest
from retail.quality.errors import QualityError
from retail.quality.validate import generate_invalid_batch, validate_all


@pytest.mark.integration
def test_cli_quality_accepts_fake_batches_then_rejects_invalid_demo(tmp_configs, fake_retail):
    app, _ = tmp_configs
    app.paths.raw_dir.mkdir(parents=True)
    xlsx = app.paths.raw_dir / "Online Retail.xlsx"
    fake_retail.to_excel(xlsx, index=False)
    with zipfile.ZipFile(app.paths.raw_dir / "online_retail.zip", "w") as zf:
        zf.write(xlsx, arcname=xlsx.name)
    xlsx.unlink()

    assert main(["data"]) == 0
    assert main(["quality"]) == 0
    weeks = [m.week for m in load_manifest(app.paths.batches_dir)]
    assert (app.paths.reports_dir / "quality" / f"{weeks[0]}.json").is_file()
    assert (app.paths.interim_dir / "panel" / f"{weeks[0]}.parquet").is_file()

    assert main(["quality", "--invalid-demo"]) == 0
    invalid_report = app.paths.reports_dir / "invalid_demo" / "2099-W01.json"
    assert invalid_report.is_file()
    assert '"status": "failed"' in invalid_report.read_text(encoding="utf-8")
    assert not (app.paths.reports_dir / "quality" / "2099-W01.json").exists()


@pytest.mark.integration
def test_validate_all_fails_fast_on_corrupt_lot(tmp_configs, fake_retail):
    app, _ = tmp_configs
    app.paths.raw_dir.mkdir(parents=True)
    xlsx = app.paths.raw_dir / "Online Retail.xlsx"
    fake_retail.to_excel(xlsx, index=False)
    with zipfile.ZipFile(app.paths.raw_dir / "online_retail.zip", "w") as zf:
        zf.write(xlsx, arcname=xlsx.name)
    xlsx.unlink()
    assert main(["data"]) == 0

    first = load_manifest(app.paths.batches_dir)[0]
    generate_invalid_batch().to_parquet(app.paths.batches_dir / first.file, index=False)
    with pytest.raises(QualityError):
        validate_all(app)
    assert not (app.paths.interim_dir / "panel" / first.file).exists()
