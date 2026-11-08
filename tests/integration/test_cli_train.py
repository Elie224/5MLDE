import os
import zipfile
from pathlib import Path

import pytest
import yaml

from retail.cli import main
from retail.common import config as config_module


def _write_xlsx_zip(raw_dir: Path, frame) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    xlsx = raw_dir / "Online Retail.xlsx"
    frame.to_excel(xlsx, index=False)
    with zipfile.ZipFile(raw_dir / "online_retail.zip", "w") as zf:
        zf.write(xlsx, arcname=xlsx.name)
    xlsx.unlink()


@pytest.mark.integration
def test_cli_train_after_quality_and_refuses_without_it(tmp_configs, fake_retail_long):
    app, _ = tmp_configs
    cfg_path = Path(os.environ["RETAIL_CONFIG"])
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["panel"]["warmup_weeks"] = 5
    cfg["panel"]["top_n_products"] = 3
    cfg["training"]["test_weeks"] = 2
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    config_module.get_settings.cache_clear()

    _write_xlsx_zip(app.paths.raw_dir, fake_retail_long)
    assert main(["data"]) == 0
    assert main(["train"]) == 2

    assert main(["quality"]) == 0
    assert main(["train", "--skip-register"]) == 0
    assert not (app.paths.models_dir / "sklearn_pipeline.joblib").exists()
    assert (app.paths.reports_dir / "training" / "metrics.json").is_file()
    assert (app.paths.interim_dir / "top_skus.json").is_file()
    assert (app.paths.interim_dir / "features.parquet").is_file()
    assert os.environ["RETAIL_MLFLOW_TRACKING_URI"].startswith("file:")
