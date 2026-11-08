from pathlib import Path

import pytest
import yaml

from retail.common.config import REPO_ROOT, ConfigError, load_app_config, load_data_config

ROOT = Path(__file__).resolve().parents[2]


def test_repository_configs_are_valid():
    app = load_app_config(ROOT / "configs/config.yaml")
    data = load_data_config(ROOT / "configs/data.yaml")
    assert app.promotion.registered_model_name == "retail-weekly-qty"
    assert app.orchestration.interval_seconds == 600
    assert app.panel.top_n_products > 0
    assert app.panel.warmup_weeks >= 5
    assert str(data.source.url).startswith("https://archive.ics.uci.edu/")
    assert app.paths.raw_dir.is_absolute()


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="introuvable"):
        load_app_config(tmp_path / "absent.yaml")


def test_invalid_value_is_rejected_at_startup(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))
    cfg["quality"]["max_share_non_positive_price"] = 3
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="max_share_non_positive_price"):
        load_app_config(path)


def test_env_selects_config_file(tmp_configs):
    app, _ = tmp_configs
    assert "configs" not in str(app.paths.raw_dir)


def test_relative_paths_resolve_to_repo_root_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from retail.common import config as config_module

    config_module.get_settings.cache_clear()
    app = load_app_config()
    assert app.paths.raw_dir == REPO_ROOT / "data" / "raw"
    assert app.paths.batches_dir == REPO_ROOT / "data" / "batches"
    config_module.get_settings.cache_clear()


def test_cwd_dotenv_is_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("RETAIL_LOG_LEVEL=CRITICAL\n", encoding="utf-8")
    from retail.common import config as config_module

    config_module.get_settings.cache_clear()
    assert config_module.get_settings().log_level != "CRITICAL"
    config_module.get_settings.cache_clear()
