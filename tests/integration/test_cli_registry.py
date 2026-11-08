import json

import pytest

from retail.cli import main


@pytest.mark.integration
def test_cli_three_promotions_then_double_rollback(tmp_configs, register_dummy):
    app, _ = tmp_configs
    first = register_dummy(app, constant=0.0)
    second = register_dummy(app, constant=4.0)
    third = register_dummy(app, constant=8.0)
    assert [first["status"], second["status"], third["status"]] == ["champion"] * 3

    assert main(["rollback"]) == 0
    meta = json.loads((app.paths.models_dir / "champion" / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == second["model_version"]

    assert main(["rollback"]) == 0
    meta = json.loads((app.paths.models_dir / "champion" / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == first["model_version"]

    assert main(["rollback"]) == 2
    assert main(["rollback", "--to-version", third["model_version"]]) == 0
    meta = json.loads((app.paths.models_dir / "champion" / "meta.json").read_text(encoding="utf-8"))
    assert meta["model_version"] == third["model_version"]
