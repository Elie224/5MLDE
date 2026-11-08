from types import SimpleNamespace

from retail.cli import build_parser
from retail.demo import DemoError, pick_demo_weeks, run_demo


def test_pick_demo_weeks_prefers_w40_w41():
    weeks = [f"2011-W{i:02d}" for i in range(27, 49)]
    n, n1 = pick_demo_weeks(weeks)
    assert (n, n1) == ("2011-W40", "2011-W41")


def test_pick_demo_weeks_follows_cursor():
    weeks = [f"2011-W{i:02d}" for i in range(27, 49)]
    assert pick_demo_weeks(weeks, last_week="2011-W41") == ("2011-W41", "2011-W42")


def test_pick_demo_weeks_skips_load_week_after_cursor():
    try:
        pick_demo_weeks(
            ["2011-W46", "2011-W47", "2011-W48"],
            last_week="2011-W47",
        )
    except DemoError as exc:
        assert "apres" in str(exc)
    else:
        raise AssertionError("attendu DemoError")


def test_pick_demo_weeks_skips_load_week():
    n, n1 = pick_demo_weeks(
        ["2011-W46", "2011-W47", "2011-W48"],
        preferred=("2011-W99", "2011-W98"),
    )
    assert (n, n1) == ("2011-W46", "2011-W47")


def test_pick_demo_weeks_requires_two():
    try:
        pick_demo_weeks(["2011-W48"])
    except DemoError as exc:
        assert "Pas assez de semaines" in str(exc)
    else:
        raise AssertionError("attendu DemoError")


def test_cli_exposes_demo():
    args = build_parser().parse_args(["demo", "--skip-docker", "--reset-cursor"])
    assert args.command == "demo"
    assert args.skip_docker is True
    assert args.reset_cursor is True


def _fake_app(tmp_path):
    return SimpleNamespace(
        paths=SimpleNamespace(models_dir=tmp_path / "models", reports_dir=tmp_path / "reports")
    )


def test_run_demo_skip_docker_restores_champion(monkeypatch, tmp_path):
    from retail import demo

    app = _fake_app(tmp_path)
    monkeypatch.setattr(demo, "load_app_config", lambda: app)
    monkeypatch.setattr(demo, "reset_demo_cursor", lambda _app: None)
    monkeypatch.setattr(
        demo,
        "step_checks",
        lambda _app, **_k: {"model_version": "2", "degraded": False},
    )
    monkeypatch.setattr(demo, "step_nominal", lambda _app: {"week_n1": "2011-W41"})
    monkeypatch.setattr(demo, "step_invalid", lambda _app: None)
    monkeypatch.setattr(demo, "step_rollback", lambda _app, orig: "1")
    restored: list = []
    monkeypatch.setattr(demo, "restore_champion", lambda *args: restored.append(args))
    monkeypatch.setattr(demo, "step_bilan", lambda: None)
    assert run_demo(skip_docker=True) == 0
    assert restored == [(app, "2", "1")]


def test_run_demo_restarts_postgres_if_down_probe_fails(monkeypatch, tmp_path):
    from retail import demo

    app = _fake_app(tmp_path)
    monkeypatch.setattr(demo, "load_app_config", lambda: app)
    monkeypatch.setattr(demo, "reset_demo_cursor", lambda _app: None)
    monkeypatch.setattr(demo, "step_checks", lambda _app, **_k: {"model_version": "2"})
    monkeypatch.setattr(demo, "step_nominal", lambda _app: {"week_n1": "2011-W41"})
    monkeypatch.setattr(demo, "step_invalid", lambda _app: None)
    monkeypatch.setattr(demo, "_get", lambda path, **_k: {"skus": ["AAA"]})
    monkeypatch.setattr(demo, "stop_postgres", lambda: None)
    monkeypatch.setattr(
        demo, "step_postgres_down", lambda *_a: (_ for _ in ()).throw(DemoError("db_ok"))
    )
    restores: list[int] = []
    monkeypatch.setattr(demo, "step_restore_postgres", lambda: restores.append(1))
    try:
        run_demo(skip_docker=False)
    except DemoError:
        pass
    else:
        raise AssertionError("attendu DemoError")
    assert restores == [1]


def test_step_nominal_predicts_week_after_cursor(monkeypatch, tmp_path):
    from retail import demo

    weeks = ["2011-W40", "2011-W41", "2011-W42"]
    app = SimpleNamespace(
        paths=SimpleNamespace(interim_dir=tmp_path, models_dir=tmp_path / "models")
    )
    ticks: list[str] = []
    monkeypatch.setattr(demo, "manifest_weeks", lambda _app: weeks)
    monkeypatch.setattr(demo, "load_cursor", lambda _app: {"last_week": "2011-W41"})
    monkeypatch.setattr(
        demo,
        "_post",
        lambda *_a, **_k: {
            "predictions": [{"model_version": "5"}],
            "model_version": "5",
            "source": "mlflow",
        },
    )
    monkeypatch.setattr(demo, "run_tick", lambda _app, week=None, train=False: ticks.append(week))
    monkeypatch.setattr(demo, "_get", lambda *_a, **_k: {"production_wape": 0.5, "n_skus": 50})
    monkeypatch.setattr(demo.time, "sleep", lambda *_a, **_k: None)
    result = demo.step_nominal(app)
    assert ticks == ["2011-W42"]
    assert result["week_n"] == "2011-W41"
    assert result["week_n1"] == "2011-W42"


def test_step_postgres_down_waits_for_prom_alert(monkeypatch):
    from retail import demo

    monkeypatch.setattr(
        demo,
        "_post",
        lambda *_a, **_k: {"prediction": 1.0, "model_version": "5"},
    )
    monkeypatch.setattr(demo, "_get", lambda *_a, **_k: {"db_ok": False, "model_version": "5"})
    states = {"n": 0}

    def firing(_name):
        states["n"] += 1
        return states["n"] >= 2

    monkeypatch.setattr(demo, "prometheus_alert_firing", firing)
    monkeypatch.setattr(demo.time, "sleep", lambda *_a, **_k: None)
    demo.step_postgres_down("2011-W42", "AAA")
    assert states["n"] >= 2


def test_prepare_rollback_chain_unwinds_losing_champion(monkeypatch):
    from unittest.mock import MagicMock

    from retail import demo
    from retail.registry.promote import TAG_DECISION, TAG_PREVIOUS

    current = {"v": "5"}
    tags = {
        "5": {
            TAG_DECISION: "challenger: MAE 173.14 > champion 1 MAE 169.31",
            TAG_PREVIOUS: "1",
        },
        "1": {TAG_DECISION: "promoted: first champion"},
    }
    client = MagicMock()
    other = MagicMock()
    other.version = "5"
    client.search_model_versions.return_value = [other]

    monkeypatch.setattr(demo, "_client", lambda: client)
    monkeypatch.setattr(demo, "_model_name", lambda _app: "retail-weekly-qty")
    monkeypatch.setattr(demo, "_alias_version", lambda *_a, **_k: current["v"])
    monkeypatch.setattr(
        demo,
        "_version_tag",
        lambda _c, _n, ver, tag: tags.get(str(ver), {}).get(tag),
    )
    rolled: list[str] = []

    def _rollback(_app, to_version=None):
        rolled.append(str(to_version))
        current["v"] = str(to_version)

    monkeypatch.setattr(demo, "rollback_champion", _rollback)
    monkeypatch.setattr(demo, "_post", lambda *_a, **_k: None)
    monkeypatch.setattr(demo, "_promote", lambda *_a, **_k: current.update(v="5"))
    legal = demo.prepare_rollback_chain(SimpleNamespace())
    assert rolled == ["1"]
    assert legal == "1"


def test_reset_demo_cursor_writes_w40(monkeypatch, tmp_path):
    from retail import demo

    saved: list[tuple] = []
    app = SimpleNamespace()
    monkeypatch.setattr(
        demo,
        "save_cursor",
        lambda _app, week, status, **_k: saved.append((week, status)),
    )
    demo.reset_demo_cursor(app)
    assert saved == [("2011-W40", "ok")]
