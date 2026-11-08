import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _task_weight(decorator: ast.expr) -> int | None:
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    if name != "task" or not decorator.args:
        return None
    arg = decorator.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
        return arg.value
    return None


def test_locustfile_mix_and_shape():
    tree = ast.parse((ROOT / "load_tests" / "locustfile.py").read_text(encoding="utf-8"))
    weights: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "RetailUser":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            for decorator in item.decorator_list:
                weight = _task_weight(decorator)
                if weight is not None:
                    weights[item.name] = weight
    assert weights == {"predict_one": 70, "predict_batch": 20, "health": 10}

    source = (ROOT / "load_tests" / "locustfile.py").read_text(encoding="utf-8")
    runner = (ROOT / "load_tests" / "run.py").read_text(encoding="utf-8")
    assert "USER_LEVELS = (2, 5, 10, 20, 40, 80, 160, 320)" in source
    assert "USER_LEVELS = (2, 5, 10, 20, 40, 80, 160, 320)" in runner
    assert "timeout=REQUEST_TIMEOUT_S" in source
    assert "X-Load-Test" in source
    assert "self.client.timeout" not in source
    assert "/health/live" in source
    assert "models" in source and "champion" in source
    assert "--csv-full-history" in runner
    assert "degraded" in runner
    assert 'source != "disk"' in runner
    assert "code not in (0, 1)" in runner
    assert "timeout_worker_healthcheck" in (ROOT / "src" / "retail" / "api" / "serve.py").read_text(
        encoding="utf-8"
    )
    assert "summary.json" in runner and "except BaseException" in runner
    assert "_stats.csv" in runner
    assert "_locust_cpu_pct" in runner
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.index("requirements.lock") < dockerfile.index("COPY src")
    assert "README.md" in dockerfile
    assert "-e ." in dockerfile

    skus_path = ROOT / "data" / "interim" / "top_skus.json"
    if skus_path.is_file():
        payload = json.loads(skus_path.read_text(encoding="utf-8"))
        items = payload if isinstance(payload, list) else payload.get("skus", [])
        assert len(items) == 50


def test_compose_api_has_cpu_limits():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "load" in text
    assert "RETAIL_SKIP_REGISTRY_PROBE" in text
    assert "API_CPUS" in text
    assert "API_WORKERS" in text
    assert "cpus:" in text
    assert "mem_limit:" in text
    assert "retail.api.serve" in text
    assert "./src:/app/src" in text
    assert "PROMETHEUS_MULTIPROC_DIR: /tmp/prom" in text
    assert "tmpfs:" in text


def test_pick_capacity_stops_at_first_failing_stage():
    from load_tests.run import pick_capacity

    windows = [
        ("u20", 0.0, 45.0),
        ("u40", 45.0, 90.0),
        ("u80", 90.0, 135.0),
        ("u160", 135.0, 180.0),
    ]
    phases = {
        "u20": {"p95_ms": 10.0, "errors": 0.0, "rps": 100.0},
        "u40": {"p95_ms": 50.0, "errors": 0.0, "rps": 200.0},
        "u80": {"p95_ms": 250.0, "errors": 0.0, "rps": 300.0},
        "u160": {"p95_ms": 80.0, "errors": 0.0, "rps": 400.0},
    }
    chosen = pick_capacity(windows, phases)
    assert chosen is not None
    assert chosen["label"] == "u40"
    assert chosen["rps"] == 200.0


def test_load_runner_covers_three_configs():
    source = ast.parse((ROOT / "load_tests" / "run.py").read_text(encoding="utf-8"))
    configs = None
    for node in source.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "CONFIGS" in names and isinstance(node.value, ast.Tuple):
                configs = node.value.elts
    assert configs is not None
    assert len(configs) == 3
