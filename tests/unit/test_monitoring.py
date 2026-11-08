import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MONITORING = ROOT / "monitoring"


def test_prometheus_scrapes_api_and_loads_alerts():
    prom = yaml.safe_load((MONITORING / "prometheus.yml").read_text(encoding="utf-8"))
    jobs = {job["job_name"]: job for job in prom["scrape_configs"]}
    assert jobs["retail-api"]["metrics_path"] == "/metrics"
    assert "host.docker.internal:8001" in jobs["retail-api"]["static_configs"][0]["targets"]
    assert "/etc/prometheus/alerts.yml" in prom["rule_files"]

    alerts = yaml.safe_load((MONITORING / "alerts.yml").read_text(encoding="utf-8"))
    names = {rule["alert"] for group in alerts["groups"] for rule in group["rules"]}
    assert names == {"RetailDegraded", "RetailHigh5xx", "RetailApiDown", "RetailDbDown"}
    rules = [rule for group in alerts["groups"] for rule in group["rules"]]
    degraded = next(rule for rule in rules if rule["alert"] == "RetailDegraded")
    assert degraded["for"] == "1m"
    assert "retail_model_degraded" in degraded["expr"]
    down = next(rule for rule in rules if rule["alert"] == "RetailApiDown")
    assert down["for"] == "30s"
    assert 'up{job="retail-api"} == 0' in down["expr"]
    db_down = next(rule for rule in rules if rule["alert"] == "RetailDbDown")
    assert db_down["for"] == "15s"
    assert "retail_db_ok" in db_down["expr"]


def test_grafana_provisioning_is_complete():
    ds_path = MONITORING / "grafana/provisioning/datasources/datasources.yml"
    sources = yaml.safe_load(ds_path.read_text(encoding="utf-8"))
    by_uid = {item["uid"]: item for item in sources["datasources"]}
    assert by_uid["prometheus"]["url"] == "http://prometheus:9090"
    assert by_uid["postgres"]["type"] == "postgres"
    assert by_uid["postgres"]["url"] == "postgres:5432"
    assert by_uid["postgres"]["database"] == "${POSTGRES_DB}"
    assert by_uid["postgres"]["jsonData"]["database"] == "${POSTGRES_DB}"

    prov_path = MONITORING / "grafana/provisioning/dashboards/dashboards.yml"
    providers = yaml.safe_load(prov_path.read_text(encoding="utf-8"))
    assert providers["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"

    dash_path = MONITORING / "grafana/dashboards/retail-mlops.json"
    dash = json.loads(dash_path.read_text(encoding="utf-8"))
    titles = {panel["title"] for panel in dash["panels"]}
    assert "Requêtes / s" in titles
    assert "Latence p50 / p95 / p99" in titles
    assert "WAPE de production par version" in titles
    assert "WAPE par semaine" in titles
    assert "Exécutions du pipeline" in titles
    assert "Alertes du pipeline (qualité, webhook)" in titles
    assert "Postgres OK" in titles
    assert "Postgres (retail_db_ok)" in titles
    assert "Alertes Prometheus (API, Postgres, mode dégradé)" in titles
    uids = {panel["datasource"]["uid"] for panel in dash["panels"]}
    assert uids == {"prometheus", "postgres"}
    by_title = {panel["title"]: panel for panel in dash["panels"]}
    rps = by_title["Requêtes / s"]["targets"][0]["expr"]
    assert 'path!~"/metrics|/health.*"' in rps
    latency = by_title["Latence p50 / p95 / p99"]["targets"][0]["expr"]
    assert "sum by (le)" in latency
    assert "sum by (le, path)" not in latency
    version = by_title["Version servie"]
    assert version["options"]["textMode"] == "name"
    assert "max by (model_version, source)" in version["targets"][0]["expr"]
    assert "retail_model_version > 0.5" in version["targets"][0]["expr"]
    assert by_title["Version servie"]["targets"][0]["instant"] is True
    runs_sql = by_title["Exécutions du pipeline"]["targets"][0]["rawSql"]
    assert "ORDER BY created_at DESC" in runs_sql
    assert "LIMIT 20" in runs_sql
    assert "trained" in runs_sql
    week_sql = by_title["WAPE par semaine"]["targets"][0]["rawSql"]
    assert "iso_week" in week_sql
    assert "loadtest" in week_sql
    assert by_title["WAPE par semaine"]["type"] == "barchart"
    assert by_title["WAPE par semaine"]["targets"][0]["format"] == "table"
    firing = by_title["Alertes Prometheus (API, Postgres, mode dégradé)"]["targets"][0]["expr"]
    assert "max by (alertname, alertstate) (ALERTS)" in firing
    assert by_title["Alertes Prometheus (API, Postgres, mode dégradé)"]["type"] == "state-timeline"
    dropped = by_title["Écritures SQL abandonnées (par minute)"]["targets"][0]["expr"]
    assert "increase(retail_db_dropped_total[1m])" in dropped
    wape_sql = by_title["WAPE de production par version"]["targets"][0]["rawSql"]
    assert "DISTINCT ON (stock_code, iso_week, model_version)" in wape_sql
    assert dash["refresh"] == "10s"
    rate_5xx = by_title["Taux 5xx"]["targets"][0]["expr"]
    assert "or vector(0)" in rate_5xx
    errors = by_title["Erreurs 4xx / 5xx"]["targets"][0]["expr"]
    assert "or vector(0)" in errors
    assert "retail_db_ok" in by_title["Postgres OK"]["targets"][0]["expr"]


def test_latency_histogram_includes_5s_and_10s_buckets():
    text = (ROOT / "src" / "retail" / "api" / "app.py").read_text(encoding="utf-8")
    block = text.split("LATENCY_BUCKETS")[1].split(")")[0]
    assert "5.0" in block
    assert "10.0" in block


def test_grafana_password_comes_from_env():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    grafana = compose["services"]["grafana"]["environment"]
    assert grafana["GF_SECURITY_ADMIN_PASSWORD"] == "${GRAFANA_ADMIN_PASSWORD:-admin}"
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GRAFANA_ADMIN_PASSWORD=" in example


def test_api_cli_exposes_workers():
    from retail.cli import build_parser

    args = build_parser().parse_args(["api", "--workers", "4"])
    assert args.workers == 4


def test_cli_does_not_import_train_at_module_level():
    import ast

    tree = ast.parse((ROOT / "src" / "retail" / "cli.py").read_text(encoding="utf-8"))
    top_modules = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            top_modules.add(node.module)
    assert "retail.training.train" not in top_modules
    assert "retail.orchestration.replay" not in top_modules
    assert "retail.api.app" not in top_modules
    assert "retail.features.panel" not in top_modules
    assert "retail.demo" not in top_modules


def test_api_serve_module_stays_light():
    import ast

    tree = ast.parse((ROOT / "src" / "retail" / "api" / "serve.py").read_text(encoding="utf-8"))
    top_modules = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            top_modules.add(node.module)
        elif isinstance(node, ast.Import):
            top_modules.update(alias.name for alias in node.names)
    assert "uvicorn" not in top_modules
    assert "retail.cli" not in top_modules
    assert "retail.api.app" not in top_modules
