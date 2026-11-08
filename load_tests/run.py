"""Protocole Locust identique pour 1 / 2 / 4 workers (limites CPU Docker)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "load_tests" / "results"
HOST = os.environ.get("LOCUST_HOST", "http://127.0.0.1:8001")
P95_LIMIT_MS = 200.0
USER_LEVELS = (2, 5, 10, 20, 40, 80, 160, 320)

CONFIGS = (
    {"name": "1w_0p5cpu", "workers": "1", "cpus": "0.50", "mem": "1g"},
    {"name": "2w_1cpu", "workers": "2", "cpus": "1.00", "mem": "1g"},
    {"name": "4w_2cpu", "workers": "4", "cpus": "2.00", "mem": "2g"},
)


def _compose(args: list[str], extra_env: dict[str, str]) -> None:
    env = {**os.environ, **extra_env}
    subprocess.run(
        ["docker", "compose", "--profile", "load", *args],
        cwd=ROOT,
        env=env,
        check=True,
    )


def _get_json(path: str, timeout: float = 5.0) -> dict:
    url = f"{HOST.rstrip('/')}{path}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_ready(timeout_s: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        try:
            live = _get_json("/health/live", timeout=5)
            if live.get("status") != "ok":
                last = f"/health/live inattendu : {live}"
                time.sleep(2)
                continue
            payload = _get_json("/health", timeout=8)
            last = str(payload)
            source = payload.get("source")
            if source != "disk":
                raise SystemExit(
                    "L'API sur le port 8001 n'est pas le conteneur de charge "
                    f"(source={source!r}, attendu 'disk'). Arrête l'API hôte "
                    "(netstat -ano | findstr :8001 puis taskkill /PID … /F) "
                    "puis relance python load_tests/run.py."
                )
            if payload.get("status") == "ok" and payload.get("degraded") is False:
                return payload
            last = f"degraded={payload.get('degraded')} {payload}"
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code} sur {exc.url}"
            if exc.code == 404:
                raise SystemExit(
                    "GET /health/live a renvoyé 404 : c'est l'ancienne API hôte, "
                    "pas le conteneur. Arrête le processus sur :8001 puis relance."
                ) from exc
        except SystemExit:
            raise
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise SystemExit(
        "API conteneur pas prête (attendu /health/live 200, source=disk, "
        f"degraded=false) sur {HOST} : {last}"
    )


def _metrics_dropped() -> float:
    try:
        with urllib.request.urlopen(f"{HOST.rstrip('/')}/metrics", timeout=5) as response:
            text = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError):
        return 0.0
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("retail_db_dropped_total"):
            continue
        try:
            total += float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return total


def _process_cpu_seconds(pid: int) -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    try:
        parent = psutil.Process(pid)
        procs = [parent, *parent.children(recursive=True)]
    except (psutil.Error, OSError):
        return None
    total = 0.0
    for proc in procs:
        try:
            times = proc.cpu_times()
        except (psutil.Error, OSError):
            continue
        total += float(times.user) + float(times.system)
    return total


def _locust_cpu_pct(pid: int, prev: dict) -> float | None:
    """Pourcentage d'un cœur (100 = 1 vCPU), via delta cpu_times — pas cpu_percent().

    psutil.Process.cpu_percent(interval=None) reste à 0 si on recrée l'objet à chaque
    échantillon (le précédent est stocké sur l'instance, pas sur le PID).
    """
    cpu = _process_cpu_seconds(pid)
    now = time.monotonic()
    if cpu is None:
        return None
    last_t = prev.get("t")
    last_cpu = prev.get("cpu")
    prev["t"] = now
    prev["cpu"] = cpu
    if last_t is None or last_cpu is None:
        return 0.0
    elapsed = now - last_t
    if elapsed <= 0:
        return 0.0
    return round(100.0 * (cpu - last_cpu) / elapsed, 2)


def _sample_stats(stop: threading.Event, rows: list[dict], locust_proc) -> None:
    prev: dict = {}
    if locust_proc is not None:
        _locust_cpu_pct(locust_proc.pid, prev)
    while not stop.wait(5):
        sample = {"ts": time.time(), "api_cpu": "", "api_mem": "", "locust_cpu_pct": None}
        try:
            raw = subprocess.check_output(
                [
                    "docker",
                    "stats",
                    "retail-api",
                    "--no-stream",
                    "--format",
                    "{{.CPUPerc}}\t{{.MemUsage}}",
                ],
                text=True,
                cwd=ROOT,
            ).strip()
            cpu, _, mem = raw.partition("\t")
            sample["api_cpu"] = cpu.strip()
            sample["api_mem"] = mem.strip()
        except subprocess.CalledProcessError:
            pass
        if locust_proc is not None:
            sample["locust_cpu_pct"] = _locust_cpu_pct(locust_proc.pid, prev)
        rows.append(sample)


def _locust(csv_prefix: Path, extra_env: dict[str, str]) -> subprocess.Popen:
    csv_prefix.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **extra_env}
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "locust",
            "-f",
            str(ROOT / "load_tests" / "locustfile.py"),
            "--headless",
            "--host",
            HOST,
            "--csv",
            str(csv_prefix),
            "--csv-full-history",
            "--only-summary",
        ],
        cwd=ROOT,
        env=env,
    )


def _f(row: dict, *keys: str) -> float:
    for key in keys:
        raw = row.get(key)
        if raw not in (None, ""):
            try:
                return float(raw)
            except ValueError:
                continue
    return 0.0


def _phase_windows(*, smoke: bool) -> list[tuple[str, float, float]]:
    if smoke:
        return [("smoke", 0.0, 30.0)]
    stage_s = float(os.environ.get("LOCUST_STAGE_SECONDS", "45"))
    windows: list[tuple[str, float, float]] = []
    previous = 0.0
    for users in USER_LEVELS:
        end = previous + stage_s
        windows.append((f"u{users}", previous, end))
        previous = end
    return windows


def _summarize_history(csv_prefix: Path, *, smoke: bool) -> dict:
    path = Path(str(csv_prefix) + "_stats_history.csv")
    phases: dict[str, dict] = {}
    endpoints: dict[str, dict] = {}
    if not path.is_file():
        return {"phases": phases, "endpoints": endpoints, "capacity": None}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"phases": phases, "endpoints": endpoints, "capacity": None}
    t0 = _f(rows[0], "Timestamp")
    windows = _phase_windows(smoke=smoke)
    names = {row.get("Name", "") for row in rows}

    def last_in_window(name: str, start: float, end: float) -> dict | None:
        chosen = None
        for row in rows:
            if row.get("Name") != name:
                continue
            elapsed = _f(row, "Timestamp") - t0
            if start <= elapsed < end:
                chosen = row
        return chosen

    def first_in_window(name: str, start: float, end: float) -> dict | None:
        for row in rows:
            if row.get("Name") != name:
                continue
            elapsed = _f(row, "Timestamp") - t0
            if start <= elapsed < end:
                return row
        return None

    for label, start, end in windows:
        snapshot = last_in_window("Aggregated", start, end) or last_in_window(
            "/predict", start, end
        )
        first = first_in_window("Aggregated", start, end)
        if snapshot is None:
            continue
        duration = max(end - start, 1.0)
        reqs = _f(snapshot, "Total Request Count", "Request Count")
        fails = _f(snapshot, "Total Failure Count", "Failure Count")
        if first is not None:
            reqs -= _f(first, "Total Request Count", "Request Count")
            fails -= _f(first, "Total Failure Count", "Failure Count")
        rps = reqs / duration
        error_pct = (fails / reqs) if reqs else 0.0
        p95 = _f(snapshot, "95%")
        phases[label] = {
            "rps": round(rps, 2),
            "p95_ms": p95,
            "errors": round(error_pct, 4),
            "requests": int(max(reqs, 0)),
            "users": snapshot.get("User Count"),
        }

    stats_path = Path(str(csv_prefix) + "_stats.csv")
    if stats_path.is_file():
        with stats_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                name = row.get("Name") or row.get("Type") or ""
                if not name or name == "None":
                    continue
                endpoints[name] = {
                    "rps": _f(row, "Requests/s"),
                    "p95_ms": _f(row, "95%"),
                    "p50_ms": _f(row, "50%"),
                    "p99_ms": _f(row, "99%"),
                    "requests": int(_f(row, "Request Count")),
                    "failures": int(_f(row, "Failure Count")),
                }
    else:
        for name in sorted(n for n in names if n and n != "None"):
            last = None
            for row in rows:
                if row.get("Name") == name:
                    last = row
            if last is None:
                continue
            endpoints[name] = {
                "rps": _f(last, "Requests/s"),
                "p95_ms": _f(last, "95%"),
                "p50_ms": _f(last, "50%"),
                "p99_ms": _f(last, "99%"),
                "requests": int(_f(last, "Total Request Count", "Request Count")),
                "failures": int(_f(last, "Total Failure Count", "Failure Count")),
            }

    capacity = pick_capacity(windows, phases)
    return {"phases": phases, "endpoints": endpoints, "capacity": capacity}


def pick_capacity(windows: list[tuple[str, float, float]], phases: dict) -> dict | None:
    """Dernier palier valide avant le premier échec (p95 ≥ 200 ms ou erreurs)."""
    capacity = None
    for label, _, _ in windows:
        phase = phases.get(label)
        if not phase or phase["p95_ms"] >= P95_LIMIT_MS or phase["errors"] > 0:
            break
        capacity = {**phase, "label": label}
    return capacity


def run_one(config: dict[str, str], *, smoke: bool) -> dict:
    env = {
        "API_WORKERS": config["workers"],
        "API_CPUS": config["cpus"],
        "API_MEM": config["mem"],
    }
    locust_env = {}
    if smoke:
        locust_env["LOCUST_SMOKE"] = "1"
    prefix = RESULTS / ("smoke_" + config["name"] if smoke else config["name"])
    print(f"== {prefix.name} workers={config['workers']} cpus={config['cpus']} ==", flush=True)
    _compose(["up", "-d", "--build", "api"], env)
    try:
        try:
            health = _wait_ready()
        except SystemExit:
            subprocess.run(["docker", "logs", "retail-api", "--tail", "80"], check=False)
            inspect = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "retail-api",
                    "--format",
                    "exit={{.State.ExitCode}} oom={{.State.OOMKilled}}",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            print(inspect.stdout or inspect.stderr, flush=True)
            raise
        dropped_before = _metrics_dropped()
        stop = threading.Event()
        samples: list[dict] = []
        proc = _locust(prefix, locust_env)
        sampler = threading.Thread(target=_sample_stats, args=(stop, samples, proc), daemon=True)
        sampler.start()
        code = proc.wait()
        stop.set()
        sampler.join(timeout=8)
        if code not in (0, 1):
            raise SystemExit(f"Locust a quitté avec le code {code}")
        dropped_after = _metrics_dropped()
        history = _summarize_history(prefix, smoke=smoke)
        return {
            "name": prefix.name,
            "workers": config["workers"],
            "cpus": config["cpus"],
            "health": {
                "degraded": health.get("degraded"),
                "source": health.get("source"),
                "model_version": health.get("model_version"),
                "db_backend": health.get("db_backend"),
            },
            "db_dropped_delta": dropped_after - dropped_before,
            "cpu_note": "docker stats CPU% = part d'un cœur (100% = 1 vCPU)",
            "single_machine": True,
            "cpu_samples": samples,
            **history,
        }
    finally:
        _compose(["stop", "api"], env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="30 s / 5 users, une seule config")
    parser.add_argument("--only", default=None, help="Nom de config (ex. 1w_0p5cpu)")
    args = parser.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    chosen = [item for item in CONFIGS if args.only is None or item["name"] == args.only]
    if not chosen:
        raise SystemExit(f"Config inconnue : {args.only}")
    if args.smoke:
        chosen = chosen[:1]
    reports: list[dict] = []
    failed = False
    for config in chosen:
        name = "smoke_" + config["name"] if args.smoke else config["name"]
        try:
            reports.append(run_one(config, smoke=args.smoke))
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            failed = True
            message = str(exc) or type(exc).__name__
            print(f"Échec {name}: {message}", flush=True)
            reports.append(
                {
                    "name": name,
                    "workers": config["workers"],
                    "cpus": config["cpus"],
                    "error": message,
                }
            )
    summary_path = RESULTS / ("summary_smoke.json" if args.smoke else "summary.json")
    summary_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))
    print(f"Résumé : {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
