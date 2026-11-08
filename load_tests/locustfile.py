"""Charge : 70 % prédiction unitaire, 20 % lot, 10 % santé légère.

Deux modes :
- par défaut : paliers 2→320 users (RetailShape) pour mesurer la capacité ;
- ``LOCUST_CONSTANT=1`` : pas de paliers, trafic continu et modéré (≈1 req/s par user)
  piloté par ``-u`` / ``-r`` / ``-t`` ; sans ``-t``, tourne jusqu'à Ctrl+C.
"""

from __future__ import annotations

import json
import os
import random
import threading
import urllib.request
from pathlib import Path

from locust import HttpUser, LoadTestShape, between, task

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEEK = "2011-W48"
USER_LEVELS = (2, 5, 10, 20, 40, 80, 160, 320)
STAGE_SECONDS = int(os.environ.get("LOCUST_STAGE_SECONDS", "45"))
CONSTANT_MODE = os.environ.get("LOCUST_CONSTANT", "").strip().lower() in {"1", "true", "yes"}

_skus_lock = threading.Lock()
_skus: list[str] | None = None


def _skus_from_file(path: Path) -> list[str]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("skus", [])
    return [str(item) for item in items if str(item).strip()]


def load_skus_from_disk() -> list[str]:
    for candidate in (
        REPO_ROOT / "models" / "champion" / "top_skus.json",
        REPO_ROOT / "data" / "interim" / "top_skus.json",
    ):
        skus = _skus_from_file(candidate)
        if skus:
            return skus
    return ["85123A"]


def load_shape_stages() -> tuple[dict, ...]:
    if os.environ.get("LOCUST_SMOKE", "").strip() in {"1", "true", "yes"}:
        return ({"duration": 30, "users": 5, "spawn_rate": 5.0, "label": "smoke"},)
    elapsed = 0
    stages: list[dict] = []
    for users in USER_LEVELS:
        elapsed += STAGE_SECONDS
        stages.append(
            {
                "duration": elapsed,
                "users": users,
                "spawn_rate": float(max(users / 5.0, 2.0)),
                "label": f"u{users}",
            }
        )
    return tuple(stages)


WEEK = os.environ.get("LOCUST_WEEK", DEFAULT_WEEK)
REQUEST_TIMEOUT_S = 5.0
LOAD_HEADERS = {"X-Load-Test": "1"}


class RetailUser(HttpUser):
    # Capacité : pause courte pour saturer ; continu : ~1 req/s par user, bien sous la capacité.
    wait_time = between(0.5, 1.5) if CONSTANT_MODE else between(0.05, 0.2)

    def on_start(self) -> None:
        global _skus
        if _skus:
            return
        with _skus_lock:
            if _skus:
                return
            fetched: list[str] = []
            try:
                url = f"{self.host.rstrip('/')}/model"
                with urllib.request.urlopen(url, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                fetched = [str(item) for item in payload.get("skus") or [] if str(item).strip()]
            except (OSError, TimeoutError, json.JSONDecodeError, ValueError):
                fetched = []
            _skus = fetched or load_skus_from_disk()

    def _sku(self) -> str:
        pool = _skus or load_skus_from_disk()
        return random.choice(pool)

    @task(70)
    def predict_one(self) -> None:
        self.client.post(
            "/predict",
            json={"StockCode": self._sku(), "week": WEEK},
            name="/predict",
            timeout=REQUEST_TIMEOUT_S,
            headers=LOAD_HEADERS,
        )

    @task(20)
    def predict_batch(self) -> None:
        self.client.post(
            "/predict/batch",
            json={"week": WEEK},
            name="/predict/batch",
            timeout=REQUEST_TIMEOUT_S,
            headers=LOAD_HEADERS,
        )

    @task(10)
    def health(self) -> None:
        self.client.get(
            "/health/live",
            name="/health/live",
            timeout=REQUEST_TIMEOUT_S,
            headers=LOAD_HEADERS,
        )


if not CONSTANT_MODE:
    # Locust applique toute LoadTestShape définie dans le module (et ignore alors -u/-r/-t) :
    # en mode continu, la classe n'est donc pas définie.

    class RetailShape(LoadTestShape):
        """Paliers 2→320 users (45 s) pour trouver la capacité ; smoke = 30 s / 5 users."""

        stages = load_shape_stages()

        def tick(self):
            run_time = self.get_run_time()
            for stage in self.stages:
                if run_time < stage["duration"]:
                    return stage["users"], stage["spawn_rate"]
            return None
