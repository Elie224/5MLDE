"""Chargement du champion : MLflow, sinon ``models/champion/``, sinon ``.champion.bak``."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from retail.api.features import (
    FeatureLookupError,
    clear_feature_cache,
    features_for_week,
    next_iso_week,
    panel_weeks,
)
from retail.common.config import AppConfig
from retail.common.logging import get_logger
from retail.common.mlflow_uri import tracking_uri
from retail.features.panel import FEATURE_COLUMNS, TOP_SKUS_FILENAME
from retail.quality.errors import QualityError
from retail.registry.promote import CHAMPION_ALIAS, META_FILENAME, load_run_skus

log = get_logger(__name__)

_MLFLOW_RETRIES = "MLFLOW_HTTP_REQUEST_MAX_RETRIES"
_MLFLOW_TIMEOUT = "MLFLOW_HTTP_REQUEST_TIMEOUT"
_ALIAS_CACHE_SECONDS = 10.0


def skip_registry_probe() -> bool:
    flag = os.environ.get("RETAIL_SKIP_REGISTRY_PROBE", "").strip().lower()
    return flag in {"1", "true", "yes"}


class ModelNotReadyError(RuntimeError):
    """Aucun champion disque ni registre."""


@dataclass(frozen=True)
class ServingState:
    pipeline: Any
    skus: frozenset[str]
    model_version: str
    source: str
    meta: dict
    registry_ok: bool = True


class ModelStore:
    def __init__(self, app: AppConfig) -> None:
        self.app = app
        self._state: ServingState | None = None
        self._lock = threading.Lock()
        self._alias_cache: tuple[float, str | None] | None = None
        os.environ[_MLFLOW_RETRIES] = "0"
        os.environ[_MLFLOW_TIMEOUT] = "2"

    def _snapshot(self) -> ServingState | None:
        return self._state

    def _publish(self, state: ServingState) -> None:
        with self._lock:
            self._state = state

    @property
    def pipeline(self):
        state = self._snapshot()
        return None if state is None else state.pipeline

    @property
    def skus(self) -> set[str]:
        state = self._snapshot()
        return set() if state is None else set(state.skus)

    @property
    def model_version(self) -> str:
        state = self._snapshot()
        return "unknown" if state is None else state.model_version

    @property
    def source(self) -> str:
        state = self._snapshot()
        return "none" if state is None else state.source

    @property
    def meta(self) -> dict:
        state = self._snapshot()
        return {} if state is None else state.meta

    @property
    def registry_ok(self) -> bool:
        state = self._snapshot()
        return False if state is None else state.registry_ok

    @property
    def degraded(self) -> bool:
        state = self._snapshot()
        if state is None:
            return True
        if skip_registry_probe():
            return state.source == "bak"
        return state.source in {"disk", "bak"} or not state.registry_ok

    def reload(self) -> None:
        state = self._load_fresh()
        self._publish(state)
        clear_feature_cache()
        self._prewarm_features()

    def refresh_registry_status(self) -> None:
        """Sonde le registre ; recharge si l'alias champion change ou si MLflow revient."""
        if skip_registry_probe():
            return
        state = self._snapshot()
        if state is None:
            return
        ok = self._probe_registry()
        if not ok:
            if not state.registry_ok:
                return
            source = state.source
            if source == "mlflow":
                source = "disk" if self._dir_ready(self._champion_dir()) else state.source
            self._publish(replace(state, registry_ok=False, source=source))
            return
        alias = self._alias_version()
        same_champion = (
            state.registry_ok
            and state.source == "mlflow"
            and alias is not None
            and state.model_version == alias
        )
        if same_champion:
            return
        try:
            self.reload()
        except ModelNotReadyError:
            self._publish(replace(state, registry_ok=True))

    def _prewarm_features(self) -> None:
        skus = sorted(self.skus)
        if not skus:
            return
        try:
            nxt = next_iso_week(panel_weeks(self.app)[-1])
            features_for_week(self.app, nxt, skus, model_version=self.model_version)
        except (
            FeatureLookupError,
            QualityError,
            OSError,
            FileNotFoundError,
            ValueError,
        ) as exc:
            log.debug("Précalcul des features ignoré", extra={"error": str(exc)})

    def _alias_version(self) -> str | None:
        if skip_registry_probe():
            return None
        now = time.monotonic()
        if self._alias_cache is not None and now - self._alias_cache[0] < _ALIAS_CACHE_SECONDS:
            return self._alias_cache[1]
        value = self._fetch_alias_version()
        self._alias_cache = (now, value)
        return value

    def _fetch_alias_version(self) -> str | None:
        os.environ[_MLFLOW_RETRIES] = "0"
        os.environ[_MLFLOW_TIMEOUT] = "2"
        try:
            uri = tracking_uri()
            mlflow.set_tracking_uri(uri)
            client = MlflowClient(tracking_uri=uri)
            name = self.app.promotion.registered_model_name
            return str(client.get_model_version_by_alias(name, CHAMPION_ALIAS).version)
        except (MlflowException, OSError, TimeoutError, ConnectionError):
            return None

    def _probe_registry(self) -> bool:
        return self._alias_version() is not None

    def _load_fresh(self) -> ServingState:
        if skip_registry_probe():
            dest = self._champion_dir()
            bak = dest.parent / ".champion.bak"
            if self._dir_ready(dest):
                return self._state_from_dir(dest, "disk", registry_ok=True)
            if self._dir_ready(bak):
                return self._state_from_dir(bak, "bak", registry_ok=False)
            raise ModelNotReadyError("Aucun champion local (models/champion/ ni .champion.bak).")
        name = self.app.promotion.registered_model_name
        uri = tracking_uri()
        mlflow.set_tracking_uri(uri)
        try:
            client = MlflowClient(tracking_uri=uri)
            mv = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
            version = str(mv.version)
            pipeline = mlflow.sklearn.load_model(f"models:/{name}/{version}")
            skus = load_run_skus(mv.run_id)
            log.info("Modèle chargé depuis MLflow", extra={"version": version})
            return ServingState(
                pipeline=pipeline,
                skus=frozenset(skus),
                model_version=version,
                source="mlflow",
                meta={
                    "registered_model": name,
                    "model_version": version,
                    "run_id": mv.run_id,
                },
                registry_ok=True,
            )
        except (MlflowException, OSError, FileNotFoundError) as exc:
            log.warning("MLflow indisponible, repli disque", extra={"error": str(exc)})
        dest = self._champion_dir()
        bak = dest.parent / ".champion.bak"
        if self._dir_ready(dest):
            return self._state_from_dir(dest, "disk", registry_ok=False)
        if self._dir_ready(bak):
            return self._state_from_dir(bak, "bak", registry_ok=False)
        raise ModelNotReadyError("Aucun champion local (models/champion/ ni .champion.bak).")

    def _champion_dir(self) -> Path:
        return self.app.paths.models_dir / "champion"

    @staticmethod
    def _dir_ready(dest: Path) -> bool:
        return dest.is_dir() and (dest / "model" / "MLmodel").is_file()

    def _state_from_dir(self, dest: Path, source: str, *, registry_ok: bool) -> ServingState:
        pipeline = mlflow.sklearn.load_model(str(dest / "model"))
        meta_path = dest / META_FILENAME
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        version = str(meta.get("model_version") or source)
        skus = _skus_from_file(dest / TOP_SKUS_FILENAME)
        log.info("Modèle chargé depuis le disque", extra={"source": source, "dir": str(dest)})
        return ServingState(
            pipeline=pipeline,
            skus=frozenset(skus),
            model_version=version,
            source=source,
            meta=meta,
            registry_ok=registry_ok,
        )

    def predict_frame(self, frame: pd.DataFrame) -> list[float]:
        state = self._snapshot()
        if state is None:
            raise ModelNotReadyError("Modèle non chargé.")
        preds = state.pipeline.predict(frame[FEATURE_COLUMNS])
        return [max(0.0, float(value)) for value in preds]


def _skus_from_file(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("skus", [])
    return {str(item) for item in items}
