"""API FastAPI : prédiction métier, santé, rechargement, alertes, métriques Prometheus."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.multiprocess import MultiProcessCollector
from starlette.exceptions import HTTPException as StarletteHTTPException

from retail.api.features import FeatureLookupError, feature_row, features_for_week
from retail.api.schemas import (
    AlertIn,
    BatchPredictRequest,
    BatchPredictResponse,
    FeatureBatchPredictRequest,
    FeaturePredictRequest,
    FeedbackRequest,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)
from retail.api.store import ModelNotReadyError, ModelStore
from retail.common.config import load_app_config
from retail.common.logging import get_logger
from retail.db.repo import (
    LOADTEST_SOURCE,
    error_detail_text,
    get_prediction,
    persist_alert,
    persist_feedback,
    persist_prediction,
    persist_predictions,
    production_wape,
    set_drop_hook,
)
from retail.db.session import database_kind, ensure_schema
from retail.db.session import ping as db_ping
from retail.db.session import ping_cached as db_ping_cached
from retail.features.panel import FEATURE_COLUMNS, NUMERIC_FEATURES
from retail.orchestration.replay import append_jsonl

log = get_logger(__name__)

RECEIVED_ALERTS_FILENAME = "alerts_received.jsonl"
LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)
CHAMPION_POLL_SECONDS = 30.0


def _multiprocess() -> bool:
    return bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))


def set_version_gauge(
    gauge: Gauge,
    previous: tuple[str, str] | None,
    current: tuple[str, str],
    *,
    multiprocess: bool,
) -> tuple[str, str]:
    """Une seule version active : l'étiquette précédente passe à 0 (multiprocessus)."""
    if not multiprocess:
        gauge.clear()
    elif previous is not None and previous != current:
        gauge.labels(model_version=previous[0], source=previous[1]).set(0)
    gauge.labels(model_version=current[0], source=current[1]).set(1)
    return current


def create_app() -> FastAPI:
    app_cfg = load_app_config()
    store = ModelStore(app_cfg)
    multiprocess = _multiprocess()
    registry = None if multiprocess else CollectorRegistry()
    metric_kw: dict = {} if multiprocess else {"registry": registry}
    http_count = Counter(
        "retail_http_requests_total",
        "Requêtes HTTP",
        ["method", "path", "status"],
        **metric_kw,
    )
    http_latency = Histogram(
        "retail_http_request_seconds",
        "Latence HTTP",
        ["method", "path"],
        buckets=LATENCY_BUCKETS,
        **metric_kw,
    )
    degraded_kw = {**metric_kw}
    version_kw = {**metric_kw}
    if multiprocess:
        degraded_kw["multiprocess_mode"] = "livemax"
        version_kw["multiprocess_mode"] = "liveall"
    degraded_gauge = Gauge("retail_model_degraded", "1 si mode dégradé", **degraded_kw)
    version_gauge = Gauge(
        "retail_model_version",
        "Version du champion servi (0 si inconnue)",
        ["model_version", "source"],
        **version_kw,
    )
    db_dropped = Counter(
        "retail_db_dropped_total",
        "Écritures SQL abandonnées (file pleine ou base injoignable trop lente)",
        **metric_kw,
    )
    db_ok_kw = {**metric_kw}
    if multiprocess:
        db_ok_kw["multiprocess_mode"] = "livemin"
    db_ok_gauge = Gauge("retail_db_ok", "1 si la base répond", **db_ok_kw)
    set_drop_hook(db_dropped.inc)
    try:
        store.reload()
    except ModelNotReadyError:
        log.warning("API démarrée sans modèle (GET /health = 503 jusqu'au chargement)")
    try:
        ensure_schema()
    except Exception as exc:
        log.warning("Schéma SQL non créé au démarrage", extra={"error": str(exc)})

    api = FastAPI(title="retail-mlops", version="0.1.0")
    api.state.store = store
    api.state.cfg = app_cfg

    version_labels: tuple[str, str] | None = None

    def _sync_gauges(*, live_db: bool = False) -> None:
        nonlocal version_labels
        degraded_gauge.set(1 if store.degraded else 0)
        db_ok_gauge.set(1 if (db_ping() if live_db else db_ping_cached()) else 0)
        current = (store.model_version, store.source)
        version_labels = set_version_gauge(
            version_gauge, version_labels, current, multiprocess=multiprocess
        )

    _sync_gauges()

    def _poll_champion() -> None:
        raw = os.environ.get("RETAIL_CHAMPION_POLL_SECONDS", str(CHAMPION_POLL_SECONDS))
        try:
            interval = float(raw)
        except ValueError:
            interval = CHAMPION_POLL_SECONDS
        if interval <= 0:
            return
        while True:
            time.sleep(interval)
            try:
                store.refresh_registry_status()
                _sync_gauges()
            except Exception as exc:
                log.debug("Sondage champion ignoré", extra={"error": str(exc)})

    threading.Thread(target=_poll_champion, name="champion-poll", daemon=True).start()

    def _metric_path(request: Request) -> str:
        route = request.scope.get("route")
        return getattr(route, "path", None) or "unmatched"

    @api.middleware("http")
    async def _http_metrics(request: Request, call_next):
        if request.url.path in {"/metrics", "/health/live"}:
            return await call_next(request)
        started = time.perf_counter()
        method = request.method
        try:
            response = await call_next(request)
        except Exception:
            path = _metric_path(request)
            http_count.labels(method, path, "500").inc()
            http_latency.labels(method, path).observe(time.perf_counter() - started)
            raise
        path = _metric_path(request)
        status = str(response.status_code)
        http_count.labels(method, path, status).inc()
        http_latency.labels(method, path).observe(time.perf_counter() - started)
        response.headers["X-Model-Version"] = store.model_version
        return response

    def _require_store():
        if store.pipeline is None:
            raise HTTPException(status_code=503, detail="Modèle non chargé")

    async def _persist_predict_error(request: Request, status_code: int, detail) -> None:
        if not request.url.path.startswith("/predict"):
            return
        payload: dict = {}
        try:
            raw = await request.body()
            if raw:
                payload = json.loads(raw)
        except Exception:
            payload = {}
        stock = str(payload.get("StockCode") or "unknown")
        if request.url.path.rstrip("/").endswith("batch"):
            stock = "*"
        persist_prediction(
            {
                "prediction_id": str(uuid.uuid4()),
                "iso_week": payload.get("week"),
                "stock_code": stock,
                "predicted_qty": 0.0,
                "model_version": store.model_version,
                "source": LOADTEST_SOURCE if _is_load_test(request) else store.source,
                "status": "error",
                "error": f"{status_code}: {error_detail_text(detail)}",
            }
        )

    @api.exception_handler(StarletteHTTPException)
    async def _on_http_error(request: Request, exc: StarletteHTTPException):
        await _persist_predict_error(request, exc.status_code, exc.detail)
        return await http_exception_handler(request, exc)

    @api.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError):
        await _persist_predict_error(request, 422, exc.errors())
        return await request_validation_exception_handler(request, exc)

    def _is_load_test(request: Request) -> bool:
        return request.headers.get("x-load-test", "").strip().lower() in {"1", "true", "yes"}

    def _trace_source(request: Request | None) -> str:
        if request is not None and _is_load_test(request):
            return LOADTEST_SOURCE
        return store.source

    def _trace(
        stock: str,
        value: float,
        week: str | None = None,
        *,
        latency_ms: float | None = None,
        status: str = "ok",
        error: str | None = None,
        prediction_id: str | None = None,
        request: Request | None = None,
    ) -> dict:
        return {
            "prediction_id": prediction_id or str(uuid.uuid4()),
            "iso_week": week,
            "stock_code": stock,
            "predicted_qty": value,
            "model_version": store.model_version,
            "source": _trace_source(request),
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
        }

    def _reply(
        stock: str,
        value: float,
        week: str | None = None,
        *,
        latency_ms: float | None = None,
        status: str = "ok",
        error: str | None = None,
        persist: bool = True,
        request: Request | None = None,
    ) -> PredictResponse:
        pred_id = str(uuid.uuid4())
        if persist:
            persist_prediction(
                _trace(
                    stock,
                    value,
                    week,
                    latency_ms=latency_ms,
                    status=status,
                    error=error,
                    prediction_id=pred_id,
                    request=request,
                )
            )
        return PredictResponse(
            StockCode=stock,
            prediction=value,
            model_version=store.model_version,
            source=_trace_source(request),
            prediction_id=pred_id,
            week=week,
        )

    def _health_body() -> HealthResponse:
        store.refresh_registry_status()
        _sync_gauges(live_db=True)
        _require_store()
        return HealthResponse(
            status="ok",
            model_version=store.model_version,
            source=store.source,
            degraded=store.degraded,
            registry_ok=store.registry_ok,
            db_ok=db_ping(),
            db_backend=database_kind(),
        )

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return _health_body()

    @api.get("/health/live", response_model=HealthResponse)
    def health_live() -> HealthResponse:
        """Liveness (sonde / Locust) : process up, modèle chargé.

        ``db_ok`` est toujours ``true`` : pas de ping SQL (contrairement à ``GET /health``).
        ``/health/live`` et ``/metrics`` sont exclus du middleware Prometheus.
        """
        _require_store()
        return HealthResponse(
            status="ok",
            model_version=store.model_version,
            source=store.source,
            degraded=store.degraded,
            registry_ok=store.registry_ok,
            db_ok=True,
            db_backend=database_kind(),
        )

    @api.get("/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        _require_store()
        return ModelInfo(
            model_version=store.model_version,
            source=store.source,
            degraded=store.degraded,
            registry_ok=store.registry_ok,
            registered_model=store.meta.get("registered_model"),
            n_skus=len(store.skus),
            skus=sorted(store.skus),
            features=list(NUMERIC_FEATURES),
            production_wape=production_wape(store.model_version),
        )

    @api.post("/model/reload", response_model=ModelInfo)
    def reload_model() -> ModelInfo:
        try:
            store.reload()
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        _sync_gauges()
        return model_info()

    def _from_features(item: FeaturePredictRequest, request: Request) -> PredictResponse:
        started = time.perf_counter()
        _require_store()
        if store.skus and item.StockCode not in store.skus:
            raise HTTPException(status_code=404, detail=f"SKU inconnu : {item.StockCode}")
        frame = pd.DataFrame([item.model_dump()])[FEATURE_COLUMNS]
        value = store.predict_frame(frame)[0]
        return _reply(
            item.StockCode,
            value,
            latency_ms=(time.perf_counter() - started) * 1000,
            request=request,
        )

    @api.post("/predict", response_model=PredictResponse)
    def predict(item: PredictRequest, request: Request) -> PredictResponse:
        started = time.perf_counter()
        _require_store()
        if store.skus and item.StockCode not in store.skus:
            raise HTTPException(status_code=404, detail=f"SKU inconnu : {item.StockCode}")
        try:
            frame = feature_row(
                app_cfg,
                item.StockCode,
                item.week,
                sorted(store.skus),
                model_version=store.model_version,
            )
        except FeatureLookupError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        value = store.predict_frame(frame)[0]
        return _reply(
            item.StockCode,
            value,
            week=item.week,
            latency_ms=(time.perf_counter() - started) * 1000,
            request=request,
        )

    @api.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(body: BatchPredictRequest, request: Request) -> BatchPredictResponse:
        started = time.perf_counter()
        _require_store()
        skus = sorted(store.skus)
        if not skus:
            raise HTTPException(status_code=503, detail="Liste de produits du champion absente")
        try:
            rows = features_for_week(app_cfg, body.week, skus, model_version=store.model_version)
        except FeatureLookupError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        preds = store.predict_frame(rows[FEATURE_COLUMNS])
        latency_ms = (time.perf_counter() - started) * 1000
        items: list[PredictResponse] = []
        traces: list[dict] = []
        for code, value in zip(rows["StockCode"].tolist(), preds, strict=True):
            reply = _reply(
                str(code),
                value,
                week=body.week,
                latency_ms=latency_ms,
                persist=False,
                request=request,
            )
            items.append(reply)
            traces.append(
                _trace(
                    str(code),
                    value,
                    body.week,
                    latency_ms=latency_ms,
                    prediction_id=reply.prediction_id,
                    request=request,
                )
            )
        persist_predictions(traces)
        return BatchPredictResponse(
            predictions=items,
            model_version=store.model_version,
            source=store.source,
            week=body.week,
        )

    @api.post("/predict/features", response_model=PredictResponse)
    def predict_features(item: FeaturePredictRequest, request: Request) -> PredictResponse:
        return _from_features(item, request)

    @api.post("/predict/features/batch", response_model=BatchPredictResponse)
    def predict_features_batch(
        body: FeatureBatchPredictRequest, request: Request
    ) -> BatchPredictResponse:
        rows = [_from_features(item, request) for item in body.items]
        return BatchPredictResponse(
            predictions=rows,
            model_version=store.model_version,
            source=store.source,
            week=None,
        )

    @api.post("/feedback", status_code=202)
    def feedback(body: FeedbackRequest) -> dict:
        _require_store()
        if store.skus and body.StockCode not in store.skus:
            raise HTTPException(status_code=404, detail=f"SKU inconnu : {body.StockCode}")
        pred = get_prediction(str(body.prediction_id))
        if pred is None:
            raise HTTPException(
                status_code=404,
                detail=f"Prédiction introuvable : {body.prediction_id}",
            )
        path = app_cfg.paths.reports_dir / "feedback.jsonl"
        payload = body.model_dump(mode="json")
        payload["received_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        append_jsonl(path, payload)
        persist_feedback(
            {
                "prediction_id": str(body.prediction_id),
                "stock_code": body.StockCode,
                "iso_week": body.iso_week,
                "actual_qty": body.actual_qty,
                "predicted_qty": pred["predicted_qty"],
                "model_version": pred["model_version"] or store.model_version,
                "origin": "http",
            }
        )
        return {"status": "accepted"}

    @api.post("/alerts", status_code=202)
    def ingest_alert(payload: AlertIn) -> dict:
        path = app_cfg.paths.reports_dir / RECEIVED_ALERTS_FILENAME
        record = {
            **payload.model_dump(),
            "received_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "via": "http",
        }
        append_jsonl(path, record)
        persist_alert({**record, "origin": "http"})
        return {"status": "accepted"}

    @api.get("/alerts")
    def list_alerts() -> dict:
        path = app_cfg.paths.reports_dir / RECEIVED_ALERTS_FILENAME
        if not path.is_file():
            return {"alerts": []}
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return {"alerts": lines[-50:]}

    @api.get("/metrics")
    def metrics() -> PlainTextResponse:
        _sync_gauges()
        if multiprocess:
            collected = CollectorRegistry()
            MultiProcessCollector(collected)
            payload = generate_latest(collected)
        else:
            payload = generate_latest(registry)
        return PlainTextResponse(payload.decode("utf-8"), media_type="text/plain; version=0.0.4")

    return api


app = create_app
