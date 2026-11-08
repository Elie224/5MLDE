"""Moteur SQLAlchemy : PostgreSQL si ``RETAIL_DATABASE_URL``, sinon SQLite local."""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from retail.common.config import REPO_ROOT, get_settings
from retail.common.logging import get_logger
from retail.db.models import Base

log = get_logger(__name__)

_LOCK = threading.Lock()
_ENGINE: Engine | None = None
_SESSION: sessionmaker[Session] | None = None
_SCHEMA_READY = False
_PING_LOCK = threading.Lock()
_LAST_PING: tuple[float, bool] | None = None


def database_url() -> str:
    raw = (get_settings().database_url or "").strip()
    if not raw:
        path = REPO_ROOT / "data" / "reports" / "retail.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        return "sqlite:///" + path.resolve().as_posix()
    if raw.startswith("postgresql+psycopg://") or raw.startswith("sqlite"):
        return raw
    if raw.startswith("postgres://"):
        return "postgresql+psycopg://" + raw[len("postgres://") :]
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg://" + raw[len("postgresql://") :]
    return raw


def database_kind() -> str:
    return "sqlite" if database_url().startswith("sqlite") else "postgresql"


def reset_engine() -> None:
    global _ENGINE, _SESSION, _SCHEMA_READY, _LAST_PING
    with _LOCK:
        if _ENGINE is not None:
            _ENGINE.dispose()
        _ENGINE = None
        _SESSION = None
        _SCHEMA_READY = False
        _LAST_PING = None
    from retail.db.repo import reset_worker

    reset_worker()


def get_engine() -> Engine:
    global _ENGINE, _SESSION
    with _LOCK:
        if _ENGINE is None:
            url = database_url()
            kwargs: dict = {"pool_pre_ping": True}
            if url.startswith("sqlite"):
                kwargs["connect_args"] = {"check_same_thread": False, "timeout": 2}
                kwargs["poolclass"] = NullPool
            else:
                kwargs["connect_args"] = {"connect_timeout": 2}
                kwargs["pool_size"] = 2
                kwargs["max_overflow"] = 2
            _ENGINE = create_engine(url, **kwargs)
            _SESSION = sessionmaker(_ENGINE, expire_on_commit=False)
            if url.startswith("sqlite"):
                log.warning(
                    "SQLite local, pas Postgres : les traces n'alimenteront pas Grafana. "
                    "Définir RETAIL_DATABASE_URL=postgresql+psycopg://…"
                )
        return _ENGINE


def ensure_schema() -> None:
    global _SCHEMA_READY
    engine = get_engine()
    with _LOCK:
        if _SCHEMA_READY:
            return
        Base.metadata.create_all(engine)
        marked = _mark_legacy_loadtest(engine)
        if marked:
            log.info(
                "Predictions W48 reetiquetees loadtest",
                extra={"rows": marked},
            )
        _SCHEMA_READY = True


def _mark_legacy_loadtest(engine: Engine) -> int:
    """Series Locust avant X-Load-Test : source=disk sur 2011-W48."""
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE predictions SET source = 'loadtest' "
                "WHERE iso_week = '2011-W48' "
                "AND (source IS NULL OR source <> 'loadtest')"
            )
        )
        return int(result.rowcount or 0)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    get_engine()
    assert _SESSION is not None
    session = _SESSION()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping() -> bool:
    try:
        with session_scope() as session:
            session.execute(text("SELECT 1"))
        ok = True
    except Exception as extra:
        log.warning("Base indisponible", extra={"error": str(extra)})
        ok = False
    global _LAST_PING
    with _PING_LOCK:
        _LAST_PING = (time.monotonic(), ok)
    return ok


def ping_cached(ttl_s: float = 2.0) -> bool:
    """Évite un SELECT à chaque scrape Prometheus (/metrics)."""
    with _PING_LOCK:
        cached = _LAST_PING
    if cached is not None and (time.monotonic() - cached[0]) < ttl_s:
        return cached[1]
    return ping()
