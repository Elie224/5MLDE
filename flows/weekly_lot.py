"""Flow Prefect : un lot ISO par run. Relances uniquement sur erreurs techniques.

prefect server start --port 4201
prefect config set PREFECT_API_URL=http://127.0.0.1:4201/api
python flows/weekly_lot.py
"""

from __future__ import annotations

import time

from prefect import flow, task
from prefect.states import State

from retail.common.config import load_app_config, load_data_config
from retail.data.download import ingest
from retail.data.make_batches import make_batches
from retail.orchestration.errors import ReplayDone, ReplayError
from retail.orchestration.lock import replay_lock
from retail.orchestration.replay import (
    LOCK_FILENAME,
    _orch_dir,
    emit_alert,
    train_step,
    validate_step,
)
from retail.quality.errors import QualityError, QualityGateBypassError
from retail.training.errors import TrainError

_NO_RETRY = (QualityError, QualityGateBypassError, TrainError, ReplayError)


def retry_if_technical(_task, _task_run, state: State) -> bool:
    try:
        state.result(raise_on_failure=True)
    except _NO_RETRY:
        return False
    except Exception:
        return True
    return False


def notify_flow_failure(_flow, flow_run, state: State) -> None:
    app = load_app_config()
    message = state.message or "flow failed"
    emit_alert(
        app,
        {
            "kind": "flow_failure",
            "flow_run": str(getattr(flow_run, "id", "")),
            "message": message,
        },
    )


@task(retries=2, retry_delay_seconds=10)
def ingest_batches() -> None:
    ingest(load_app_config(), load_data_config())
    make_batches(load_app_config())


@task(retries=0)
def validate_week_task(week: str | None = None) -> str | None:
    try:
        return validate_step(load_app_config(), week, time.monotonic())
    except ReplayDone:
        return None  # tous les lots rejoués : rien à faire, ce n'est pas une panne


@task(retries=2, retry_delay_seconds=10, retry_condition_fn=retry_if_technical)
def train_register_task(week: str) -> dict:
    return train_step(load_app_config(), week, time.monotonic())


@flow(name="retail-weekly-lot", on_failure=[notify_flow_failure])
def weekly_lot(week: str | None = None, *, ingest_first: bool = False) -> dict:
    app = load_app_config()
    with replay_lock(_orch_dir(app) / LOCK_FILENAME):
        if ingest_first:
            ingest_batches()
        chosen = validate_week_task(week)
        if chosen is None:
            return {"status": "done", "message": "Tous les lots du manifeste ont été rejoués."}
        return train_register_task(chosen)


def serve_weekly_lot() -> None:
    app = load_app_config()
    orch = app.orchestration
    train_register_task.retries = orch.train_retries
    train_register_task.retry_delay_seconds = orch.train_retry_delay_seconds
    weekly_lot.serve(
        name="retail-weekly-lot",
        interval=orch.interval_seconds,
        limit=1,
    )


if __name__ == "__main__":
    serve_weekly_lot()
