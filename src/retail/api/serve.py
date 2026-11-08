"""Entrée API légère : le spawn Uvicorn réimporte ``__main__`` avant le fil ping.

Ne pas importer ``retail.cli`` ni ``create_app`` ici : chaque worker le ferait
avant de pouvoir répondre au healthcheck (5 s par défaut).
"""

from __future__ import annotations

import os
from pathlib import Path


def _prepare_multiprocess() -> None:
    directory = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not directory:
        directory = "/tmp/prom"
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = directory
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def main() -> None:
    import uvicorn

    workers = max(1, int(os.environ.get("API_WORKERS", "1")))
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT") or "8001")
    if workers > 1:
        _prepare_multiprocess()
    uvicorn.run(
        "retail.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        timeout_worker_healthcheck=60,
    )


if __name__ == "__main__":
    main()
