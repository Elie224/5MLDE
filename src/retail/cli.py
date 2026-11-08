"""Point d'entrée en ligne de commande : ``retail <commande>`` (ou ``python -m retail``).

Le Makefile ne fait qu'appeler ces commandes : elles fonctionnent aussi sous Windows
sans ``make``.
"""

from __future__ import annotations

import argparse
import os
import sys

from retail.common.config import (
    ConfigError,
    get_settings,
    load_app_config,
    load_data_config,
)
from retail.common.logging import get_logger, setup_logging
from retail.orchestration.errors import ReplayError
from retail.quality.errors import QualityError, QualityGateBypassError
from retail.registry.errors import RegistryError
from retail.training.errors import TrainError

log = get_logger("retail.cli")


def _cmd_download(_: argparse.Namespace) -> None:
    from retail.data.download import ingest

    ingest(load_app_config(), load_data_config())


def _cmd_batches(_: argparse.Namespace) -> None:
    from retail.data.make_batches import make_batches

    make_batches(load_app_config())


def _cmd_data(args: argparse.Namespace) -> None:
    _cmd_download(args)
    _cmd_batches(args)


def _cmd_quality(args: argparse.Namespace) -> None:
    from retail.quality.validate import run_invalid_demo, validate_all

    app = load_app_config()
    if args.invalid_demo:
        run_invalid_demo(app)
        return
    validate_all(app, week=args.week)


def _cmd_features(args: argparse.Namespace) -> None:
    from retail.features.panel import build_features

    build_features(load_app_config(), until_week=args.until_week)


def _cmd_train(args: argparse.Namespace) -> None:
    from retail.training.train import train_and_evaluate

    train_and_evaluate(
        load_app_config(), until_week=args.until_week, register=not args.skip_register
    )


def _cmd_rollback(args: argparse.Namespace) -> None:
    from retail.registry.promote import rollback_champion

    rollback_champion(load_app_config(), to_version=args.to_version)


def _cmd_replay(args: argparse.Namespace) -> None:
    from retail.orchestration.replay import run_tick

    run_tick(load_app_config(), week=args.week, train=not args.skip_train)


def _cmd_db_init(_: argparse.Namespace) -> None:
    from retail.db.session import database_url, ensure_schema

    ensure_schema()
    log.info("Schéma SQL prêt", extra={"url": database_url()})


def _cmd_api(args: argparse.Namespace) -> None:
    import uvicorn

    port = int(os.environ.get("API_PORT") or args.port)
    workers = max(1, int(args.workers))
    if workers == 1:
        from retail.api.app import create_app

        uvicorn.run(create_app(), host=args.host, port=port)
        return
    from retail.api.serve import main as serve_main

    os.environ["API_WORKERS"] = str(workers)
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(port)
    serve_main()


def _cmd_check_config(_: argparse.Namespace) -> None:
    load_app_config()
    load_data_config()
    log.info("Configuration valide")


def _cmd_demo(args: argparse.Namespace) -> None:
    from retail.demo import run_demo

    run_demo(skip_docker=args.skip_docker, reset_cursor=args.reset_cursor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retail", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check-config", help="Valider les fichiers de configuration").set_defaults(
        func=_cmd_check_config
    )
    sub.add_parser(
        "download", help="Télécharger, vérifier et convertir la source UCI"
    ).set_defaults(func=_cmd_download)
    sub.add_parser("batches", help="Découper l'historique en lots hebdomadaires").set_defaults(
        func=_cmd_batches
    )
    sub.add_parser("data", help="download + batches").set_defaults(func=_cmd_data)
    quality = sub.add_parser(
        "quality",
        help="Contrôles Pandera bloquants + nettoyage tracé (un lot ou tous)",
    )
    quality.add_argument("--week", default=None, help="Semaine ISO (ex. 2010-W49)")
    quality.add_argument(
        "--invalid-demo",
        action="store_true",
        help="Écrire un lot corrompu : code 0 s'il est rejeté, 1 s'il passe",
    )
    quality.set_defaults(func=_cmd_quality)
    features = sub.add_parser(
        "features",
        help="Top 50 figé sur le warmup + lags / calendrier / prix",
    )
    features.add_argument(
        "--until-week",
        default=None,
        help="Ne garder que les lots ≤ cette semaine ISO",
    )
    features.set_defaults(func=_cmd_features)
    train = sub.add_parser(
        "train",
        help="Features + Ridge vs naïf (MAE/WAPE), split temporel, run MLflow",
    )
    train.add_argument(
        "--until-week",
        default=None,
        help="Ne garder que les lots ≤ cette semaine ISO",
    )
    train.add_argument(
        "--skip-register",
        action="store_true",
        help="Entraîner sans enregistrer ni promouvoir dans le registre",
    )
    train.set_defaults(func=_cmd_train)
    rollback = sub.add_parser(
        "rollback",
        help="Ré-alias une version précédente et réexporte models/champion/",
    )
    rollback.add_argument(
        "--to-version",
        default=None,
        help="Version du registre à restaurer (sinon previous_champion du champion actuel)",
    )
    rollback.set_defaults(func=_cmd_rollback)
    replay = sub.add_parser(
        "replay",
        help="Livrer un lot ISO (qualité de la semaine seulement, puis train --until-week)",
    )
    replay.add_argument("--week", default=None, help="Semaine ISO à livrer (sinon la suivante)")
    replay.add_argument(
        "--skip-train",
        action="store_true",
        help="Valider le lot sans entraîner",
    )
    replay.set_defaults(func=_cmd_replay)
    sub.add_parser("db-init", help="Créer les tables SQL (SQLite ou PostgreSQL)").set_defaults(
        func=_cmd_db_init
    )
    api = sub.add_parser("api", help="Servir l'API FastAPI (prédiction, santé, alertes)")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8001)
    api.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processus Uvicorn (si > 1 : PROMETHEUS_MULTIPROC_DIR, agrégation des métriques)",
    )
    api.set_defaults(func=_cmd_api)
    demo = sub.add_parser(
        "demo",
        help="Scénario 10 min : nominal, lot invalide, Postgres down, rollback",
    )
    demo.add_argument(
        "--skip-docker",
        action="store_true",
        help="Ne pas arrêter Postgres (CI / sans Docker)",
    )
    demo.add_argument(
        "--reset-cursor",
        action="store_true",
        help="Remettre le curseur à 2011-W40 (cycle W40/W41 pour la soutenance)",
    )
    demo.set_defaults(func=_cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(get_settings().log_level)
    try:
        args.func(args)
    except QualityGateBypassError as exc:
        log.error(str(exc))
        return 1
    except (
        ConfigError,
        QualityError,
        TrainError,
        RegistryError,
        ReplayError,
        FileNotFoundError,
    ) as exc:
        log.error(str(exc))
        return 2
    except Exception as exc:
        from retail.data.download import DataSourceError

        if isinstance(exc, DataSourceError):
            log.error(str(exc))
            return 2
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
