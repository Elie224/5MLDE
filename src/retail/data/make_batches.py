"""Découpage de l'historique en lots hebdomadaires pour simuler l'arrivée de données.

Hypothèse documentée (docs/DATA.md) : le dataset historique est **rejoué** semaine par
semaine ; chaque fichier ``data/batches/<AAAA-Www>.parquet`` représente les factures
reçues pendant une semaine ISO (lundi 00:00 -> dimanche 23:59).

Les semaines incomplètes en bord de période (début déc. 2010, fin déc. 2011) sont
écartées par défaut : elles fausseraient les volumes hebdomadaires. Un trou interne
(ex. 2010-W52, fermeture de Noël) est listé dans ``missing_weeks`` du manifeste.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from retail.common.config import AppConfig
from retail.common.logging import get_logger
from retail.data.download import INTERIM_FILENAME, content_hash

log = get_logger(__name__)

MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class BatchInfo:
    week: str
    week_start: str
    week_end: str
    rows: int
    invoices: int
    content_hash: str
    file: str


def week_label(week_start: pd.Timestamp) -> str:
    iso = week_start.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def assign_week(dates: pd.Series) -> pd.Series:
    """Date de début (lundi) de la semaine de chaque facture."""
    return dates.dt.to_period("W-SUN").dt.start_time


def split_weekly(
    df: pd.DataFrame, drop_partial_weeks: bool = True
) -> dict[pd.Timestamp, pd.DataFrame]:
    if df.empty:
        return {}
    weeks = assign_week(df["InvoiceDate"])
    first_day = df["InvoiceDate"].min().normalize()
    last_day = df["InvoiceDate"].max().normalize()

    batches: dict[pd.Timestamp, pd.DataFrame] = {}
    for start, part in df.groupby(weeks, sort=True):
        end = start + pd.Timedelta(days=6)
        if drop_partial_weeks and (start < first_day or end > last_day):
            log.info("Semaine incomplète écartée", extra={"week": week_label(start)})
            continue
        batches[start] = part.reset_index(drop=True)
    return batches


def write_batches(batches: dict[pd.Timestamp, pd.DataFrame], out_dir: Path) -> list[BatchInfo]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.parquet"):
        stale.unlink()

    infos: list[BatchInfo] = []
    for start, part in sorted(batches.items()):
        label = week_label(start)
        path = out_dir / f"{label}.parquet"
        part.to_parquet(path, index=False)
        infos.append(
            BatchInfo(
                week=label,
                week_start=start.date().isoformat(),
                week_end=(start + pd.Timedelta(days=6)).date().isoformat(),
                rows=len(part),
                invoices=int(part["InvoiceNo"].nunique()),
                content_hash=content_hash(part),
                file=path.name,
            )
        )
    missing = missing_weeks(infos)
    if missing:
        log.info("Semaines ISO sans facture", extra={"missing_weeks": missing})
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {"batches": [asdict(i) for i in infos], "missing_weeks": missing},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return infos


def make_batches(app_cfg: AppConfig) -> list[BatchInfo]:
    src = app_cfg.paths.interim_dir / INTERIM_FILENAME
    if not src.is_file():
        raise FileNotFoundError(f"{src} introuvable : lancer d'abord `retail download`.")
    df = pd.read_parquet(src)
    batches = split_weekly(df, app_cfg.batching.drop_partial_weeks)
    infos = write_batches(batches, app_cfg.paths.batches_dir)
    log.info(
        "Lots hebdomadaires écrits",
        extra={
            "count": len(infos),
            "first": infos[0].week if infos else None,
            "last": infos[-1].week if infos else None,
            "dir": str(app_cfg.paths.batches_dir),
        },
    )
    return infos


def missing_weeks(infos: list[BatchInfo]) -> list[str]:
    """Trous internes entre la première et la dernière semaine du manifeste."""
    if len(infos) < 2:
        return []
    starts = [pd.Timestamp(item.week_start) for item in infos]
    expected: list[str] = []
    cursor, last = min(starts), max(starts)
    while cursor <= last:
        expected.append(week_label(cursor))
        cursor += pd.Timedelta(days=7)
    present = {item.week for item in infos}
    return [week for week in expected if week not in present]


def _manifest_payload(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {"batches": raw, "missing_weeks": []}
    return raw


def load_manifest(batches_dir: Path) -> list[BatchInfo]:
    path = batches_dir / MANIFEST_FILENAME
    return [BatchInfo(**item) for item in _manifest_payload(path)["batches"]]


def load_missing_weeks(batches_dir: Path) -> list[str]:
    path = batches_dir / MANIFEST_FILENAME
    return list(_manifest_payload(path).get("missing_weeks") or [])
