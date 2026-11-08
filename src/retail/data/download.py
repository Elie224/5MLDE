"""Téléchargement UCI, contrôle SHA256, conversion Parquet. Repli : zip dans ``data/raw/``."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from retail.common.config import REPO_ROOT, AppConfig, DataConfig
from retail.common.logging import get_logger

log = get_logger(__name__)

RAW_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]
INTERIM_FILENAME = "online_retail.parquet"
MANIFEST_FILENAME = "ingestion_manifest.json"
USER_AGENT = "retail-mlops/0.1 (UCI Online Retail; academic MLOps pipeline)"


class DataSourceError(RuntimeError):
    """Source de données indisponible ou non conforme à la version attendue."""


@dataclass(frozen=True)
class IngestionManifest:
    source_url: str
    archive_sha256: str | None
    expected_sha256: str | None
    parquet_path: str
    parquet_content_hash: str
    rows: int
    first_invoice_date: str
    last_invoice_date: str
    ingested_at: str


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(df: pd.DataFrame) -> str:
    """Empreinte du contenu (indépendante du format de fichier et des métadonnées)."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _path_relative_to_repo(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def download_archive(url: str, dest: Path, timeout: int, max_attempts: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, max_attempts + 1):
        try:
            log.info("Téléchargement", extra={"url": url, "attempt": attempt})
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)
            return dest
        except requests.RequestException as exc:
            last_error = exc
            log.warning("Échec du téléchargement", extra={"attempt": attempt, "error": str(exc)})
            tmp.unlink(missing_ok=True)
            if attempt < max_attempts:
                time.sleep(2 * attempt)
    raise DataSourceError(
        f"Impossible de télécharger {url} après {max_attempts} essai(s) ({last_error}).\n"
        f"Repli manuel : télécharger l'archive depuis la page UCI et la déposer dans "
        f"{dest} (ou déposer directement le fichier Excel dans {dest.parent}), puis relancer."
    )


def verify_checksum(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if not expected:
        log.warning(
            "Aucune empreinte de référence configurée : recopier celle-ci dans "
            "configs/data.yaml (source.sha256) pour figer la version des données",
            extra={"sha256": actual},
        )
    elif actual.lower() != expected.lower():
        raise DataSourceError(
            f"Empreinte SHA256 inattendue pour {path.name} : {actual} (attendu {expected}). "
            "La source a changé : ingestion interrompue."
        )
    return actual


def extract_excel(archive: Path, excel_filename: str, dest_dir: Path) -> Path:
    with zipfile.ZipFile(archive) as zf:
        if excel_filename not in zf.namelist():
            raise DataSourceError(
                f"{excel_filename!r} absent de l'archive (contenu : {zf.namelist()})"
            )
        zf.extract(excel_filename, dest_dir)
    return dest_dir / excel_filename


def normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Typage strict des colonnes UCI. Aucun filtrage métier ici (c'est le rôle de la phase 2)."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DataSourceError(f"Colonnes manquantes dans la source : {missing}")
    out = df[RAW_COLUMNS].copy()
    out["InvoiceNo"] = out["InvoiceNo"].astype(str).str.strip()
    out["StockCode"] = out["StockCode"].astype(str).str.strip().str.upper()
    out["Description"] = out["Description"].astype("string").str.strip()
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="raise").astype("int64")
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="raise")
    out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="raise").astype("float64")
    out["CustomerID"] = pd.to_numeric(out["CustomerID"], errors="coerce").astype("Int64")
    out["Country"] = out["Country"].astype("string").str.strip()
    return out.sort_values(["InvoiceDate", "InvoiceNo", "StockCode"], kind="stable").reset_index(
        drop=True
    )


def convert_to_parquet(excel_path: Path, dest: Path) -> pd.DataFrame:
    log.info("Lecture du fichier Excel (peut prendre ~1 min)", extra={"path": str(excel_path)})
    df = pd.read_excel(excel_path, dtype={"InvoiceNo": str, "StockCode": str})
    df = normalize_raw(df)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return df


def ingest(app_cfg: AppConfig, data_cfg: DataConfig) -> IngestionManifest:
    raw_dir, interim_dir = app_cfg.paths.raw_dir, app_cfg.paths.interim_dir
    src = data_cfg.source
    excel_path = raw_dir / src.excel_filename
    archive_path = raw_dir / src.archive_filename
    archive_sha: str | None = None

    if excel_path.is_file() and not archive_path.is_file():
        log.warning(
            "Fichier Excel déposé manuellement : empreinte d'archive non vérifiable",
            extra={"path": str(excel_path)},
        )
    else:
        if not archive_path.is_file():
            download_archive(
                str(src.url),
                archive_path,
                data_cfg.download.timeout_seconds,
                data_cfg.download.max_attempts,
            )
        archive_sha = verify_checksum(archive_path, src.sha256)
        excel_path = extract_excel(archive_path, src.excel_filename, raw_dir)

    parquet_path = interim_dir / INTERIM_FILENAME
    df = convert_to_parquet(excel_path, parquet_path)

    manifest = IngestionManifest(
        source_url=str(src.url),
        archive_sha256=archive_sha,
        expected_sha256=src.sha256 or None,
        parquet_path=_path_relative_to_repo(parquet_path),
        parquet_content_hash=content_hash(df),
        rows=len(df),
        first_invoice_date=df["InvoiceDate"].min().isoformat(),
        last_invoice_date=df["InvoiceDate"].max().isoformat(),
        ingested_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (interim_dir / MANIFEST_FILENAME).write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "Ingestion terminée",
        extra={"rows": manifest.rows, "hash": manifest.parquet_content_hash},
    )
    return manifest
