"""Pull SADA 2022 from Kaggle and verify the on-disk layout."""

from __future__ import annotations

import logging
import shutil
import subprocess
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "sdaiancai/sada2022"
EXPECTED_CSVS = ("train.csv", "valid.csv", "test.csv")


class KaggleNotConfiguredError(RuntimeError):
    """Raised when the kaggle CLI is missing or has no API token."""


def _find_kaggle_cli() -> str:
    cli = shutil.which("kaggle")
    if cli is None:
        raise KaggleNotConfiguredError(
            "`kaggle` CLI not found. Install with `uv pip install kaggle` and place "
            "your API token at ~/.kaggle/kaggle.json (download it from "
            "https://www.kaggle.com/settings → 'Create New Token'). "
            "Then visit https://www.kaggle.com/datasets/sdaiancai/sada2022 once in a "
            "browser to accept the dataset terms."
        )
    return cli


def download(dest: Path, *, dataset: str = KAGGLE_DATASET, force: bool = False) -> Path:
    """Download the SADA zip into `dest`. Returns the path to the zip."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_name = dataset.split("/", 1)[1] + ".zip"
    zip_path = dest / zip_name

    if zip_path.is_file() and not force:
        logger.info("Zip already present, skipping download: %s", zip_path)
        return zip_path

    cli = _find_kaggle_cli()
    cmd = [cli, "datasets", "download", "-d", dataset, "-p", str(dest)]
    if force:
        cmd.append("--force")

    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        if "403" in msg or "Forbidden" in msg:
            raise KaggleNotConfiguredError(
                "Kaggle returned 403. Open "
                "https://www.kaggle.com/datasets/sdaiancai/sada2022 in a browser, "
                "accept the dataset terms, then re-run."
            )
        raise RuntimeError(f"kaggle download failed: {msg}")

    if not zip_path.is_file():
        # Some dataset versions append a -version suffix; fall back to first .zip.
        candidates = sorted(dest.glob("*.zip"))
        if not candidates:
            raise RuntimeError(f"No zip in {dest} after download")
        zip_path = candidates[-1]
    return zip_path


def unzip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Unzipping %s → %s", zip_path, dest)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def verify_layout(root: Path) -> dict[str, Path]:
    """Locate the CSVs and audio root under `root`. Searches one level deep."""
    candidates: list[Path] = [root, *(p for p in root.iterdir() if p.is_dir())]
    found: dict[str, Path] = {}
    for base in candidates:
        for csv_name in EXPECTED_CSVS:
            p = base / csv_name
            if p.is_file() and csv_name not in found:
                found[csv_name] = p

    missing = [c for c in EXPECTED_CSVS if c not in found]
    if missing:
        raise FileNotFoundError(
            f"SADA layout invalid under {root}. Missing: {missing}. "
            f"Found: {sorted(p.name for p in found.values())}"
        )

    csv_dir = found["train.csv"].parent
    audio_root = csv_dir
    if not any(audio_root.glob("**/*.wav")):
        raise FileNotFoundError(f"No .wav files found under {audio_root}")

    found["__root__"] = csv_dir
    return found


def pull(dest: Path, *, force: bool = False) -> dict[str, Path]:
    zip_path = download(dest, force=force)
    extract_dir = dest / "extracted"
    if force or not any(extract_dir.glob("**/train.csv")):
        unzip(zip_path, extract_dir)
    return verify_layout(extract_dir)
