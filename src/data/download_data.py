"""Download MovieLens 1M dataset from GroupLens.

Source: https://grouplens.org/datasets/movielens/1m/
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_MD5 = "c4d9eecfca2ab87c1945afe126590906"

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_movielens_1m(target_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Download and extract MovieLens 1M into ``target_dir``.

    Returns the path to the extracted ``ml-1m`` folder.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    extract_root = target_dir / "ml-1m"
    if extract_root.exists() and not force:
        print(f"[skip] Already extracted at {extract_root}")
        return extract_root

    zip_path = target_dir / "ml-1m.zip"
    if not zip_path.exists() or force:
        print(f"[download] {MOVIELENS_1M_URL}")
        with requests.get(MOVIELENS_1M_URL, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with open(zip_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc="ml-1m.zip"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    pbar.update(len(chunk))

    actual_md5 = _md5(zip_path)
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(
            f"MD5 mismatch for {zip_path} (got {actual_md5}, expected {EXPECTED_MD5})."
        )

    print(f"[extract] {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    zip_path.unlink(missing_ok=True)
    return extract_root


if __name__ == "__main__":
    out = download_movielens_1m(force=("--force" in sys.argv))
    print(f"Dataset ready at: {out}")
