"""Run a bounded TMDB synchronization from the command line.

Usage: ``python -m recommender_api.workers.sync_tmdb``
"""
from __future__ import annotations

import argparse
import asyncio

from ..config import get_settings
from ..database import SessionLocal
from ..services.sync_service import TMDBSyncService


async def run(max_pages: int) -> None:
    with SessionLocal() as db:
        result = await TMDBSyncService(db, get_settings()).sync(max_pages=max_pages)
        print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize the TMDB movie catalog")
    parser.add_argument("--max-pages", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(run(args.max_pages))


if __name__ == "__main__":
    main()
