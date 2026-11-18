#!/usr/bin/env python3
"""Hold a SQLite write lock for a fixed duration.

Used by lock-recovery tests to simulate the on-call engineer's long write
transaction: the process opens ``BEGIN IMMEDIATE`` (taking the WAL write lock),
sleeps for ``--seconds`` and rolls back on exit.  Workers and the API that try
to write during that window must survive the busy period and recover.

Acquisition is retried because a worker may be inside its own short commit
when this process starts; the lock is only announced once it is really held.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time

from app.db import init_db


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--busy-timeout", type=float, default=0.2)
    args = parser.parse_args()

    # Retry-safe schema creation (the API/workers may be committing right now).
    init_db(args.db)
    conn = sqlite3.connect(args.db, timeout=args.busy_timeout, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")

    deadline = time.time() + 30
    while True:
        try:
            conn.execute("BEGIN IMMEDIATE")
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or time.time() >= deadline:
                raise
            time.sleep(0.05)
    conn.execute("CREATE TABLE IF NOT EXISTS lock_probe(marker TEXT)")
    print(f"holding write lock for {args.seconds}s", flush=True)
    time.sleep(args.seconds)
    conn.execute("ROLLBACK")
    conn.close()
    print("released write lock", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
