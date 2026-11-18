"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


@dataclass(frozen=True)
class Settings:
    """Paths and timing shared by the API and the workers.

    ``db_path`` is one SQLite file on a shared Docker volume, mounted into the
    API and both worker containers.  Jobs, checkpoints, leases and heartbeats
    live in it, so a checkpoint commit and the lease check that authorises it
    are a single atomic transaction.

    A temporary write lock never stops training: each SQLite write waits up to
    ``sqlite_busy_timeout`` seconds, then is retried with backoff for up to
    ``lock_retry_seconds`` in total; the worker keeps the job and re-enters from
    the newest committed checkpoint when an outage outlasts that budget.
    """

    db_path: Path
    lease_seconds: float
    heartbeat_interval: float
    poll_interval: float
    batch_delay_seconds: float
    sqlite_busy_timeout: float = 30.0
    lock_retry_seconds: float = 300.0

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("TRAINING_DATA_DIR", str(ROOT / "data")))
        db_path = Path(os.environ.get("TRAINING_DB_PATH", str(data_dir / "training.db")))
        return cls(
            db_path=db_path.resolve(),
            lease_seconds=_float("TRAINING_LEASE_SECONDS", 6.0),
            heartbeat_interval=_float("TRAINING_HEARTBEAT_INTERVAL", 1.0),
            poll_interval=_float("TRAINING_POLL_INTERVAL", 0.3),
            batch_delay_seconds=_float("TRAINING_BATCH_DELAY_SECONDS", 0.0),
            sqlite_busy_timeout=_float("TRAINING_SQLITE_BUSY_TIMEOUT", 30.0),
            lock_retry_seconds=_float("TRAINING_LOCK_RETRY_SECONDS", 300.0),
        )

    def validate(self) -> None:
        if self.lease_seconds <= 0:
            raise ValueError("TRAINING_LEASE_SECONDS must be positive")
        if not (0 < self.heartbeat_interval < self.lease_seconds):
            raise ValueError("heartbeat interval must be positive and shorter than the lease")
        if self.poll_interval <= 0:
            raise ValueError("TRAINING_POLL_INTERVAL must be positive")
        if self.batch_delay_seconds < 0:
            raise ValueError("TRAINING_BATCH_DELAY_SECONDS must not be negative")
        if self.sqlite_busy_timeout <= 0:
            raise ValueError("TRAINING_SQLITE_BUSY_TIMEOUT must be positive")
        if self.lock_retry_seconds < 0:
            raise ValueError("TRAINING_LOCK_RETRY_SECONDS must not be negative")
