"""SQLite schema and connection helpers.

One database file on a shared Docker volume, mounted into the API and both
worker containers.  Keeping jobs, checkpoints, leases and heartbeats in one
file means a checkpoint commit and the lease check that authorises it are a
single atomic ``BEGIN IMMEDIATE`` transaction.

SQLite runs in WAL mode with ``synchronous=FULL``: a committed batch survives a
process crash and a container restart.

A *temporary* write lock (another connection holding a long write transaction)
must never stop training permanently.  Every mutating entry point therefore
goes through :func:`retry_on_busy`: when SQLite reports ``SQLITE_BUSY`` /
``SQLITE_LOCKED`` the operation is retried with a capped backoff until the
database becomes writable again, the retry budget is exhausted, or the worker
asks to stop.  A unit of work is only retried as a whole, so a batch is only
ever reported as committed after its transaction has really committed.
"""
from __future__ import annotations

import functools
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

logger = logging.getLogger("app.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    status          TEXT NOT NULL,
    feature_dim     INTEGER NOT NULL,
    epochs          INTEGER NOT NULL,
    batch_size      INTEGER NOT NULL,
    record_count    INTEGER NOT NULL,
    total_batches   INTEGER NOT NULL,
    learning_rate   REAL NOT NULL,
    momentum        REAL NOT NULL,
    features_json   TEXT NOT NULL,
    labels_json     TEXT NOT NULL,
    weights_json    TEXT NOT NULL,
    velocity_json   TEXT NOT NULL,
    loss            REAL NOT NULL,
    completed_steps INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    finished_at     REAL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    job_id        TEXT NOT NULL,
    step          INTEGER NOT NULL,
    weights_json  TEXT NOT NULL,
    velocity_json TEXT NOT NULL,
    loss          REAL NOT NULL,
    batch_loss    REAL NOT NULL,
    committed_at  REAL NOT NULL,
    worker_id     TEXT NOT NULL,
    lease_epoch   INTEGER NOT NULL,
    PRIMARY KEY (job_id, step),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);

CREATE TABLE IF NOT EXISTS job_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT NOT NULL,
    event      TEXT NOT NULL,
    detail     TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS workers (
    worker_id    TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    started_at   REAL NOT NULL,
    heartbeat_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS leases (
    job_id       TEXT PRIMARY KEY,
    worker_id    TEXT NOT NULL,
    epoch        INTEGER NOT NULL,
    heartbeat_at REAL NOT NULL
);

-- Monotonic per-job lease counter.  Leases themselves are deleted on release,
-- but the counter is never reset, so a *new* lease can never reuse an epoch a
-- previous holder still holds a reference to.
CREATE TABLE IF NOT EXISTS lease_epochs (
    job_id TEXT PRIMARY KEY,
    epoch  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_job ON job_events(job_id, id);
"""

# SQLite result codes for lock contention.  SQLITE_BUSY (5) is the write-lock
# timeout; SQLITE_LOCKED (6) is a table-level conflict, treated the same way.
_BUSY_CODES = frozenset(
    {
        getattr(sqlite3, "SQLITE_BUSY", 5),
        getattr(sqlite3, "SQLITE_LOCKED", 6),
    }
)
_BUSY_MESSAGES = ("database is locked", "database table is locked")

DEFAULT_BUSY_TIMEOUT = 30.0
DEFAULT_LOCK_RETRY_SECONDS = 300.0
_INITIAL_BACKOFF = 0.05
_BACKOFF_FACTOR = 1.5
_MAX_BACKOFF = 1.0

T = TypeVar("T")

# Optional cooperative stop hook (a worker's ``stop_event.is_set``).  While a
# mutating call is waiting for the write lock it is interrupted as soon as the
# hook returns true, so SIGTERM still shuts a blocked worker down promptly.
_stop_hook: ContextVar[Callable[[], bool] | None] = ContextVar(
    "db_stop_hook", default=None
)


class RetryStopRequested(Exception):
    """Raised when the stop hook fires while waiting for the write lock."""


def set_stop_hook(hook: Callable[[], bool]):
    """Install a cooperative stop check for lock retries on this context."""
    return _stop_hook.set(hook)


def reset_stop_hook(token) -> None:
    _stop_hook.reset(token)


def busy_timeout() -> float:
    """Per-attempt SQLite busy timeout (``TRAINING_SQLITE_BUSY_TIMEOUT``)."""
    raw = os.environ.get("TRAINING_SQLITE_BUSY_TIMEOUT")
    return DEFAULT_BUSY_TIMEOUT if raw is None else float(raw)


def lock_retry_timeout() -> float:
    """Total budget for retrying one operation against a busy write lock."""
    raw = os.environ.get("TRAINING_LOCK_RETRY_SECONDS")
    return DEFAULT_LOCK_RETRY_SECONDS if raw is None else float(raw)


def is_busy_error(exc: BaseException) -> bool:
    """True when ``exc`` is a transient SQLite lock-contention error."""
    if not isinstance(exc, sqlite3.Error):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None:
        return int(code) in _BUSY_CODES
    message = str(exc).lower()
    return any(needle in message for needle in _BUSY_MESSAGES)


def retry_on_busy(
    func: Callable[..., T],
    *args: Any,
    timeout: float | None = None,
    **kwargs: Any,
) -> T:
    """Call ``func`` and retry only on transient SQLite lock errors.

    The whole call is retried, so ``func`` must be a unit of work that is safe
    to re-execute: every mutating caller re-reads the database state inside a
    fresh ``BEGIN IMMEDIATE`` transaction, and a busy error means nothing was
    committed.  Non-busy errors propagate immediately.  When the budget runs
    out the last busy error propagates so the caller can back off and try the
    whole cycle again.
    """
    budget = lock_retry_timeout() if timeout is None else timeout
    if budget <= 0:
        return func(*args, **kwargs)
    deadline = time.monotonic() + budget
    delay = _INITIAL_BACKOFF
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except sqlite3.Error as exc:
            if not is_busy_error(exc):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            hook = _stop_hook.get()
            if hook is not None and hook():
                raise RetryStopRequested(
                    "stop requested while waiting for the database write lock"
                ) from exc
            attempt += 1
            sleep_for = min(delay, remaining)
            logger.warning(
                "database write lock busy (%s); retry %d in %.2fs",
                exc,
                attempt,
                sleep_for,
            )
            time.sleep(sleep_for)
            delay = min(delay * _BACKOFF_FACTOR, _MAX_BACKOFF)


def resilient(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator making a whole mutating function survive temporary locks."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return retry_on_busy(func, *args, **kwargs)

    return wrapper


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=busy_timeout(), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout() * 1000)}")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(path: Path | str) -> Iterator[sqlite3.Connection]:
    """A write transaction that takes the write lock immediately.

    ``BEGIN IMMEDIATE`` makes writers in separate containers queue on the busy
    timeout instead of failing on a deferred-to-exclusive lock upgrade.

    If ``COMMIT`` itself reports ``SQLITE_BUSY`` (possible in WAL mode while a
    checkpoint runs), the transaction is still open per SQLite's rules, so the
    commit is retried on the same connection.  Only after that fails is the
    transaction rolled back and the busy error re-raised for
    :func:`retry_on_busy` to redo the whole unit of work.
    """
    conn = connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        _commit_with_retry(conn)
    finally:
        conn.close()


def _commit_with_retry(conn: sqlite3.Connection) -> None:
    # In WAL mode a COMMIT can briefly return SQLITE_BUSY while a checkpoint
    # runs.  Retry on the same open transaction; if it stays busy, roll back and
    # let retry_on_busy redo the whole unit of work (every mutating caller
    # re-reads its inputs inside a fresh transaction, so nothing is duplicated).
    for attempt in (1, 2, 3):
        try:
            conn.execute("COMMIT")
            return
        except sqlite3.Error as exc:
            if not is_busy_error(exc) or attempt == 3:
                conn.execute("ROLLBACK")
                raise
            logger.warning("COMMIT hit a busy database; retrying commit (%d)", attempt)
            time.sleep(0.1 * attempt)


@contextmanager
def reader(path: Path | str) -> Iterator[sqlite3.Connection]:
    """A consistent read snapshot (WAL readers do not block writers)."""
    conn = connect(path)
    try:
        conn.execute("BEGIN")
        try:
            yield conn
        finally:
            conn.execute("COMMIT")
    finally:
        conn.close()


def init_db(path: Path | str) -> None:
    """Create the schema if missing.

    ``executescript`` commits implicitly, so it must not run inside our
    hand-managed transaction; concurrent starts serialise on SQLite's schema
    lock and every statement is idempotent.  Retried on busy errors so a worker
    starting while another connection holds the write lock does not crash.
    """

    def _create() -> None:
        conn = connect(path)
        try:
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    retry_on_busy(_create)
