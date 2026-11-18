"""Lock-outage recovery tests.

Regression coverage for the incident where a long SQLite write transaction made
both workers wait past their 30 s busy timeout and exit, leaving queued and
in-flight jobs permanently unprocessed.  These tests prove that:

* during a write-lock outage no worker reports a committed batch or completion;
* after the lock is released the same worker resumes by itself from the newest
  *committed* checkpoint and finishes;
* when both workers recover together, the worker that lost the lease cannot
  write or roll the checkpoint back, and the final weights are identical to an
  uninterrupted run.
"""
from __future__ import annotations

import sqlite3
import threading
import time

import numpy as np
import pytest

from app import lease as lm
from app import store
from app.config import Settings
from app.db import init_db, is_busy_error, retry_on_busy
from app.worker import Worker


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "training.db",
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        poll_interval=0.02,
        batch_delay_seconds=0.01,
    )
    base.update(overrides)
    return Settings(**base)


def sample(records: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 3)).tolist()
    labels = [int(i) % 2 for i in range(records)]
    return features, labels


def run_worker(settings: Settings, worker_id: str):
    worker = Worker(settings, worker_id)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    return worker, thread


def stop(worker: Worker, thread: threading.Thread) -> None:
    worker.request_stop()
    thread.join(timeout=10)
    assert not thread.is_alive()


class WriteLock:
    """Hold BEGIN IMMEDIATE in a dedicated thread while the test runs.

    The connection is created and locked inside the holder thread, so one
    thread owns the connection for its whole lifetime and the WAL write lock is
    genuinely held for the duration of the context.
    """

    def __init__(self, db_path, busy_timeout: float = 0.2) -> None:
        self._db_path = str(db_path)
        self._busy_timeout = busy_timeout
        self._acquired = threading.Event()
        self._acquire_error: Exception | None = None
        self._release = threading.Event()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        conn = None
        try:
            conn = sqlite3.connect(self._db_path, timeout=self._busy_timeout, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("CREATE TABLE IF NOT EXISTS lock_probe(marker TEXT)")
            self._acquired.set()
            self._release.wait()
        except Exception as exc:  # surfaced to the test thread on __enter__
            self._acquire_error = exc
            self._acquired.set()
        finally:
            if conn is not None:
                try:
                    conn.execute("ROLLBACK")
                finally:
                    conn.close()
            self._closed.set()

    def __enter__(self) -> "WriteLock":
        self._thread.start()
        self._acquired.wait(timeout=5)
        if self._acquire_error is not None:
            self._thread.join(timeout=5)
            raise self._acquire_error
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        if not self._release.is_set():
            self._release.set()
            self._closed.wait(timeout=5)


def assert_write_lock_held(db_path) -> None:
    """Block until a separate writer provably cannot take the write lock."""
    deadline = time.time() + 5
    while time.time() < deadline:
        probe = sqlite3.connect(str(db_path), timeout=0.1, isolation_level=None)
        try:
            probe.execute("PRAGMA journal_mode=WAL")
            probe.execute("BEGIN IMMEDIATE")
            probe.execute("ROLLBACK")
        except sqlite3.OperationalError as exc:
            if is_busy_error(exc):
                probe.close()
                return
        finally:
            probe.close()
        time.sleep(0.02)
    raise AssertionError("write lock was never observed as held")


def wait_until(predicate, timeout: float, message: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {message}")


def test_is_busy_error_recognises_lock_contention():
    exc = sqlite3.OperationalError("database is locked")
    exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
    assert is_busy_error(exc)
    assert not is_busy_error(sqlite3.OperationalError("no such table: x"))
    assert not is_busy_error(ValueError("database is locked"))


def test_retry_on_busy_succeeds_after_lock_releases(tmp_path):
    settings = make_settings(tmp_path)
    init_db(settings.db_path)
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            exc = sqlite3.OperationalError("database is locked")
            exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise exc
        return "ok"

    assert retry_on_busy(flaky, timeout=10.0) == "ok"
    assert attempts["n"] == 3


def test_retry_on_busy_does_not_retry_non_busy_errors():
    calls = {"n": 0}

    def broken() -> None:
        calls["n"] += 1
        raise sqlite3.OperationalError("no such table")

    with pytest.raises(sqlite3.OperationalError):
        retry_on_busy(broken, timeout=5.0)
    assert calls["n"] == 1


def test_retry_on_busy_honours_stop_hook(tmp_path):
    from app import db as db_mod

    stop_event = threading.Event()
    token = db_mod.set_stop_hook(stop_event.is_set)
    stop_event.set()
    try:
        def always_busy() -> None:
            exc = sqlite3.OperationalError("database is locked")
            exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise exc

        with pytest.raises(db_mod.RetryStopRequested):
            retry_on_busy(always_busy, timeout=30.0)
    finally:
        db_mod.reset_stop_hook(token)


def test_worker_survives_lock_longer_than_busy_timeout_and_resumes(tmp_path, monkeypatch):
    # One second of total retry budget in the DB layer; the held lock outlasts
    # it, exercising both the inner retries and the worker's outer backoff.
    monkeypatch.setenv("TRAINING_LOCK_RETRY_SECONDS", "1")
    monkeypatch.setenv("TRAINING_SQLITE_BUSY_TIMEOUT", "0.3")
    settings = make_settings(tmp_path, lease_seconds=5.0, heartbeat_interval=0.1)
    init_db(settings.db_path)
    job = store.create_job(settings.db_path, *sample(24, seed=4), epochs=3)
    total = job["total_steps"]

    worker, thread = run_worker(settings, "w1")
    try:
        wait_until(
            lambda: store.get_job(settings.db_path, job["job_id"])["completed_steps"] >= 2,
            timeout=20,
            message="initial progress",
        )
        # Reference: what an uninterrupted run of the same input must produce.
        ref_path = tmp_path / "ref.db"
        init_db(ref_path)
        ref_job = store.create_job(ref_path, *sample(24, seed=4), epochs=3)
        lm.register_worker(ref_path, "ref")
        ref_lease = lm.acquire(ref_path, ref_job["job_id"], "ref", 5.0)
        while True:
            result = store.advance_job(ref_path, ref_lease)
            if result["status"] == "completed":
                break
        reference = store.get_job(ref_path, ref_job["job_id"])

        with WriteLock(settings.db_path):
            assert_write_lock_held(settings.db_path)
            snapshot = store.get_job(settings.db_path, job["job_id"])
            time.sleep(3.0)
            # While the lock is held nothing may be reported as committed.
            blocked = store.get_job(settings.db_path, job["job_id"])
            assert blocked["completed_steps"] == snapshot["completed_steps"]
            assert blocked["weights"] == snapshot["weights"]
            assert blocked["status"] == "running"
            checkpoints_during = store.list_checkpoints(settings.db_path, job["job_id"])
            assert [c["step"] for c in checkpoints_during][-1] == snapshot["completed_steps"]
            assert thread.is_alive(), "worker must not exit while the database is locked"

        wait_until(
            lambda: store.get_job(settings.db_path, job["job_id"])["status"] == "completed",
            timeout=30,
            message="completion after lock release",
        )
    finally:
        stop(worker, thread)

    final = store.get_job(settings.db_path, job["job_id"])
    assert final["completed_steps"] == total
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    assert [c["step"] for c in checkpoints] == list(range(1, total + 1))
    assert final["final_weights"] == reference["final_weights"]
    assert final["momentum_velocity"] == reference["momentum_velocity"]
    assert final["loss"] == reference["loss"]


def test_two_workers_recovering_together_keep_lease_fence_and_equivalence(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TRAINING_LOCK_RETRY_SECONDS", "1")
    monkeypatch.setenv("TRAINING_SQLITE_BUSY_TIMEOUT", "0.3")
    settings = make_settings(tmp_path, lease_seconds=0.8, heartbeat_interval=0.1)
    init_db(settings.db_path)
    job = store.create_job(settings.db_path, *sample(32, seed=8), epochs=4)
    total = job["total_steps"]

    # Uninterrupted oracle in a separate database.
    ref_path = tmp_path / "ref.db"
    init_db(ref_path)
    ref_job = store.create_job(ref_path, *sample(32, seed=8), epochs=4)
    lm.register_worker(ref_path, "ref")
    ref_lease = lm.acquire(ref_path, ref_job["job_id"], "ref", 5.0)
    while True:
        result = store.advance_job(ref_path, ref_lease)
        if result["status"] == "completed":
            break
    reference = store.get_job(ref_path, ref_job["job_id"])

    # A pauses (no heartbeats) on demand, so its lease can only be taken over.
    # The per-worker variable keeps B unaffected inside the same process.
    pause_file = tmp_path / "pause-a"
    monkeypatch.setenv("TRAINING_DEMO_PAUSE_FILE_w_a", str(pause_file))
    worker_a, thread_a = run_worker(settings, "w-a")
    try:
        wait_until(
            lambda: store.get_job(settings.db_path, job["job_id"])["completed_steps"] >= 2,
            timeout=20,
            message="A initial progress",
        )
        pause_file.write_text("paused")
        worker_b, thread_b = run_worker(settings, "w-b")
        try:
            def lease_expired_or_moved():
                held = lm.holder(settings.db_path, job["job_id"])
                if held is None:
                    return True
                worker_id, _epoch, heartbeat_at = held
                return (
                    worker_id == "w-b"
                    or time.time() - heartbeat_at >= settings.lease_seconds
                )

            # Deterministic takeover condition: A is paused and its lease is
            # stale, so B - never A - can own the next epoch.
            wait_until(lease_expired_or_moved, timeout=10, message="A lease expiry/move")
            paused_snapshot = store.get_job(settings.db_path, job["job_id"])

            # A write-lock outage during the handover must not kill either
            # worker, and nothing may commit while it is held.
            with WriteLock(settings.db_path):
                assert_write_lock_held(settings.db_path)
                time.sleep(settings.lease_seconds + 1.0)
                assert thread_a.is_alive() and thread_b.is_alive()
                assert (
                    store.get_job(settings.db_path, job["job_id"])["completed_steps"]
                    == paused_snapshot["completed_steps"]
                )

            wait_until(
                lambda: store.get_job(settings.db_path, job["job_id"])["status"] == "completed",
                timeout=40,
                message="completion after simultaneous recovery",
            )
            # Wake A only after completion: the fenced-out old owner must not
            # write or roll anything back.
            before_wake = store.get_job(settings.db_path, job["job_id"])
            pause_file.unlink()
            time.sleep(0.5)
            after_wake = store.get_job(settings.db_path, job["job_id"])
            final = store.get_job(settings.db_path, job["job_id"])
            checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
        finally:
            stop(worker_b, thread_b)
    finally:
        pause_file.unlink(missing_ok=True)
        stop(worker_a, thread_a)

    assert after_wake == before_wake
    steps = [c["step"] for c in checkpoints]
    assert steps == list(range(1, total + 1))
    assert len(steps) == len(set(steps)), "no duplicated or rolled-back checkpoints"
    epochs = [c["lease_epoch"] for c in checkpoints]
    assert epochs == sorted(epochs) and epochs[-1] >= 2, "a takeover epoch must exist"
    # The old owner's writes stop exactly where it paused: no writes above the
    # takeover step, and the checkpoint can never be rolled back.
    old_owner = [c for c in checkpoints if c["worker_id"] == "w-a"]
    assert max(c["step"] for c in old_owner) == paused_snapshot["completed_steps"]
    assert final["final_weights"] == reference["final_weights"]
    assert final["momentum_velocity"] == reference["momentum_velocity"]
    assert final["loss"] == reference["loss"]


def test_heartbeat_failure_under_lock_does_not_kill_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINING_LOCK_RETRY_SECONDS", "0.8")
    monkeypatch.setenv("TRAINING_SQLITE_BUSY_TIMEOUT", "0.25")
    settings = make_settings(tmp_path, lease_seconds=5.0, heartbeat_interval=0.1)
    init_db(settings.db_path)
    job = store.create_job(settings.db_path, *sample(16, seed=2), epochs=2)

    worker, thread = run_worker(settings, "w1")
    try:
        with WriteLock(settings.db_path):
            assert_write_lock_held(settings.db_path)
            time.sleep(2.0)
            assert thread.is_alive()
        wait_until(
            lambda: store.get_job(settings.db_path, job["job_id"])["status"] == "completed",
            timeout=20,
            message="completion",
        )
    finally:
        stop(worker, thread)

    final = store.get_job(settings.db_path, job["job_id"])
    assert final["completed_steps"] == final["total_steps"]


def test_queued_job_claimed_after_lock_outage(tmp_path, monkeypatch):
    """A job that is still queued when the outage starts is picked up after."""
    monkeypatch.setenv("TRAINING_LOCK_RETRY_SECONDS", "1")
    monkeypatch.setenv("TRAINING_SQLITE_BUSY_TIMEOUT", "0.3")
    settings = make_settings(tmp_path, lease_seconds=5.0, heartbeat_interval=0.1)
    init_db(settings.db_path)
    job = store.create_job(settings.db_path, *sample(16, seed=12), epochs=2)
    assert store.get_job(settings.db_path, job["job_id"])["status"] == "queued"

    # Hold the lock before the worker can even register/claim anything.
    with WriteLock(settings.db_path):
        assert_write_lock_held(settings.db_path)
        worker, thread = run_worker(settings, "w1")
        time.sleep(2.5)
        assert thread.is_alive()
        assert store.get_job(settings.db_path, job["job_id"])["status"] == "queued"
    try:
        wait_until(
            lambda: store.get_job(settings.db_path, job["job_id"])["status"] == "completed",
            timeout=25,
            message="queued job completion after outage",
        )
    finally:
        stop(worker, thread)

    final = store.get_job(settings.db_path, job["job_id"])
    assert final["completed_steps"] == final["total_steps"]
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    assert [c["step"] for c in checkpoints] == list(range(1, final["total_steps"] + 1))
