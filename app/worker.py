"""Background training worker.

Both worker containers run this process with a different ``WORKER_ID``.  Each
worker registers itself, publishes a heartbeat, polls for an unfinished job
whose lease is free or expired, and then trains one batch at a time.  Every
batch commit writes the full checkpoint (weights, momentum, batch loss, dataset
loss, step, timestamp, worker and lease epoch) plus the job's progress in one
transaction.

If the process dies, its heartbeat stops; the other worker waits for the lease
to expire and resumes from the newest committed step.  If a worker is merely
suspended and wakes up later, its committed fence check fails and it cannot
write, so it can neither roll the breakpoint back nor report a false completion.

A *temporary* SQLite write lock must not stop training either.  While another
connection holds a long write transaction, the database layer retries each
mutating operation with a backoff; if that budget is ever exhausted, this
worker drops back to its main loop, keeps heartbeating as best it can and tries
again once the database is writable.  A batch is only announced after its
transaction has actually committed, so a lock timeout can never produce a false
"batch committed" or "job completed" report.  The newest committed checkpoint
remains the single source of truth, so every recovery continues exactly where
the last confirmed batch stopped.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sqlite3
import sys
import threading
import time
import uuid
from pathlib import Path

from . import lease as lease_mod
from . import store
from .config import Settings
from .db import (
    RetryStopRequested,
    init_db,
    is_busy_error,
    reset_stop_hook,
    set_stop_hook,
)
from .lease import Lease, LeaseLost

logger = logging.getLogger("app.worker")


class Worker:
    def __init__(self, settings: Settings, worker_id: str) -> None:
        self.settings = settings
        self.worker_id = worker_id
        self.stop_event = threading.Event()
        self.lease: Lease | None = None
        self._last_heartbeat = 0.0
        self._last_status = ""
        # Backoff state for lock outages that outlast the per-operation budget.
        self._lock_backoff = 0.05

    def request_stop(self, *_: object) -> None:
        self.stop_event.set()

    def beat(self, status: str) -> bool:
        """Publish liveness and renew the lease.

        Returns ``False`` when the lease has been taken over, which the training
        loop treats as a hard stop.  A busy database does not count as a lost
        lease: the heartbeat is retried by the database layer and any residual
        busy error propagates to the main loop, which waits and retries instead
        of dying.
        """
        now = time.monotonic()
        heartbeat_due = (
            status != self._last_status
            or now - self._last_heartbeat >= self.settings.heartbeat_interval
        )
        if heartbeat_due:
            lease_mod.heartbeat(self.settings.db_path, self.worker_id, status)
            self._last_heartbeat = now
            self._last_status = status
        if self.lease is None:
            return True
        return lease_mod.renew(self.settings.db_path, self.lease, self.settings.lease_seconds)

    def run(self) -> None:
        self.settings.validate()
        # Make every lock wait interruptible by SIGTERM/SIGINT.
        token = set_stop_hook(self.stop_event.is_set)
        try:
            # Startup itself must survive a database that is already locked:
            # retry registration until the write lock frees or we are stopped.
            self._wait_for_database(init_db, self.settings.db_path)
            self._wait_for_database(
                lease_mod.register_worker, self.settings.db_path, self.worker_id
            )
            self._wait_for_database(self.beat, "idle")
            print(f"[{self.worker_id}] started", flush=True)
            while not self.stop_event.is_set():
                try:
                    self._run_cycle()
                except LeaseLost:
                    # _train normally consumes this itself; this is the
                    # defensive outer guard so it can never kill the process.
                    logger.warning("lease lost; returning to the idle pool")
                    self.lease = None
                except RetryStopRequested:
                    # SIGTERM/SIGINT arrived while waiting on the write lock.
                    break
                except sqlite3.Error as exc:
                    if not is_busy_error(exc):
                        raise
                    # The database stayed locked longer than the per-operation
                    # retry budget. Stay alive: wait and re-enter the same
                    # cycle, keeping any lease so training resumes from the
                    # newest committed checkpoint instead of restarting.
                    self._handle_lock_outage(exc)
        finally:
            reset_stop_hook(token)
            if self.lease is not None:
                self._best_effort_release()
            self._best_effort_stopped_heartbeat()
            print(f"[{self.worker_id}] stopped", flush=True)

    def _wait_for_database(self, func, *args) -> None:
        """Run a startup action, backing off across temporary lock outages."""
        while not self.stop_event.is_set():
            try:
                func(*args)
                return
            except RetryStopRequested:
                return
            except sqlite3.Error as exc:
                if not is_busy_error(exc):
                    raise
                self._handle_lock_outage(exc)

    def _run_cycle(self) -> None:
        # A lease kept here means training was interrupted by a lock outage:
        # re-enter the same job rather than re-polling and dropping the lease.
        if self.lease is None and not self._run_one_job():
            self.beat("idle")
            self.stop_event.wait(self.settings.poll_interval)
            return
        train_failed: BaseException | None = None
        try:
            self._train()
        except BaseException as exc:  # keep the lease on lock outages
            train_failed = exc
            raise
        finally:
            if self.lease is not None:
                if train_failed is None:
                    lease_mod.release(self.settings.db_path, self.lease)
                    self.lease = None
                else:
                    # Re-enter with the same lease after the backoff so the job
                    # continues from its newest committed checkpoint.
                    logger.info(
                        "keeping lease for %s after interrupted training cycle",
                        self.lease.job_id,
                    )

    def _handle_lock_outage(self, exc: sqlite3.Error) -> None:
        delay = min(self._lock_backoff, self.settings.lease_seconds)
        print(
            f"[{self.worker_id}] database write lock unavailable "
            f"({exc}); backing off {delay:.2f}s and will retry",
            flush=True,
        )
        self.stop_event.wait(delay)
        self._lock_backoff = min(self._lock_backoff * 1.5, 5.0)

    def _best_effort_release(self) -> None:
        try:
            lease_mod.release(self.settings.db_path, self.lease)
        except sqlite3.Error as exc:
            if is_busy_error(exc):
                logger.warning("could not release lease while database is locked: %s", exc)
            else:
                raise
        self.lease = None

    def _best_effort_stopped_heartbeat(self) -> None:
        try:
            lease_mod.heartbeat(self.settings.db_path, self.worker_id, "stopped")
        except sqlite3.Error as exc:
            if is_busy_error(exc):
                logger.warning("could not publish stopped heartbeat: %s", exc)
            else:
                raise

    def _run_one_job(self) -> bool:
        """Claim one job's lease. Returns ``False`` when nothing is claimable."""
        for job_id in store.claimable_jobs(self.settings.db_path):
            lease = lease_mod.acquire(
                self.settings.db_path,
                job_id,
                self.worker_id,
                self.settings.lease_seconds,
            )
            if lease is not None:
                self.lease = lease
                self._lock_backoff = 0.05
                return True
        return False

    def _train(self) -> None:
        assert self.lease is not None
        job_id = self.lease.job_id
        lease_key = (self.lease.worker_id, self.lease.epoch)
        start_recorded = False

        def record_start() -> None:
            nonlocal start_recorded
            if start_recorded:
                return
            job = store.get_job(self.settings.db_path, job_id)
            if job is None:
                return
            store.record_event(
                self.settings.db_path,
                job_id,
                "training_started",
                {
                    "worker_id": self.worker_id,
                    "lease_epoch": self.lease.epoch,
                    "resumed_from_step": job["completed_steps"],
                },
            )
            start_recorded = True
            print(
                f"[{self.worker_id}] training {job_id} from step "
                f"{job['completed_steps']}/{job['total_steps']} "
                f"(lease epoch {self.lease.epoch})",
                flush=True,
            )

        while not self.stop_event.is_set():
            try:
                # Re-read on every cycle: after a lock outage another worker may
                # have made progress, taken over, or finished the job.
                job = store.get_job(self.settings.db_path, job_id)
                if job is None:
                    return
                if job["status"] == "completed":
                    print(f"[{self.worker_id}] {job_id} already completed; releasing", flush=True)
                    return
                # Renew first: if another worker took over while we were
                # blocked, leave without touching the job.  The commit below is
                # independently fenced by the lease epoch.
                if not self.beat("training"):
                    raise LeaseLost("lease was taken over")
                record_start()
                result = store.advance_job(
                    self.settings.db_path,
                    self.lease,
                    self.settings.batch_delay_seconds,
                )
            except LeaseLost:
                print(f"[{self.worker_id}] lost lease on {job_id}; stopping", flush=True)
                self._record_lease_lost(job_id, lease_key)
                self.lease = None
                return
            # A successful commit proves the database is writable again.
            self._lock_backoff = 0.05
            print(
                f"[{self.worker_id}] {job_id} step {result['step']}/{result['total_steps']} "
                f"loss={result['loss']:.6f}",
                flush=True,
            )
            if result["status"] == "completed":
                print(f"[{self.worker_id}] completed {job_id}", flush=True)
                return
            self._pause_after_batch(job_id)

    def _record_lease_lost(self, job_id: str, lease_key: tuple[str, int]) -> None:
        try:
            store.record_event(
                self.settings.db_path,
                job_id,
                "lease_lost",
                {"worker_id": lease_key[0], "lease_epoch": lease_key[1]},
            )
        except sqlite3.Error as exc:
            if is_busy_error(exc):
                logger.warning("could not record lease_lost event while locked: %s", exc)
            else:
                raise

    def _pause_after_batch(self, job_id: str) -> None:
        """Simulated freezes for the failover demo; disabled by default.

        ``TRAINING_DEMO_PAUSE_FILE`` (a path shared with the demo) pauses this
        worker between batches, outside any SQLite transaction, without
        heartbeating - so its lease expires and the other worker takes over.

        With ``TRAINING_DEMO_CRASH_AFTER`` set to a step number, the process
        exits hard right after that step commits, modelling a container crash.
        Both are no-ops unless the environment variables are set.
        """
        crash_after = os.environ.get("TRAINING_DEMO_CRASH_AFTER")
        if crash_after and int(crash_after) == int(
            store.get_job(self.settings.db_path, job_id)["completed_steps"]
        ):
            print(f"[{self.worker_id}] simulated crash after step {crash_after}", flush=True)
            os._exit(70)

        pause_file = (
            os.environ.get(f"TRAINING_DEMO_PAUSE_FILE_{self.worker_id.replace('-', '_')}")
            or os.environ.get("TRAINING_DEMO_PAUSE_FILE")
        )
        if not pause_file:
            return
        path = Path(pause_file)
        if path.exists():
            print(f"[{self.worker_id}] paused by {path}", flush=True)
            # No heartbeat while paused: this is what lets the lease expire.
            while path.exists() and not self.stop_event.is_set():
                self.stop_event.wait(self.settings.poll_interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Training worker")
    parser.add_argument("--worker-id", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("TRAINING_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    settings = Settings.from_env()
    worker_id = args.worker_id or f"worker-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
    worker = Worker(settings, worker_id)
    signal.signal(signal.SIGTERM, worker.request_stop)
    signal.signal(signal.SIGINT, worker.request_stop)
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
