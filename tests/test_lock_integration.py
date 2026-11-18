"""Process-level lock-outage integration test.

Reproduces the incident with real processes (one uvicorn API, two real
``app.worker`` processes) and a third process that holds a long
``BEGIN IMMEDIATE`` write transaction - the same shape as the engineer's long
SQLite write transaction, compressed in time:

* the per-attempt busy timeout is shortened to 1 s and the retry budget to 3 s
  so the test exercises the exact "busy timeout exhausted" code path without
  waiting the production 30 s per attempt;
* the lock is held long enough to outlast the retry budget (and, in the second
  case, the training lease);
* both workers must stay alive, nothing may be reported as committed during the
  outage, and after the lock is released the job is finished from the newest
  committed checkpoint with weights identical to an uninterrupted run.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEARTBEAT_INTERVAL = 0.3
BATCH_DELAY = 0.2


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def env_for(db: Path, lease: float = 2.0, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "TRAINING_DB_PATH": str(db),
            "TRAINING_LEASE_SECONDS": str(lease),
            "TRAINING_HEARTBEAT_INTERVAL": str(HEARTBEAT_INTERVAL),
            "TRAINING_POLL_INTERVAL": "0.05",
            "TRAINING_BATCH_DELAY_SECONDS": str(BATCH_DELAY),
            # Compressed lock budgets: the production defaults (30 s / 300 s)
            # take the same code paths, just slower.
            "TRAINING_SQLITE_BUSY_TIMEOUT": "1",
            "TRAINING_LOCK_RETRY_SECONDS": "3",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(extra)
    return env


class Server:
    def __init__(self, db: Path, port: int, lease: float = 2.0) -> None:
        self.db = db
        self.port = port
        self.lease = lease
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "Server":
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--log-level",
                "warning",
            ],
            cwd=ROOT,
            env=env_for(self.db, self.lease),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                self.get("/health")
                return self
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("API did not start")

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.send_signal(signal.SIGTERM)
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)

    def get(self, path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
            return json.loads(response.read())

    def post(self, path: str, body: dict):
        data = json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


def start_worker(db: Path, worker_id: str, lease: float = 2.0, **env: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "app.worker", "--worker-id", worker_id],
        cwd=ROOT,
        env=env_for(db, lease, **env),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_until_write_lock_held(db: Path) -> None:
    """Block until a separate writer provably cannot take the write lock."""
    deadline = time.time() + 10
    while time.time() < deadline:
        probe = sqlite3.connect(str(db), timeout=0.1, isolation_level=None)
        try:
            probe.execute("PRAGMA journal_mode=WAL")
            probe.execute("BEGIN IMMEDIATE")
            probe.execute("ROLLBACK")
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower():
                return
        finally:
            probe.close()
        time.sleep(0.05)
    raise AssertionError("write lock was never observed as held")


def hold_write_lock(db: Path, seconds: float) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests._lock_holder", "--db", str(db), "--seconds", str(seconds)],
        cwd=ROOT,
        env=env_for(db),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    line = proc.stdout.readline()
    assert "holding write lock" in line, line
    return proc


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def wait_for(predicate, timeout: float, message: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {message}")


def wait_completed(server: Server, job_id: str, timeout: float = 90):
    return wait_for(
        lambda: (state := server.get(f"/jobs/{job_id}"))["status"] == "completed" and state,
        timeout=timeout,
        message="job completion",
    )


def dataset(records: int = 40, seed: int = 21):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 4)).tolist()
    labels = [1 if row[0] + row[1] > 0 else 0 for row in features]
    return {"features": features, "labels": labels, "epochs": 3}


def run_reference(tmp_path: Path, data: dict):
    """Uninterrupted oracle run for one concrete dataset."""
    db = tmp_path / ("ref-" + uuid.uuid4().hex + ".db")
    with Server(db, free_port()) as server:
        _, job = server.post("/jobs", data)
        worker = start_worker(db, "solo")
        try:
            return wait_completed(server, job["job_id"])
        finally:
            stop(worker)


def test_worker_recovers_after_long_write_lock_and_matches_reference(tmp_path):
    data = dataset()
    reference = run_reference(tmp_path, data)
    db = tmp_path / "training.db"
    # A long lease so the blocked owner never loses the job during the outage.
    with Server(db, free_port(), lease=30.0) as server:
        _, job = server.post("/jobs", data)
        worker = start_worker(db, "worker-a", lease=30.0)
        try:
            wait_for(
                lambda: server.get(f"/jobs/{job['job_id']}")["completed_steps"] >= 3,
                timeout=60,
                message="first batches",
            )
            before = server.get(f"/jobs/{job['job_id']}")

            holder = hold_write_lock(db, 6.0)
            try:
                # The outage outlasts both the 1 s busy timeout and the 3 s
                # retry budget; reads still work in WAL mode.
                time.sleep(3.5)
                blocked = server.get(f"/jobs/{job['job_id']}")
                assert blocked["completed_steps"] == before["completed_steps"]
                assert blocked["weights"] == before["weights"]
                assert blocked["status"] == "running"
                checkpoints = server.get(f"/jobs/{job['job_id']}/checkpoints")
                assert checkpoints[-1]["step"] == before["completed_steps"]
                assert worker.poll() is None, "worker exited during the write-lock outage"
            finally:
                holder.wait(timeout=15)

            final = wait_completed(server, job["job_id"])
            checkpoints = server.get(f"/jobs/{job['job_id']}/checkpoints")
        finally:
            stop(worker)

    assert final["completed_steps"] == final["total_steps"]
    assert final["final_weights"] == reference["final_weights"]
    assert final["momentum_velocity"] == reference["momentum_velocity"]
    assert final["loss"] == reference["loss"]
    steps = [c["step"] for c in checkpoints]
    assert steps == list(range(1, final["total_steps"] + 1))


def test_two_workers_recover_together_with_lease_fencing(tmp_path):
    data = dataset(records=32, seed=8)
    reference = run_reference(tmp_path, data)
    db = tmp_path / "training.db"
    pause_file = tmp_path / "pause-a"
    with Server(db, free_port(), lease=2.0) as server:
        _, job = server.post("/jobs", data)
        # A pauses (without heartbeating) whenever the pause file exists, so
        # after it pauses its lease can only be taken over - never renewed.
        worker_a = start_worker(
            db, "worker-a", lease=2.0, TRAINING_DEMO_PAUSE_FILE=str(pause_file)
        )
        try:
            wait_for(
                lambda: server.get(f"/jobs/{job['job_id']}")["completed_steps"] >= 2,
                timeout=60,
                message="A progress",
            )
            # A may still commit one in-flight batch before reaching the pause
            # point; the true frozen step is snapshotted once its lease expires.
            pause_file.write_text("paused")
            worker_b = start_worker(db, "worker-b", lease=2.0)
            try:
                def a_lease_expired_or_moved():
                    leases = server.get("/workers")["leases"]
                    held = [l for l in leases if l["job_id"] == job["job_id"]]
                    if not held:
                        return True  # already taken over and released (job done)
                    lease = held[0]
                    return lease["expired"] or lease["worker_id"] == "worker-b"

                # Wait for A's lease to expire (WAL reads work during outages).
                # At this instant A is paused inside the pause loop, so this is
                # the true last step it can ever own.
                wait_for(a_lease_expired_or_moved, timeout=15, message="A lease expiry/move")
                paused_snapshot = server.get(f"/jobs/{job['job_id']}")
                # Now hold the write lock: B's takeover and first commits must
                # survive the busy period instead of either worker dying.
                holder = hold_write_lock(db, 4.0)
                try:
                    wait_until_write_lock_held(db)
                    locked_snapshot = server.get(f"/jobs/{job['job_id']}")
                    time.sleep(2.0)
                    assert worker_a.poll() is None and worker_b.poll() is None
                    # Nothing can commit while the write lock is held.
                    assert (
                        server.get(f"/jobs/{job['job_id']}")["completed_steps"]
                        == locked_snapshot["completed_steps"]
                    )
                finally:
                    holder.wait(timeout=15)

                # Lock free: B takes over (A cannot renew while paused) and
                # finishes from the committed checkpoint.
                final = wait_completed(server, job["job_id"])
                checkpoints = server.get(f"/jobs/{job['job_id']}/checkpoints")

                # Only then wake A: it must discover the lost lease and write
                # nothing, so the completed state stands unchanged.
                before_resume = server.get(f"/jobs/{job['job_id']}")
                pause_file.unlink()
                time.sleep(2.0)
                after_resume = server.get(f"/jobs/{job['job_id']}")
            finally:
                stop(worker_b)
        finally:
            pause_file.unlink(missing_ok=True)
            stop(worker_a)

    assert after_resume == before_resume
    steps = [c["step"] for c in checkpoints]
    assert steps == list(range(1, final["total_steps"] + 1))
    assert len(steps) == len(set(steps)), "no duplicated or rolled-back checkpoints"
    epochs = [c["lease_epoch"] for c in checkpoints]
    assert epochs == sorted(epochs) and epochs[-1] >= 2, "a takeover epoch must exist"
    # The old owner's writes stop exactly where it paused: it can neither
    # write above the takeover nor roll the breakpoint back.
    old_owner = [c for c in checkpoints if c["worker_id"] == "worker-a"]
    assert max(c["step"] for c in old_owner) == paused_snapshot["completed_steps"]
    assert final["final_weights"] == reference["final_weights"]


def test_api_returns_retryable_503_while_lock_budget_exhausted(tmp_path):
    db = tmp_path / "training.db"
    with Server(db, free_port()) as server:
        holder = hold_write_lock(db, 6.0)
        try:
            status, body = server.post("/jobs", dataset(records=8))
            assert status == 503
            assert "retry" in body["detail"].lower()
        finally:
            holder.wait(timeout=15)
        # Once writable again the same submission succeeds.
        status, job = server.post("/jobs", dataset(records=8))
        assert status == 201
        worker = start_worker(db, "solo")
        try:
            final = wait_completed(server, job["job_id"])
        finally:
            stop(worker)
        assert final["status"] == "completed"
