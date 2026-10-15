# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from openviking.message import Message, TextPart
from openviking.service.session_requeue import (
    DEFAULT_BASE_BACKOFF_SECONDS,
    DEFAULT_MAX_BACKOFF_SECONDS,
    DEFAULT_MAX_REQUEUE_ATTEMPTS,
    RequeueCandidate,
    SessionRequeueSweeper,
    calculate_requeue_backoff,
    clear_requeue,
    register_failed_requeue,
)
from openviking.service.task_tracker import TaskStatus, TaskTracker, set_task_tracker
from openviking.session.session import Session
from openviking.session.session_lock import (
    SessionExtractionBusyError,
    SessionExtractionLockManager,
    get_session_extraction_lock_manager,
)
from openviking.storage.queuefs import QueueManager
from openviking.storage.queuefs.session_commit_msg import SessionCommitMsg


class _FakeTaskStore:
    def __init__(self):
        self.tasks = {}

    async def create(self, task):
        self.tasks[task.task_id] = task

    async def update(self, task):
        self.tasks[task.task_id] = task

    async def get(self, task_id, *, account_id=None, user_id=None):
        return self.tasks.get(task_id)

    async def list(self, account_id, *, user_id=None):
        return list(self.tasks.values())

    async def delete(self, task_id, *, account_id, user_id=None):
        self.tasks.pop(task_id, None)


class _FakeAsyncAGFS:
    def __init__(self, files_dict):
        self.files = files_dict
        self._pathlocks = set()

    async def ls(self, path):
        prefix = path.rstrip("/") + "/"
        entries = set()
        for p in self.files:
            if p.startswith(prefix):
                remainder = p[len(prefix) :]
                parts = remainder.split("/")
                name = parts[0]
                is_dir = len(parts) > 1
                entries.add((name, is_dir))
        return [{"name": name, "isDir": is_dir} for name, is_dir in sorted(entries)]

    async def read(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        content = self.files[path]
        return content.encode("utf-8") if isinstance(content, str) else content

    async def write(self, path, data):
        self.files[path] = data.decode("utf-8") if isinstance(data, bytes) else str(data)

    async def rm(self, path, force=False):
        self.files.pop(path, None)

    async def ensure_parent_dirs(self, path, mode="755"):
        pass

    async def pathlock_acquire_exact(self, target, timeout_secs=0.0):
        if target in self._pathlocks:
            if timeout_secs <= 0.0:
                raise RuntimeError(f"Lock busy on {target}")
            # wait simulation
            while target in self._pathlocks:
                await asyncio.sleep(0.01)
        self._pathlocks.add(target)
        return target

    async def pathlock_release(self, lease):
        self._pathlocks.discard(lease)


class _FakeVikingFS:
    def __init__(self, files=None):
        self.files = files if files is not None else {}
        self._async_agfs = _FakeAsyncAGFS(self.files)

    def _uri_to_path(self, uri, ctx=None):
        return uri.replace("viking://user/", "/local/").replace("viking://", "/local/")

    async def read_file(self, uri, ctx=None):
        path = self._uri_to_path(uri, ctx)
        if path in self.files:
            return self.files[path]
        if uri in self.files:
            return self.files[uri]
        raise FileNotFoundError(uri)

    async def write_file(self, uri, content, ctx=None, lease_ref=None):
        path = self._uri_to_path(uri, ctx)
        self.files[path] = content

    async def rm(self, uri, ctx=None, **kwargs):
        path = self._uri_to_path(uri, ctx)
        self.files.pop(path, None)
        self.files.pop(uri, None)

    async def exists(self, uri, ctx=None):
        path = self._uri_to_path(uri, ctx)
        return path in self.files or uri in self.files


# ── 1. Exponential Backoff & Serde ──


def test_calculate_requeue_backoff():
    b1 = calculate_requeue_backoff(1, base_delay=10.0, max_delay=3600.0)
    assert b1 == 10.0
    b2 = calculate_requeue_backoff(2, base_delay=10.0, max_delay=3600.0)
    assert b2 == 20.0
    b3 = calculate_requeue_backoff(3, base_delay=10.0, max_delay=3600.0)
    assert b3 == 40.0
    b4 = calculate_requeue_backoff(4, base_delay=10.0, max_delay=3600.0)
    assert b4 == 80.0
    b_max = calculate_requeue_backoff(15, base_delay=10.0, max_delay=3600.0)
    assert b_max <= 3600.0


def test_requeue_candidate_serde():
    candidate = RequeueCandidate(
        account_id="acc-1",
        user_id="user-1",
        session_id="sess-1",
        session_uri="viking://user/acc-1/sessions/sess-1",
        archive_uri="viking://user/acc-1/sessions/sess-1/history/archive_001",
        requeue_count=3,
        next_requeue_at=1700000000.0,
        quarantined=False,
        quarantine_reason=None,
        last_error="APITimeoutError",
        created_at="2026-10-15T12:00:00Z",
    )
    d = candidate.to_dict()
    assert d["account_id"] == "acc-1"
    assert d["requeue_count"] == 3
    restored = RequeueCandidate.from_dict(d)
    assert restored.account_id == candidate.account_id
    assert restored.requeue_count == candidate.requeue_count
    assert restored.last_error == "APITimeoutError"


# ── 2. Per-Session Serialization (AC-5) ──


@pytest.mark.asyncio
async def test_session_extraction_lock_serialization():
    manager = SessionExtractionLockManager()
    events = []

    async def task_1():
        async with manager.acquire("sess-1", "viking://user/default/sessions/sess-1", wait=True):
            events.append("task_1_start")
            await asyncio.sleep(0.05)
            events.append("task_1_end")

    async def task_2_wait():
        await asyncio.sleep(0.01)  # ensure task_1 holds lock
        async with manager.acquire("sess-1", "viking://user/default/sessions/sess-1", wait=True):
            events.append("task_2_start")
            events.append("task_2_end")

    async def task_2_nowait():
        await asyncio.sleep(0.01)  # ensure task_1 holds lock
        with pytest.raises(SessionExtractionBusyError):
            async with manager.acquire(
                "sess-1", "viking://user/default/sessions/sess-1", wait=False
            ):
                events.append("task_2_nowait_should_not_run")

    async def task_diff_session():
        await asyncio.sleep(0.01)
        async with manager.acquire(
            "sess-diff", "viking://user/default/sessions/sess-diff", wait=False
        ):
            events.append("task_diff_session_ran")

    # Run task 1, task 2 (wait), task 2 (nowait), and task diff session
    await asyncio.gather(task_1(), task_2_wait(), task_2_nowait(), task_diff_session())

    assert "task_1_start" in events
    assert "task_1_end" in events
    assert "task_diff_session_ran" in events
    # task 2 (wait) must start AFTER task 1 ends
    t1_end_idx = events.index("task_1_end")
    t2_start_idx = events.index("task_2_start")
    assert t2_start_idx > t1_end_idx
    assert "task_2_nowait_should_not_run" not in events


# ── 3. Resume Queued Commit Serialization and Deferral (AC-5) ──


@pytest.mark.asyncio
async def test_resume_queued_commit_busy_lock_returns_false(monkeypatch):
    session_uri = "viking://user/default/sessions/sess-busy"
    archive_uri = f"{session_uri}/history/archive_001"
    archived = Message(id="archived", role="user", parts=[TextPart("old")])
    vfs = _FakeVikingFS(
        {
            f"/local/default/sessions/sess-busy/messages.jsonl": "",
            f"/local/default/sessions/sess-busy/.meta.json": json.dumps({"session_id": "sess-busy"}),
            f"/local/default/sessions/sess-busy/history/archive_001/messages.jsonl": f"{archived.to_jsonl()}\n",
            f"/local/default/sessions/sess-busy/history/archive_001/.meta.json": json.dumps({}),
        }
    )
    session = Session(viking_fs=vfs, session_id="sess-busy", session_uri=session_uri)
    tracker = TaskTracker(_FakeTaskStore())
    set_task_tracker(tracker)
    monkeypatch.setattr(session, "_can_run_archive", AsyncMock(return_value=True))

    lock_mgr = get_session_extraction_lock_manager()
    msg = SessionCommitMsg(
        task_id="task-busy",
        session_id="sess-busy",
        session_uri=session_uri,
        archive_uri=archive_uri,
        user={"account_id": "default", "user_id": "default"},
    )

    try:
        # Pre-acquire the lock to simulate Trigger 1 running Phase 2
        async with lock_mgr.acquire("sess-busy", session_uri, wait=True):
            # Trigger 2 arrives with wait=False (default from SessionCommitProcessor)
            result = await session.resume_queued_commit(msg, wait=False)
            assert result is False  # Must return False so QueueFS re-enqueues without blocking
    finally:
        set_task_tracker(None)


# ── 4. Completed Step Skip and Resume (AC-4) ──


@pytest.mark.asyncio
async def test_resume_queued_commit_skips_completed_memory_steps(monkeypatch):
    session_uri = "viking://user/default/sessions/sess-resume"
    archive_uri = f"{session_uri}/history/archive_001"
    msg_1 = Message(id="msg-1", role="user", parts=[TextPart("step 1")])
    msg_2 = Message(id="msg-2", role="user", parts=[TextPart("step 2")])
    files = {
        f"/local/default/sessions/sess-resume/messages.jsonl": "",
        f"/local/default/sessions/sess-resume/.meta.json": json.dumps({"session_id": "sess-resume"}),
        f"/local/default/sessions/sess-resume/history/archive_001/messages.jsonl": f"{msg_1.to_jsonl()}\n{msg_2.to_jsonl()}\n",
        f"/local/default/sessions/sess-resume/history/archive_001/.meta.json": json.dumps({}),
        f"/local/default/sessions/sess-resume/history/archive_001/.failed.json": json.dumps(
            {
                "stage": "memory_extraction",
                "error": "APITimeoutError",
                "completed_memory_steps": {"long_term": ["msg-1"]},
                "requeue_count": 1,
            }
        ),
    }
    vfs = _FakeVikingFS(files)
    session = Session(viking_fs=vfs, session_id="sess-resume", session_uri=session_uri)
    tracker = TaskTracker(_FakeTaskStore())
    set_task_tracker(tracker)
    monkeypatch.setattr(session, "_can_run_archive", AsyncMock(return_value=True))

    extracted_messages = []

    async def mock_run_memory_extraction(**kwargs):
        # In _prepare_phase2_archive_messages, verify completed_memory_steps was merged
        extracted_messages.extend(kwargs.get("messages", []))

    monkeypatch.setattr(session, "_run_memory_extraction", mock_run_memory_extraction)

    msg = SessionCommitMsg(
        task_id="task-resume",
        session_id="sess-resume",
        session_uri=session_uri,
        archive_uri=archive_uri,
        user={"account_id": "default", "user_id": "default"},
    )

    try:
        res = await session.resume_queued_commit(msg, wait=False)
        assert res is True
        # Verify completed steps are merged from .failed.json
        messages, _, _, _, completed = await session._prepare_phase2_archive_messages(
            archive_uri, [msg_1, msg_2]
        )
        assert "long_term" in completed
        assert "msg-1" in completed["long_term"]
    finally:
        set_task_tracker(None)


# ── 5. Requeue Registration and Quarantine Valve (AC-4 & AC-5) ──


@pytest.mark.asyncio
async def test_write_failed_marker_registers_requeue_and_quarantine():
    vfs = _FakeVikingFS()
    session_uri = "viking://user/default/sessions/sess-fail"
    archive_uri = f"{session_uri}/history/archive_001"
    session = Session(viking_fs=vfs, session_id="sess-fail", session_uri=session_uri)

    # 1st failure: requeue_count becomes 1
    await session._write_failed_marker(archive_uri, stage="memory_extraction", error="timeout 1")
    failed_1 = json.loads(vfs.files[f"/local/default/sessions/sess-fail/history/archive_001/.failed.json"])
    assert failed_1["requeue_count"] == 1
    assert failed_1["quarantined"] is False
    assert "/local/default/_system/session_requeue/sess-fail_archive_001.json" in vfs.files

    # Advance to 9th failure
    failed_1["requeue_count"] = 9
    vfs.files[f"/local/default/sessions/sess-fail/history/archive_001/.failed.json"] = json.dumps(failed_1)

    # 10th failure: reaches DEFAULT_MAX_REQUEUE_ATTEMPTS -> triggers quarantine valve
    await session._write_failed_marker(archive_uri, stage="memory_extraction", error="timeout 10")
    failed_10 = json.loads(vfs.files[f"/local/default/sessions/sess-fail/history/archive_001/.failed.json"])
    assert failed_10["requeue_count"] == 10
    assert failed_10["quarantined"] is True
    assert "Exceeded 10 requeue attempts" in failed_10["quarantine_reason"]
    # Requeue file moved to quarantine directory
    assert "/local/default/_system/session_quarantine/sess-fail_archive_001.json" in vfs.files


@pytest.mark.asyncio
async def test_write_done_file_cleans_up_failed_and_requeue():
    vfs = _FakeVikingFS(
        {
            "/local/default/_system/session_requeue/sess-done_archive_001.json": "{}",
            "/local/default/sessions/sess-done/history/archive_001/.failed.json": "{}",
        }
    )
    session_uri = "viking://user/default/sessions/sess-done"
    archive_uri = f"{session_uri}/history/archive_001"
    session = Session(viking_fs=vfs, session_id="sess-done", session_uri=session_uri)

    await session._write_done_file(archive_uri, first_message_id="m1", last_message_id="m2")

    assert "/local/default/sessions/sess-done/history/archive_001/.done" in vfs.files
    assert "/local/default/sessions/sess-done/history/archive_001/.failed.json" not in vfs.files
    assert "/local/default/_system/session_requeue/sess-done_archive_001.json" not in vfs.files


# ── 6. Sweeper Operations & Restart Drill ──


@pytest.mark.asyncio
async def test_sweeper_sweep_once_re_enqueues_and_respects_backoff():
    files = {}
    vfs = _FakeVikingFS(files)
    queue_mgr = SimpleNamespace(enqueue=AsyncMock())
    session_service = SimpleNamespace()
    sweeper = SessionRequeueSweeper(
        session_service=session_service,
        queue_manager=queue_mgr,
        viking_fs=vfs,
        check_interval=1.0,
    )

    now = time.time()
    # Candidate 1: due now
    c1 = RequeueCandidate(
        account_id="default",
        user_id="default",
        session_id="s1",
        session_uri="viking://user/default/sessions/s1",
        archive_uri="viking://user/default/sessions/s1/history/archive_001",
        requeue_count=1,
        next_requeue_at=now - 5.0,  # due
    )
    # Candidate 2: not due yet (in future)
    c2 = RequeueCandidate(
        account_id="default",
        user_id="default",
        session_id="s2",
        session_uri="viking://user/default/sessions/s2",
        archive_uri="viking://user/default/sessions/s2/history/archive_001",
        requeue_count=1,
        next_requeue_at=now + 500.0,  # not due
    )
    files["/local/default/_system/session_requeue/s1_archive_001.json"] = json.dumps(c1.to_dict())
    files["/local/default/_system/session_requeue/s2_archive_001.json"] = json.dumps(c2.to_dict())
    files["/local/default/sessions/s1/history/archive_001/.meta.json"] = json.dumps({})
    files["/local/default/sessions/s2/history/archive_001/.meta.json"] = json.dumps({})

    re_enqueued = await sweeper.sweep_once()
    assert re_enqueued == 1
    queue_mgr.enqueue.assert_awaited_once()
    # Check updated backoff on s1
    updated_c1 = json.loads(files["/local/default/_system/session_requeue/s1_archive_001.json"])
    assert updated_c1["next_requeue_at"] > now


@pytest.mark.asyncio
async def test_sweeper_startup_recovery():
    files = {
        "/local/default/_system/session_requeue/surviving_001.json": json.dumps({"archive_uri": "a1"}),
        "/local/default/_system/session_quarantine/quarantined_001.json": json.dumps({"archive_uri": "q1"}),
    }
    vfs = _FakeVikingFS(files)
    queue_mgr = SimpleNamespace(enqueue=AsyncMock())
    session_service = SimpleNamespace()
    sweeper = SessionRequeueSweeper(
        session_service=session_service,
        queue_manager=queue_mgr,
        viking_fs=vfs,
    )

    # Calling start should execute _recover_on_startup without error
    await sweeper.start()
    await sweeper.stop()

    quarantined = await sweeper.get_quarantined_sessions("default")
    assert len(quarantined) == 1
    assert quarantined[0]["archive_uri"] == "q1"

    pending = await sweeper.get_pending_requeues("default")
    assert len(pending) == 1
    assert pending[0]["archive_uri"] == "a1"
