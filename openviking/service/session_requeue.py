# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Persistent requeue sweeper and quarantine valve for failed session extractions."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from openviking.server.error_mapping import is_not_found_error
from openviking.server.identity import RequestContext, Role
from openviking.session.session_lock import get_session_extraction_lock_manager
from openviking.storage.queuefs.session_commit_msg import SessionCommitMsg
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.logger import get_logger

if TYPE_CHECKING:
    from openviking.service.session_service import SessionService
    from openviking.storage.queuefs import QueueManager
    from openviking.storage.viking_fs import VikingFS

logger = get_logger(__name__)

AGFS_REQUEUE_DIR_TEMPLATE = "/local/{account_id}/_system/session_requeue"
AGFS_QUARANTINE_DIR_TEMPLATE = "/local/{account_id}/_system/session_quarantine"
DEFAULT_CHECK_INTERVAL_SECONDS = 15.0
DEFAULT_MAX_REQUEUE_ATTEMPTS = 10
DEFAULT_QUARANTINE_WINDOW_HOURS = 24.0
DEFAULT_BASE_BACKOFF_SECONDS = 10.0
DEFAULT_MAX_BACKOFF_SECONDS = 3600.0


def calculate_requeue_backoff(
    attempt: int,
    base_delay: float = DEFAULT_BASE_BACKOFF_SECONDS,
    max_delay: float = DEFAULT_MAX_BACKOFF_SECONDS,
) -> float:
    """Calculate exponential backoff delay for requeue attempts."""
    exponent = min(max(0, attempt - 1), 8)
    return min(max_delay, base_delay * (2**exponent))


@dataclass
class RequeueCandidate:
    account_id: str
    user_id: str
    session_id: str
    session_uri: str
    archive_uri: str
    requeue_count: int
    next_requeue_at: float
    quarantined: bool = False
    quarantine_reason: Optional[str] = None
    last_error: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequeueCandidate":
        return cls(
            account_id=str(data.get("account_id") or "default"),
            user_id=str(data.get("user_id") or "default"),
            session_id=str(data.get("session_id") or ""),
            session_uri=str(data.get("session_uri") or ""),
            archive_uri=str(data.get("archive_uri") or ""),
            requeue_count=int(data.get("requeue_count") or 0),
            next_requeue_at=float(data.get("next_requeue_at") or 0.0),
            quarantined=bool(data.get("quarantined", False)),
            quarantine_reason=data.get("quarantine_reason"),
            last_error=data.get("last_error") or data.get("error"),
            created_at=data.get("created_at"),
        )


class SessionRequeueSweeper:
    """Background sweeper that discovers failed session extractions and re-fires them with backoff."""

    def __init__(
        self,
        session_service: "SessionService",
        queue_manager: "QueueManager",
        viking_fs: "VikingFS",
        *,
        check_interval: float = DEFAULT_CHECK_INTERVAL_SECONDS,
        max_requeue_attempts: int = DEFAULT_MAX_REQUEUE_ATTEMPTS,
        quarantine_window_hours: float = DEFAULT_QUARANTINE_WINDOW_HOURS,
        base_delay_seconds: float = DEFAULT_BASE_BACKOFF_SECONDS,
        max_delay_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS,
    ) -> None:
        self._session_service = session_service
        self._queue_manager = queue_manager
        self._viking_fs = viking_fs
        self._check_interval = check_interval
        self._max_requeue_attempts = max_requeue_attempts
        self._quarantine_window_hours = quarantine_window_hours
        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock_manager = get_session_extraction_lock_manager()

    async def start(self) -> None:
        """Start the background requeue sweeper."""
        if self._running:
            return
        self._running = True
        logger.info(
            "SessionRequeueSweeper starting (interval=%.1fs, max_attempts=%d, quarantine_window=%.1fh)",
            self._check_interval,
            self._max_requeue_attempts,
            self._quarantine_window_hours,
        )
        # Perform startup recovery scan
        try:
            await self._recover_on_startup()
        except Exception as exc:
            logger.warning("SessionRequeueSweeper startup recovery failed: %s", exc, exc_info=True)

        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        """Stop the background sweeper."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("SessionRequeueSweeper stopped")

    async def _run_loop(self) -> None:
        """Main periodic sweeping loop."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                await self.sweep_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("SessionRequeueSweeper loop error: %s", exc, exc_info=True)

    async def sweep_once(self) -> int:
        """Perform one scan and re-enqueue pass. Returns count of re-enqueued jobs."""
        if not self._viking_fs or not hasattr(self._viking_fs, "_async_agfs"):
            return 0

        agfs = self._viking_fs._async_agfs
        re_enqueued_count = 0
        now = time.time()

        try:
            account_entries = await agfs.ls("/local")
        except Exception as exc:
            if not is_not_found_error(exc):
                logger.warning("SessionRequeueSweeper failed to list accounts: %s", exc)
            return 0

        for account_entry in account_entries:
            account_id = str(account_entry.get("name") or "").strip()
            if not account_id or account_id == "_system" or not account_entry.get("isDir", False):
                continue

            requeue_dir = AGFS_REQUEUE_DIR_TEMPLATE.format(account_id=account_id)
            try:
                requeue_files = await agfs.ls(requeue_dir)
            except Exception as exc:
                if not is_not_found_error(exc):
                    logger.debug("No requeue directory for account %s (%s)", account_id, exc)
                continue

            for file_entry in requeue_files:
                file_name = str(file_entry.get("name") or "").strip()
                if not file_name.endswith(".json") or file_entry.get("isDir", False):
                    continue

                item_path = f"{requeue_dir}/{file_name}"
                try:
                    content_raw = await agfs.read(item_path)
                    content_str = (
                        content_raw.decode("utf-8")
                        if isinstance(content_raw, bytes)
                        else str(content_raw)
                    )
                    data = json.loads(content_str)
                    candidate = RequeueCandidate.from_dict(data)
                except Exception as exc:
                    logger.warning(
                        "Invalid requeue item at %s: %s; cleaning up", item_path, exc
                    )
                    try:
                        await agfs.rm(item_path, force=True)
                    except Exception:
                        pass
                    continue

                # Check if archive already completed
                archive_uri = candidate.archive_uri
                done_marker_uri = f"{archive_uri}/.done"
                if await self._viking_fs.exists(done_marker_uri):
                    logger.info(
                        "[REQUEUE_SWEEPER] Archive %s already completed (.done found); removing requeue item",
                        archive_uri,
                    )
                    try:
                        await agfs.rm(item_path, force=True)
                    except Exception:
                        pass
                    continue

                # Check if quarantined
                if candidate.quarantined or candidate.requeue_count >= self._max_requeue_attempts:
                    logger.error(
                        "[QUARANTINE_VALVE] Archive %s exceeded max requeue attempts (%d/%d); moving to quarantine",
                        archive_uri,
                        candidate.requeue_count,
                        self._max_requeue_attempts,
                    )
                    quarantine_dir = AGFS_QUARANTINE_DIR_TEMPLATE.format(account_id=account_id)
                    candidate.quarantined = True
                    candidate.quarantine_reason = (
                        candidate.quarantine_reason
                        or f"Exceeded {self._max_requeue_attempts} requeue attempts ({candidate.last_error})"
                    )
                    try:
                        await agfs.ensure_parent_dirs(f"{quarantine_dir}/{file_name}")
                        await agfs.write(
                            f"{quarantine_dir}/{file_name}",
                            json.dumps(candidate.to_dict(), ensure_ascii=False).encode("utf-8"),
                        )
                        await agfs.rm(item_path, force=True)
                    except Exception as q_exc:
                        logger.warning("Failed to move %s to quarantine: %s", item_path, q_exc)
                    continue

                # Check if due for requeue
                if candidate.next_requeue_at > now:
                    continue

                # Check if session extraction is currently locked
                if self._lock_manager.is_locked(candidate.session_uri or candidate.session_id):
                    logger.debug(
                        "[REQUEUE_SWEEPER] Session %s extraction is currently busy; deferring requeue",
                        candidate.session_id,
                    )
                    continue

                # Re-fire: read queue message from archive meta and enqueue
                try:
                    re_enqueued = await self._re_enqueue_archive(candidate)
                    if re_enqueued:
                        re_enqueued_count += 1
                        # Update next_requeue_at for next cycle backoff in case it fails again
                        next_delay = calculate_requeue_backoff(
                            candidate.requeue_count + 1,
                            base_delay=self._base_delay_seconds,
                            max_delay=self._max_delay_seconds,
                        )
                        candidate.next_requeue_at = now + next_delay
                        await agfs.write(
                            item_path,
                            json.dumps(candidate.to_dict(), ensure_ascii=False).encode("utf-8"),
                        )
                except Exception as exc:
                    logger.error(
                        "[REQUEUE_SWEEPER] Failed to re-enqueue archive %s: %s",
                        archive_uri,
                        exc,
                        exc_info=True,
                    )

        return re_enqueued_count

    async def _re_enqueue_archive(self, candidate: RequeueCandidate) -> bool:
        """Construct SessionCommitMsg and enqueue to QueueManager.SESSION_COMMIT."""
        ctx = RequestContext(
            user=UserIdentifier(account_id=candidate.account_id, user_id=candidate.user_id),
            role=Role.USER,
        )
        archive_uri = candidate.archive_uri
        meta_content = ""
        try:
            meta_content = await self._viking_fs.read_file(f"{archive_uri}/.meta.json", ctx=ctx)
        except Exception as exc:
            logger.warning("Failed to read archive meta from %s: %s", archive_uri, exc)
            return False

        meta_dict = json.loads(meta_content or "{}")
        phase1 = meta_dict.get("phase1", {})
        queue_msg_dict = phase1.get("queue_message") if isinstance(phase1, dict) else None

        if isinstance(queue_msg_dict, dict):
            msg = SessionCommitMsg.from_dict(queue_msg_dict)
            msg.task_id = (
                f"requeue-{candidate.session_id}-{candidate.requeue_count}-{int(time.time())}"
            )
        else:
            # Reconstruct SessionCommitMsg from archive metadata
            msg = SessionCommitMsg(
                task_id=f"requeue-{candidate.session_id}-{candidate.requeue_count}-{int(time.time())}",
                session_id=candidate.session_id,
                session_uri=candidate.session_uri,
                archive_uri=archive_uri,
                user={"account_id": candidate.account_id, "user_id": candidate.user_id},
                memory_policy=meta_dict.get("memory_policy") or {},
                auto_commit_policy=meta_dict.get("auto_commit_policy") or {},
            )

        from openviking.storage.queuefs import QueueManager

        await self._queue_manager.enqueue(
            QueueManager.SESSION_COMMIT,
            msg.to_dict(),
        )
        logger.info(
            "[REQUEUE_SWEEPER] Re-enqueued failed archive %s for session %s (attempt %d)",
            archive_uri,
            candidate.session_id,
            candidate.requeue_count,
        )
        return True

    async def _recover_on_startup(self) -> None:
        """Scan and log surviving requeue items upon server startup."""
        if not self._viking_fs or not hasattr(self._viking_fs, "_async_agfs"):
            return
        agfs = self._viking_fs._async_agfs
        try:
            account_entries = await agfs.ls("/local")
        except Exception:
            return

        total_surviving = 0
        total_quarantined = 0
        for account_entry in account_entries:
            account_id = str(account_entry.get("name") or "").strip()
            if not account_id or account_id == "_system" or not account_entry.get("isDir", False):
                continue
            requeue_dir = AGFS_REQUEUE_DIR_TEMPLATE.format(account_id=account_id)
            try:
                entries = await agfs.ls(requeue_dir)
                total_surviving += len([e for e in entries if not e.get("isDir", False)])
            except Exception:
                pass
            quarantine_dir = AGFS_QUARANTINE_DIR_TEMPLATE.format(account_id=account_id)
            try:
                q_entries = await agfs.ls(quarantine_dir)
                total_quarantined += len([e for e in q_entries if not e.get("isDir", False)])
            except Exception:
                pass

        logger.info(
            "[REQUEUE_SWEEPER] Startup recovery complete: %d surviving requeue(s), %d quarantined item(s)",
            total_surviving,
            total_quarantined,
        )

    async def get_quarantined_sessions(
        self, account_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all quarantined sessions for observability / admin review."""
        if not self._viking_fs or not hasattr(self._viking_fs, "_async_agfs"):
            return []
        agfs = self._viking_fs._async_agfs
        quarantined = []
        accounts = [account_id] if account_id else []
        if not accounts:
            try:
                entries = await agfs.ls("/local")
                accounts = [
                    str(e.get("name")).strip()
                    for e in entries
                    if e.get("name") and e.get("name") != "_system" and e.get("isDir")
                ]
            except Exception:
                accounts = ["default"]

        for acct in accounts:
            q_dir = AGFS_QUARANTINE_DIR_TEMPLATE.format(account_id=acct)
            try:
                files = await agfs.ls(q_dir)
                for f in files:
                    if f.get("isDir"):
                        continue
                    content = await agfs.read(f"{q_dir}/{f.get('name')}")
                    quarantined.append(json.loads(content.decode("utf-8")))
            except Exception:
                continue
        return quarantined

    async def get_pending_requeues(
        self, account_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all pending requeue candidates."""
        if not self._viking_fs or not hasattr(self._viking_fs, "_async_agfs"):
            return []
        agfs = self._viking_fs._async_agfs
        pending = []
        accounts = [account_id] if account_id else []
        if not accounts:
            try:
                entries = await agfs.ls("/local")
                accounts = [
                    str(e.get("name")).strip()
                    for e in entries
                    if e.get("name") and e.get("name") != "_system" and e.get("isDir")
                ]
            except Exception:
                accounts = ["default"]

        for acct in accounts:
            r_dir = AGFS_REQUEUE_DIR_TEMPLATE.format(account_id=acct)
            try:
                files = await agfs.ls(r_dir)
                for f in files:
                    if f.get("isDir"):
                        continue
                    content = await agfs.read(f"{r_dir}/{f.get('name')}")
                    pending.append(json.loads(content.decode("utf-8")))
            except Exception:
                continue
        return pending


async def register_failed_requeue(
    viking_fs: Any,
    candidate: RequeueCandidate,
) -> None:
    """Register or update a failed archive in the persistent AGFS requeue directory."""
    if not viking_fs or not hasattr(viking_fs, "_async_agfs"):
        return
    agfs = viking_fs._async_agfs
    archive_id = candidate.archive_uri.rstrip("/").split("/")[-1]
    file_name = f"{candidate.session_id}_{archive_id}.json"
    requeue_dir = AGFS_REQUEUE_DIR_TEMPLATE.format(account_id=candidate.account_id)
    quarantine_dir = AGFS_QUARANTINE_DIR_TEMPLATE.format(account_id=candidate.account_id)
    payload_bytes = json.dumps(candidate.to_dict(), ensure_ascii=False).encode("utf-8")

    if candidate.quarantined:
        try:
            await agfs.ensure_parent_dirs(f"{quarantine_dir}/{file_name}")
            await agfs.write(f"{quarantine_dir}/{file_name}", payload_bytes)
            await agfs.rm(f"{requeue_dir}/{file_name}", force=True)
        except Exception as exc:
            logger.warning("Failed to register quarantined session %s: %s", candidate.session_id, exc)
    else:
        try:
            await agfs.ensure_parent_dirs(f"{requeue_dir}/{file_name}")
            await agfs.write(f"{requeue_dir}/{file_name}", payload_bytes)
        except Exception as exc:
            logger.warning("Failed to register requeue candidate %s: %s", candidate.session_id, exc)


async def clear_requeue(
    viking_fs: Any,
    account_id: str,
    session_id: str,
    archive_uri: str,
) -> None:
    """Clear requeue and quarantine entries for an archive that has completed successfully."""
    if not viking_fs or not hasattr(viking_fs, "_async_agfs"):
        return
    agfs = viking_fs._async_agfs
    archive_id = archive_uri.rstrip("/").split("/")[-1]
    file_name = f"{session_id}_{archive_id}.json"
    requeue_dir = AGFS_REQUEUE_DIR_TEMPLATE.format(account_id=account_id)
    quarantine_dir = AGFS_QUARANTINE_DIR_TEMPLATE.format(account_id=account_id)
    try:
        await agfs.rm(f"{requeue_dir}/{file_name}", force=True)
    except Exception:
        pass
    try:
        await agfs.rm(f"{quarantine_dir}/{file_name}", force=True)
    except Exception:
        pass

