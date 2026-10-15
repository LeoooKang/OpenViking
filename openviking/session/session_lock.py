# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Per-session extraction lock manager for serializing Phase 2 memory extractions."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Optional

from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


class SessionExtractionBusyError(Exception):
    """Raised when an extraction lock for a session is already held and wait=False."""

    pass


class SessionExtractionLockManager:
    """Manages in-process and distributed filesystem locks per session."""

    def __init__(self) -> None:
        self._in_process_locks: Dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()

    async def _get_in_process_lock(self, key: str) -> asyncio.Lock:
        async with self._registry_lock:
            if key not in self._in_process_locks:
                self._in_process_locks[key] = asyncio.Lock()
            return self._in_process_locks[key]

    def is_locked(self, key: str) -> bool:
        """Check if the in-process lock is currently acquired."""
        lock = self._in_process_locks.get(key)
        return lock.locked() if lock is not None else False

    @asynccontextmanager
    async def acquire(
        self,
        session_id: str,
        session_uri: str,
        *,
        session_path: Optional[str] = None,
        viking_fs: Optional[Any] = None,
        wait: bool = True,
        timeout_secs: float = 3600.0,
    ) -> AsyncIterator[None]:
        """Acquire a per-session extraction lock.

        Serializes extractions for the same session both in-process (via asyncio.Lock)
        and across processes (via AGFS pathlock on f"{session_path}/.phase2_lock").
        """
        key = (session_uri or session_id).rstrip("/")
        in_proc_lock = await self._get_in_process_lock(key)

        if not wait:
            if in_proc_lock.locked():
                raise SessionExtractionBusyError(
                    f"Session {session_id} extraction already in progress (in-process lock held)"
                )
            try:
                await asyncio.wait_for(in_proc_lock.acquire(), timeout=0.001)
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise SessionExtractionBusyError(
                    f"Session {session_id} extraction already in progress (in-process race)"
                ) from exc
        else:
            if in_proc_lock.locked():
                logger.info(
                    "Session %s waiting for in-process extraction lock (%s)...", session_id, key
                )
            await in_proc_lock.acquire()
            logger.info("Session %s acquired in-process extraction lock (%s).", session_id, key)

        lease = None
        try:
            if viking_fs and session_path and hasattr(viking_fs, "_async_agfs"):
                pathlock_target = f"{session_path}/.phase2_lock"
                if not wait:
                    try:
                        lease = await viking_fs._async_agfs.pathlock_acquire_exact(
                            pathlock_target, timeout_secs=0.0
                        )
                    except Exception as exc:
                        raise SessionExtractionBusyError(
                            f"Session {session_id} extraction locked across processes: {exc}"
                        ) from exc
                else:
                    lease = await viking_fs._async_agfs.pathlock_acquire_exact(
                        pathlock_target, timeout_secs=timeout_secs
                    )
        except Exception:
            if in_proc_lock.locked():
                in_proc_lock.release()
            raise

        try:
            yield
        finally:
            if lease and viking_fs and hasattr(viking_fs, "_async_agfs"):
                try:
                    await viking_fs._async_agfs.pathlock_release(lease)
                except Exception as e:
                    logger.warning(
                        "Failed to release Phase 2 pathlock for %s: %s", session_id, e
                    )
            if in_proc_lock.locked():
                in_proc_lock.release()


_GLOBAL_LOCK_MANAGER = SessionExtractionLockManager()


def get_session_extraction_lock_manager() -> SessionExtractionLockManager:
    """Get the global SessionExtractionLockManager singleton."""
    return _GLOBAL_LOCK_MANAGER
