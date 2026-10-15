# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Query shape classifier for retrieval metrics segmentation (P-20261015-02).

Classifies retrieval queries into three distinct shapes:
- ``id_like``: queries dominated by identifier tokens (task/proposal/broadcast IDs,
  alert IDs, commit SHAs, UUID fragments).
- ``natural_language``: queries composed of natural language prose or semantic concepts.
- ``mixed``: queries containing both identifier tokens and natural language terms.
"""

from __future__ import annotations

import re
from typing import Optional

ID_LIKE = "id_like"
NATURAL_LANGUAGE = "natural_language"
MIXED = "mixed"

# Unified regex matching common system and code identifiers:
# - Alert IDs: ALT-1791979752-1, ALT-TEST-ROUNDTRIP-2
# - Task/Proposal/Broadcast/Error/Followup IDs: T-20261014-01, P-20261015-02, B-20260918-01, ERR-20261013-01, FU-20261012-01
# - Defect/Decision IDs: D4-05, D-01, B-2
# - Golden queries / questions: Q29, Q01
# - UUIDs: e4c616a4-1bc0-4226-a94b-7440dec8f358
# - Git commit SHAs: 7-40 hex chars containing at least one digit and one letter a-f
_ID_PATTERN = re.compile(
    r"\b(?:"
    r"[A-Za-z]{1,4}-\d+(?:-[A-Za-z0-9_-]+)*"
    r"|[A-Za-z]\d+-[A-Za-z0-9_-]+"
    r"|Q\d+"
    r"|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    r"|(?=[0-9a-fA-F]*[0-9])(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{7,40}"
    r")\b"
)


def classify_query_shape(query: Optional[str]) -> str:
    """Classify a query into id_like, natural_language, or mixed.

    Args:
        query: Raw query text from the user or agent.

    Returns:
        One of ``id_like``, ``natural_language``, or ``mixed``.
    """
    if not query or not query.strip():
        return NATURAL_LANGUAGE

    text = query.strip()

    # Find all identifier tokens matching the ID pattern.
    found_ids = _ID_PATTERN.findall(text)

    if not found_ids:
        return NATURAL_LANGUAGE

    # Remove the found identifiers from the text.
    remaining = _ID_PATTERN.sub("", text)

    # Check if there are remaining meaningful characters (alphanumeric, CJK).
    # Discard common punctuation and whitespace separators.
    remaining_clean = re.sub(r"[\s\W_]+", "", remaining)

    if not remaining_clean:
        # The query was completely dominated by identifier tokens and separators.
        return ID_LIKE

    return MIXED


def extract_caller_identity(ctx: object = None) -> tuple[str, str]:
    """Extract (caller_account, caller_user) from RequestContext or request headers.

    Falls back to ("default", "default") when unauthenticated/unspecified.
    """
    account: Optional[str] = None
    user: Optional[str] = None
    if ctx is not None:
        account = getattr(ctx, "account_id", None)
        user_obj = getattr(ctx, "user", None)
        if user_obj is not None:
            if not account:
                account = getattr(user_obj, "account_id", None)
            user = getattr(user_obj, "user_id", None)
        if not user:
            user = getattr(ctx, "user_id", None)

    # Inspect call stack for explicit HTTP request header overrides (e.g. x-openviking-account / x-openviking-user)
    try:
        import sys

        frame = sys._getframe(1)
        for _ in range(8):
            if frame is None:
                break
            for var_name in ("http_request", "request"):
                req = frame.f_locals.get(var_name)
                if req is not None and hasattr(req, "headers"):
                    hdr_acct = req.headers.get("x-openviking-account") or req.headers.get("X-OpenViking-Account")
                    hdr_user = req.headers.get("x-openviking-user") or req.headers.get("X-OpenViking-User")
                    if hdr_acct and str(hdr_acct).strip():
                        account = str(hdr_acct).strip()
                    if hdr_user and str(hdr_user).strip():
                        user = str(hdr_user).strip()
                    break
            frame = frame.f_back
    except Exception:
        pass

    return str(account or "default"), str(user or "default")

