# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for query shape classification (P-20261015-02)."""

import pytest
from openviking.retrieve.query_shape import (
    ID_LIKE,
    MIXED,
    NATURAL_LANGUAGE,
    classify_query_shape,
)


class TestQueryShapeClassification:
    """Test suite for classify_query_shape."""

    @pytest.mark.parametrize(
        "query",
        [
            "watchdog",
            "watchdog slow channel",
            "how does three agent collaboration work",
            "find watchdog",
            "semantic search relevance and ranking",
            "who is the arbiter of this project",
            "three-agent collaboration system",
            "retrieval zero result rate",
        ],
    )
    def test_natural_language_queries(self, query: str):
        assert classify_query_shape(query) == NATURAL_LANGUAGE

    @pytest.mark.parametrize(
        "query",
        [
            # Alert ID (AC-1 probe target)
            "ALT-1791979752-1",
            # Task ID
            "T-20261014-01",
            # Proposal ID
            "P-20261015-02",
            # Broadcast ID
            "B-20260918-01",
            # Error ID
            "ERR-20261013-01",
            # Decision / defect ID
            "D4-05",
            "D-01",
            # Golden query ID
            "Q29",
            # Git commit SHAs
            "3ee036e",
            "aa9bcf9e08f6",
            # UUID
            "e4c616a4-1bc0-4226-a94b-7440dec8f358",
            # Quoted identifier
            '"ALT-1791979752-1"',
            "`T-20261014-01`",
            # Multiple identifier tokens
            "ALT-1791979752-1 T-20261014-01",
            "P-20261015-02, B-20260918-01",
        ],
    )
    def test_id_like_queries(self, query: str):
        assert classify_query_shape(query) == ID_LIKE

    @pytest.mark.parametrize(
        "query",
        [
            "what happened to ALT-1791979752-1 in the watchdog",
            "status of task T-20261014-01",
            "ALT-1791979752-1: 0 results",
            "investigate error ERR-20261013-01",
            "check commit 3ee036e",
            "Q29 separation threshold ruling",
        ],
    )
    def test_mixed_queries(self, query: str):
        assert classify_query_shape(query) == MIXED

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("三智能体协作系统", NATURAL_LANGUAGE),
            ("知识库检索质量评估", NATURAL_LANGUAGE),
            ("ALT-1791979752-1 告警分析", MIXED),
            ("任务 T-20261014-01 执行状态", MIXED),
            ("提案 P-20261015-02 验收标准", MIXED),
        ],
    )
    def test_cjk_queries(self, query: str, expected: str):
        assert classify_query_shape(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "",
            "   ",
            None,
        ],
    )
    def test_empty_and_none_queries(self, query):
        assert classify_query_shape(query) == NATURAL_LANGUAGE


class TestExtractCallerIdentity:
    def test_none_context_falls_back_to_default(self):
        from openviking.retrieve.query_shape import extract_caller_identity

        acct, user = extract_caller_identity(None)
        assert acct == "default"
        assert user == "default"

    def test_context_with_account_and_user(self):
        from types import SimpleNamespace
        from openviking.retrieve.query_shape import extract_caller_identity

        ctx = SimpleNamespace(
            account_id="custom_account",
            user=SimpleNamespace(account_id="custom_account", user_id="custom_user"),
        )
        acct, user = extract_caller_identity(ctx)
        assert acct == "custom_account"
        assert user == "custom_user"
