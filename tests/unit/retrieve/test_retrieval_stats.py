# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for retrieval statistics and observer."""

from openviking.retrieve.retrieval_stats import RetrievalStats, RetrievalStatsCollector
from openviking.storage.observers.retrieval_observer import RetrievalObserver


class TestRetrievalStats:
    def test_defaults(self):
        stats = RetrievalStats()
        assert stats.total_queries == 0
        assert stats.avg_results_per_query == 0.0
        assert stats.zero_result_rate == 0.0
        assert stats.avg_score == 0.0
        assert stats.avg_latency_ms == 0.0

    def test_to_dict_empty(self):
        d = RetrievalStats().to_dict()
        assert d["total_queries"] == 0
        assert d["max_score"] == 0.0
        assert d["min_score"] == 0.0


class TestRetrievalStatsCollector:
    def test_record_single_query(self):
        collector = RetrievalStatsCollector()
        collector.record_query(
            context_type="memory",
            result_count=3,
            scores=[0.9, 0.7, 0.5],
            latency_ms=42.0,
        )
        stats = collector.snapshot()
        assert stats.total_queries == 1
        assert stats.total_results == 3
        assert stats.zero_result_queries == 0
        assert stats.max_score == 0.9
        assert stats.min_score == 0.5
        assert stats.queries_by_type == {"memory": 1}
        assert stats.avg_latency_ms == 42.0

    def test_record_empty_query_as_success(self):
        collector = RetrievalStatsCollector()
        collector.record_query(
            context_type="resource",
            result_count=0,
            scores=[],
            latency_ms=10.0,
        )
        stats = collector.snapshot()
        assert stats.total_queries == 1
        assert stats.total_results == 0
        assert stats.zero_result_queries == 1
        assert stats.zero_result_rate == 1.0

    def test_record_multiple_queries(self):
        collector = RetrievalStatsCollector()
        collector.record_query("memory", 2, [0.8, 0.6], latency_ms=30.0)
        collector.record_query("resource", 1, [0.5], latency_ms=20.0)
        collector.record_query("memory", 0, [], latency_ms=5.0)

        stats = collector.snapshot()
        assert stats.total_queries == 3
        assert stats.total_results == 3
        assert stats.zero_result_queries == 1
        assert stats.queries_by_type == {"memory": 2, "resource": 1}
        assert stats.avg_latency_ms == (30 + 20 + 5) / 3

    def test_rerank_tracking(self):
        collector = RetrievalStatsCollector()
        collector.record_query("memory", 1, [0.9], rerank_used=True)
        collector.record_query("memory", 1, [0.7], rerank_fallback=True)

        stats = collector.snapshot()
        assert stats.rerank_used == 1
        assert stats.rerank_fallback == 1

    def test_max_latency(self):
        collector = RetrievalStatsCollector()
        collector.record_query("memory", 1, [0.5], latency_ms=10.0)
        collector.record_query("memory", 1, [0.5], latency_ms=100.0)
        collector.record_query("memory", 1, [0.5], latency_ms=50.0)

        stats = collector.snapshot()
        assert stats.max_latency_ms == 100.0

    def test_reset(self):
        collector = RetrievalStatsCollector()
        collector.record_query("memory", 3, [0.9, 0.7, 0.5])
        collector.reset()
        stats = collector.snapshot()
        assert stats.total_queries == 0
        assert stats.total_results == 0

    def test_snapshot_is_copy(self):
        collector = RetrievalStatsCollector()
        collector.record_query("memory", 1, [0.9])
        snap = collector.snapshot()
        collector.record_query("memory", 1, [0.8])
        assert snap.total_queries == 1

    def test_to_dict(self):
        collector = RetrievalStatsCollector()
        collector.record_query("memory", 2, [0.9, 0.6], latency_ms=25.0)
        d = collector.snapshot().to_dict()
        assert d["total_queries"] == 1
        assert d["total_results"] == 2
        assert d["avg_results_per_query"] == 2.0
        assert d["max_score"] == 0.9
        assert d["min_score"] == 0.6
        assert d["avg_latency_ms"] == 25.0

    def test_record_per_lane_latency(self):
        collector = RetrievalStatsCollector()
        collector.record_query(
            context_type="memory",
            result_count=2,
            scores=[0.9, 0.8],
            latency_ms=150.0,
            rerank_used=True,
            lane="light",
            rerank_latency_ms=120.0,
        )
        collector.record_query(
            context_type="memory",
            result_count=3,
            scores=[0.95],
            latency_ms=800.0,
            rerank_used=True,
            lane="heavy",
            rerank_latency_ms=750.0,
        )
        collector.record_query(
            context_type="memory",
            result_count=1,
            scores=[0.85],
            latency_ms=160.0,
            rerank_used=True,
            lane="light",
            rerank_latency_ms=140.0,
        )

        snap = collector.snapshot()
        assert snap.rerank_latency_by_lane == {"light": 260.0, "heavy": 750.0}
        assert snap.avg_rerank_latency_by_lane == {"light": 130.0, "heavy": 750.0}

        d = snap.to_dict()
        assert d["rerank_by_lane"] == {"light": 2, "heavy": 1}
        assert "avg_rerank_latency_by_lane" in d
        assert d["avg_rerank_latency_by_lane"] == {"light": 130.0, "heavy": 750.0}


class TestRetrievalObserver:
    def _setup_collector(self):
        """Replace the global collector with a fresh one for testing."""
        import openviking.retrieve.retrieval_stats as mod

        collector = RetrievalStatsCollector()
        mod._collector = collector
        return collector

    def test_healthy_when_no_queries(self):
        self._setup_collector()
        observer = RetrievalObserver()
        assert observer.is_healthy() is True
        assert observer.has_errors() is False

    def test_healthy_with_good_results(self):
        collector = self._setup_collector()
        for _ in range(10):
            collector.record_query("memory", 3, [0.9, 0.7, 0.5])
        observer = RetrievalObserver()
        assert observer.is_healthy() is True
        assert observer.has_errors() is False

    def test_healthy_with_only_empty_results(self):
        collector = self._setup_collector()
        for _ in range(10):
            collector.record_query("memory", 0, [])
        observer = RetrievalObserver()
        assert observer.is_healthy() is True
        assert observer.has_errors() is False
        assert "Zero-Result Rate" in observer.get_status_table()

    def test_status_table_no_data(self):
        self._setup_collector()
        observer = RetrievalObserver()
        table = observer.get_status_table()
        assert "No retrieval queries recorded" in table

    def test_status_table_with_data(self):
        collector = self._setup_collector()
        collector.record_query("memory", 2, [0.9, 0.7], latency_ms=30.0)
        collector.record_query("resource", 1, [0.5], latency_ms=20.0)
        observer = RetrievalObserver()
        table = observer.get_status_table()
        assert "Total Queries" in table
        assert "Zero-Result Rate" in table
        assert "memory" in table
        assert "resource" in table

    def test_str(self):
        self._setup_collector()
        observer = RetrievalObserver()
        assert str(observer) == observer.get_status_table()

    def test_record_query_shape_and_account_tracking(self):
        collector = self._setup_collector()
        collector.record_query(
            "search",
            result_count=10,
            scores=[0.8],
            caller_account="account_a",
            caller_user="user_1",
            query_shape="natural_language",
        )
        collector.record_query(
            "search",
            result_count=0,
            scores=[],
            caller_account="account_a",
            caller_user="user_1",
            query_shape="id_like",
        )
        collector.record_query(
            "search",
            result_count=5,
            scores=[0.6],
            caller_account="account_b",
            caller_user="user_2",
            query_shape="mixed",
        )

        snap = collector.snapshot()
        assert snap.queries_by_shape == {
            "natural_language": 1,
            "id_like": 1,
            "mixed": 1,
        }
        assert snap.zero_result_by_shape == {"id_like": 1}
        assert snap.queries_by_account == {"account_a": 2, "account_b": 1}
        assert snap.zero_result_by_account == {"account_a": 1}
        assert snap.queries_by_account_shape == {
            "account_a/natural_language": 1,
            "account_a/id_like": 1,
            "account_b/mixed": 1,
        }
        assert snap.zero_result_by_account_shape == {"account_a/id_like": 1}

        d = snap.to_dict()
        assert "queries_by_shape" in d
        assert "zero_result_by_shape" in d
        assert "queries_by_account_shape" in d
        assert "zero_result_by_account_shape" in d
        assert d["zero_result_by_account_shape"]["account_a/id_like"] == 1

    def test_observer_table_with_account_shape_breakdown(self):
        collector = self._setup_collector()
        collector.record_query(
            "search",
            result_count=0,
            scores=[],
            caller_account="acct_x",
            caller_user="user_x",
            query_shape="id_like",
        )
        collector.record_query(
            "search",
            result_count=2,
            scores=[0.9],
            caller_account="acct_x",
            caller_user="user_x",
            query_shape="natural_language",
        )
        observer = RetrievalObserver()
        table = observer.get_status_table()
        assert "Account" in table
        assert "Shape" in table
        assert "acct_x" in table
        assert "id_like" in table
        assert "natural_language" in table
        assert "100.0%" in table
        assert "0.0%" in table
