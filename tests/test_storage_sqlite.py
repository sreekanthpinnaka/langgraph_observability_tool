from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sqlite3
import tempfile
import pytest

from langgraph_observe.core.models import (
    HTTPMetadata,
    Span,
    SpanStatus,
    SpanType,
    Trace,
)
from langgraph_observe.server.storage.sqlite import SQLiteStorage


def _create_sample_trace(trace_id: str, name: str, status: SpanStatus = SpanStatus.COMPLETED) -> Trace:
    t = Trace(
        id=trace_id,
        name=name,
        http=HTTPMetadata(method="POST", path=f"/test/{name}", url=f"http://localhost/test/{name}"),
    )
    s = Span(id=f"span_{trace_id}", trace_id=trace_id, name="node_a", span_type=SpanType.NODE)
    s.finish(status=status)
    t.spans.append(s)
    t.finish(status=status)
    return t


def test_sqlite_storage_crud():
    """Verify SQLiteStorage save, get, search, stats, delete, and clear."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_obs.db")
        storage = SQLiteStorage(db_path=db_path)

        try:
            t1 = _create_sample_trace("sql_1", "checkout_flow")
            t2 = _create_sample_trace("sql_2", "search_flow", status=SpanStatus.FAILED)

            storage.save_trace(t1)
            storage.save_trace(t2)

            loaded = storage.get_trace("sql_1")
            assert loaded is not None
            assert loaded.id == "sql_1"
            assert len(loaded.spans) == 1
            assert loaded.spans[0].name == "node_a"

            # Search
            results = storage.list_traces(search="checkout")
            assert len(results) == 1
            assert results[0].id == "sql_1"

            # Stats
            stats = storage.get_stats()
            assert stats.total_traces == 2
            assert stats.completed_traces == 1
            assert stats.failed_traces == 1

            # Delete
            assert storage.delete_trace("sql_1") is True
            assert storage.get_trace("sql_1") is None

            # Clear
            storage.clear()
            assert len(storage.list_traces()) == 0
        finally:
            storage.close()


def test_sqlite_storage_node_counts_and_migration():
    """Verify node execution counts JSON column serialization and aggregation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = os.path.join(tmp_dir, "test_node_counts.db")
        storage = SQLiteStorage(db_file)

        try:
            t1 = Trace(id="tr-1", name="WF1")
            n1 = Span(id="n1", trace_id="tr-1", name="retrieve", span_type=SpanType.NODE, status=SpanStatus.COMPLETED)
            n2 = Span(id="n2", trace_id="tr-1", name="generate", span_type=SpanType.NODE, status=SpanStatus.COMPLETED)
            n3 = Span(id="n3", trace_id="tr-1", name="generate", span_type=SpanType.NODE, status=SpanStatus.COMPLETED)
            t1.spans.extend([n1, n2, n3])
            t1.finish(SpanStatus.COMPLETED)

            storage.save_trace(t1)

            # Check raw SQL table content
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            cur.execute("SELECT node_counts FROM traces WHERE id = 'tr-1'")
            row = cur.fetchone()
            assert row is not None
            counts = json.loads(row[0])
            assert counts == {"retrieve": 1, "generate": 2}
            conn.close()

            # Check get_stats aggregation
            stats = storage.get_stats()
            assert stats.node_execution_counts.get("retrieve") == 1
            assert stats.node_execution_counts.get("generate") == 2
        finally:
            storage.close()


def test_sqlite_connection_auto_recovery():
    """Verify that a closed or stale thread-local connection is automatically revived."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "recovery.db")
        store = SQLiteStorage(db_path=db_path)

        try:
            t1 = Trace(id="t-rec-1", name="before-close")
            t1.finish(SpanStatus.COMPLETED)
            store.save_trace(t1)

            # Intentionally close underlying thread-local connection
            raw_conn = store._local.conn
            assert raw_conn is not None
            raw_conn.close()

            # Next operation should transparently detect closed connection and reconnect
            t2 = Trace(id="t-rec-2", name="after-close")
            t2.finish(SpanStatus.COMPLETED)
            store.save_trace(t2)

            retrieved = store.get_trace(t2.id)
            assert retrieved is not None
            assert retrieved.name == "after-close"
        finally:
            store.close()


def test_sqlite_multithreaded_close():
    """Verify SQLiteStorage tracks connections across multiple worker threads and closes all safely."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = os.path.join(tmp_dir, "threads.db")
        store = SQLiteStorage(db_file)

        try:
            def worker(idx: int):
                t = Trace(id=f"t-th-{idx}", name=f"name-{idx}")
                t.finish(SpanStatus.COMPLETED)
                store.save_trace(t)
                store.get_trace(f"t-th-{idx}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(worker, i) for i in range(8)]
                concurrent.futures.wait(futures)

            assert len(store._all_connections) > 0
            store.close()
            assert len(store._all_connections) == 0
        finally:
            store.close()


def test_sqlite_closed_guard():
    """Verify that calling database operations on a closed SQLiteStorage raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = os.path.join(tmp_dir, "closed.db")
        store = SQLiteStorage(db_file)
        store.close()

        with pytest.raises(RuntimeError, match="SQLiteStorage is closed"):
            store._get_connection()


def test_sqlite_wildcard_escaping_and_counting():
    """Verify that LIKE special characters (% and _) are properly escaped during search."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "search_test.db")
        store = SQLiteStorage(db_path=db_path)

        try:
            t1 = Trace(id="t-special-1", name="discount 50% promo")
            t1.finish(SpanStatus.COMPLETED)
            t2 = Trace(id="t-special-2", name="discount 500 promo")
            t2.finish(SpanStatus.COMPLETED)
            t3 = Trace(id="t-special-3", name="my_custom_node")
            t3.finish(SpanStatus.COMPLETED)
            t4 = Trace(id="t-special-4", name="myxcustomynode")
            t4.finish(SpanStatus.COMPLETED)

            for t in [t1, t2, t3, t4]:
                store.save_trace(t)

            assert store.count_traces() == 4

            # Search with '%' should only match t1, not t2
            pct_res = store.list_traces(search="50%")
            assert len(pct_res) == 1
            assert pct_res[0].id == "t-special-1"

            # Search with '_' should only match t3, not t4
            underscore_res = store.list_traces(search="my_custom")
            assert len(underscore_res) == 1
            assert underscore_res[0].id == "t-special-3"
        finally:
            store.close()


def test_sqlite_time_range_and_eval_filtering():
    """Verify SQLiteStorage supports from_time, to_time, and eval_status filtering."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "eval_test.db")
        store = SQLiteStorage(db_path=db_path)
        try:
            t1 = Trace(id="t1", name="wf1", start_time="2026-02-01T10:00:00Z")
            t1.metadata["evals"] = [{"name": "safety", "passed": True}]
            t1.finish(SpanStatus.COMPLETED)

            t2 = Trace(id="t2", name="wf2", start_time="2026-02-02T10:00:00Z")
            t2.metadata["evals"] = [{"name": "safety", "passed": False}]
            t2.finish(SpanStatus.COMPLETED)

            t3 = Trace(id="t3", name="wf3", start_time="2026-02-03T10:00:00Z")
            t3.finish(SpanStatus.COMPLETED)

            for t in [t1, t2, t3]:
                store.save_trace(t)

            # Time range
            mid_traces = store.list_traces(from_time="2026-02-02T00:00:00Z", to_time="2026-02-02T23:59:59Z")
            assert len(mid_traces) == 1
            assert mid_traces[0].id == "t2"

            # Eval status: passed
            passed_traces = store.list_traces(eval_status="passed")
            assert len(passed_traces) == 1
            assert passed_traces[0].id == "t1"
            assert passed_traces[0].eval_status == "passed"
            assert passed_traces[0].eval_count == 1

            # Eval status: failed
            failed_traces = store.list_traces(eval_status="failed")
            assert len(failed_traces) == 1
            assert failed_traces[0].id == "t2"
            assert failed_traces[0].eval_status == "failed"

            # Total count with eval filter
            assert store.count_traces(eval_status="passed") == 1
            assert store.count_traces(eval_status="failed") == 1
        finally:
            store.close()


def test_sqlite_model_counts_and_node_metrics_columns():
    """Verify model_counts and node_metrics pre-aggregated columns are populated and queried."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "metrics_test.db")
        store = SQLiteStorage(db_path=db_path)
        try:
            t = Trace(id="t-metrics-1", name="qa-pipeline", duration_ms=250.0)
            llm_span = Span(
                id="sp-llm",
                trace_id="t-metrics-1",
                name="Chat: gpt-4o",
                span_type=SpanType.LLM,
                metadata={
                    "model": "gpt-4o",
                    "usage": {
                        "model": "gpt-4o",
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                        "total_cost": 0.005,
                    },
                },
            )
            llm_span.finish(SpanStatus.COMPLETED)
            node_span = Span(
                id="sp-node",
                trace_id="t-metrics-1",
                name="format_answer",
                span_type=SpanType.NODE,
                duration_ms=120.0,
            )
            node_span.finish(SpanStatus.COMPLETED)
            t.spans.extend([llm_span, node_span])
            t.finish(SpanStatus.COMPLETED)
            t.duration_ms = 250.0

            store.save_trace(t)

            # 1. Verify columns in raw SQLite table
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT model_counts, node_metrics, duration_ms FROM traces WHERE id = 't-metrics-1'")
            row = cur.fetchone()
            assert row is not None
            model_counts = json.loads(row[0])
            node_metrics = json.loads(row[1])
            duration_ms = row[2]
            conn.close()

            assert model_counts.get("gpt-4o") == 1
            assert any(n["name"] == "format_answer" for n in node_metrics)
            assert duration_ms == 250.0

            # 2. Verify get_timeseries_analytics utilizes model_counts
            analytics = store.get_timeseries_analytics(time_window="24h")
            assert analytics["total_traces"] == 1
            bucket_models = [b["models"] for b in analytics["buckets"] if "models" in b]
            assert any(m.get("gpt-4o") == 1 for m in bucket_models)

            # 3. Verify get_node_analytics utilizes node_metrics
            node_analytics = store.get_node_analytics(time_window="24h")
            assert node_analytics["total_traces"] == 1
            assert any(n["node_name"] == "format_answer" for n in node_analytics["nodes"])
        finally:
            store.close()


def test_sqlite_stats_ttl_cache_and_trace_lru_cache():
    """Verify stats in-memory TTL caching and trace LRU caching with invalidation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "cache_test.db")
        store = SQLiteStorage(db_path=db_path)
        try:
            t1 = Trace(id="t-cache-1", name="wf1")
            t1.finish(SpanStatus.COMPLETED)
            store.save_trace(t1)

            # 1. Stats caching
            stats1 = store.get_stats()
            assert stats1.total_traces == 1
            assert store._stats_cache is not None

            # Directly insert row into SQLite behind the storage's back
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("INSERT INTO traces (id, name, status, start_time, data_json) VALUES ('t-direct', 'ghost', 'completed', '2026-01-01T00:00:00Z', '{}')")
            conn.commit()
            conn.close()

            # get_stats within TTL should still return cached 1
            stats2 = store.get_stats()
            assert stats2.total_traces == 1

            # Saving through store invalidates stats cache
            t2 = Trace(id="t-cache-2", name="wf2")
            t2.finish(SpanStatus.COMPLETED)
            store.save_trace(t2)
            assert store._stats_cache is None

            # New get_stats reflects both added traces
            stats3 = store.get_stats()
            assert stats3.total_traces == 3

            # 2. Trace LRU cache
            trace_fetched = store.get_trace("t-cache-1")
            assert trace_fetched is not None
            assert "t-cache-1" in store._trace_lru

            # Mutate via save_trace updates/invalidates cache
            t1_updated = Trace(id="t-cache-1", name="wf1-renamed")
            t1_updated.finish(SpanStatus.COMPLETED)
            store.save_trace(t1_updated)
            assert store.get_trace("t-cache-1").name == "wf1-renamed"

            # Delete invalidates cache
            store.delete_trace("t-cache-1")
            assert "t-cache-1" not in store._trace_lru
            assert store.get_trace("t-cache-1") is None
        finally:
            store.close()


