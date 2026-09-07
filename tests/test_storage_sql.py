from __future__ import annotations

import os
import tempfile
import pytest

from langgraph_observe.core.models import (
    HTTPMetadata,
    Span,
    SpanStatus,
    SpanType,
    Trace,
)
from langgraph_observe.server.storage.sql import SQLAlchemyStorage, create_storage_from_url


def _make_trace(trace_id: str, name: str, status: SpanStatus = SpanStatus.COMPLETED) -> Trace:
    t = Trace(
        id=trace_id,
        name=name,
        http=HTTPMetadata(method="POST", path=f"/api/{name}", url=f"http://localhost/api/{name}"),
    )
    s = Span(id=f"sp_{trace_id}", trace_id=trace_id, name="node_exec", span_type=SpanType.NODE)
    s.finish(status=status)
    t.spans.append(s)
    t.finish(status=status)
    return t


def test_sqlalchemy_storage_crud():
    """Verify SQLAlchemyStorage CRUD lifecycle: save, get, search, stats, delete, and clear."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = os.path.join(tmp_dir, "test_sql.db")
        storage = SQLAlchemyStorage(database_url=f"sqlite:///{db_file}")

        try:
            t1 = _make_trace("sql-1", "order_service")
            t2 = _make_trace("sql-2", "billing_service", status=SpanStatus.FAILED)

            storage.save_trace(t1)
            storage.save_trace(t2)

            loaded = storage.get_trace("sql-1")
            assert loaded is not None
            assert loaded.id == "sql-1"
            assert loaded.name == "order_service"
            assert len(loaded.spans) == 1

            # List and search
            traces = storage.list_traces()
            assert len(traces) == 2

            search_traces = storage.list_traces(search="billing")
            assert len(search_traces) == 1
            assert search_traces[0].id == "sql-2"

            # Stats
            stats = storage.get_stats()
            assert stats.total_traces == 2
            assert stats.completed_traces == 1
            assert stats.failed_traces == 1
            assert stats.node_execution_counts.get("node_exec") == 2

            # Delete & clear
            assert storage.delete_trace("sql-1") is True
            assert storage.get_trace("sql-1") is None

            storage.clear()
            assert len(storage.list_traces()) == 0
        finally:
            storage.close()


def test_sqlalchemy_atomic_upsert():
    """Verify that updating an existing trace and adding spans performs an atomic upsert."""
    store = SQLAlchemyStorage("sqlite:///:memory:")

    trace = Trace(id="t-upsert-1", name="initial-run")
    span1 = Span(id="s-up-1", trace_id=trace.id, name="node-1", span_type=SpanType.NODE)
    trace.spans.append(span1)
    trace.finish(SpanStatus.RUNNING)
    store.save_trace(trace)

    # Upsert with updated status and extra span
    span2 = Span(id="s-up-2", trace_id=trace.id, name="node-2", span_type=SpanType.NODE)
    trace.spans.append(span2)
    trace.status = SpanStatus.COMPLETED
    store.save_trace(trace)

    fetched = store.get_trace(trace.id)
    assert fetched is not None
    assert fetched.status == SpanStatus.COMPLETED
    assert len(fetched.spans) == 2

    store.close()


def test_mysql_url_normalization_and_cloud_sql_settings():
    """Verify that plain mysql:// URLs are normalized to pymysql driver."""
    raw_mysql_url = "mysql://observe_user:secret_pass@127.0.0.1:3306/observe_db"

    raw_url = raw_mysql_url
    if raw_url.startswith("mysql://"):
        raw_url = raw_url.replace("mysql://", "mysql+pymysql://", 1)

    assert raw_url.startswith("mysql+pymysql://")
    assert "pymysql" in raw_url


def test_create_storage_from_url_env(monkeypatch):
    """Verify factory initializes SQLAlchemyStorage from DATABASE_URL env."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    storage = create_storage_from_url()
    assert isinstance(storage, SQLAlchemyStorage)
    assert storage.is_sqlite is True
    storage.close()


def test_sqlalchemy_time_range_and_eval_filtering():
    """Verify SQLAlchemyStorage supports from_time, to_time, and eval_status filtering."""
    storage = SQLAlchemyStorage(database_url="sqlite:///:memory:")
    try:
        t1 = Trace(id="sa_t1", name="wf1", start_time="2026-03-01T10:00:00Z")
        t1.metadata["evals"] = [{"name": "clarity", "passed": True}]
        t1.finish(SpanStatus.COMPLETED)

        t2 = Trace(id="sa_t2", name="wf2", start_time="2026-03-02T10:00:00Z")
        t2.metadata["evals"] = [{"name": "clarity", "passed": False}]
        t2.finish(SpanStatus.COMPLETED)

        t3 = Trace(id="sa_t3", name="wf3", start_time="2026-03-03T10:00:00Z")
        t3.finish(SpanStatus.COMPLETED)

        for t in [t1, t2, t3]:
            storage.save_trace(t)

        # Time range
        mid_traces = storage.list_traces(from_time="2026-03-02T00:00:00Z", to_time="2026-03-02T23:59:59Z")
        assert len(mid_traces) == 1
        assert mid_traces[0].id == "sa_t2"

        # Eval status: passed
        passed_traces = storage.list_traces(eval_status="passed")
        assert len(passed_traces) == 1
        assert passed_traces[0].id == "sa_t1"
        assert passed_traces[0].eval_status == "passed"
        assert passed_traces[0].eval_count == 1

        # Eval status: failed
        failed_traces = storage.list_traces(eval_status="failed")
        assert len(failed_traces) == 1
        assert failed_traces[0].id == "sa_t2"
        assert failed_traces[0].eval_status == "failed"

        # Total count with eval filter
        assert storage.count_traces(eval_status="passed") == 1
        assert storage.count_traces(eval_status="failed") == 1
    finally:
        storage.close()

