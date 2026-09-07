from __future__ import annotations

import os
import shutil
import tempfile
import pytest

from langgraph_observe.core.models import (
    HTTPMetadata,
    Span,
    SpanStatus,
    SpanType,
    Trace,
)
from langgraph_observe.server.storage.memory import MemoryStorage
from langgraph_observe.server.storage.sql import SQLAlchemyStorage
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


def test_memory_storage_crud_and_filters():
    """Verify MemoryStorage save, get, search, stats, delete, and clear."""
    storage = MemoryStorage()
    t1 = _create_sample_trace("t1", "workflow_a")
    t2 = _create_sample_trace("t2", "workflow_b", status=SpanStatus.FAILED)

    storage.save_trace(t1)
    storage.save_trace(t2)

    loaded = storage.get_trace("t1")
    assert loaded is not None
    assert loaded.id == "t1"
    assert loaded.name == "workflow_a"

    # List & filter
    all_traces = storage.list_traces()
    assert len(all_traces) == 2

    failed = storage.list_traces(status="failed")
    assert len(failed) == 1
    assert failed[0].id == "t2"

    search_res = storage.list_traces(search="workflow_a")
    assert len(search_res) == 1
    assert search_res[0].id == "t1"

    # Stats
    stats = storage.get_stats()
    assert stats.total_traces == 2
    assert stats.completed_traces == 1
    assert stats.failed_traces == 1
    assert stats.node_execution_counts.get("node_a") == 2

    # Delete
    assert storage.delete_trace("t1") is True
    assert storage.get_trace("t1") is None

    # Clear
    storage.clear()
    assert len(storage.list_traces()) == 0


def test_list_traces_with_count_on_all_storages():
    """Verify list_traces_with_count returns both sliced trace summaries and total count across all storages."""
    tmp_dir = tempfile.mkdtemp()
    try:
        sqlite_file = os.path.join(tmp_dir, "atomic_count.db")
        sql_file = os.path.join(tmp_dir, "atomic_sql.db")

        mem_store = MemoryStorage()
        sqlite_store = SQLiteStorage(sqlite_file)
        sql_store = SQLAlchemyStorage(f"sqlite:///{sql_file}")

        storages = [mem_store, sqlite_store, sql_store]

        for store in storages:
            for i in range(12):
                t = Trace(id=f"t-at-{i}", name=f"trace-{i}", project="proj_a" if i < 6 else "proj_b")
                t.finish(SpanStatus.COMPLETED)
                store.save_trace(t)

            # Test pagination with limit 5, offset 0
            items, total = store.list_traces_with_count(limit=5, offset=0)
            assert len(items) == 5
            assert total == 12

            # Test offset 10, limit 5
            items, total = store.list_traces_with_count(limit=5, offset=10)
            assert len(items) == 2
            assert total == 12

            # Test with filter
            items_filt, total_filt = store.list_traces_with_count(limit=10, offset=0, project="proj_a")
            assert len(items_filt) == 6
            assert total_filt == 6

            # Test offset past total
            items_empty, total_past = store.list_traces_with_count(limit=5, offset=50)
            assert len(items_empty) == 0
            assert total_past == 12

        sqlite_store.close()
        sql_store.close()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
