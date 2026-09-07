from __future__ import annotations

import pytest

from langgraph_observe.client.collector import TraceCollector, set_default_storage
from langgraph_observe.core.models import SpanStatus, SpanType
from langgraph_observe.server.storage.memory import MemoryStorage


def test_collector_trace_lifecycle_and_spans():
    """Verify standard trace creation, span lifecycle, and finish handling."""
    storage = MemoryStorage()
    collector = TraceCollector(name="test-workflow", storage=storage)

    span1 = collector.start_span(name="step-1", span_type=SpanType.NODE, inputs={"input": 42})
    assert span1.status == SpanStatus.RUNNING
    assert span1.id in collector._spans_by_id

    collector.finish_span(span1.id, outputs={"output": 84})
    assert span1.status == SpanStatus.COMPLETED
    assert span1.outputs == {"output": 84}

    trace = collector.finish_trace(status=SpanStatus.COMPLETED)
    assert trace.status == SpanStatus.COMPLETED
    assert len(trace.spans) == 1
    assert len(storage.list_traces()) == 1


def test_collector_max_spans_bounding():
    """Verify that TraceCollector limits total spans to max_spans to prevent runaway memory usage."""
    collector = TraceCollector(name="bounded-test", max_spans=5)

    for i in range(12):
        s = collector.start_span(name=f"span-{i}", span_type=SpanType.NODE)
        collector.finish_span(s.id)

    assert len(collector.trace.spans) == 5
    assert collector.trace.metadata.get("spans_truncated") is True
    assert collector.trace.metadata.get("dropped_spans_count") == 7


def test_collector_node_state_updates_capped():
    """Verify that node state updates dictionary array is capped to max_spans bounds."""
    collector = TraceCollector(name="capped-updates", max_spans=5)

    for i in range(10):
        collector.apply_node_state_update(f"node_{i}", {"key": f"val_{i}"})

    updates = collector.trace.metadata.get("node_state_updates", [])
    assert len(updates) == 5
    assert collector.trace.metadata.get("node_state_updates_truncated") is True
