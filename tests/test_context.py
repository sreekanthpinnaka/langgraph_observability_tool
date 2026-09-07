from __future__ import annotations

import asyncio
import pytest
from langgraph.graph import StateGraph, START, END

from langgraph_observe.client.collector import TraceCollector
from langgraph_observe.client.langgraph import observe_graph
from langgraph_observe.core.context import (
    get_current_collector,
    get_current_trace,
    get_current_trace_id,
    record_eval,
    reset_current_collector,
    reset_current_trace,
    set_current_collector,
    set_current_trace,
    set_trace_metadata,
)
from langgraph_observe.core.models import SpanStatus, SpanType


def test_context_token_restoration_nested_graphs():
    """Verify that nested graph execution properly preserves and restores parent context tokens."""
    parent_collector = TraceCollector(name="parent_workflow")
    tok_c = set_current_collector(parent_collector)
    tok_t = set_current_trace(parent_collector.trace)

    try:
        builder = StateGraph(dict)
        builder.add_node("sub_step", lambda s: {"sub": "ok"})
        builder.add_edge(START, "sub_step")
        builder.add_edge("sub_step", END)

        sub_graph = observe_graph(builder.compile(), name="SubGraph")
        res = sub_graph.invoke({"start": 1})
        assert res == {"sub": "ok"}

        # Context MUST be restored to parent_collector and parent_trace
        assert get_current_collector() is parent_collector
        assert get_current_trace() is parent_collector.trace

        # Verify context preservation when an inner graph raises an exception
        builder_fail = StateGraph(dict)

        def crashing_node(s):
            raise RuntimeError("Inner failure")

        builder_fail.add_node("crash", crashing_node)
        builder_fail.add_edge(START, "crash")
        builder_fail.add_edge("crash", END)
        failing_graph = observe_graph(builder_fail.compile(), name="CrashGraph")

        with pytest.raises(RuntimeError, match="Inner failure"):
            failing_graph.invoke({})

        # Context MUST still be restored to parent_collector after error
        assert get_current_collector() is parent_collector
        assert get_current_trace() is parent_collector.trace
    finally:
        reset_current_collector(tok_c)
        reset_current_trace(tok_t)


@pytest.mark.asyncio
async def test_context_isolation_across_async_tasks():
    """Verify that independent async tasks maintain their own isolated context."""
    collector_a = TraceCollector(name="workflow_a")
    collector_b = TraceCollector(name="workflow_b")

    async def task_a():
        tok_c = set_current_collector(collector_a)
        await asyncio.sleep(0.01)
        assert get_current_collector() is collector_a
        reset_current_collector(tok_c)

    async def task_b():
        tok_c = set_current_collector(collector_b)
        await asyncio.sleep(0.01)
        assert get_current_collector() is collector_b
        reset_current_collector(tok_c)

    await asyncio.gather(task_a(), task_b())


def test_set_trace_metadata_public_api():
    """Verify set_trace_metadata adds metadata to active trace via dict or key-value pair, and is safe when no trace is active."""
    # 1. Safe when no trace is active (no-op, no crash)
    set_trace_metadata("should_not_crash", "val")
    set_trace_metadata({"should_not": "crash"})

    # 2. Active collector/trace
    collector = TraceCollector(name="test_meta_workflow")
    tok = set_current_collector(collector)
    try:
        # Key-value pair
        set_trace_metadata("tenant_id", "tenant-xyz")
        assert collector.trace.metadata.get("tenant_id") == "tenant-xyz"

        # Dictionary update
        set_trace_metadata({"user_tier": "enterprise", "region": "us-east-1"})
        assert collector.trace.metadata.get("user_tier") == "enterprise"
        assert collector.trace.metadata.get("region") == "us-east-1"
        assert collector.trace.metadata.get("tenant_id") == "tenant-xyz"
    finally:
        reset_current_collector(tok)


def test_get_current_trace_id_public_api():
    """Verify get_current_trace_id retrieves the active trace ID or None when no trace is active."""
    # 1. When no trace is active, returns None
    assert get_current_trace_id() is None

    # 2. When active collector is set
    collector = TraceCollector(name="test_id_workflow")
    tok_c = set_current_collector(collector)
    try:
        assert get_current_trace_id() == collector.trace.id
    finally:
        reset_current_collector(tok_c)

    # 3. When active trace is set directly via set_current_trace
    from langgraph_observe.core.models import Trace
    custom_trace = Trace(id="trace_direct_123", name="direct_trace")
    tok_t = set_current_trace(custom_trace)
    try:
        assert get_current_trace_id() == "trace_direct_123"
    finally:
        reset_current_trace(tok_t)


def test_record_eval_public_api():
    """Verify record_eval records evaluation metadata and EVAL spans."""
    # 1. When no trace is active, safely returns False
    assert record_eval(name="faithfulness", score=0.95, passed=True) is False

    # 2. With active collector
    collector = TraceCollector(name="test_eval_workflow")
    tok_c = set_current_collector(collector)
    try:
        ok1 = record_eval(name="hallucination_check", score=1.0, passed=True, reason="Grounded in context")
        assert ok1 is True
        ok2 = record_eval(name="toxicity_check", score=0.2, passed=False, reason="Unsafe language detected")
        assert ok2 is True

        evals = collector.trace.metadata.get("evals", [])
        assert len(evals) == 2
        assert evals[0]["name"] == "hallucination_check"
        assert evals[0]["passed"] is True
        assert evals[0]["score"] == 1.0
        assert evals[1]["name"] == "toxicity_check"
        assert evals[1]["passed"] is False

        # Verify EVAL spans were created
        eval_spans = [s for s in collector.trace.spans if s.span_type == SpanType.EVAL]
        assert len(eval_spans) == 2
        assert eval_spans[0].name == "Eval: hallucination_check"
        assert eval_spans[0].status == SpanStatus.COMPLETED
        assert eval_spans[1].name == "Eval: toxicity_check"
        assert eval_spans[1].status == SpanStatus.FAILED
    finally:
        reset_current_collector(tok_c)


