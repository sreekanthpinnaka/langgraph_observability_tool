from __future__ import annotations

import time
from langgraph_observe.core.models import (
    HTTPMetadata,
    Span,
    SpanStatus,
    SpanType,
    Trace,
    TraceSummary,
)


def test_span_lifecycle():
    span = Span(
        id="s1",
        trace_id="t1",
        name="test_span",
        span_type=SpanType.NODE,
        inputs={"query": "test"},
    )
    assert span.status == SpanStatus.RUNNING
    assert span.duration_ms is None

    time.sleep(0.01)
    span.finish(status=SpanStatus.COMPLETED, outputs={"result": 100})
    assert span.status == SpanStatus.COMPLETED
    assert span.outputs == {"result": 100}
    assert span.duration_ms is not None
    assert span.duration_ms >= 5.0


def test_trace_lifecycle_and_metrics():
    trace = Trace(
        id="t1",
        name="test_workflow",
        http=HTTPMetadata(method="POST", path="/api/run", url="http://test/api/run"),
    )

    span1 = Span(
        id="s1",
        trace_id="t1",
        name="node1",
        span_type=SpanType.NODE,
    )
    span1.finish(status=SpanStatus.COMPLETED)

    span2 = Span(
        id="s2",
        trace_id="t1",
        name="llm1",
        span_type=SpanType.LLM,
        metadata={"usage": {"total_tokens": 150, "prompt_tokens": 50, "completion_tokens": 100}},
    )
    span2.finish(status=SpanStatus.COMPLETED)

    trace.spans = [span1, span2]
    trace.finish(status=SpanStatus.COMPLETED)
    trace.compute_metrics()

    assert trace.status == SpanStatus.COMPLETED
    assert trace.duration_ms is not None
    assert trace.llm_metrics.total_tokens == 150
    assert trace.llm_metrics.prompt_tokens == 50
    assert trace.llm_metrics.completion_tokens == 100
    assert trace.llm_metrics.llm_calls == 1


def test_metrics_computed_flag_excluded_from_serialization():
    """Verify that internal metrics computation flags are excluded from JSON dumps."""
    trace = Trace(id="t-ex-1", name="exclude-test")
    trace.compute_metrics()
    assert trace.metrics_computed is True

    dumped = trace.model_dump()
    assert "metrics_computed" not in dumped

    # Deserializing should not inherit True from JSON
    restored = Trace.model_validate(dumped)
    assert restored.metrics_computed is False


def test_trace_finish_idempotence():
    """Verify that calling finish() multiple times does not corrupt metrics or timestamps."""
    trace = Trace(id="t-fin-1", name="finish-test")
    span = Span(
        id="s-fin-1",
        trace_id=trace.id,
        name="llm",
        span_type=SpanType.LLM,
        metadata={"usage": {"model": "gpt-4o", "prompt_tokens": 100, "completion_tokens": 50}},
    )
    trace.spans.append(span)

    trace.finish(SpanStatus.COMPLETED)
    trace.compute_metrics()
    first_duration = trace.duration_ms
    first_cost = trace.estimated_cost
    assert first_cost > 0

    # Second finish call must be idempotent
    trace.finish(SpanStatus.COMPLETED)
    assert trace.duration_ms == first_duration
    assert trace.estimated_cost == first_cost


def test_compute_metrics_idempotence_and_cost_safety():
    """Ensure repeatedly computing metrics does not multiply token counts or costs."""
    trace = Trace(id="t-metrics-1", name="test-trace")
    span = Span(
        id="s-metrics-1",
        trace_id=trace.id,
        name="llm_step",
        span_type=SpanType.LLM,
        metadata={
            "usage": {
                "model": "gpt-4o",
                "prompt_tokens": 1000,
                "completion_tokens": 500,
            }
        },
    )
    trace.spans.append(span)
    trace.finish(SpanStatus.COMPLETED)
    trace.compute_metrics()

    assert trace.estimated_cost > 0
    cost1 = trace.estimated_cost
    tokens1 = trace.llm_metrics.total_tokens

    # Compute metrics multiple times
    trace.compute_metrics()
    trace.compute_metrics()

    assert trace.estimated_cost == cost1
    assert trace.llm_metrics.total_tokens == tokens1


def test_compute_metrics_force_recomputation():
    """Verify that force=True re-calculates metrics even if already computed."""
    trace = Trace(id="t-force-1", name="force-test")
    span1 = Span(
        id="s-f-1",
        trace_id=trace.id,
        name="llm-1",
        span_type=SpanType.LLM,
        metadata={"usage": {"model": "gpt-4o-mini", "prompt_tokens": 100, "completion_tokens": 50}},
    )
    trace.spans.append(span1)
    trace.compute_metrics()
    initial_tokens = trace.llm_metrics.total_tokens
    assert initial_tokens == 150

    # Add a new span
    span2 = Span(
        id="s-f-2",
        trace_id=trace.id,
        name="llm-2",
        span_type=SpanType.LLM,
        metadata={"usage": {"model": "gpt-4o-mini", "prompt_tokens": 200, "completion_tokens": 100}},
    )
    trace.spans.append(span2)

    # Without force, metrics should remain unchanged
    trace.compute_metrics(force=False)
    assert trace.llm_metrics.total_tokens == 150

    # With force=True, metrics should be re-calculated
    trace.compute_metrics(force=True)
    assert trace.llm_metrics.total_tokens == 450


def test_p95_duration_calculation():
    """Verify that p95 latency is correctly calculated from sorted sample durations."""
    from langgraph_observe.server.storage.memory import MemoryStorage

    storage = MemoryStorage()
    # Insert 100 traces with durations 1..100
    for i in range(1, 101):
        t = Trace(id=f"t-p95-{i}", name="test", duration_ms=float(i))
        storage.save_trace(t)

    stats = storage.get_stats()
    # 95th percentile of 1..100 is between 94.0 and 96.0
    assert 94.0 <= stats.p95_duration_ms <= 96.0


def test_tool_loop_detection():
    """Verify automated detection of infinite tool call loops (Class 3 failure)."""
    # 1. Normal trace: 3 tool calls with distinct inputs -> No loop
    trace_ok = Trace(id="t-loop-ok", name="normal-agent")
    for i in range(3):
        trace_ok.spans.append(
            Span(
                id=f"span-tool-{i}",
                trace_id=trace_ok.id,
                name="web_search",
                span_type=SpanType.TOOL,
                inputs={"query": f"search topic {i}"},
            )
        )
    trace_ok.compute_metrics()
    assert trace_ok.has_tool_loop is False
    assert "tool_loop_detected" not in trace_ok.metadata

    summary_ok = TraceSummary.from_trace(trace_ok)
    assert summary_ok.has_tool_loop is False

    # 2. Tool loop via 5 or more invocations of the same tool
    trace_loop_count = Trace(id="t-loop-count", name="looping-agent-5")
    for i in range(5):
        trace_loop_count.spans.append(
            Span(
                id=f"span-loop-{i}",
                trace_id=trace_loop_count.id,
                name="calculator",
                span_type=SpanType.TOOL,
                inputs={"expression": f"2 + {i}"},
            )
        )
    trace_loop_count.compute_metrics()
    assert trace_loop_count.has_tool_loop is True
    assert trace_loop_count.metadata.get("tool_loop_detected") is True
    assert trace_loop_count.metadata.get("tool_loop_tool") == "calculator"
    assert trace_loop_count.metadata.get("tool_loop_count") == 5

    summary_loop = TraceSummary.from_trace(trace_loop_count)
    assert summary_loop.has_tool_loop is True

    # 3. Tool loop via repeated duplicate inputs (3 times)
    trace_loop_dup = Trace(id="t-loop-dup", name="looping-agent-dup")
    for i in range(3):
        trace_loop_dup.spans.append(
            Span(
                id=f"span-dup-{i}",
                trace_id=trace_loop_dup.id,
                name="api_fetch",
                span_type=SpanType.TOOL,
                inputs={"url": "https://api.example.com/status"},
            )
        )
    trace_loop_dup.compute_metrics()
    assert trace_loop_dup.has_tool_loop is True
    assert trace_loop_dup.metadata.get("tool_loop_tool") == "api_fetch"

