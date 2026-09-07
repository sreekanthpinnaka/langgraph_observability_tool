from __future__ import annotations

from contextlib import contextmanager
import contextvars
from typing import Any, Generator, List, Optional
import uuid
from langgraph_observe.core.models import Span, SpanStatus, SpanType, Trace, current_iso_time

_current_trace: contextvars.ContextVar[Optional[Trace]] = contextvars.ContextVar(
    "_current_trace", default=None
)

_current_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_span_id", default=None
)

_current_collector: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "_current_collector", default=None
)


def get_current_trace() -> Optional[Trace]:
    """Retrieve the active Trace from the current async context."""
    return _current_trace.get()


def set_current_trace(trace: Optional[Trace]) -> contextvars.Token:
    """Set the active Trace in the current async context."""
    return _current_trace.set(trace)


def get_current_span_id() -> Optional[str]:
    """Retrieve the ID of the currently executing span in this task's context."""
    return _current_span_id.get()


def set_current_span_id(span_id: Optional[str]) -> contextvars.Token:
    """Set the active span ID in this task's context, returning a restoration token."""
    return _current_span_id.set(span_id)


def reset_current_span_id(token: contextvars.Token) -> None:
    """Reset the active span ID using its task-specific restoration token."""
    _current_span_id.reset(token)


@contextmanager
def active_span(span_id: str) -> Generator[None, None, None]:
    """Context manager to safely set and restore the active span ID for this task."""
    token = _current_span_id.set(span_id)
    try:
        yield
    finally:
        _current_span_id.reset(token)


@contextmanager
def trace_span(
    name: str,
    span_type: SpanType = SpanType.CUSTOM,
    inputs: Optional[Any] = None,
    metadata: Optional[Any] = None,
) -> Generator[Optional[Span], None, None]:
    """Context manager to create, nest, and automatically record a child span within the current trace."""
    collector = get_current_collector()
    if collector is None:
        yield None
        return

    parent_id = get_current_span_id()
    span = collector.start_span(
        name=name,
        span_type=span_type,
        parent_id=parent_id,
        inputs=inputs,
        metadata=metadata,
    )
    token = set_current_span_id(span.id)
    try:
        yield span
        if span.status == SpanStatus.RUNNING:
            collector.finish_span(span.id, status=SpanStatus.COMPLETED)
    except Exception as exc:
        if span.status == SpanStatus.RUNNING:
            collector.finish_span(
                span.id,
                status=SpanStatus.FAILED,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
        raise
    finally:
        reset_current_span_id(token)


# Backwards compatibility wrappers
def push_span_id(span_id: str) -> None:
    _current_span_id.set(span_id)


def pop_span_id() -> Optional[str]:
    old = _current_span_id.get()
    _current_span_id.set(None)
    return old


def reset_current_trace(token: contextvars.Token) -> None:
    """Reset the active Trace using its context restoration token."""
    _current_trace.reset(token)


def get_current_collector() -> Optional[Any]:
    """Retrieve the active TraceCollector."""
    return _current_collector.get()


def set_current_collector(collector: Optional[Any]) -> contextvars.Token:
    """Set the active TraceCollector."""
    return _current_collector.set(collector)


def reset_current_collector(token: contextvars.Token) -> None:
    """Reset the active TraceCollector using its context restoration token."""
    _current_collector.reset(token)


def set_trace_metadata(key_or_dict: str | dict[str, Any], value: Any = None) -> bool:
    """Attach arbitrary key/value metadata directly to the active Trace in the current context.

    Can be invoked with a key and value:
        set_trace_metadata("user_id", "user_123")

    Or with a dictionary:
        set_trace_metadata({"user_id": "user_123", "variant": "B"})

    Returns True if metadata was attached to an active trace, False if no trace is active.
    """
    trace = get_current_trace()
    if not trace:
        collector = get_current_collector()
        if collector and hasattr(collector, "trace"):
            trace = collector.trace
    if not trace:
        return False

    if isinstance(key_or_dict, dict):
        trace.metadata.update(key_or_dict)
    else:
        trace.metadata[str(key_or_dict)] = value
    return True


def get_current_trace_id() -> Optional[str]:
    """Retrieve the ID of the active Trace from the current async context, or None if no trace is active."""
    trace = get_current_trace()
    if trace and trace.id:
        return trace.id
    collector = get_current_collector()
    if collector and hasattr(collector, "trace") and collector.trace and collector.trace.id:
        return collector.trace.id
    return None


def record_eval(
    name: str,
    score: float,
    passed: bool,
    reason: Optional[str] = None,
) -> bool:
    """Record an evaluation result for the current active trace.

    Appends an evaluation record to `trace.metadata['evals']` and emits an EVAL span.
    Returns True if recorded, False if no trace is active.
    """
    trace = get_current_trace()
    collector = get_current_collector()
    if not trace and collector and hasattr(collector, "trace"):
        trace = collector.trace
    if not trace:
        return False

    eval_data = {
        "name": str(name),
        "score": float(score),
        "passed": bool(passed),
        "reason": str(reason) if reason is not None else None,
        "timestamp": current_iso_time(),
    }

    if "evals" not in trace.metadata or not isinstance(trace.metadata["evals"], list):
        trace.metadata["evals"] = []
    trace.metadata["evals"].append(eval_data)

    if collector and hasattr(collector, "start_span") and hasattr(collector, "finish_span"):
        span = collector.start_span(
            name=f"Eval: {name}",
            span_type=SpanType.EVAL,
            parent_id=get_current_span_id(),
            inputs={"score": score, "passed": passed, "reason": reason},
            metadata={"eval": eval_data},
        )
        collector.finish_span(
            span.id,
            status=SpanStatus.COMPLETED if passed else SpanStatus.FAILED,
            outputs={"passed": passed, "score": score, "reason": reason},
        )
    else:
        now = current_iso_time()
        span = Span(
            id=f"span_{uuid.uuid4().hex[:12]}",
            trace_id=trace.id,
            parent_id=get_current_span_id(),
            name=f"Eval: {name}",
            span_type=SpanType.EVAL,
            status=SpanStatus.COMPLETED if passed else SpanStatus.FAILED,
            start_time=now,
            end_time=now,
            duration_ms=0.0,
            inputs={"score": score, "passed": passed, "reason": reason},
            outputs={"passed": passed, "score": score, "reason": reason},
            metadata={"eval": eval_data},
        )
        trace.spans.append(span)

    return True




