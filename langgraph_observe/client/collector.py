from __future__ import annotations

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional, Union
import os
import uuid

from langgraph_observe.core.models import (
    HTTPMetadata,
    Span,
    SpanStatus,
    SpanType,
    Trace,
)
from langgraph_observe.core.serializer import calculate_state_diff, safe_serialize
from langgraph_observe.server.storage.base import BaseStorage

logger = logging.getLogger("langgraph_observe.client.collector")

_default_storage: Optional[BaseStorage] = None


def get_default_storage() -> BaseStorage:
    """Get the global default storage instance for the client (lazily imports RemoteStorage if None)."""
    global _default_storage
    if _default_storage is None:
        from langgraph_observe.client.exporter import RemoteStorage
        _default_storage = RemoteStorage()
    return _default_storage


def set_default_storage(storage: BaseStorage) -> None:
    """Set the global default storage instance for the client."""
    global _default_storage
    _default_storage = storage


class TraceCollector:
    """Collects and aggregates spans for a single trace and coordinates delivery."""

    def __init__(
        self,
        trace_id: Optional[str] = None,
        name: str = "workflow",
        workflow_name: Optional[str] = None,
        http_metadata: Optional[HTTPMetadata] = None,
        storage: Optional[BaseStorage] = None,
        metadata: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
        mask_pii: bool = False,
        pii_masker: Optional[Any] = None,
        max_spans: Optional[int] = None,
    ) -> None:
        self.trace_id = trace_id or str(uuid.uuid4())
        self.storage = storage or get_default_storage()
        self.mask_pii = mask_pii
        self.pii_masker = pii_masker
        env_val = environment or os.getenv("OBSERVE_ENV") or os.getenv("ENVIRONMENT") or "production"
        proj_val = project or os.getenv("OBSERVE_PROJECT") or None
        wf_name = workflow_name or name

        env_max_spans = os.getenv("OBSERVE_MAX_SPANS")
        if max_spans is not None:
            self.max_spans = max_spans
        elif env_max_spans:
            try:
                self.max_spans = int(env_max_spans)
            except ValueError:
                self.max_spans = 1000
        else:
            self.max_spans = 1000

        self._max_spans_warned = False

        self.trace = Trace(
            id=self.trace_id,
            name=wf_name,
            environment=env_val,
            project=proj_val,
            http=http_metadata,
            metadata=safe_serialize(metadata, mask_pii=mask_pii, pii_masker=pii_masker) if metadata else {},
        )
        self._spans_by_id: Dict[str, Span] = {}

    def start_span(
        self,
        span_id: Optional[str] = None,
        name: str = "span",
        span_type: SpanType = SpanType.CUSTOM,
        parent_id: Optional[str] = None,
        inputs: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Span:
        sid = span_id or str(uuid.uuid4())
        span = Span(
            id=sid,
            trace_id=self.trace_id,
            parent_id=parent_id,
            name=name,
            span_type=span_type,
            status=SpanStatus.RUNNING,
            inputs=safe_serialize(inputs, mask_pii=self.mask_pii, pii_masker=self.pii_masker) if inputs is not None else None,
            metadata=safe_serialize(metadata, mask_pii=self.mask_pii, pii_masker=self.pii_masker) if metadata else {},
        )
        span._raw_inputs = inputs
        self._spans_by_id[sid] = span

        if len(self.trace.spans) < self.max_spans:
            self.trace.spans.append(span)
        else:
            if not self._max_spans_warned:
                logger.warning(
                    f"Trace {self.trace_id} reached max_spans limit ({self.max_spans}). "
                    "Subsequent spans are dropped from memory while trace-level metrics continue to accumulate."
                )
                self._max_spans_warned = True
            self.trace.metadata["spans_truncated"] = True
            self.trace.metadata["dropped_spans_count"] = self.trace.metadata.get("dropped_spans_count", 0) + 1

        return span

    def finish_span(
        self,
        span_id: Union[str, Span],
        status: SpanStatus = SpanStatus.COMPLETED,
        outputs: Optional[Any] = None,
        error: Optional[Dict[str, Any]] = None,
        state_diff: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        sid = span_id.id if isinstance(span_id, Span) else str(span_id)
        span = self._spans_by_id.get(sid)
        if not span:
            return None

        clean_outputs = safe_serialize(outputs, mask_pii=self.mask_pii, pii_masker=self.pii_masker) if outputs is not None else None
        clean_error = safe_serialize(error, mask_pii=self.mask_pii, pii_masker=self.pii_masker) if error is not None else None

        if metadata:
            span.metadata.update(safe_serialize(metadata, mask_pii=self.mask_pii, pii_masker=self.pii_masker))

        if state_diff is not None:
            span.state_diff = safe_serialize(state_diff, mask_pii=self.mask_pii, pii_masker=self.pii_masker)
        elif span.span_type == SpanType.NODE and (span._raw_inputs is not None or span.inputs is not None) and outputs is not None:
            raw_in = span._raw_inputs if span._raw_inputs is not None else span.inputs
            span.state_diff = calculate_state_diff(raw_in, outputs, mask_pii=self.mask_pii, pii_masker=self.pii_masker)

        span.finish(status=status, outputs=clean_outputs, error=clean_error)
        return span

    def apply_node_state_update(self, node_name: str, state_update: Dict[str, Any]) -> None:
        """Record a node's state update into trace metadata for timeline visualization."""
        if "node_state_updates" not in self.trace.metadata:
            self.trace.metadata["node_state_updates"] = []
        updates = self.trace.metadata["node_state_updates"]
        if len(updates) < self.max_spans:
            clean_update = safe_serialize(state_update, mask_pii=self.mask_pii, pii_masker=self.pii_masker)
            updates.append({
                "node": node_name,
                "update": clean_update,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        else:
            self.trace.metadata["node_state_updates_truncated"] = True

    def get_span(self, span_id: str) -> Optional[Span]:
        return self._spans_by_id.get(span_id)

    def set_trace_input(self, input_data: Any) -> None:
        self.trace.input = safe_serialize(input_data, mask_pii=self.mask_pii, pii_masker=self.pii_masker)

    def finish_trace(
        self,
        status: SpanStatus = SpanStatus.COMPLETED,
        output: Optional[Any] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> Trace:
        clean_output = safe_serialize(output, mask_pii=self.mask_pii, pii_masker=self.pii_masker) if output is not None else None
        clean_error = safe_serialize(error, mask_pii=self.mask_pii, pii_masker=self.pii_masker) if error is not None else None
        self.trace.finish(status=status, output=clean_output, error=clean_error)
        self.trace.compute_metrics()
        self.flush()
        return self.trace

    async def finish_trace_async(
        self,
        status: SpanStatus = SpanStatus.COMPLETED,
        output: Optional[Any] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> Trace:
        clean_output = safe_serialize(output, mask_pii=self.mask_pii, pii_masker=self.pii_masker) if output is not None else None
        clean_error = safe_serialize(error, mask_pii=self.mask_pii, pii_masker=self.pii_masker) if error is not None else None
        self.trace.finish(status=status, output=clean_output, error=clean_error)
        self.trace.compute_metrics()
        try:
            await self.storage.save_trace_async(self.trace)
        except Exception as e:
            logger.warning(f"Failed to persist trace {self.trace_id} asynchronously: {e}")
        return self.trace

    def flush(self) -> None:
        """Ship the current trace to storage without blocking."""
        try:
            self.storage.save_trace(self.trace)
        except Exception as e:
            logger.warning(f"Failed to persist trace {self.trace_id}: {e}")

