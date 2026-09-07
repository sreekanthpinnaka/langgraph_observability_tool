from __future__ import annotations

import enum
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, PrivateAttr


def current_iso_time() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class SpanType(str, enum.Enum):
    WORKFLOW = "workflow"
    NODE = "node"
    LLM = "llm"
    TOOL = "tool"
    HTTP = "http"
    CUSTOM = "custom"
    EVAL = "eval"


class SpanStatus(str, enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Span(BaseModel):
    id: str
    trace_id: str
    parent_id: Optional[str] = None
    name: str
    span_type: SpanType = SpanType.CUSTOM
    status: SpanStatus = SpanStatus.RUNNING
    start_time: str = Field(default_factory=current_iso_time)
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    inputs: Optional[Any] = None
    outputs: Optional[Any] = None
    state_diff: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[Dict[str, Any]] = None
    _raw_inputs: Optional[Any] = PrivateAttr(default=None)

    def finish(
        self,
        status: SpanStatus = SpanStatus.COMPLETED,
        outputs: Optional[Any] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.status = status
        self.end_time = current_iso_time()
        if outputs is not None:
            self.outputs = outputs
        if error is not None:
            self.error = error
        try:
            start_dt = datetime.fromisoformat(self.start_time)
            end_dt = datetime.fromisoformat(self.end_time)
            self.duration_ms = round((end_dt - start_dt).total_seconds() * 1000.0, 2)
        except Exception:
            self.duration_ms = 0.0


class HTTPMetadata(BaseModel):
    method: str
    path: str
    url: str
    status_code: Optional[int] = None
    client_ip: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    query_params: Dict[str, str] = Field(default_factory=dict)


from langgraph_observe.core.pricing import calculate_cost


class LLMMetrics(BaseModel):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    total_cost: float = 0.0
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    model_usage: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class Trace(BaseModel):
    id: str
    name: str
    status: SpanStatus = SpanStatus.RUNNING
    environment: str = Field(default="production")
    project: Optional[str] = None
    start_time: str = Field(default_factory=current_iso_time)
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    http: Optional[HTTPMetadata] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    spans: List[Span] = Field(default_factory=list)
    llm_metrics: LLMMetrics = Field(default_factory=LLMMetrics)
    estimated_cost: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    graph_topology: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    has_tool_loop: bool = False
    metrics_computed: bool = Field(default=False, exclude=True)
    _metrics_computed: bool = PrivateAttr(default=False)

    def finish(
        self,
        status: SpanStatus = SpanStatus.COMPLETED,
        output: Optional[Any] = None,
        error: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.end_time is not None:
            if status != SpanStatus.RUNNING:
                self.status = status
            if output is not None:
                self.output = output
            if error is not None:
                self.error = error
            return

        self.status = status
        self.end_time = current_iso_time()
        self._metrics_computed = False
        self.metrics_computed = False
        if output is not None:
            self.output = output
        if error is not None:
            self.error = error
        try:
            start_dt = datetime.fromisoformat(self.start_time)
            end_dt = datetime.fromisoformat(self.end_time)
            self.duration_ms = round((end_dt - start_dt).total_seconds() * 1000.0, 2)
        except Exception:
            self.duration_ms = 0.0

    def compute_metrics(self, force: bool = False) -> None:
        """Aggregate metrics from all spans."""
        if (self._metrics_computed or self.metrics_computed) and not force:
            return
        total_tok = 0
        prompt_tok = 0
        comp_tok = 0
        llm_count = 0
        total_cost = 0.0
        prompt_cost = 0.0
        comp_cost = 0.0
        model_usage: Dict[str, Dict[str, Any]] = {}
        has_failed_span = False

        for span in self.spans:
            if span.status == SpanStatus.FAILED:
                has_failed_span = True
            if span.span_type == SpanType.LLM:
                llm_count += 1
                usage = span.metadata.get("usage", {})
                s_tot = usage.get("total_tokens", 0)
                s_in = usage.get("prompt_tokens", 0)
                s_out = usage.get("completion_tokens", 0)
                if s_tot == 0 and (s_in > 0 or s_out > 0):
                    s_tot = s_in + s_out

                total_tok += s_tot
                prompt_tok += s_in
                comp_tok += s_out

                model = (
                    usage.get("model")
                    or span.metadata.get("model")
                    or span.name.replace("LLM: ", "").replace("Chat: ", "")
                )

                # If cost was already computed on the span, use it; otherwise compute now
                if "total_cost" in usage:
                    s_pc = float(usage.get("prompt_cost", 0.0))
                    s_cc = float(usage.get("completion_cost", 0.0))
                    s_tc = float(usage.get("total_cost", 0.0))
                elif s_tot > 0:
                    cost_info = calculate_cost(model, s_in, s_out)
                    s_pc = cost_info["prompt_cost"]
                    s_cc = cost_info["completion_cost"]
                    s_tc = cost_info["total_cost"]
                    usage["prompt_cost"] = s_pc
                    usage["completion_cost"] = s_cc
                    usage["total_cost"] = s_tc
                    usage["model"] = cost_info["model"]
                    span.metadata["usage"] = usage
                else:
                    s_pc = 0.0
                    s_cc = 0.0
                    s_tc = 0.0

                prompt_cost += s_pc
                comp_cost += s_cc
                total_cost += s_tc

                norm_model = usage.get("model", model)
                if norm_model not in model_usage:
                    model_usage[norm_model] = {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "total_cost": 0.0,
                    }
                model_usage[norm_model]["calls"] += 1
                model_usage[norm_model]["prompt_tokens"] += s_in
                model_usage[norm_model]["completion_tokens"] += s_out
                model_usage[norm_model]["total_tokens"] += s_tot
                model_usage[norm_model]["total_cost"] = round(
                    model_usage[norm_model]["total_cost"] + s_tc, 6
                )

        self.estimated_cost = round(total_cost, 6)
        self.llm_metrics = LLMMetrics(
            total_tokens=total_tok,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            llm_calls=llm_count,
            total_cost=round(total_cost, 6),
            prompt_cost=round(prompt_cost, 6),
            completion_cost=round(comp_cost, 6),
            model_usage=model_usage,
        )
        if has_failed_span and self.status == SpanStatus.COMPLETED:
            self.status = SpanStatus.FAILED

        # Check for tool call loops (Class 3: Tool Call Loop / Infinite Agent Cycles)
        tool_counts: Dict[str, int] = {}
        tool_inputs: Dict[str, List[str]] = {}
        for span in self.spans:
            if span.span_type == SpanType.TOOL:
                t_name = span.name.replace("Tool: ", "").strip()
                tool_counts[t_name] = tool_counts.get(t_name, 0) + 1
                try:
                    inp_str = json.dumps(span.inputs, sort_keys=True) if span.inputs is not None else ""
                except Exception:
                    inp_str = str(span.inputs)
                tool_inputs.setdefault(t_name, []).append(inp_str)

        has_loop = False
        loop_tool = None
        loop_count = 0
        for t_name, count in tool_counts.items():
            inps = tool_inputs.get(t_name, [])
            # If called >= 5 times, or >= 3 times with identical inputs
            has_dups = (len(inps) - len(set(inps))) >= 2 if len(inps) >= 3 else False
            if count >= 5 or has_dups:
                has_loop = True
                loop_tool = t_name
                loop_count = count
                break

        if has_loop:
            self.has_tool_loop = True
            self.metadata["tool_loop_detected"] = True
            self.metadata["tool_loop_tool"] = loop_tool
            self.metadata["tool_loop_count"] = loop_count

        self._metrics_computed = True
        self.metrics_computed = True


class TraceSummary(BaseModel):
    id: str
    name: str
    status: SpanStatus
    environment: str = "production"
    project: Optional[str] = None
    start_time: str
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    http_method: Optional[str] = None
    http_path: Optional[str] = None
    http_status_code: Optional[int] = None
    total_spans: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    estimated_cost: float = 0.0
    has_error: bool = False
    eval_status: Optional[str] = None
    eval_count: int = 0
    has_tool_loop: bool = False

    @classmethod
    def from_trace(cls, trace: Trace) -> TraceSummary:
        return cls(
            id=trace.id,
            name=trace.name,
            status=trace.status,
            environment=trace.environment,
            project=trace.project,
            start_time=trace.start_time,
            end_time=trace.end_time,
            duration_ms=trace.duration_ms,
            http_method=trace.http.method if trace.http else None,
            http_path=trace.http.path if trace.http else None,
            http_status_code=trace.http.status_code if trace.http else None,
            total_spans=len(trace.spans),
            total_tokens=trace.llm_metrics.total_tokens,
            llm_calls=trace.llm_metrics.llm_calls,
            estimated_cost=trace.estimated_cost,
            has_error=(trace.status == SpanStatus.FAILED or trace.error is not None),
            eval_status=trace.metadata.get("eval_status"),
            eval_count=int(trace.metadata.get("eval_count", 0)),
            has_tool_loop=trace.has_tool_loop,
        )


class StatsSummary(BaseModel):
    total_traces: int = 0
    completed_traces: int = 0
    failed_traces: int = 0
    success_rate: float = 0.0
    avg_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    total_tokens: int = 0
    total_llm_calls: int = 0
    total_cost: float = 0.0
    node_execution_counts: Dict[str, int] = Field(default_factory=dict)

