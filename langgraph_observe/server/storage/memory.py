from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from langgraph_observe.core.models import (
    SpanStatus,
    SpanType,
    StatsSummary,
    Trace,
    TraceSummary,
)
from langgraph_observe.server.storage.base import BaseStorage


class MemoryStorage(BaseStorage):
    """Thread-safe in-memory storage for traces and metrics."""

    def __init__(self, max_traces: int = 1000) -> None:
        self._lock = threading.Lock()
        self._traces: Dict[str, Trace] = {}
        self._max_traces = max_traces
        self._stats_cache: Optional[StatsSummary] = None
        self._stats_cache_ts: float = 0.0
        self._stats_cache_ttl: float = 3.0

    def save_trace(self, trace: Trace) -> None:
        trace.compute_metrics()
        with self._lock:
            self._stats_cache = None
            if len(self._traces) >= self._max_traces and trace.id not in self._traces:
                oldest_id = next(iter(self._traces))
                del self._traces[oldest_id]
            self._traces[trace.id] = trace

    def save_traces_batch(self, traces: List[Trace]) -> int:
        count = 0
        with self._lock:
            self._stats_cache = None
            for trace in traces:
                trace.compute_metrics()
                if len(self._traces) >= self._max_traces and trace.id not in self._traces:
                    oldest_id = next(iter(self._traces))
                    del self._traces[oldest_id]
                self._traces[trace.id] = trace
                count += 1
        return count


    def get_trace(self, trace_id: str) -> Optional[Trace]:
        with self._lock:
            trace = self._traces.get(trace_id)
            return trace.model_copy(deep=True) if trace else None

    def _extract_eval_info(self, t: Trace) -> Tuple[Optional[str], int]:
        evals = []
        if isinstance(t.metadata.get("evals"), list):
            evals.extend(t.metadata["evals"])
        for s in t.spans:
            if s.span_type == SpanType.EVAL:
                eval_meta = s.metadata.get("eval")
                if eval_meta and eval_meta not in evals:
                    evals.append(eval_meta)
                elif not eval_meta:
                    evals.append({"name": s.name, "passed": s.status == SpanStatus.COMPLETED})
        eval_count = len(evals)
        eval_status = None
        if eval_count > 0:
            all_passed = all(e.get("passed", True) for e in evals)
            eval_status = "passed" if all_passed else "failed"
        return eval_status, eval_count

    def count_traces(
        self,
        status: Optional[str] = None,
        search: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        eval_status: Optional[str] = None,
    ) -> int:
        with self._lock:
            traces_list = list(self._traces.values())

        count = 0
        for t in traces_list:
            if status and t.status.value.lower() != status.lower():
                continue
            if environment and (t.environment or "production").lower() != environment.lower():
                continue
            if project and (t.project or "").lower() != project.lower():
                continue
            if from_time and t.start_time < from_time:
                continue
            if to_time and t.start_time > to_time:
                continue
            t_eval_status, _ = self._extract_eval_info(t)
            if eval_status and (t_eval_status or "").lower() != eval_status.lower():
                continue
            if search:
                search_lower = search.lower()
                http_path = t.http.path.lower() if t.http and t.http.path else ""
                proj_str = (t.project or "").lower()
                name_match = search_lower in t.name.lower()
                path_match = search_lower in http_path
                id_match = search_lower in t.id.lower()
                proj_match = search_lower in proj_str
                if not (name_match or path_match or id_match or proj_match):
                    continue
            count += 1
        return count

    def list_traces_with_count(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        search: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        eval_status: Optional[str] = None,
    ) -> Tuple[List[TraceSummary], int]:
        with self._lock:
            traces_list = list(self._traces.values())

        traces_list.sort(key=lambda t: t.start_time, reverse=True)

        filtered: List[TraceSummary] = []
        for t in traces_list:
            if status and t.status.value.lower() != status.lower():
                continue
            if environment and (t.environment or "production").lower() != environment.lower():
                continue
            if project and (t.project or "").lower() != project.lower():
                continue
            if from_time and t.start_time < from_time:
                continue
            if to_time and t.start_time > to_time:
                continue
            t_eval_status, t_eval_count = self._extract_eval_info(t)
            if eval_status and (t_eval_status or "").lower() != eval_status.lower():
                continue
            if search:
                search_lower = search.lower()
                http_path = t.http.path.lower() if t.http and t.http.path else ""
                proj_str = (t.project or "").lower()
                name_match = search_lower in t.name.lower()
                path_match = search_lower in http_path
                id_match = search_lower in t.id.lower()
                proj_match = search_lower in proj_str
                if not (name_match or path_match or id_match or proj_match):
                    continue

            summary = TraceSummary(
                id=t.id,
                name=t.name,
                status=t.status,
                start_time=t.start_time,
                end_time=t.end_time,
                duration_ms=t.duration_ms,
                http_method=t.http.method if t.http else None,
                http_path=t.http.path if t.http else None,
                http_status_code=t.http.status_code if t.http else None,
                total_spans=len(t.spans),
                total_tokens=t.llm_metrics.total_tokens,
                llm_calls=t.llm_metrics.llm_calls,
                estimated_cost=float(t.estimated_cost or 0.0),
                has_error=bool(t.error or any(s.status == SpanStatus.FAILED for s in t.spans)),
                environment=t.environment or "production",
                project=t.project,
                eval_status=t_eval_status,
                eval_count=t_eval_count,
                has_tool_loop=getattr(t, "has_tool_loop", False),
            )
            filtered.append(summary)

        total = len(filtered)
        return filtered[offset : offset + limit], total

    def list_traces(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        search: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        eval_status: Optional[str] = None,
    ) -> List[TraceSummary]:
        items, _ = self.list_traces_with_count(
            limit=limit,
            offset=offset,
            status=status,
            search=search,
            environment=environment,
            project=project,
            from_time=from_time,
            to_time=to_time,
            eval_status=eval_status,
        )
        return items

    def get_stats(self) -> StatsSummary:
        now_ts = time.monotonic()
        with self._lock:
            if self._stats_cache is not None and (now_ts - self._stats_cache_ts < self._stats_cache_ttl):
                return self._stats_cache
            traces = list(self._traces.values())

        total = len(traces)
        if total == 0:
            res = StatsSummary()
            with self._lock:
                self._stats_cache = res
                self._stats_cache_ts = now_ts
            return res

        completed = sum(1 for t in traces if t.status == SpanStatus.COMPLETED)
        failed = sum(1 for t in traces if t.status == SpanStatus.FAILED)
        durations = [t.duration_ms for t in traces if t.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        durations_sorted = sorted(durations)
        if durations_sorted:
            idx = int(0.95 * len(durations_sorted))
            p95 = durations_sorted[min(idx, len(durations_sorted) - 1)]
        else:
            p95 = 0.0

        total_tokens = sum(t.llm_metrics.total_tokens for t in traces)
        total_llm_calls = sum(t.llm_metrics.llm_calls for t in traces)
        total_cost = sum(t.estimated_cost for t in traces)

        node_counts: Dict[str, int] = {}
        for t in traces:
            for s in t.spans:
                if s.span_type == SpanType.NODE:
                    node_counts[s.name] = node_counts.get(s.name, 0) + 1

        stats = StatsSummary(
            total_traces=total,
            completed_traces=completed,
            failed_traces=failed,
            success_rate=round((completed / total * 100.0), 1) if total > 0 else 0.0,
            avg_duration_ms=round(avg_duration, 2),
            p95_duration_ms=round(p95, 2),
            total_tokens=total_tokens,
            total_llm_calls=total_llm_calls,
            total_cost=round(total_cost, 6),
            node_execution_counts=node_counts,
        )
        with self._lock:
            self._stats_cache = stats
            self._stats_cache_ts = now_ts
        return stats


    def get_timeseries_analytics(
        self,
        time_window: str = "24h",
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            traces_list = list(self._traces.values())

        now_dt = datetime.now(timezone.utc)
        if from_time:
            try:
                start_dt = datetime.fromisoformat(from_time.replace("Z", "+00:00"))
            except Exception:
                start_dt = now_dt - timedelta(hours=24)
        else:
            window_lower = (time_window or "24h").lower()
            if window_lower == "15m":
                start_dt = now_dt - timedelta(minutes=15)
            elif window_lower == "1h":
                start_dt = now_dt - timedelta(hours=1)
            elif window_lower == "7d":
                start_dt = now_dt - timedelta(days=7)
            elif window_lower == "30d":
                start_dt = now_dt - timedelta(days=30)
            else:
                start_dt = now_dt - timedelta(hours=24)

        end_dt = now_dt
        if to_time:
            try:
                end_dt = datetime.fromisoformat(to_time.replace("Z", "+00:00"))
            except Exception:
                end_dt = now_dt

        total_seconds = max(60, int((end_dt - start_dt).total_seconds()))

        if total_seconds <= 15 * 60:
            num_buckets = 15
            bucket_sec = 60
        elif total_seconds <= 3600:
            num_buckets = 12
            bucket_sec = 300
        elif total_seconds <= 86400:
            num_buckets = 24
            bucket_sec = 3600
        elif total_seconds <= 7 * 86400:
            num_buckets = 14
            bucket_sec = 12 * 3600
        else:
            num_buckets = 30
            bucket_sec = 86400

        # Filter traces
        matching_traces = []
        for t in traces_list:
            if environment and (t.environment or "production").lower() != environment.lower():
                continue
            if project and (t.project or "").lower() != project.lower():
                continue
            if t.start_time < start_dt.isoformat() or t.start_time > end_dt.isoformat():
                continue
            matching_traces.append(t)

        buckets = []
        for i in range(num_buckets):
            b_start = start_dt + timedelta(seconds=i * bucket_sec)
            buckets.append({
                "timestamp": b_start.isoformat(),
                "trace_count": 0,
                "error_count": 0,
                "error_rate": 0.0,
                "durations": [],
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "models": {},
            })

        for t in matching_traces:
            try:
                row_dt = datetime.fromisoformat(t.start_time.replace("Z", "+00:00"))
                offset_sec = (row_dt - start_dt).total_seconds()
                b_idx = int(offset_sec // bucket_sec)
                if 0 <= b_idx < num_buckets:
                    b = buckets[b_idx]
                    b["trace_count"] += 1
                    if t.status == SpanStatus.FAILED or t.error or any(s.status == SpanStatus.FAILED for s in t.spans):
                        b["error_count"] += 1
                    if t.duration_ms is not None:
                        b["durations"].append(float(t.duration_ms))
                    b["total_cost"] += float(t.estimated_cost or 0.0)
                    b["total_tokens"] += int(t.llm_metrics.total_tokens or 0)

                    if t.llm_metrics and t.llm_metrics.model_usage:
                        for m_name, m_data in t.llm_metrics.model_usage.items():
                            calls = m_data.get("calls", 1) if isinstance(m_data, dict) else 1
                            b["models"][m_name] = b["models"].get(m_name, 0) + calls
                    elif t.spans:
                        for sp in t.spans:
                            if sp.span_type == SpanType.LLM:
                                m = (sp.metadata or {}).get("model") or sp.name.replace("LLM: ", "").replace("Chat: ", "")
                                if m:
                                    b["models"][m] = b["models"].get(m, 0) + 1
            except Exception:
                continue

        for b in buckets:
            tc = b["trace_count"]
            b["total_requests"] = tc
            b["success_count"] = max(0, tc - b["error_count"])
            b["error_rate"] = round((b["error_count"] / tc * 100.0), 1) if tc > 0 else 0.0
            durs = sorted(b.pop("durations"))
            if durs:
                avg_val = round(sum(durs) / len(durs), 2)
                p95_idx = min(len(durs) - 1, max(0, int(math.ceil(0.95 * len(durs))) - 1))
                p95_val = round(durs[p95_idx], 2)
                if p95_val < avg_val:
                    p95_val = avg_val
                b["avg_latency_ms"] = avg_val
                b["avg_duration_ms"] = avg_val
                b["p95_latency_ms"] = p95_val
                b["p95_duration_ms"] = p95_val
            else:
                b["avg_duration_ms"] = 0.0
                b["p95_duration_ms"] = 0.0
            b["total_cost"] = round(b["total_cost"], 4)
            try:
                b["label"] = datetime.fromisoformat(b["timestamp"]).strftime("%H:%M")
            except Exception:
                b["label"] = ""

        return {
            "time_window": time_window,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "total_traces": len(matching_traces),
            "buckets": buckets,
        }

    def get_node_analytics(
        self,
        time_window: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            traces_list = list(self._traces.values())

        if time_window:
            now_dt = datetime.now(timezone.utc)
            window_lower = time_window.lower()
            if window_lower == "15m":
                s_dt = now_dt - timedelta(minutes=15)
            elif window_lower == "1h":
                s_dt = now_dt - timedelta(hours=1)
            elif window_lower == "7d":
                s_dt = now_dt - timedelta(days=7)
            else:
                s_dt = now_dt - timedelta(hours=24)
            traces_list = [t for t in traces_list if t.start_time >= s_dt.isoformat()]

        if environment:
            traces_list = [t for t in traces_list if (t.environment or "production").lower() == environment.lower()]
        if project:
            traces_list = [t for t in traces_list if (t.project or "").lower() == project.lower()]

        total_traces = len(traces_list)
        if total_traces == 0:
            return {
                "time_window": time_window,
                "total_traces": 0,
                "nodes": [],
            }

        node_stats: Dict[str, Dict[str, Any]] = {}

        for t in traces_list:
            visited_in_trace: Set[str] = set()
            for s in t.spans:
                if s.span_type == SpanType.NODE and s.name:
                    n_name = s.name
                    visited_in_trace.add(n_name)
                    if n_name not in node_stats:
                        node_stats[n_name] = {
                            "node_name": n_name,
                            "execution_count": 0,
                            "traces_executed": 0,
                            "error_count": 0,
                            "durations": [],
                        }
                    ns = node_stats[n_name]
                    ns["execution_count"] += 1
                    if s.status == SpanStatus.FAILED:
                        ns["error_count"] += 1
                    if s.duration_ms is not None:
                        ns["durations"].append(float(s.duration_ms))

            for n_name in visited_in_trace:
                if n_name in node_stats:
                    node_stats[n_name]["traces_executed"] += 1

        result = []
        for n_name, ns in node_stats.items():
            exec_count = ns["execution_count"]
            durs = sorted(ns.pop("durations"))
            avg_dur = round(sum(durs) / len(durs), 2) if durs else 0.0
            p95_dur = round(durs[int(0.95 * (len(durs) - 1))], 2) if durs else 0.0
            min_dur = round(durs[0], 2) if durs else 0.0
            max_dur = round(durs[-1], 2) if durs else 0.0
            traces_exec = ns["traces_executed"]
            freq_pct = round((traces_exec / total_traces * 100.0), 1) if total_traces > 0 else 0.0
            err_count = ns["error_count"]
            err_rate = round((err_count / exec_count * 100.0), 1) if exec_count > 0 else 0.0

            result.append({
                "node_name": n_name,
                "executions": exec_count,
                "execution_count": exec_count,
                "traces_executed": traces_exec,
                "frequency_pct": freq_pct,
                "error_count": err_count,
                "error_rate": err_rate,
                "avg_duration_ms": avg_dur,
                "p95_duration_ms": p95_dur,
                "min_duration_ms": min_dur,
                "max_duration_ms": max_dur,
            })

        result.sort(key=lambda x: x["frequency_pct"], reverse=True)
        return {
            "time_window": time_window,
            "total_traces": total_traces,
            "nodes": result,
        }

    def delete_trace(self, trace_id: str) -> bool:
        with self._lock:
            self._stats_cache = None
            return self._traces.pop(trace_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._stats_cache = None
            self._traces.clear()

