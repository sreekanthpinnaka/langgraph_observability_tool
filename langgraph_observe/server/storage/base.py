from __future__ import annotations

import abc
from typing import List, Optional, Tuple
from langgraph_observe.core.models import StatsSummary, Trace, TraceSummary


class BaseStorage(abc.ABC):
    """Abstract base class for trace storage backends."""

    @abc.abstractmethod
    def save_trace(self, trace: Trace) -> None:
        """Save or update a trace synchronously."""
        pass

    def save_traces_batch(self, traces: List[Trace]) -> int:
        """Save a batch of traces. Subclasses should override for atomic single-transaction batching."""
        count = 0
        for t in traces:
            try:
                self.save_trace(t)
                count += 1
            except Exception:
                pass
        return count

    async def save_trace_async(self, trace: Trace) -> None:
        """Save or update a trace asynchronously (dispatches to worker thread pool)."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.save_trace, trace)
        except RuntimeError:
            self.save_trace(trace)

    @abc.abstractmethod
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Retrieve a full trace by its ID."""
        pass

    @abc.abstractmethod
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
        """Retrieve a paginated list of trace summaries with optional filters."""
        pass

    @abc.abstractmethod
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
        """Count total matching traces for pagination. Subclasses must provide an efficient count."""
        pass

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
        """Retrieve paginated trace summaries alongside total matching count."""
        items = self.list_traces(
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
        total = self.count_traces(
            status=status,
            search=search,
            environment=environment,
            project=project,
            from_time=from_time,
            to_time=to_time,
            eval_status=eval_status,
        )
        return items, total

    @abc.abstractmethod
    def get_stats(self) -> StatsSummary:
        """Retrieve aggregated metrics and performance statistics."""
        pass

    @abc.abstractmethod
    def get_timeseries_analytics(
        self,
        time_window: str = "24h",
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve bucketed time-series analytics for requests, errors, latency, cost, and models."""
        pass

    @abc.abstractmethod
    def get_node_analytics(
        self,
        time_window: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve aggregate node execution metrics (frequency %, P95 latency, errors) across traces."""
        pass

    def get_dashboard_data(
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
    ) -> Dict[str, Any]:
        """Retrieve stats and paginated trace list in a single combined payload."""
        traces, total = self.list_traces_with_count(
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
        stats = self.get_stats()
        return {
            "stats": stats,
            "traces": traces,
            "total_count": total,
            "has_more": (offset + len(traces)) < total,
        }

    @abc.abstractmethod
    def delete_trace(self, trace_id: str) -> bool:
        """Delete a single trace by ID."""
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all stored traces."""
        pass

