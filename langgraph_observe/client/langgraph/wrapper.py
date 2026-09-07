from __future__ import annotations

import functools
import inspect
import logging
import random
from typing import Any, Callable, Dict, Optional, TypeVar

from langgraph_observe.client.collector import TraceCollector, get_default_storage
from langgraph_observe.client.langgraph.callback import LangGraphTraceCallbackHandler
from langgraph_observe.core.context import (
    get_current_collector,
    reset_current_collector,
    reset_current_trace,
    set_current_collector,
    set_current_trace,
)
from langgraph_observe.core.masking import PIIMasker
from langgraph_observe.core.models import SpanStatus
from langgraph_observe.server.storage.base import BaseStorage

logger = logging.getLogger("langgraph_observe.wrapper")

F = TypeVar("F", bound=Callable[..., Any])



def _extract_graph_topology(graph: Any) -> Optional[Dict[str, Any]]:
    """Inspect a compiled LangGraph and extract its node & edge topology."""
    try:
        nodes = []
        edges = []

        # Priority 1: Inspect graph.builder (StateGraph builder definition)
        builder = getattr(graph, "builder", None)
        if builder and hasattr(builder, "nodes") and hasattr(builder, "edges"):
            node_ids = set()
            for nid in builder.nodes.keys():
                nodes.append({"id": str(nid), "name": str(nid)})
                node_ids.add(str(nid))

            for src, dst in builder.edges:
                edges.append({
                    "source": str(src),
                    "target": str(dst),
                    "conditional": False,
                    "data": None,
                })
                if str(src) not in node_ids and str(src) in ("__start__", "__end__", "START", "END"):
                    nodes.append({"id": str(src), "name": str(src)})
                    node_ids.add(str(src))
                if str(dst) not in node_ids and str(dst) in ("__start__", "__end__", "START", "END"):
                    nodes.append({"id": str(dst), "name": str(dst)})
                    node_ids.add(str(dst))

            branches = getattr(builder, "branches", {}) or {}
            for src, b_dict in branches.items():
                items = b_dict.items() if isinstance(b_dict, dict) else [("", b_dict)]
                for _, b_spec in items:
                    ends = getattr(b_spec, "ends", None)
                    if isinstance(ends, dict):
                        for cond_val, target in ends.items():
                            edges.append({
                                "source": str(src),
                                "target": str(target),
                                "conditional": True,
                                "data": str(cond_val),
                            })
                    elif isinstance(ends, (list, tuple, set)):
                        for target in ends:
                            edges.append({
                                "source": str(src),
                                "target": str(target),
                                "conditional": True,
                                "data": None,
                            })

            if not any(n["id"] in ("__start__", "START") for n in nodes):
                nodes.insert(0, {"id": "__start__", "name": "__start__"})
            if not any(n["id"] in ("__end__", "END") for n in nodes):
                nodes.append({"id": "__end__", "name": "__end__"})

            return {"nodes": nodes, "edges": edges}

        # Priority 2: graph.get_graph() fallback
        if hasattr(graph, "get_graph"):
            gd = graph.get_graph()
            nodes = [{"id": str(n.id), "name": str(getattr(n, "name", n.id))} for n in gd.nodes.values()]
            edges = [{
                "source": str(e.source),
                "target": str(e.target),
                "conditional": bool(getattr(e, "conditional", False)),
                "data": str(getattr(e, "data", "")) if getattr(e, "data", None) else None,
            } for e in gd.edges]
            return {"nodes": nodes, "edges": edges}
    except Exception as e:
        logger.debug(f"Topology extraction skipped: {e}")
    return None


class ObservedGraph:
    """Transparent proxy wrapper around a compiled LangGraph (Pregel / CompiledStateGraph)
    that automatically attaches tracing callbacks to invoke, ainvoke, stream, and astream.
    """

    def __init__(
        self,
        graph: Any,
        name: Optional[str] = None,
        storage: Optional[BaseStorage] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
        mask_pii: bool = False,
        pii_masker: Optional[PIIMasker] = None,
        sample_rate: float = 1.0,
    ) -> None:
        self._graph = graph
        self._name = name or getattr(graph, "name", "LangGraph")
        self._storage = storage
        self._environment = environment
        self._project = project
        self._mask_pii = mask_pii
        self._pii_masker = pii_masker
        self._sample_rate = sample_rate
        self._topology = _extract_graph_topology(graph)

    def _create_collector(self) -> TraceCollector:
        return TraceCollector(
            name=self._name,
            storage=self._storage or get_default_storage(),
            environment=self._environment,
            project=self._project,
            mask_pii=self._mask_pii,
            pii_masker=self._pii_masker,
        )

    def _attach_topology(self, collector: TraceCollector) -> None:
        if self._topology and not collector.trace.graph_topology:
            collector.trace.graph_topology = self._topology

    def _prepare_config(
        self, config: Optional[Dict[str, Any]], collector: TraceCollector
    ) -> Dict[str, Any]:
        cfg = dict(config or {})
        callbacks = list(cfg.get("callbacks") or [])
        # Check if already attached
        if not any(isinstance(cb, LangGraphTraceCallbackHandler) for cb in callbacks):
            callbacks.append(
                LangGraphTraceCallbackHandler(
                    collector=collector,
                    workflow_name=self._name,
                )
            )
        cfg["callbacks"] = callbacks
        return cfg

    def invoke(
        self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        ctx_collector = get_current_collector()
        is_root = ctx_collector is None
        if is_root and self._sample_rate < 1.0 and random.random() > self._sample_rate:
            return self._graph.invoke(input, config=config, **kwargs)

        collector = ctx_collector or self._create_collector()
        self._attach_topology(collector)

        token_c = set_current_collector(collector)
        token_t = set_current_trace(collector.trace)
        cfg = self._prepare_config(config, collector)

        try:
            result = self._graph.invoke(input, config=cfg, **kwargs)
            if is_root:
                collector.finish_trace(status=SpanStatus.COMPLETED, output=result)
            return result
        except Exception as e:
            if is_root:
                collector.finish_trace(
                    status=SpanStatus.FAILED,
                    error={"type": type(e).__name__, "message": str(e)},
                )
            raise
        finally:
            reset_current_collector(token_c)
            reset_current_trace(token_t)

    async def ainvoke(
        self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        ctx_collector = get_current_collector()
        is_root = ctx_collector is None
        if is_root and self._sample_rate < 1.0 and random.random() > self._sample_rate:
            return await self._graph.ainvoke(input, config=config, **kwargs)

        collector = ctx_collector or self._create_collector()
        self._attach_topology(collector)

        token_c = set_current_collector(collector)
        token_t = set_current_trace(collector.trace)
        cfg = self._prepare_config(config, collector)

        try:
            result = await self._graph.ainvoke(input, config=cfg, **kwargs)
            if is_root:
                collector.finish_trace(status=SpanStatus.COMPLETED, output=result)
            return result
        except Exception as e:
            if is_root:
                collector.finish_trace(
                    status=SpanStatus.FAILED,
                    error={"type": type(e).__name__, "message": str(e)},
                )
            raise
        finally:
            reset_current_collector(token_c)
            reset_current_trace(token_t)

    def stream(
        self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        ctx_collector = get_current_collector()
        is_root = ctx_collector is None
        if is_root and self._sample_rate < 1.0 and random.random() > self._sample_rate:
            for item in self._graph.stream(input, config=config, **kwargs):
                yield item
            return

        collector = ctx_collector or self._create_collector()
        self._attach_topology(collector)

        token_c = set_current_collector(collector)
        token_t = set_current_trace(collector.trace)
        cfg = self._prepare_config(config, collector)

        try:
            for item in self._graph.stream(input, config=cfg, **kwargs):
                yield item
            if is_root:
                collector.finish_trace(status=SpanStatus.COMPLETED)
        except Exception as e:
            if is_root:
                collector.finish_trace(
                    status=SpanStatus.FAILED,
                    error={"type": type(e).__name__, "message": str(e)},
                )
            raise
        finally:
            reset_current_collector(token_c)
            reset_current_trace(token_t)

    async def astream(
        self, input: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        ctx_collector = get_current_collector()
        is_root = ctx_collector is None
        if is_root and self._sample_rate < 1.0 and random.random() > self._sample_rate:
            async for item in self._graph.astream(input, config=config, **kwargs):
                yield item
            return

        collector = ctx_collector or self._create_collector()
        self._attach_topology(collector)

        token_c = set_current_collector(collector)
        token_t = set_current_trace(collector.trace)
        cfg = self._prepare_config(config, collector)

        try:
            async for item in self._graph.astream(input, config=cfg, **kwargs):
                yield item
            if is_root:
                collector.finish_trace(status=SpanStatus.COMPLETED)
        except Exception as e:
            if is_root:
                collector.finish_trace(
                    status=SpanStatus.FAILED,
                    error={"type": type(e).__name__, "message": str(e)},
                )
            raise
        finally:
            reset_current_collector(token_c)
            reset_current_trace(token_t)

    def __getattr__(self, name: str) -> Any:
        """Forward any other attributes (e.g. get_state, update_state, nodes, get_graph) to underlying graph."""
        return getattr(self._graph, name)


def observe_graph(
    graph: Any,
    name: Optional[str] = None,
    storage: Optional[BaseStorage] = None,
    environment: Optional[str] = None,
    project: Optional[str] = None,
    mask_pii: bool = False,
    pii_masker: Optional[PIIMasker] = None,
    sample_rate: float = 1.0,
) -> ObservedGraph:
    """Wrap a compiled LangGraph workflow to automatically trace every execution.

    Example:
        graph = builder.compile()
        observed_graph = observe_graph(graph, name="ChatAssistant", environment="production", mask_pii=True)
        response = await observed_graph.ainvoke({"messages": [...]})
    """
    return ObservedGraph(
        graph,
        name=name,
        storage=storage,
        environment=environment,
        project=project,
        mask_pii=mask_pii,
        pii_masker=pii_masker,
        sample_rate=sample_rate,
    )


def observe_workflow(
    name: Optional[str] = None,
    storage: Optional[BaseStorage] = None,
    environment: Optional[str] = None,
    project: Optional[str] = None,
    mask_pii: bool = False,
    pii_masker: Optional[PIIMasker] = None,
    sample_rate: float = 1.0,
) -> Callable[[F], F]:
    """Decorator for functions or route handlers that execute LangGraph workflows.

    Example:
        @app.post("/chat")
        @observe_workflow(name="customer_support", environment="production", mask_pii=True)
        async def chat_endpoint(req: ChatRequest):
            return await graph.ainvoke(...)
    """
    def decorator(fn: F) -> F:
        wf_name = name or fn.__name__

        def _make_collector() -> TraceCollector:
            return TraceCollector(
                name=wf_name,
                storage=storage or get_default_storage(),
                environment=environment,
                project=project,
                mask_pii=mask_pii,
                pii_masker=pii_masker,
            )

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx_collector = get_current_collector()
                is_root = ctx_collector is None
                if is_root and sample_rate < 1.0 and random.random() > sample_rate:
                    return await fn(*args, **kwargs)

                collector = ctx_collector or _make_collector()

                token_c = set_current_collector(collector)
                token_t = set_current_trace(collector.trace)

                try:
                    result = await fn(*args, **kwargs)
                    if is_root:
                        collector.finish_trace(status=SpanStatus.COMPLETED, output=result)
                    return result
                except Exception as e:
                    if is_root:
                        collector.finish_trace(
                            status=SpanStatus.FAILED,
                            error={"type": type(e).__name__, "message": str(e)},
                        )
                    raise
                finally:
                    reset_current_collector(token_c)
                    reset_current_trace(token_t)

            return async_wrapper  # type: ignore
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx_collector = get_current_collector()
                is_root = ctx_collector is None
                if is_root and sample_rate < 1.0 and random.random() > sample_rate:
                    return fn(*args, **kwargs)

                collector = ctx_collector or _make_collector()

                token_c = set_current_collector(collector)
                token_t = set_current_trace(collector.trace)

                try:
                    result = fn(*args, **kwargs)
                    if is_root:
                        collector.finish_trace(status=SpanStatus.COMPLETED, output=result)
                    return result
                except Exception as e:
                    if is_root:
                        collector.finish_trace(
                            status=SpanStatus.FAILED,
                            error={"type": type(e).__name__, "message": str(e)},
                        )
                    raise
                finally:
                    reset_current_collector(token_c)
                    reset_current_trace(token_t)

            return sync_wrapper  # type: ignore

    return decorator
