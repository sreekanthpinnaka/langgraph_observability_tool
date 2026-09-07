from __future__ import annotations

import asyncio
from typing import Any, Dict
from typing_extensions import TypedDict

import pytest
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from langgraph_observe.client.collector import set_default_storage
from langgraph_observe.client.langgraph import observe_graph
from langgraph_observe.core.models import SpanStatus, SpanType
from langgraph_observe.server.storage import MemoryStorage


class FlowState(TypedDict):
    val: str
    counter: int


def test_sync_langgraph_tracing():
    """Verify synchronous graph execution, node span creation, and state diff tracking."""
    mem_store = MemoryStorage()
    set_default_storage(mem_store)

    builder = StateGraph(FlowState)

    def step1(s: FlowState) -> Dict[str, Any]:
        return {"val": s["val"] + " -> step1", "counter": s["counter"] + 1}

    def step2(s: FlowState) -> Dict[str, Any]:
        return {"val": s["val"] + " -> step2", "counter": s["counter"] + 10}

    builder.add_node("step1", step1)
    builder.add_node("step2", step2)
    builder.add_edge(START, "step1")
    builder.add_edge("step1", "step2")
    builder.add_edge("step2", END)

    raw_graph = builder.compile()
    graph = observe_graph(raw_graph, name="TestSyncGraph")

    out = graph.invoke({"val": "init", "counter": 0})
    assert out["val"] == "init -> step1 -> step2"
    assert out["counter"] == 11

    traces = mem_store.list_traces()
    assert len(traces) == 1
    t = mem_store.get_trace(traces[0].id)
    assert t is not None
    assert t.name == "TestSyncGraph"
    assert t.status == SpanStatus.COMPLETED

    node_spans = [s for s in t.spans if s.span_type == SpanType.NODE]
    assert len(node_spans) == 2
    assert [s.name for s in node_spans] == ["step1", "step2"]

    # Verify state diffs
    step1_span = next(s for s in node_spans if s.name == "step1")
    assert step1_span.state_diff is not None
    assert "val" in step1_span.state_diff
    assert "counter" in step1_span.state_diff


@pytest.mark.asyncio
async def test_async_langgraph_tracing_and_error():
    """Verify asynchronous graph execution and error tracking when a node raises an exception."""
    mem_store = MemoryStorage()
    set_default_storage(mem_store)

    builder = StateGraph(FlowState)

    async def bad_node(s: FlowState) -> Dict[str, Any]:
        raise ValueError("Intentional crash in bad_node")

    builder.add_node("bad_node", bad_node)
    builder.add_edge(START, "bad_node")
    builder.add_edge("bad_node", END)

    graph = observe_graph(builder.compile(), name="FailingAsyncGraph")

    with pytest.raises(ValueError, match="Intentional crash"):
        await graph.ainvoke({"val": "err", "counter": 0})

    traces = mem_store.list_traces()
    assert len(traces) == 1
    t = mem_store.get_trace(traces[0].id)
    assert t is not None
    assert t.status == SpanStatus.FAILED
    assert any(s.status == SpanStatus.FAILED for s in t.spans)


class ParallelFlowState(TypedDict, total=False):
    branch_a: str
    branch_b: str
    result: str


@pytest.mark.asyncio
async def test_parallel_branches_interleaved_children():
    """Verify that parallel branches executing concurrently maintain deterministic parent-child relationships."""
    mem_store = MemoryStorage()
    set_default_storage(mem_store)

    @tool
    def tool_a(q: str) -> str:
        """Tool A"""
        return f"result_a:{q}"

    @tool
    def tool_b(q: str) -> str:
        """Tool B"""
        return f"result_b:{q}"

    builder = StateGraph(ParallelFlowState)

    async def run_branch_a(s: ParallelFlowState) -> Dict[str, Any]:
        await asyncio.sleep(0.02)
        res = tool_a.invoke({"q": "slow"})
        await asyncio.sleep(0.04)
        return {"branch_a": res}

    async def run_branch_b(s: ParallelFlowState) -> Dict[str, Any]:
        res = tool_b.invoke({"q": "fast"})
        await asyncio.sleep(0.01)
        return {"branch_b": res}

    def join_node(s: ParallelFlowState) -> Dict[str, Any]:
        return {"result": f"{s.get('branch_a')} + {s.get('branch_b')}"}

    builder.add_node("branch_a", run_branch_a)
    builder.add_node("branch_b", run_branch_b)
    builder.add_node("join_node", join_node)

    builder.add_edge(START, "branch_a")
    builder.add_edge(START, "branch_b")
    builder.add_edge("branch_a", "join_node")
    builder.add_edge("branch_b", "join_node")
    builder.add_edge("join_node", END)

    graph = observe_graph(builder.compile(), name="ParallelGraph")
    out = await graph.ainvoke({})

    assert "result_a:slow" in out["result"]
    assert "result_b:fast" in out["result"]

    traces = mem_store.list_traces()
    assert len(traces) == 1
    t = mem_store.get_trace(traces[0].id)
    assert t is not None

    spans_by_name = {s.name: s for s in t.spans}
    assert "branch_a" in spans_by_name
    assert "branch_b" in spans_by_name
    assert "Tool: tool_a" in spans_by_name
    assert "Tool: tool_b" in spans_by_name

    root_span = next(s for s in t.spans if s.span_type == SpanType.WORKFLOW)
    branch_a_span = spans_by_name["branch_a"]
    branch_b_span = spans_by_name["branch_b"]
    tool_a_span = spans_by_name["Tool: tool_a"]
    tool_b_span = spans_by_name["Tool: tool_b"]

    # Both parallel branches must be direct children of the root workflow
    assert branch_a_span.parent_id == root_span.id
    assert branch_b_span.parent_id == root_span.id

    # Deterministic parenting: child tools must point to their respective branch
    assert tool_a_span.parent_id == branch_a_span.id
    assert tool_b_span.parent_id == branch_b_span.id


def test_graph_topology_extraction():
    """Verify that graph static structure (nodes, edges, conditional branches) is extracted into trace."""
    mem_store = MemoryStorage()
    set_default_storage(mem_store)

    builder = StateGraph(dict)
    builder.add_node("classify", lambda s: {"route": "fast"})
    builder.add_node("fast_track", lambda s: {"done": True})
    builder.add_node("slow_track", lambda s: {"done": True})

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        lambda s: s.get("route"),
        {"fast": "fast_track", "slow": "slow_track"},
    )
    builder.add_edge("fast_track", END)
    builder.add_edge("slow_track", END)

    graph = observe_graph(builder.compile(), name="ConditionalRouter")
    res = graph.invoke({"input": "test"})
    assert res.get("done") is True

    traces = mem_store.list_traces()
    assert len(traces) == 1
    t = mem_store.get_trace(traces[0].id)
    assert t is not None
    assert t.graph_topology is not None

    node_ids = {n["id"] for n in t.graph_topology["nodes"]}
    assert "__start__" in node_ids
    assert "classify" in node_ids
    assert "fast_track" in node_ids
    assert "slow_track" in node_ids
    assert "__end__" in node_ids

    # Check edges
    edges = t.graph_topology["edges"]
    assert any(e["source"] == "__start__" and e["target"] == "classify" for e in edges)
    # Check conditional edges
    cond_edges = [e for e in edges if e.get("conditional")]
    assert len(cond_edges) >= 2
    assert any(e["source"] == "classify" and e["target"] == "fast_track" for e in cond_edges)
    assert any(e["source"] == "classify" and e["target"] == "slow_track" for e in cond_edges)
