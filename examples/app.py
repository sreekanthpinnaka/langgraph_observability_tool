from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from langgraph_observe import (
    SpanType,
    mount_observability,
    observe_graph,
    observe_workflow,
    trace_span,
)


# 1. Define LangGraph Workflow State
class WorkflowState(TypedDict):
    query: str
    category: Optional[str]
    retrieved_docs: List[str]
    summary: Optional[str]
    review_status: Optional[str]
    score: float


# 2. Define Workflow Nodes
def classify_intent(state: WorkflowState) -> Dict[str, Any]:
    query = state["query"].lower()
    if "error" in query or "bug" in query:
        category = "technical_support"
    elif "price" in query or "billing" in query:
        category = "billing"
    else:
        category = "general_inquiry"

    return {"category": category, "score": 0.95}


def retrieve_context(state: WorkflowState) -> Dict[str, Any]:
    category = state.get("category")
    mock_docs = [
        f"Knowledge base document for {category} - Policy A",
        f"Standard operating procedure for {category} - SOP 101",
    ]
    return {"retrieved_docs": mock_docs}


def generate_summary(state: WorkflowState) -> Dict[str, Any]:
    category = state.get("category", "unknown")
    docs_count = len(state.get("retrieved_docs", []))
    prompt = f"Summarize user query '{state['query']}' under category '{category}' with {docs_count} docs"

    with trace_span(
        name="LLM: gpt-4o-mini",
        span_type=SpanType.LLM,
        inputs={"prompt": prompt},
        metadata={
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 340,
                "completion_tokens": 120,
                "total_tokens": 460,
            },
        },
    ) as span:
        summary = (
            f"Processed query '{state['query']}' under category '{category}'. "
            f"Consulted {docs_count} reference docs."
        )
        if span:
            span.outputs = {"summary": summary}

    return {"summary": summary}


def review_guardrail(state: WorkflowState) -> Dict[str, Any]:
    with trace_span(
        name="LLM: claude-3-5-haiku",
        span_type=SpanType.LLM,
        inputs={"summary": state.get("summary")},
        metadata={
            "model": "claude-3-5-haiku",
            "usage": {
                "prompt_tokens": 195,
                "completion_tokens": 35,
                "total_tokens": 230,
            },
        },
    ) as span:
        review_status = "APPROVED"
        if span:
            span.outputs = {"review_status": review_status}

    return {"review_status": review_status}


def route_review(state: WorkflowState) -> str:
    if state.get("category") == "technical_support":
        return "review_guardrail"
    return END


# 3. Build & Compile LangGraph
builder = StateGraph(WorkflowState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("generate_summary", generate_summary)
builder.add_node("review_guardrail", review_guardrail)

builder.add_edge(START, "classify_intent")
builder.add_edge("classify_intent", "retrieve_context")
builder.add_edge("retrieve_context", "generate_summary")
builder.add_conditional_edges("generate_summary", route_review, ["review_guardrail", END])
builder.add_edge("review_guardrail", END)

raw_graph = builder.compile()

# 4. Wrap with observe_graph
workflow_graph = observe_graph(raw_graph, name="CustomerSupportAssistant")


# 5. Build an error-generating graph to demonstrate failure tracing
class FailState(TypedDict):
    input_text: str


def failing_node(state: FailState) -> Dict[str, Any]:
    raise RuntimeError("Simulated internal node failure in step 'compute_vector_embedding'!")


fail_builder = StateGraph(FailState)
fail_builder.add_node("compute_vector_embedding", failing_node)
fail_builder.add_edge(START, "compute_vector_embedding")
fail_builder.add_edge("compute_vector_embedding", END)
failing_graph = observe_graph(fail_builder.compile(), name="FaultyWorkflow")


# 6. Setup FastAPI App & Mount Observability
app = FastAPI(
    title="LangGraph Observability Demo",
    description="Sample FastAPI application instrumented with langgraph-observe.",
)

# One-line instrumentation: mounts dashboard at /observe, API at /observe/api, and adds middleware
mount_observability(app, path="/observe")


class QueryRequest(BaseModel):
    query: str


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to the observability dashboard."""
    return RedirectResponse(url="/observe")


@app.post("/api/run")
async def run_workflow_endpoint(request: QueryRequest) -> Dict[str, Any]:
    """Execute the multi-step LangGraph customer support workflow."""
    initial_state: WorkflowState = {
        "query": request.query,
        "category": None,
        "retrieved_docs": [],
        "summary": None,
        "review_status": None,
        "score": 0.0,
    }
    result = await workflow_graph.ainvoke(initial_state)
    return {"status": "success", "result": result}


@app.post("/api/fail")
async def trigger_failure_endpoint() -> Dict[str, Any]:
    """Trigger a workflow that deliberately fails to test error observability."""
    return await failing_graph.ainvoke({"input_text": "trigger error"})


if __name__ == "__main__":
    import uvicorn

    print("\n🚀 Starting LangGraph Observability Demo Server...")
    print("👉 Dashboard: http://localhost:8000/observe")
    print("👉 API Docs:  http://localhost:8000/docs\n")
    uvicorn.run("examples.app:app", host="127.0.0.1", port=8000, reload=True)
