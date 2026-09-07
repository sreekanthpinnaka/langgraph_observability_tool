from __future__ import annotations

import threading
import time
from typing import Any, Dict
from typing_extensions import TypedDict

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.graph import END, START, StateGraph

from langgraph_observe import instrument_fastapi, mount_observability, observe_graph
from langgraph_observe.server import create_server_app
from langgraph_observe.server.storage.memory import MemoryStorage


class ChatState(TypedDict):
    text: str
    response: str


def make_test_app(storage: MemoryStorage) -> FastAPI:
    app = FastAPI()

    builder = StateGraph(ChatState)

    def process_node(state: ChatState) -> Dict[str, Any]:
        return {"response": f"Echo: {state['text']}"}

    builder.add_node("process_node", process_node)
    builder.add_edge(START, "process_node")
    builder.add_edge("process_node", END)
    graph = observe_graph(builder.compile(), name="EchoWorkflow")

    mount_observability(app, path="/observe", storage=storage)

    @app.post("/chat")
    async def chat_endpoint(data: Dict[str, str]):
        res = await graph.ainvoke({"text": data.get("message", ""), "response": ""})
        return res

    return app


def test_fastapi_observability_flow():
    """Verify FastAPI middleware automatically injects trace context and captures HTTP metadata."""
    storage = MemoryStorage()
    app = make_test_app(storage)
    client = TestClient(app)

    # 1. Send request to monitored endpoint
    resp = client.post("/chat", json={"message": "hello observer"})
    assert resp.status_code == 200
    assert resp.json() == {"text": "hello observer", "response": "Echo: hello observer"}

    # 2. Check X-Trace-ID header
    trace_id = resp.headers.get("X-Trace-ID")
    assert trace_id is not None

    # 3. Check Trace in storage
    trace = storage.get_trace(trace_id)
    assert trace is not None
    assert trace.http is not None
    assert trace.http.method == "POST"
    assert trace.http.path == "/chat"
    assert trace.http.status_code == 200

    # 4. Verify spans
    assert len(trace.spans) >= 1
    node_span = next(s for s in trace.spans if s.name == "process_node")
    assert node_span is not None
    assert node_span.outputs == {"response": "Echo: hello observer"}

    # 5. Check UI & API endpoints
    ui_resp = client.get("/observe")
    assert ui_resp.status_code == 200
    assert "LangGraph" in ui_resp.text

    traces_api_resp = client.get("/observe/api/traces")
    assert traces_api_resp.status_code == 200
    assert len(traces_api_resp.json()) == 1


def test_decoupled_fastapi_backend_to_server():
    """Verify instrument_fastapi successfully pipes request traces to an external standalone server."""
    server_storage = MemoryStorage()
    server_app = create_server_app(storage=server_storage)

    test_port = 18777
    config = uvicorn.Config(server_app, host="127.0.0.1", port=test_port, log_level="error")
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True)
    th.start()

    ready = False
    start_wait = time.time()
    while time.time() - start_wait < 4.0:
        try:
            with httpx.Client(timeout=0.2) as c:
                if c.get(f"http://127.0.0.1:{test_port}/health").status_code == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.05)
    assert ready, "Test uvicorn server failed to start within timeout"

    backend_app = FastAPI()
    remote_store = instrument_fastapi(backend_app, server_url=f"http://127.0.0.1:{test_port}")

    class SimpleState(TypedDict):
        text: str
        done: bool

    builder = StateGraph(SimpleState)
    builder.add_node("compute", lambda s: {"done": True})
    builder.add_edge(START, "compute")
    builder.add_edge("compute", END)
    graph = observe_graph(builder.compile(), name="BackendServiceGraph")

    @backend_app.post("/process")
    async def process_endpoint(data: Dict[str, str]):
        return await graph.ainvoke({"text": data.get("text", ""), "done": False})

    backend_client = TestClient(backend_app)
    try:
        resp = backend_client.post("/process", json={"text": "hello from backend"})
        assert resp.status_code == 200
        assert resp.json()["done"] is True
        trace_id = resp.headers.get("X-Trace-ID")
        assert trace_id is not None

        remote_store.flush(timeout=2.0)

        server_trace = server_storage.get_trace(trace_id)
        assert server_trace is not None
        assert server_trace.http is not None
        assert server_trace.http.path == "/process"
        assert len(server_trace.spans) >= 1
    finally:
        remote_store.close()
        server.should_exit = True
        th.join(timeout=1.0)
