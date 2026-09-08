# langgraph-observe

A self-hosted dashboard and tracing tool for LangGraph agents running in FastAPI.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Tests Passing](https://img.shields.io/badge/tests-104%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## Why this exists (The Purpose)

When you build multi-step agents or stateful workflows with **LangGraph**, diagnosing what happened during a run is difficult:
- **State changes are opaque**: You cannot easily see which node modified what data in the shared state.
- **Tools can loop**: An agent might repeatedly call the same tool with the same arguments, burning tokens without making progress.
- **Costs are hidden**: Calculating how much each request cost across different models (OpenAI, Anthropic, Gemini, DeepSeek) requires manual math.
- **Existing tools are heavy or cloud-locked**: Services like LangSmith require sending your data to third-party servers and paying monthly subscriptions. Traditional APM tools (like Datadog) don't understand LangGraph states or node transitions.

**`langgraph-observe` gives you a local, private dashboard to see inside your agent runs.** It runs on your own machine (or private server), captures traces in the background without slowing down your API, and lets you inspect state, timing, and errors step-by-step.

---

## How it works

The tool has two parts:

1. **The Server (Dashboard)**: A lightweight local web app that collects traces and displays them at `http://localhost:8765`.
2. **The Client SDK**: A few lines of code you add to your FastAPI application. It records workflow runs and sends them to the dashboard asynchronously. If the dashboard is ever offline, your application continues running normally without errors.

```
Your FastAPI App (Port 8000)
  ├── Receives user request
  ├── Runs your LangGraph agent
  └── Sends trace in background (non-blocking)
            │
            ▼
Dashboard Server (Port 8765)
  ├── Stores traces in local SQLite (or MySQL)
  └── Shows graphs, state diffs, tokens, and errors in your browser
```

---

## What you can see in the Dashboard

- **State Inspector (Before vs. After)**: See exactly what changed in the state after each node runs—highlighting added keys, updated values, and appended chat messages.
- **Interactive Graph Diagram**: Automatically draws your compiled LangGraph structure (nodes, edges, and conditional routing) using Mermaid.js.
- **Timing Timeline**: A waterfall chart showing how long each step took—from the initial HTTP request down to individual LLM completions and tool calls.
- **Token & Cost Estimation**: Automatically counts prompt and completion tokens and estimates dollar cost for models from OpenAI, Anthropic, Google Gemini, DeepSeek, and Meta Llama.
- **Infinite Loop Warnings**: Detects when an agent repeatedly invokes the same tool (e.g. 5+ times or identical inputs) and highlights it on the timeline.
- **PII & Secret Scrubbing**: Strips out API keys (`sk-...`), Bearer tokens, passwords, emails, and credit cards client-side before traces are saved.
- **Search & Filter**: Filter past runs by status (Completed, Failed), project name, environment, date, or search query.

---

## Quickstart

### Step 1: Start the Dashboard Server

You can run the server using Python:

```bash
python -m langgraph_observe server
```

*(On Windows, you can also just double-click `dist/langgraph-observe.exe`—no Python installation needed).*

Open your browser at: **`http://localhost:8765/`**

---

### Step 2: Add it to your FastAPI app

Install the package:
```bash
pip install dist/langgraph_observe-0.1.0-py3-none-any.whl
# or: pip install -e .
```

Wrap your graph and app:

```python
from fastapi import FastAPI
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Import langgraph-observe
from langgraph_observe import instrument_fastapi, observe_graph

app = FastAPI()

# 2. Attach observability (points to the dashboard server)
instrument_fastapi(app, server_url="http://localhost:8765")

# 3. Define your LangGraph workflow
class AgentState(TypedDict):
    query: str
    result: str

builder = StateGraph(AgentState)

def process_node(state: AgentState) -> dict:
    return {"result": f"Answer for: {state['query']}"}

builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)

# 4. Wrap your compiled graph
workflow = observe_graph(builder.compile(), name="SearchAgent")

# 5. Use it in your endpoints
@app.post("/ask")
async def ask(payload: dict):
    return await workflow.ainvoke({"query": payload["query"], "result": ""})
```

Start your FastAPI app as usual:
```bash
uvicorn main:app --port 8000
```

When you send requests to `/ask`, the run, node timings, state changes, and token counts will immediately show up in your dashboard.

---

## Other Usage Examples

### Streaming responses

If your FastAPI endpoint streams tokens or chunks with `astream()`:

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/chat/stream")
async def stream_chat(payload: dict):
    async def event_generator():
        async for chunk in workflow.astream({"query": payload["query"]}):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

Traces are collected seamlessly during streaming.

---

### Non-FastAPI functions (Background jobs, Celery, scripts)

You can trace any async function using the `@observe_workflow` decorator:

```python
from langgraph_observe import observe_workflow

@observe_workflow(name="nightly_batch_job")
async def run_batch(items: list):
    for item in items:
        await workflow.ainvoke({"query": item})
```

---

### Running inside an existing FastAPI app (Embedded Mode)

If you don't want a separate server and prefer viewing the dashboard directly inside your existing FastAPI service (e.g. at `/observe`):

```python
from fastapi import FastAPI
from langgraph_observe import mount_observability, observe_graph

app = FastAPI()

# Mount dashboard at /observe
mount_observability(app, path="/observe")
```

---

## Configuration Options

### `instrument_fastapi()` settings

```python
instrument_fastapi(
    app,
    server_url="http://localhost:8765",  # Dashboard server address
    environment="production",            # Tag runs (e.g. dev, staging, prod)
    project="customer-support",          # Group runs by project
    mask_pii=True,                       # Redact emails, keys, and cards
    sample_rate=1.0,                     # 1.0 = 100% of traces, 0.5 = 50%
)
```

### Database storage

By default, traces are stored in a local SQLite file (`observe.db`) using Write-Ahead Logging (WAL) for speed.

If you have multiple workers or want a shared database, you can set `DATABASE_URL` in a `.env` file or pass `--db-url`:

```ini
# MySQL / Google Cloud SQL
DATABASE_URL="mysql+pymysql://user:password@host:3306/observe_db"
```

Connection pooling, pre-ping checks, and automatic reconnects are handled automatically.

---

## Standalone Windows Executable

If team members or QA testers want to run the dashboard without installing Python or setting up virtual environments:

1. Go to `dist/`
2. Double-click `langgraph-observe.exe`

It starts the server and automatically opens `http://localhost:8765` in the default web browser.

To rebuild the executable from source:
```bash
python build_exe.py
```

---

## Running Tests

The test suite covers callback tree mapping, concurrent async execution, state serialization, PII redaction, pricing models, storage engines, and server APIs:

```bash
pytest
```

```text
====================== 101 passed, 2 warnings in 21s =======================
```

---

## Project Structure

```
langgraph_observe/
├── core/                  # Core logic & utilities
│   ├── models.py          # Trace, Span, and metrics data structures
│   ├── serializer.py      # State diffing & cyclic serialization
│   ├── masking.py         # PII & API key regex scrubber
│   ├── pricing.py         # Token cost calculator for OpenAI, Anthropic, Gemini, etc.
│   └── context.py         # Task-local ContextVar tracking
│
├── client/                # Python client SDK for FastAPI
│   ├── instrument.py      # instrument_fastapi() entry point
│   ├── middleware.py      # Starlette HTTP middleware
│   ├── collector.py       # Span aggregator
│   ├── exporter.py        # Background non-blocking HTTP exporter
│   └── langgraph/         # Callback handler & graph wrapper
│
└── server/                # Observability dashboard server
    ├── app.py             # FastAPI ingestion & query REST endpoints
    ├── cli.py             # CLI runner (`langgraph-observe server`)
    ├── storage/           # SQLite and MySQL/PostgreSQL backends
    └── ui/                # Embedded single-page dashboard
```

---

## License

MIT License. Free for commercial and personal use.
