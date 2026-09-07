# 🔭 langgraph-observe

> **Enterprise-grade, lightweight, and decoupled observability server, interactive dark-mode dashboard, and universal client SDK for LangGraph workflows in ANY FastAPI backend.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-FF4F00.svg?logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![Tests Passing](https://img.shields.io/badge/tests-101%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📑 Table of Contents

- [🏛️ Architecture: Completely Decoupled](#️-architecture-completely-decoupled)
- [🌟 Key Features](#-key-features)
- [⚡ Quickstart (2 Minutes)](#-quickstart-2-minutes)
  - [Step 1: Start the Observability Server](#step-1-start-the-observability-server)
  - [Step 2: Instrument Your FastAPI Application](#step-2-instrument-your-fastapi-application)
- [📦 Client SDK Installation & Setup](#-client-sdk-installation--setup)
- [🛠️ Detailed Code Examples](#️-detailed-code-examples)
  - [Example 1: Standard FastAPI Workflow](#example-1-standard-fastapi-workflow)
  - [Example 2: Streaming LangGraph Workflows](#example-2-streaming-langgraph-workflows)
  - [Example 3: Function Decorator (@observe_workflow)](#example-3-function-decorator-observe_workflow)
- [📖 API & Parameter Reference](#-api--parameter-reference)
  - [instrument_fastapi() Parameters](#instrument_fastapi-parameters)
  - [observe_graph() Parameters](#observe_graph-parameters)
  - [@observe_workflow() Parameters](#observe_workflow-parameters)
- [🖱️ Standalone Executable (.exe) for Windows](#️-standalone-executable-exe-for-windows)
  - [One-Click Launch (No Python Required)](#one-click-launch-no-python-required)
  - [Rebuilding the Executable](#rebuilding-the-executable)
- [🗄️ Database & Storage Configuration](#️-database--storage-configuration)
  - [SQLite (Default / Local Dev)](#sqlite-default--local-dev)
  - [Google Cloud SQL / MySQL (Production)](#google-cloud-sql--mysql-production)
- [🔒 Security & Authentication](#-security--authentication)
- [🌐 Environment Variables](#-environment-variables)
- [📡 Server REST APIs](#-server-rest-apis)
- [🧪 Running the Test Suite](#-running-the-test-suite)
- [📂 Project Structure](#-project-structure)
- [📄 License](#-license)

---

## 🏛️ Architecture: Completely Decoupled

`langgraph-observe` is built with a **microservices-first, decoupled architecture**. Your core FastAPI applications stay lean, fast, and free of database/UI clutter:

```
┌────────────────────────────────────────────────────────┐
│               ANY FastAPI Backend (Anywhere)           │
│    (Localhost, Docker, Kubernetes, AWS, Cloud Run)     │
│                                                        │
│  FastAPI Backend (e.g. Port 8000)                      │
│    ├── instrument_fastapi(app, server_url=...)         │
│    ├── workflow = observe_graph(compiled_graph)        │
│    └── Non-Blocking Async Queue (zero user latency)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                 Async HTTP POST /api/v1/traces
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│     Standalone Observability Server (Port 8765)        │
│     (Run via Python CLI OR standalone Windows .exe)    │
│                                                        │
│  ├── Ingestion API:  POST /api/v1/traces               │
│  ├── Live Dashboard: GET  / (Interactive Dark-Mode UI) │
│  ├── Query APIs:     GET  /api/v1/traces, /stats       │
│  └── Pluggable DB:   SQLite (WAL mode) or Cloud SQL    │
└──────────────────────────┬─────────────────────────────┘
                           │
             http://localhost:8765/
                           │
                           ▼
           👨‍💻 Engineering Team / Live Monitoring
```

- **Zero Performance Penalty**: Traces are queued in memory and dispatched asynchronously in the background.
- **Fault-Tolerant**: If the observability server is offline, restarting, or unreachable, client applications continue functioning normally without blocking or throwing errors.
- **Distributed Ready**: Multiple backend instances across multiple servers or containers can send traces to one centralized observability dashboard.

---

## 🌟 Key Features

| Capability | Description |
|---|---|
| 🧬 **Automatic State Diffs** | Automatically inspects LangGraph state before and after each node executes, highlighting added, updated, and appended values. |
| 📊 **Waterfall Gantt Timeline** | Visual breakdown displaying exact execution timing across incoming HTTP requests, workflow nodes, tool executions, and LLM calls. |
| 🗺️ **Visual Graph Topology** | Automatically parses your compiled graph structure and renders an interactive node/edge DAG diagram inside the UI. |
| 🤖 **LLM & Tool Observability** | Inspect model parameters, prompts, chat completions, tool arguments, and output payloads. |
| 💰 **Token Tracking & Cost Estimation** | Automatic token counts and dollar-cost calculations across OpenAI, Anthropic Claude, Google Gemini, and Llama models. |
| 🛡️ **Built-in PII Redaction** | Client-side scrubbing of API keys, emails, credit cards, SSNs, phone numbers, and custom sensitive dictionary keys. |
| 🖱️ **One-Click Windows Executable** | Packaged standalone `langgraph-observe.exe` allowing anyone to run the server on Windows without installing Python. |
| 🗄️ **Dual Storage Support** | High-speed SQLite with WAL mode out-of-the-box, or Google Cloud SQL (MySQL) with automatic reconnection and connection pooling. |

---

## ⚡ Quickstart (2 Minutes)

### Step 1: Start the Observability Server

Choose **either** the Python CLI or the Standalone Windows Executable:

#### Option A: Via Python CLI
```bash
# Start server on default port 8765
py -m langgraph_observe server

# Or with custom port
py -m langgraph_observe server --port 8765
```

#### Option B: Via Windows One-Click Executable
If you are on Windows, simply navigate to `dist/` and double-click:
```
dist/langgraph-observe.exe
```
*(No Python installation required! See [Standalone Executable](#️-standalone-executable-exe-for-windows) for details).*

Once started, open your browser at:
👉 **`http://localhost:8765/`**

---

### Step 2: Instrument Your FastAPI Application

In any existing or new FastAPI backend:

```python
from fastapi import FastAPI
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Import observability utilities
from langgraph_observe import instrument_fastapi, observe_graph

# 2. Create FastAPI app and attach observability
app = FastAPI(title="Payment & Order Service")
instrument_fastapi(app, server_url="http://localhost:8765")

# 3. Define any standard LangGraph workflow
class WorkflowState(TypedDict):
    order_id: str
    status: str

builder = StateGraph(WorkflowState)

def process_order(state: WorkflowState) -> dict:
    return {"status": f"Order {state['order_id']} confirmed"}

builder.add_node("process", process_order)
builder.add_edge(START, "process")
builder.add_edge("process", END)

# 4. Wrap compiled workflow
workflow = observe_graph(builder.compile(), name="OrderWorkflow")

# 5. Call in your endpoint
@app.post("/checkout")
async def checkout(payload: dict):
    result = await workflow.ainvoke({"order_id": payload["order_id"], "status": "pending"})
    return result
```

Start your backend:
```bash
py -m uvicorn main:app --port 8000
```

Send a test request:
```bash
curl -X POST http://localhost:8000/checkout -H "Content-Type: application/json" -d "{\"order_id\": \"ORD-9912\"}"
```

**That's it!** The execution trace, node latencies, and state diffs will immediately appear on your live dashboard at `http://localhost:8765/`.

---

## 📦 Client SDK Installation & Setup

You can install `langgraph-observe` in your client applications using whichever method fits your setup:

### Method 1: Install from pre-built Wheel (`.whl`)
```bash
pip install dist/langgraph_observe-0.1.0-py3-none-any.whl
```

### Method 2: Install from source directory
```bash
pip install /path/to/observe
```

### Method 3: In `requirements.txt`
```text
# Include in your application's requirements.txt
./libs/langgraph_observe-0.1.0-py3-none-any.whl
```

---

## 🛠️ Detailed Code Examples

### Example 1: Standard FastAPI Workflow

```python
from fastapi import FastAPI
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph_observe import instrument_fastapi, observe_graph

app = FastAPI()

# Instrument with production metadata and PII scrubbing
instrument_fastapi(
    app,
    server_url="http://localhost:8765",
    environment="production",
    project="recommendation-engine",
    mask_pii=True,
    sample_rate=1.0,
)

class State(TypedDict):
    user_query: str
    validated: bool
    response: str

def validate_node(state: State) -> dict:
    return {"validated": bool(state["user_query"].strip())}

def generate_node(state: State) -> dict:
    return {"response": f"Results for: {state['user_query']}"}

builder = StateGraph(State)
builder.add_node("validate", validate_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "validate")
builder.add_edge("validate", "generate")
builder.add_edge("generate", END)

# Wrap with descriptive workflow name
recommendation_graph = observe_graph(builder.compile(), name="RecommendationGraph")

@app.post("/recommend")
async def recommend(data: dict):
    return await recommendation_graph.ainvoke({
        "user_query": data.get("query", ""),
        "validated": False,
        "response": ""
    })
```

---

### Example 2: Streaming LangGraph Workflows

`observe_graph` natively supports streaming via `.stream()` and `.astream()`:

```python
from fastapi.responses import StreamingResponse
import json

@app.post("/chat/stream")
async def stream_chat(data: dict):
    async def event_generator():
        # Traces are captured seamlessly during streaming
        async for chunk in recommendation_graph.astream({"user_query": data.get("query")}):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

### Example 3: Function Decorator (`@observe_workflow`)

For arbitrary functions, jobs, or Celery/background tasks that don't use FastAPI:

```python
from langgraph_observe import observe_workflow

@observe_workflow(name="nightly_batch_job", environment="production")
async def process_batch(items: list):
    # Any LangGraph invocations inside are automatically correlated and traced
    results = []
    for item in items:
        res = await recommendation_graph.ainvoke({"user_query": item})
        results.append(res)
    return results
```

---

## 📖 API & Parameter Reference

### `instrument_fastapi()` Parameters

Call this function once when setting up your FastAPI application:

```python
instrument_fastapi(app: FastAPI, **kwargs) -> RemoteStorage
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `app` | `FastAPI` | **Required** | The target FastAPI application instance. |
| `server_url` | `str` | `None` | URL of the observability server (e.g. `http://localhost:8765`). If `None`, reads from `OBSERVE_SERVER_URL` in `.env` / environment. |
| `api_key` | `str` | `None` | Optional API key matching the server's `OBSERVE_API_KEY` for secure trace ingestion. |
| `environment` | `str` | `None` | Environment tag (e.g. `production`, `staging`, `development`). Displayed in dashboard filters. |
| `project` | `str` | `None` | Project/service name (e.g. `auth-service`, `billing-agent`). Displayed in dashboard filters. |
| `mask_pii` | `bool` | `False` | When `True`, automatically masks emails, credit cards, SSNs, API keys, and sensitive dictionary fields before sending. |
| `sample_rate` | `float` | `1.0` | Sampling rate between `0.0` (0%) and `1.0` (100%). Useful for high-throughput backends. |
| `exclude_paths` | `List[str]` | `None` | URL paths to exclude from HTTP tracing. By default, `/docs`, `/openapi.json`, and `/favicon.ico` are excluded. |
| `trace_id_header`| `str` | `"X-Trace-ID"` | Response header name injected into client responses for distributed correlation. |

---

### `observe_graph()` Parameters

Wrap any compiled LangGraph workflow:

```python
observe_graph(graph: Any, **kwargs) -> ObservedGraph
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `graph` | `CompiledGraph` | **Required** | Compiled LangGraph instance (`builder.compile()`). |
| `name` | `str` | `None` | Custom identifier for the workflow shown in the dashboard. Defaults to the graph class name. |
| `storage` | `BaseStorage`| `None` | Explicit storage target. Defaults to the global storage configured by `instrument_fastapi`. |
| `environment` | `str` | `None` | Environment tag override for this specific workflow. |
| `project` | `str` | `None` | Project name override for this specific workflow. |
| `mask_pii` | `bool` | `False` | Enable PII masking for state, inputs, and outputs of this workflow. |
| `sample_rate` | `float` | `1.0` | Sampling rate override (`0.0` to `1.0`). |

---

### `@observe_workflow()` Parameters

Decorator for custom functions executing LangGraph steps:

```python
@observe_workflow(name="MyFunction", environment="production", mask_pii=True)
async def my_function():
    ...
```

Accepts `name`, `storage`, `environment`, `project`, `mask_pii`, and `sample_rate`.

---

## 🖱️ Standalone Executable (.exe) for Windows

For developers, QA, or clients on Windows who do not want to install Python, virtual environments, or manage dependencies, `langgraph-observe` includes a **standalone single-file executable**.

### One-Click Launch (No Python Required)

1. Open the folder:
   ```
   dist/
   ├── langgraph-observe.exe   (Self-contained ~49MB binary)
   └── .env                    (Configuration file)
   ```
2. **Double-click `langgraph-observe.exe`**:
   - A console window opens displaying the current database configuration.
   - **Database Prompt**: Press <kbd>Enter</kbd> to keep the current database, or paste a new Database URL and press <kbd>Enter</kbd>.
   - **Auto-Launch**: The server starts and **automatically opens your web browser** to `http://127.0.0.1:8765/`.
   - **Exit Protection**: If stopped or if an error occurs, the window remains open with `Press Enter to exit...` so errors are never lost.

### Rebuilding the Executable

If you make modifications to the codebase and wish to recompile the `.exe`:
```bash
py build_exe.py
```
This runs PyInstaller with all required hidden imports and asset bundles, generating a clean binary in `dist/langgraph-observe.exe`.

---

## 🗄️ Database & Storage Configuration

`langgraph-observe` supports two primary persistence options configured via `.env` or the `--db-url` CLI flag.

### SQLite (Default / Local Dev)

- Zero configuration needed.
- Traces are stored in a local SQLite database (`observe.db`).
- **WAL Mode Enabled**: Configured with Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and indexed queries for fast concurrent reads and writes.

---

### Google Cloud SQL / MySQL (Production)

For shared team dashboards or high-volume horizontal scaling, configure a MySQL or Google Cloud SQL database.

#### 1. In `.env`:
```ini
# Google Cloud SQL / MySQL format:
DATABASE_URL="mysql+pymysql://<DB_USER>:<DB_PASSWORD>@<HOST_OR_PUBLIC_IP>:3306/<DB_NAME>"
```

#### 2. Production Optimizations Built-in:
- **Connection Pre-Ping (`pool_pre_ping=True`)**: Verifies connection liveness before queries to eliminate broken pipe errors.
- **Automatic Connection Recycling (`pool_recycle=1800`)**: Recycles idle connections every 30 minutes, preventing Cloud SQL disconnects.
- **Auto-Migration**: Schema tables, JSON columns, and indexes on `status`, `start_time`, and `name` are created automatically on first run.

---

## 🔒 Security & Authentication

Protect your observability server and trace ingestion using an API key:

1. **Server Configuration**: Set `OBSERVE_API_KEY` in the server's `.env`:
   ```ini
   OBSERVE_API_KEY=your-secure-secret-key-123
   ```
2. **Client Configuration**: Supply the same key in your client apps:
   ```python
   instrument_fastapi(app, server_url="http://localhost:8765", api_key="your-secure-secret-key-123")
   ```
3. **Payload Size Enforcement**: Ingestion endpoints automatically enforce a 10MB request body limit to prevent memory exhaustion (configurable via `OBSERVE_MAX_BODY_SIZE`).
4. **CORS Control**: Restrict allowed origins using `OBSERVE_CORS_ORIGINS="https://mycompany.internal"`.

---

## 🌐 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | SQLAlchemy connection string (MySQL, PostgreSQL, etc.) | `None` (uses SQLite `observe.db`) |
| `OBSERVE_SERVER_URL` | Observability server URL for client SDKs | `http://localhost:8765` |
| `OBSERVE_API_KEY` | Secret token required for trace ingestion and queries | `None` (open access) |
| `OBSERVE_CORS_ORIGINS`| Comma-separated list of allowed CORS domains | `*` (allow all) |
| `OBSERVE_MAX_BODY_SIZE`| Maximum request body size in bytes | `10485760` (10 MB) |

---

## 📡 Server REST APIs

The server exposes standard REST APIs for custom dashboard integration or CI/CD export:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Live interactive web dashboard (HTML) |
| `GET` | `/health` | Server health check (`{"status": "healthy"}`) |
| `POST` | `/api/v1/traces` | Ingest single trace payload |
| `POST` | `/api/v1/traces/batch` | Ingest a batch of trace payloads |
| `GET` | `/api/v1/traces` | Query traces (supports `limit`, `offset`, `status`, `search`, `project`, `environment`) |
| `GET` | `/api/v1/traces/{id}` | Retrieve complete trace details (spans, topology, state diffs) |
| `GET` | `/api/v1/traces-count` | Get total count matching search/filter criteria |
| `GET` | `/api/v1/stats` | Aggregate performance statistics and node execution breakdown |
| `GET` | `/api/v1/export/{id}` | Download complete trace as a formatted JSON file |
| `DELETE` | `/api/v1/traces/{id}` | Delete a specific trace |
| `DELETE` | `/api/v1/traces` | Clear all trace history |

---

## 🧪 Running the Test Suite

`langgraph-observe` includes a comprehensive test suite covering middleware, graph wrappers, state serialization, PII masking, token pricing, storage backends, and CLI commands:

```bash
py -m pytest -v
```

Expected output:
```text
======================= 82 passed, 2 warnings in 7.91s ========================
```

---

## 📂 Project Structure

```
observe/
├── langgraph_observe/
│   ├── core/                     # Shared domain models & utilities
│   │   ├── models.py             # Trace, Span, SpanType, HTTPMetadata schemas
│   │   ├── serializer.py         # Safe recursion serialization & state diffing
│   │   ├── masking.py            # PII masking & regex scrubbing engine
│   │   ├── pricing.py            # Model token pricing & USD estimation
│   │   └── context.py            # Async-safe task-local ContextVar manager
│   │
│   ├── server/                   # Standalone Observability Server & Dashboard
│   │   ├── app.py                # FastAPI ingestion server, security & REST APIs
│   │   ├── cli.py                # CLI runner (langgraph-observe server)
│   │   ├── storage/              # Pluggable persistence layer
│   │   │   ├── base.py           # BaseStorage interface
│   │   │   ├── sql.py            # SQLAlchemyStorage (Cloud SQL MySQL & generic SQL)
│   │   │   ├── sqlite.py         # High-speed SQLite (WAL mode, indexed queries)
│   │   │   └── memory.py         # Ephemeral memory storage (testing)
│   │   └── ui/                   # Embedded Web Dashboard
│   │       ├── __init__.py       # HTML template loader (PyInstaller frozen safe)
│   │       └── index.html        # Interactive dark-mode dashboard UI
│   │
│   └── client/                   # Lightweight Tracing SDK for ANY FastAPI Backend
│       ├── instrument.py         # instrument_fastapi(app, server_url)
│       ├── middleware.py         # LangGraphObserveMiddleware (Starlette HTTP interceptor)
│       ├── collector.py          # TraceCollector aggregator
│       ├── exporter.py           # Non-blocking async background exporter
│       └── langgraph/            # LangGraph integration
│           ├── callback.py       # LangGraphTraceCallbackHandler (tree run_id tracking)
│           └── wrapper.py        # observe_graph wrapper & @observe_workflow decorator
│
├── dist/                         # Distribution artifacts
│   ├── langgraph-observe.exe     # Standalone Windows executable
│   ├── langgraph_observe-*.whl   # Universal pip package wheel
│   ├── SETUP_GUIDE.html          # Interactive standalone setup guide
│   └── .env                      # Configuration file template
│
├── tests/                        # 82 comprehensive unit & integration tests
├── build_exe.py                  # PyInstaller build script
├── SETUP_GUIDE.html              # Standalone interactive HTML setup guide
├── pyproject.toml                # Project packaging specification
└── README.md                     # Project documentation
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.
