from __future__ import annotations

import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from langgraph_observe.client.collector import TraceCollector
from langgraph_observe.client.exporter import RemoteStorage
from langgraph_observe.core.models import HTTPMetadata, Span, SpanStatus, SpanType, Trace
from langgraph_observe.server.app import create_server_app
from langgraph_observe.server.storage.memory import MemoryStorage


def test_server_standalone_endpoints():
    """Verify health checks, UI dashboard rendering, and standard single/batch ingestion endpoints."""
    server_storage = MemoryStorage()
    server_app = create_server_app(storage=server_storage)
    client = TestClient(server_app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # 2. UI Dashboard
    res_ui = client.get("/")
    assert res_ui.status_code == 200
    assert "LangGraph" in res_ui.text

    # 3. Ingest trace via POST /api/v1/traces
    sample_trace = Trace(
        id="remote-1",
        name="test_remote_wf",
        http=HTTPMetadata(method="POST", path="/api/v1/execute", url="http://localhost:8000/api/v1/execute"),
    )
    span = Span(id="sp-1", trace_id="remote-1", name="node_remote", span_type=SpanType.NODE)
    span.finish(status=SpanStatus.COMPLETED)
    sample_trace.spans.append(span)
    sample_trace.finish(status=SpanStatus.COMPLETED)

    ingest_res = client.post("/api/v1/traces", json=sample_trace.model_dump(mode="json"))
    assert ingest_res.status_code == 201
    assert ingest_res.json()["status"] == "ok"

    # 4. Ingest batch via POST /api/v1/traces/batch
    batch_trace = Trace(id="remote-2", name="test_batch_wf")
    batch_trace.finish(status=SpanStatus.COMPLETED)
    batch_res = client.post("/api/v1/traces/batch", json=[batch_trace.model_dump(mode="json")])
    assert batch_res.status_code == 201
    assert batch_res.json()["count"] == 1

    # 5. Query traces & stats
    list_res = client.get("/api/v1/traces")
    assert list_res.status_code == 200
    assert len(list_res.json()) == 2

    stats_res = client.get("/api/v1/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_traces"] == 2
    assert stats["completed_traces"] == 2


def test_server_batch_ingestion():
    """Verify atomic multi-trace batch ingestion via POST /api/v1/traces/batch."""
    storage = MemoryStorage()
    app = create_server_app(storage=storage)
    client = TestClient(app)

    traces = []
    for i in range(5):
        t = Trace(id=f"t-batch-{i}", name=f"batch-trace-{i}")
        t.spans.append(Span(id=f"s-batch-{i}", trace_id=t.id, name=f"step-{i}", span_type=SpanType.NODE))
        t.finish(SpanStatus.COMPLETED)
        traces.append(t)

    batch_payload = [t.model_dump() for t in traces]
    resp = client.post("/api/v1/traces/batch", json=batch_payload)
    assert resp.status_code == 201
    assert resp.json()["status"] == "ok"
    assert resp.json()["count"] == 5
    assert len(storage.list_traces()) == 5


@pytest.mark.asyncio
async def test_server_async_trace_persistence():
    """Verify async trace saving across collector and memory storage."""
    store = MemoryStorage()
    collector = TraceCollector(name="async-test", storage=store)
    s = collector.start_span(name="step1", span_type=SpanType.NODE)
    collector.finish_span(s.id)

    # Test finish_trace_async
    await collector.finish_trace_async()
    assert store.get_trace(collector.trace.id) is not None

    # Test storage.save_trace_async
    t2 = Trace(id="t-async-2", name="async-store-test")
    t2.finish(SpanStatus.COMPLETED)
    await store.save_trace_async(t2)
    assert store.get_trace(t2.id) is not None


def test_server_pagination_and_cache_control():
    """Verify X-Total-Count, X-Has-More pagination headers and Cache-Control headers."""
    mem_store = MemoryStorage()
    for i in range(15):
        t = Trace(id=f"t-page-{i}", name=f"trace-{i}")
        t.finish(SpanStatus.COMPLETED)
        mem_store.save_trace(t)

    app = create_server_app(storage=mem_store)
    client = TestClient(app)

    # 1. Dashboard HTML cache header
    dash_resp = client.get("/")
    assert dash_resp.status_code == 200
    assert "public, max-age=300" in dash_resp.headers.get("Cache-Control", "")

    # 2. Trace list pagination headers and no-store cache control
    resp = client.get("/api/v1/traces?limit=5&offset=0")
    assert resp.status_code == 200
    assert resp.headers.get("X-Total-Count") == "15"
    assert resp.headers.get("X-Has-More") == "true"
    assert "no-store" in resp.headers.get("Cache-Control", "")
    assert len(resp.json()) == 5

    # Offset 10, limit 5 -> last page
    resp_last = client.get("/api/v1/traces?limit=5&offset=10")
    assert resp_last.status_code == 200
    assert resp_last.headers.get("X-Total-Count") == "15"
    assert resp_last.headers.get("X-Has-More") == "false"

    # 3. Stats endpoint Cache-Control
    stats_resp = client.get("/api/v1/stats")
    assert stats_resp.status_code == 200
    assert "no-store" in stats_resp.headers.get("Cache-Control", "")


def test_server_traces_count_endpoint():
    """Verify lightweight /api/v1/traces-count endpoint returns exact counts with filtering."""
    mem_store = MemoryStorage()
    for i in range(7):
        t = Trace(id=f"t-cnt-{i}", name=f"trace-{i}", project="alpha" if i < 4 else "beta")
        t.finish(SpanStatus.COMPLETED)
        mem_store.save_trace(t)

    app = create_server_app(storage=mem_store)
    client = TestClient(app)

    # 1. Test endpoint directly
    resp = client.get("/api/v1/traces-count")
    assert resp.status_code == 200
    assert resp.json() == {"count": 7}

    # Filtered count
    resp_filtered = client.get("/api/v1/traces-count?project=alpha")
    assert resp_filtered.status_code == 200
    assert resp_filtered.json() == {"count": 4}

    # 2. Test RemoteStorage count_traces using test client
    rs = RemoteStorage(server_url="http://testserver", client=client)
    assert rs.count_traces() == 7
    assert rs.count_traces(project="beta") == 3
    rs.close()


def test_server_cors_origins_configuration():
    """Verify OBSERVE_CORS_ORIGINS environment variable is parsed and injected into CORSMiddleware."""
    with patch.dict(os.environ, {"OBSERVE_CORS_ORIGINS": "https://myapp.internal, https://dashboard.org"}):
        app = create_server_app(storage=MemoryStorage())
        cors_mw = [m for m in app.user_middleware if "CORSMiddleware" in str(m.cls)]
        assert len(cors_mw) == 1
        options = getattr(cors_mw[0], "kwargs", getattr(cors_mw[0], "options", {}))
        assert "https://myapp.internal" in options["allow_origins"]


def test_server_health_check_storage_probe():
    """Verify GET /health probes storage liveness, returning 200 when healthy and 503 when degraded."""
    storage = MemoryStorage()
    app = create_server_app(storage=storage)
    client = TestClient(app)

    # 1. Healthy storage
    res_ok = client.get("/health")
    assert res_ok.status_code == 200
    data = res_ok.json()
    assert data["status"] == "healthy"
    assert data["storage"] == "connected"
    assert "traces_count" in data

    # 2. Degraded storage (count_traces raises)
    with patch.object(storage, "count_traces", side_effect=RuntimeError("Database connection lost")):
        res_degraded = client.get("/health")
        assert res_degraded.status_code == 503
        err_data = res_degraded.json()
        assert err_data["status"] == "degraded"
        assert err_data["storage"] == "unreachable"
        assert "Database connection lost" in err_data["error"]


def test_timeseries_analytics_endpoint():
    """Verify GET /api/v1/analytics/timeseries returns bucketed metrics and model tracking."""
    import time
    from datetime import datetime, timezone
    from langgraph_observe.core.models import LLMMetrics

    storage = MemoryStorage()
    now = time.time()

    # Trace 1: Success with gpt-4o
    t1 = Trace(
        id="t-ts-1",
        name="trace-1",
        start_time=datetime.fromtimestamp(now - 3600, tz=timezone.utc).isoformat(),
        duration_ms=100.0,
    )
    t1.spans.append(
        Span(
            id="s-ts-1",
            trace_id=t1.id,
            name="Chat: gpt-4o",
            span_type=SpanType.LLM,
            metadata={"usage": {"model": "gpt-4o", "prompt_tokens": 100, "completion_tokens": 50, "total_cost": 0.002}},
        )
    )
    t1.finish(SpanStatus.COMPLETED)
    storage.save_trace(t1)

    # Trace 2: Failed with gpt-4o-mini
    t2 = Trace(
        id="t-ts-2",
        name="trace-2",
        start_time=datetime.fromtimestamp(now - 1800, tz=timezone.utc).isoformat(),
        duration_ms=250.0,
    )
    t2.spans.append(
        Span(
            id="s-ts-2",
            trace_id=t2.id,
            name="Chat: gpt-4o-mini",
            span_type=SpanType.LLM,
            metadata={"usage": {"model": "gpt-4o-mini", "prompt_tokens": 50, "completion_tokens": 25, "total_cost": 0.0005}},
        )
    )
    t2.finish(SpanStatus.FAILED)
    storage.save_trace(t2)

    app = create_server_app(storage=storage)
    client = TestClient(app)

    res = client.get("/api/v1/analytics/timeseries?time_window=24h")
    assert res.status_code == 200
    data = res.json()

    assert data["time_window"] == "24h"
    assert data["total_traces"] == 2
    assert len(data["buckets"]) > 0

    total_reqs = sum(b["total_requests"] for b in data["buckets"])
    total_errs = sum(b["error_count"] for b in data["buckets"])
    assert total_reqs == 2
    assert total_errs == 1

    # Check model usage is tracked in buckets
    all_models = {}
    for b in data["buckets"]:
        for m, cnt in b.get("models", {}).items():
            all_models[m] = all_models.get(m, 0) + cnt
    assert all_models.get("gpt-4o") == 1
    assert all_models.get("gpt-4o-mini") == 1


def test_node_analytics_endpoint():
    """Verify GET /api/v1/analytics/nodes returns cross-trace frequency and P95 latency."""
    import time
    from datetime import datetime, timezone

    storage = MemoryStorage()
    now = time.time()

    # Trace 1: Executes 'retrieve' (100ms) and 'generate' (400ms, completed)
    t1 = Trace(
        id="t-node-1",
        name="qa-flow",
        start_time=datetime.fromtimestamp(now - 100, tz=timezone.utc).isoformat(),
    )
    s1 = Span(id="s-1", trace_id=t1.id, name="retrieve", span_type=SpanType.NODE)
    s1.finish(SpanStatus.COMPLETED)
    s1.duration_ms = 100.0
    s2 = Span(id="s-2", trace_id=t1.id, name="generate", span_type=SpanType.NODE)
    s2.finish(SpanStatus.COMPLETED)
    s2.duration_ms = 400.0
    t1.spans.extend([s1, s2])
    t1.finish(SpanStatus.COMPLETED)
    storage.save_trace(t1)

    # Trace 2: Executes only 'retrieve' (200ms, failed)
    t2 = Trace(
        id="t-node-2",
        name="qa-flow",
        start_time=datetime.fromtimestamp(now - 50, tz=timezone.utc).isoformat(),
    )
    s3 = Span(id="s-3", trace_id=t2.id, name="retrieve", span_type=SpanType.NODE)
    s3.finish(SpanStatus.FAILED)
    s3.duration_ms = 200.0
    t2.spans.append(s3)
    t2.finish(SpanStatus.FAILED)
    storage.save_trace(t2)

    app = create_server_app(storage=storage)
    client = TestClient(app)

    res = client.get("/api/v1/analytics/nodes?time_window=24h")
    assert res.status_code == 200
    data = res.json()

    assert data["time_window"] == "24h"
    assert data["total_traces"] == 2
    assert len(data["nodes"]) == 2

    # Node map
    node_map = {n["node_name"]: n for n in data["nodes"]}

    # 'retrieve' ran in 2/2 traces -> frequency_pct = 100.0%
    assert node_map["retrieve"]["executions"] == 2
    assert node_map["retrieve"]["frequency_pct"] == 100.0
    assert node_map["retrieve"]["error_count"] == 1
    assert node_map["retrieve"]["error_rate"] == 50.0

    # 'generate' ran in 1/2 traces -> frequency_pct = 50.0%
    assert node_map["generate"]["executions"] == 1
    assert node_map["generate"]["frequency_pct"] == 50.0
    assert node_map["generate"]["p95_duration_ms"] == 400.0
    assert node_map["generate"]["error_count"] == 0


def test_server_etag_304():
    """Verify SHA-256 ETag generation and HTTP 304 Not Modified responses on UI endpoints."""
    storage = MemoryStorage()
    app = create_server_app(storage=storage)
    client = TestClient(app)

    # Initial GET /
    res1 = client.get("/")
    assert res1.status_code == 200
    etag = res1.headers.get("ETag")
    assert etag is not None
    assert etag.startswith('"') and etag.endswith('"')

    # Conditional GET with matching If-None-Match
    res2 = client.get("/", headers={"If-None-Match": etag})
    assert res2.status_code == 304
    assert res2.headers.get("ETag") == etag

    # Conditional GET with non-matching If-None-Match
    res3 = client.get("/", headers={"If-None-Match": '"stale-etag"'})
    assert res3.status_code == 200

    # Verify mounted /observe endpoint
    res_obs = client.get("/observe")
    assert res_obs.status_code == 200
    assert res_obs.headers.get("ETag") == etag

    res_obs_304 = client.get("/observe", headers={"If-None-Match": etag})
    assert res_obs_304.status_code == 304


def test_server_dashboard_combined_endpoint():
    """Verify single-flight combined /api/v1/dashboard and /api/dashboard endpoints."""
    storage = MemoryStorage()
    for i in range(12):
        t = Trace(id=f"dash-trace-{i}", name=f"workflow-{i}")
        status = SpanStatus.COMPLETED if i % 2 == 0 else SpanStatus.FAILED
        t.finish(status)
        storage.save_trace(t)

    app = create_server_app(storage=storage)
    client = TestClient(app)

    # 1. Test /api/v1/dashboard with pagination
    res = client.get("/api/v1/dashboard?limit=5&page=1")
    assert res.status_code == 200
    assert res.headers.get("X-Total-Count") == "12"
    assert res.headers.get("X-Has-More") == "true"
    data = res.json()
    assert "stats" in data
    assert "traces" in data
    assert data["total_count"] == 12
    assert data["has_more"] is True
    assert len(data["traces"]) == 5
    assert data["stats"]["total_traces"] == 12
    assert data["stats"]["completed_traces"] == 6
    assert data["stats"]["failed_traces"] == 6

    # Page 3 (items 10-11)
    res_p3 = client.get("/api/v1/dashboard?limit=5&page=3")
    assert res_p3.status_code == 200
    assert res_p3.headers.get("X-Has-More") == "false"
    data_p3 = res_p3.json()
    assert len(data_p3["traces"]) == 2
    assert data_p3["has_more"] is False

    # 2. Test unversioned alias /api/dashboard
    res_alias = client.get("/api/dashboard?limit=5&page=1")
    assert res_alias.status_code == 200
    assert res_alias.json()["total_count"] == 12

    # 3. Test filtering via /api/v1/dashboard
    res_filtered = client.get("/api/v1/dashboard?status=failed")
    assert res_filtered.status_code == 200
    data_filtered = res_filtered.json()
    assert data_filtered["total_count"] == 6
    assert all(t["status"] == "failed" for t in data_filtered["traces"])


