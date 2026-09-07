from __future__ import annotations

import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from langgraph_observe.core.models import HTTPMetadata, Span, SpanStatus, SpanType, Trace
from langgraph_observe.server.app import create_server_app
from langgraph_observe.server.storage.memory import MemoryStorage
from langgraph_observe.server.storage.sql import SQLAlchemyStorage
from langgraph_observe.server.storage.sqlite import SQLiteStorage


def _make_trace(trace_id: str, name: str, env: str = "production", project: str = "default") -> Trace:
    t = Trace(
        id=trace_id,
        name=name,
        environment=env,
        project=project,
        http=HTTPMetadata(method="POST", path=f"/api/{name}", url=f"http://localhost/{name}"),
    )
    s = Span(id=f"span_{trace_id}", trace_id=trace_id, name="execute", span_type=SpanType.NODE)
    s.finish(status=SpanStatus.COMPLETED)
    t.spans.append(s)
    t.finish(status=SpanStatus.COMPLETED)
    return t


def test_api_key_auth_required_on_ingestion_and_deletion():
    """Verify that unauthorized write and delete requests are rejected with 401 when api_key is configured."""
    secret_key = "test-secret-key-xyz"
    app = create_server_app(storage=MemoryStorage(), api_key=secret_key)
    client = TestClient(app)

    # Health check is public
    res = client.get("/health")
    assert res.status_code == 200

    # Ingestion without key -> 401
    sample_payload = _make_trace("t1", "auth_trace").model_dump(mode="json")
    res = client.post("/api/v1/traces", json=sample_payload)
    assert res.status_code == 401
    assert "Unauthorized" in res.json()["detail"]

    # Ingestion with wrong key -> 401
    res = client.post("/api/v1/traces", json=sample_payload, headers={"X-API-Key": "wrong-key"})
    assert res.status_code == 401

    # Ingestion with valid X-API-Key -> 201
    res = client.post("/api/v1/traces", json=sample_payload, headers={"X-API-Key": secret_key})
    assert res.status_code == 201
    assert res.json()["id"] == "t1"

    # Batch ingestion with valid Authorization Bearer -> 201
    sample_payload2 = _make_trace("t2", "auth_trace2").model_dump(mode="json")
    res = client.post(
        "/api/v1/traces/batch",
        json=[sample_payload2],
        headers={"Authorization": f"Bearer {secret_key}"},
    )
    assert res.status_code == 201
    assert res.json()["count"] == 1

    # Deletion without key -> 401
    res = client.delete("/api/v1/traces/t1")
    assert res.status_code == 401

    # Deletion with key -> 200
    res = client.delete("/api/v1/traces/t1", headers={"X-API-Key": secret_key})
    assert res.status_code == 200


def test_api_key_auth_disabled_when_not_set():
    """Verify that when no api_key is configured, the server operates in open local development mode."""
    app = create_server_app(storage=MemoryStorage())
    client = TestClient(app)

    sample_payload = _make_trace("t_open", "open_trace").model_dump(mode="json")
    res = client.post("/api/v1/traces", json=sample_payload)
    assert res.status_code == 201


def test_read_endpoint_authentication_and_methods():
    """Verify read endpoints (/traces, /stats, /export) enforce API key via header, bearer, or query param."""
    mem_store = MemoryStorage()
    t = _make_trace("t-auth-read-1", "test-auth-trace")
    mem_store.save_trace(t)

    secured_app = create_server_app(storage=mem_store, api_key="secret-token-123")
    client = TestClient(secured_app)

    # 1. Unauthenticated read requests -> 401
    assert client.get("/api/v1/traces").status_code == 401
    assert client.get("/api/v1/stats").status_code == 401
    assert client.get(f"/api/v1/export/{t.id}").status_code == 401

    # 2. Authenticated via X-API-Key -> 200
    headers = {"X-API-Key": "secret-token-123"}
    assert client.get("/api/v1/traces", headers=headers).status_code == 200
    assert client.get("/api/v1/stats", headers=headers).status_code == 200
    assert client.get(f"/api/v1/export/{t.id}", headers=headers).status_code == 200

    # 3. Authenticated via ?api_key= query parameter (for direct browser downloads)
    assert client.get("/api/v1/traces?api_key=secret-token-123").status_code == 200
    assert client.get("/api/v1/stats?api_key=secret-token-123").status_code == 200

    # 4. Authenticated via Authorization Bearer token
    assert client.get("/api/v1/traces", headers={"Authorization": "Bearer secret-token-123"}).status_code == 200


def test_multi_tenancy_sqlite_filtering():
    """Verify multi-tenant trace filtering by environment and project in SQLiteStorage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "tenants.db")
        storage = SQLiteStorage(db_path=db_path)
        try:
            t_prod_pay = _make_trace("t1", "pay_run", env="production", project="payments")
            t_stag_pay = _make_trace("t2", "pay_test", env="staging", project="payments")
            t_dev_chat = _make_trace("t3", "chat_dev", env="development", project="chatbot")

            storage.save_trace(t_prod_pay)
            storage.save_trace(t_stag_pay)
            storage.save_trace(t_dev_chat)

            prod_traces = storage.list_traces(environment="production")
            assert len(prod_traces) == 1
            assert prod_traces[0].id == "t1"
            assert prod_traces[0].environment == "production"

            stag_traces = storage.list_traces(environment="staging")
            assert len(stag_traces) == 1
            assert stag_traces[0].id == "t2"
            assert stag_traces[0].environment == "staging"

            pay_traces = storage.list_traces(project="payments")
            assert len(pay_traces) == 2
            assert {t.id for t in pay_traces} == {"t1", "t2"}

            combined = storage.list_traces(environment="production", project="payments")
            assert len(combined) == 1
            assert combined[0].id == "t1"
        finally:
            storage.close()


def test_multi_tenancy_sql_alchemy_filtering():
    """Verify multi-tenant trace filtering by environment and project in SQLAlchemyStorage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_url = f"sqlite:///{os.path.join(tmpdir, 'tenants_sa.db')}"
        storage = SQLAlchemyStorage(database_url=db_url)
        try:
            t_prod = _make_trace("sa_t1", "workflow_prod", env="production", project="core_ai")
            t_stag = _make_trace("sa_t2", "workflow_staging", env="staging", project="core_ai")

            storage.save_trace(t_prod)
            storage.save_trace(t_stag)

            prod_res = storage.list_traces(environment="production")
            assert len(prod_res) == 1
            assert prod_res[0].id == "sa_t1"
            assert prod_res[0].environment == "production"

            stag_res = storage.list_traces(environment="staging")
            assert len(stag_res) == 1
            assert stag_res[0].id == "sa_t2"
        finally:
            storage.close()


def test_server_query_endpoint_filters():
    """Verify server query endpoint properly filters traces by environment and project query parameters."""
    storage = MemoryStorage()
    t1 = _make_trace("t1", "wf1", env="production", project="proj_alpha")
    t2 = _make_trace("t2", "wf2", env="staging", project="proj_alpha")
    t3 = _make_trace("t3", "wf3", env="production", project="proj_beta")
    storage.save_trace(t1)
    storage.save_trace(t2)
    storage.save_trace(t3)

    app = create_server_app(storage=storage)
    client = TestClient(app)

    res = client.get("/api/v1/traces?environment=production")
    assert res.status_code == 200
    traces = res.json()
    assert len(traces) == 2
    assert all(t["environment"] == "production" for t in traces)

    res = client.get("/api/v1/traces?project=proj_beta")
    assert res.status_code == 200
    traces = res.json()
    assert len(traces) == 1
    assert traces[0]["id"] == "t3"

    res = client.get("/api/v1/traces?environment=staging&project=proj_alpha")
    assert res.status_code == 200
    traces = res.json()
    assert len(traces) == 1
    assert traces[0]["id"] == "t2"


def test_constant_time_api_key_comparison():
    """Verify constant-time comparison prevents prefix matching or unauthorized access."""
    secret = "sk-live-super-secret-key-12345"
    app = create_server_app(storage=MemoryStorage(), api_key=secret)
    client = TestClient(app)

    # Prefix of secret should fail
    res = client.get("/api/v1/traces", headers={"X-API-Key": "sk-live-super-secret-key"})
    assert res.status_code == 401

    # Suffix of secret should fail
    res = client.get("/api/v1/traces", headers={"X-API-Key": "super-secret-key-12345"})
    assert res.status_code == 401

    # Exact secret succeeds
    res = client.get("/api/v1/traces", headers={"X-API-Key": secret})
    assert res.status_code == 200


def test_time_range_and_eval_status_filters():
    """Verify from_time, to_time, and eval_status query filtering across API and storage."""
    storage = MemoryStorage()
    
    t1 = _make_trace("t1", "trace_old")
    t1.start_time = "2026-01-01T10:00:00Z"
    t1.metadata["evals"] = [{"name": "factuality", "passed": True, "score": 1.0}]

    t2 = _make_trace("t2", "trace_mid")
    t2.start_time = "2026-01-02T10:00:00Z"
    t2.metadata["evals"] = [{"name": "factuality", "passed": False, "score": 0.4}]

    t3 = _make_trace("t3", "trace_recent")
    t3.start_time = "2026-01-03T10:00:00Z"

    storage.save_trace(t1)
    storage.save_trace(t2)
    storage.save_trace(t3)

    app = create_server_app(storage=storage)
    client = TestClient(app)

    # Time range: from_time filter
    res = client.get("/api/v1/traces?from_time=2026-01-02T00:00:00Z")
    assert res.status_code == 200
    traces = res.json()
    assert len(traces) == 2
    assert {t["id"] for t in traces} == {"t2", "t3"}

    # Time range: to_time filter
    res = client.get("/api/v1/traces?to_time=2026-01-02T12:00:00Z")
    assert res.status_code == 200
    traces = res.json()
    assert len(traces) == 2
    assert {t["id"] for t in traces} == {"t1", "t2"}

    # Eval status: passed
    res = client.get("/api/v1/traces?eval_status=passed")
    assert res.status_code == 200
    traces = res.json()
    assert len(traces) == 1
    assert traces[0]["id"] == "t1"
    assert traces[0]["eval_status"] == "passed"
    assert traces[0]["eval_count"] == 1

    # Eval status: failed
    res = client.get("/api/v1/traces?eval_status=failed")
    assert res.status_code == 200
    traces = res.json()
    assert len(traces) == 1
    assert traces[0]["id"] == "t2"
    assert traces[0]["eval_status"] == "failed"

    # Traces count endpoint with eval_status
    count_res = client.get("/api/v1/traces-count?eval_status=passed")
    assert count_res.status_code == 200
    assert count_res.json()["count"] == 1

