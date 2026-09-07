from __future__ import annotations

import time
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from langgraph_observe.client.exporter import RemoteStorage
from langgraph_observe.core.models import Span, SpanStatus, SpanType, Trace
from langgraph_observe.server import create_server_app
from langgraph_observe.server.storage.memory import MemoryStorage


def test_remote_storage_async_streaming():
    """Verify that RemoteStorage queues and asynchronously flushes traces to the remote server."""
    server_storage = MemoryStorage()
    server_app = create_server_app(storage=server_storage)
    test_client = TestClient(server_app)

    remote_store = RemoteStorage(
        server_url="http://testserver",
        client=test_client,
        flush_interval_seconds=0.1,
    )

    try:
        trace = Trace(id="streamed-trace-1", name="async_workflow")
        trace.finish(status=SpanStatus.COMPLETED)

        remote_store.save_trace(trace)
        remote_store.flush(timeout=2.0)

        server_trace = server_storage.get_trace("streamed-trace-1")
        assert server_trace is not None
        assert server_trace.name == "async_workflow"
    finally:
        remote_store.close()


def test_remote_storage_offline_resilience():
    """Verify that RemoteStorage buffers traces in-memory when the server is offline without crashing."""
    remote_store = RemoteStorage(
        server_url="http://127.0.0.1:59998",
        timeout=0.2,
        flush_interval_seconds=0.1,
    )

    try:
        trace = Trace(id="offline-trace-1", name="resilience_test")
        trace.finish(status=SpanStatus.COMPLETED)

        # Must not raise an exception
        remote_store.save_trace(trace)
        remote_store.flush(timeout=0.5)
    finally:
        remote_store.close()


def test_remote_storage_retry_backoff_on_server_errors():
    """Verify exponential backoff retry when receiving 5xx server responses."""
    mock_transport = MagicMock()
    resp_503 = MagicMock(status_code=503)
    resp_201 = MagicMock(status_code=201)
    mock_transport.post.side_effect = [resp_503, resp_503, resp_201]

    remote = RemoteStorage(
        server_url="http://mock-server:8000",
        batch_size=5,
        flush_interval_seconds=10.0,
    )

    trace = Trace(id="t-retry-1", name="retry-test")
    trace.finish(SpanStatus.COMPLETED)

    success = remote._send_batch(mock_transport, [trace.model_dump()])
    assert success is True
    assert mock_transport.post.call_count == 3
    remote.close()


def test_remote_storage_running_trace_and_worker_recovery():
    """Verify that unfinalized traces stay unfinalized and that dead worker threads are revived."""
    remote = RemoteStorage(server_url="http://mock-server:8000", batch_size=5)

    running_trace = Trace(id="t-run-1", name="still-running")
    assert running_trace.status == SpanStatus.RUNNING
    remote.save_trace(running_trace)
    assert running_trace.metrics_computed is False

    # Simulate worker death
    assert remote._worker_thread.is_alive()
    remote._stop_event.set()
    remote._worker_thread.join(timeout=2.0)
    assert not remote._worker_thread.is_alive()

    # Reset stop event and trigger save_trace to revive worker
    remote._stop_event.clear()
    t = Trace(id="t-revive-1", name="revive-test")
    t.finish(SpanStatus.COMPLETED)
    remote.save_trace(t)

    assert remote._worker_thread.is_alive()
    remote.close()


def test_remote_storage_sync_client_reuse_and_close():
    """Verify that the internal sync HTTP client is reused across queries and closed on shutdown."""
    remote = RemoteStorage(server_url="http://mock-server:8000")

    c1 = remote._get_sync_client()
    c2 = remote._get_sync_client()
    assert c1 is c2
    assert not c1.is_closed

    remote.close()
    assert c1.is_closed


def test_remote_storage_unified_client_and_dynamic_headers():
    """Verify dynamic header generation for runtime API key updates and client cleanup."""
    rs = RemoteStorage(server_url="http://localhost:9999", api_key="initial-key")
    headers = rs._get_headers()
    assert headers["X-API-Key"] == "initial-key"

    client1 = rs._get_client()
    client2 = rs._get_client()
    assert client1 is client2

    # Dynamically change API key
    rs.api_key = "updated-key"
    updated_headers = rs._get_headers()
    assert updated_headers["X-API-Key"] == "updated-key"

    rs.close()
    assert rs._client is None


def test_remote_storage_drain():
    """Verify that drain() completes quickly when queue is empty."""
    remote = RemoteStorage(server_url="http://127.0.0.1:59999", timeout=0.1)
    try:
        assert remote.drain(timeout=0.5) is True
    finally:
        remote.close()
