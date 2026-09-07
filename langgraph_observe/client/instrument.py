from __future__ import annotations

import atexit
import logging
from typing import List, Optional
from fastapi import FastAPI

from langgraph_observe.client.collector import set_default_storage
from langgraph_observe.client.exporter import RemoteStorage
from langgraph_observe.client.middleware import LangGraphObserveMiddleware

logger = logging.getLogger("langgraph_observe.client")


def instrument_fastapi(
    app: FastAPI,
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    environment: Optional[str] = None,
    project: Optional[str] = None,
    mask_pii: bool = False,
    sample_rate: float = 1.0,
    exclude_paths: Optional[List[str]] = None,
    trace_id_header: str = "X-Trace-ID",
) -> RemoteStorage:
    """Instrument ANY FastAPI backend to stream LangGraph execution traces to a standalone
    observability server asynchronously and non-blockingly.

    Example:
        app = FastAPI()
        instrument_fastapi(app, server_url="http://localhost:8765", api_key="secret", environment="production")
    """
    remote_store = RemoteStorage(server_url=server_url, api_key=api_key)
    set_default_storage(remote_store)

    exclusions = list(exclude_paths or [])
    exclusions.extend(["/docs", "/openapi.json", "/redoc", "/favicon.ico"])

    app.add_middleware(
        LangGraphObserveMiddleware,
        storage=remote_store,
        exclude_paths=exclusions,
        trace_id_header=trace_id_header,
        environment=environment,
        project=project,
        mask_pii=mask_pii,
        sample_rate=sample_rate,
    )

    # Register process shutdown hook to flush remaining buffered traces
    atexit.register(remote_store.close)

    logger.info(f"langgraph-observe: Instrumented FastAPI app to stream traces to {remote_store.server_url}")
    return remote_store
