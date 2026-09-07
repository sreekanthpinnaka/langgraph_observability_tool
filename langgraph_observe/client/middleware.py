from __future__ import annotations

import logging
import random
import time
import traceback
from typing import Any, Callable, List, Optional
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from langgraph_observe.client.collector import TraceCollector, get_default_storage
from langgraph_observe.core.context import (
    reset_current_collector,
    reset_current_trace,
    set_current_collector,
    set_current_trace,
)
from langgraph_observe.core.models import HTTPMetadata, SpanStatus
from langgraph_observe.server.storage.base import BaseStorage

logger = logging.getLogger("langgraph_observe.client.middleware")


class LangGraphObserveMiddleware(BaseHTTPMiddleware):
    """FastAPI / Starlette middleware that intercepts requests, tracks their lifecycle,

    links LangGraph executions to the incoming HTTP request, and adds an X-Trace-ID header.
    """

    def __init__(
        self,
        app: Any,
        storage: Optional[BaseStorage] = None,
        exclude_paths: Optional[List[str]] = None,
        trace_id_header: str = "X-Trace-ID",
        environment: Optional[str] = None,
        project: Optional[str] = None,
        mask_pii: bool = False,
        sample_rate: float = 1.0,
    ) -> None:
        super().__init__(app)
        self.storage = storage or get_default_storage()
        self.exclude_paths = exclude_paths or ["/observe", "/favicon.ico", "/docs", "/openapi.json"]
        self.trace_id_header = trace_id_header
        self.environment = environment
        self.project = project
        self.mask_pii = mask_pii
        self.sample_rate = sample_rate

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        path = request.url.path
        if any(path.startswith(prefix) for prefix in self.exclude_paths):
            return await call_next(request)

        if self.sample_rate < 1.0 and random.random() > self.sample_rate:
            return await call_next(request)

        trace_id = request.headers.get(self.trace_id_header) or str(uuid.uuid4())
        client_ip = request.client.host if request.client else None
        query_params = dict(request.query_params)

        safe_headers = {}
        for k, v in request.headers.items():
            if k.lower() not in ("authorization", "cookie", "set-cookie", "x-api-key"):
                safe_headers[k] = v

        http_metadata = HTTPMetadata(
            method=request.method,
            path=path,
            url=str(request.url),
            client_ip=client_ip,
            headers=safe_headers,
            query_params=query_params,
        )

        collector = TraceCollector(
            trace_id=trace_id,
            name=f"{request.method} {path}",
            http_metadata=http_metadata,
            storage=self.storage,
            environment=self.environment,
            project=self.project,
            mask_pii=self.mask_pii,
        )

        token_c = set_current_collector(collector)
        token_t = set_current_trace(collector.trace)

        start_time = time.time()

        try:
            response: Response = await call_next(request)
            http_metadata.status_code = response.status_code
            status = SpanStatus.FAILED if response.status_code >= 500 else SpanStatus.COMPLETED

            await collector.finish_trace_async(status=status)
            response.headers[self.trace_id_header] = trace_id
            return response

        except Exception as exc:
            http_metadata.status_code = 500
            err_dict = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            await collector.finish_trace_async(status=SpanStatus.FAILED, error=err_dict)
            raise exc

        finally:
            reset_current_collector(token_c)
            reset_current_trace(token_t)
