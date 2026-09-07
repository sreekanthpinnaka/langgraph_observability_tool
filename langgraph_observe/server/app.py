from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import logging
import os
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from langgraph_observe.core.models import StatsSummary, Trace, TraceSummary
from langgraph_observe.server.storage.base import BaseStorage
from langgraph_observe.server.storage.sql import SQLAlchemyStorage, create_storage_from_url
from langgraph_observe.server.storage.sqlite import SQLiteStorage
from langgraph_observe.server.ui import get_dashboard_html

logger = logging.getLogger("langgraph_observe.server")



DEFAULT_MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB
DEFAULT_RATE_LIMIT_PER_MINUTE = 1200


class RateLimiterMiddleware:
    """ASGI middleware implementing in-memory sliding window rate limiting per IP."""

    def __init__(
        self,
        app: Any,
        requests_per_minute: Optional[int] = None,
    ) -> None:
        self.app = app
        if requests_per_minute is not None:
            self.requests_per_minute = requests_per_minute
        else:
            env_val = os.getenv("OBSERVE_RATE_LIMIT")
            if env_val is not None:
                try:
                    self.requests_per_minute = int(env_val.strip())
                except (ValueError, TypeError):
                    self.requests_per_minute = DEFAULT_RATE_LIMIT_PER_MINUTE
            else:
                self.requests_per_minute = DEFAULT_RATE_LIMIT_PER_MINUTE

        self._history: Dict[str, List[float]] = {}
        self._last_cleanup: float = time.time()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or self.requests_per_minute <= 0:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        forwarded = headers.get(b"x-forwarded-for")
        client_ip = "unknown"
        if forwarded:
            try:
                client_ip = forwarded.decode("latin1").split(",")[0].strip()
            except Exception:
                pass
        if client_ip == "unknown":
            client = scope.get("client")
            if client and len(client) > 0:
                client_ip = str(client[0])

        now = time.time()

        # Periodic cleanup of timestamps older than 60s
        if now - self._last_cleanup > 60.0:
            cutoff = now - 60.0
            expired = [k for k, v in self._history.items() if not v or v[-1] < cutoff]
            for k in expired:
                self._history.pop(k, None)
            self._last_cleanup = now

        window_start = now - 60.0
        records = [t for t in self._history.get(client_ip, []) if t > window_start]

        if len(records) >= self.requests_per_minute:
            response = JSONResponse(
                {"detail": "Rate limit exceeded. Too many requests."},
                status_code=429,
                headers={"Retry-After": "60"},
            )
            await response(scope, receive, send)
            return

        records.append(now)
        self._history[client_ip] = records
        await self.app(scope, receive, send)


class _PayloadTooLargeError(Exception):
    """Internal sentinel exception for request payload exceeding configured limit."""
    def __init__(self, message: str) -> None:
        self.message = message


class RequestSizeLimitMiddleware:
    """ASGI middleware to enforce maximum request body size (default 10MB)."""

    def __init__(self, app: Any, max_body_size: Optional[int] = None) -> None:
        self.app = app
        if max_body_size is not None:
            self.max_body_size = max_body_size
        else:
            env_val = os.getenv("OBSERVE_MAX_BODY_SIZE")
            if env_val is not None:
                try:
                    self.max_body_size = int(env_val.strip())
                except (ValueError, TypeError):
                    self.max_body_size = DEFAULT_MAX_BODY_SIZE
            else:
                self.max_body_size = DEFAULT_MAX_BODY_SIZE

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length_raw = headers.get(b"content-length")
        if content_length_raw is not None:
            try:
                if int(content_length_raw) > self.max_body_size:
                    response = JSONResponse(
                        {"detail": f"Request body too large. Maximum size is {self.max_body_size} bytes."},
                        status_code=413,
                    )
                    await response(scope, receive, send)
                    return
            except (ValueError, TypeError):
                pass

        received_bytes = 0
        response_started = False

        async def tracked_send(message: Any) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                received_bytes += len(body)
                if received_bytes > self.max_body_size:
                    raise _PayloadTooLargeError(
                        f"Request body exceeded limit of {self.max_body_size} bytes"
                    )
            return message

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _PayloadTooLargeError as exc:
            if not response_started:
                response = JSONResponse({"detail": exc.message}, status_code=413)
                await response(scope, receive, send)



def _make_dashboard_response(request: Request) -> Response:
    html_content = get_dashboard_html()
    etag = f'"{hashlib.sha256(html_content.encode("utf-8")).hexdigest()[:16]}"'
    if_none_match = request.headers.get("If-None-Match")
    if if_none_match and if_none_match.strip() == etag:
        return Response(
            status_code=304,
            headers={
                "ETag": etag,
                "Cache-Control": "public, max-age=300, must-revalidate",
            },
        )
    return HTMLResponse(
        content=html_content,
        headers={
            "ETag": etag,
            "Cache-Control": "public, max-age=300, must-revalidate",
        },
    )


def get_server_router(
    storage: BaseStorage,
    api_key: Optional[str] = None,
) -> APIRouter:
    """Create an APIRouter with all observability endpoints (UI, ingestion, queries)."""
    router = APIRouter()
    server_key = api_key or os.getenv("OBSERVE_API_KEY")

    async def verify_ingestion_key(
        request: Request,
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
        api_key_query: Optional[str] = Query(None, alias="api_key"),
    ) -> None:
        if not server_key:
            return  # Ingestion/read auth disabled if no key configured

        provided_key = x_api_key or api_key_query
        if not provided_key:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                provided_key = auth_header[7:].strip()

        if not provided_key or not hmac.compare_digest(str(provided_key), str(server_key)):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # Health check
    @router.get("/health")
    async def health_check() -> Any:
        try:
            total_traces = await asyncio.to_thread(storage.count_traces)
            return {
                "status": "healthy",
                "service": "langgraph-observe",
                "storage": "connected",
                "traces_count": total_traces,
            }
        except Exception as e:
            logger.error(f"Health check failed storage probe: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "degraded",
                    "service": "langgraph-observe",
                    "storage": "unreachable",
                    "error": str(e),
                },
            )

    # Dashboard UI with ETag caching
    @router.get("/", include_in_schema=False)
    @router.get("/observe", include_in_schema=False)
    async def dashboard_view(request: Request) -> Response:
        return _make_dashboard_response(request)

    # Combined Dashboard data API (stats + traces in single network round-trip)
    @router.get("/api/v1/dashboard", dependencies=[Depends(verify_ingestion_key)])
    @router.get("/api/dashboard", include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def get_dashboard(
        response: Response,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        page: Optional[int] = Query(None, ge=1),
        status: Optional[str] = Query(None),
        search: Optional[str] = Query(None),
        environment: Optional[str] = Query(None),
        project: Optional[str] = Query(None),
        from_time: Optional[str] = Query(None),
        to_time: Optional[str] = Query(None),
        eval_status: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        """Retrieve stats and paginated trace list in a single combined network call."""
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        calc_offset = (page - 1) * limit if (page is not None and page >= 1) else offset
        data = await asyncio.to_thread(
            storage.get_dashboard_data,
            limit=limit,
            offset=calc_offset,
            status=status,
            search=search,
            environment=environment,
            project=project,
            from_time=from_time,
            to_time=to_time,
            eval_status=eval_status,
        )
        response.headers["X-Total-Count"] = str(data.get("total_count", 0))
        response.headers["X-Has-More"] = "true" if data.get("has_more") else "false"
        return data

    # --- INGESTION APIS ---

    @router.post("/api/v1/traces", status_code=201, dependencies=[Depends(verify_ingestion_key)])
    @router.post("/api/traces", status_code=201, include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def ingest_trace(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest a single trace payload from any remote FastAPI backend."""
        try:
            trace = Trace.model_validate(payload)
            await asyncio.to_thread(storage.save_trace, trace)
            return {"status": "ok", "id": trace.id}
        except Exception as e:
            logger.warning(f"Failed to ingest trace: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid trace format: {e}")

    @router.post("/api/v1/traces/batch", status_code=201, dependencies=[Depends(verify_ingestion_key)])
    @router.post("/api/traces/batch", status_code=201, include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def ingest_batch_traces(payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest a batch of traces from remote backends."""
        valid_traces: List[Trace] = []
        for item in payload:
            try:
                trace = Trace.model_validate(item)
                valid_traces.append(trace)
            except Exception as e:
                logger.debug(f"Skipping malformed trace in batch: {e}")
        ingested = 0
        if valid_traces:
            ingested = await asyncio.to_thread(storage.save_traces_batch, valid_traces)
        return {"status": "ok", "count": ingested}

    # --- QUERY APIS ---

    @router.get("/api/v1/traces", response_model=List[TraceSummary], dependencies=[Depends(verify_ingestion_key)])
    @router.get("/api/traces", response_model=List[TraceSummary], include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def list_traces(
        response: Response,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        status: Optional[str] = Query(None),
        search: Optional[str] = Query(None),
        environment: Optional[str] = Query(None),
        project: Optional[str] = Query(None),
        from_time: Optional[str] = Query(None),
        to_time: Optional[str] = Query(None),
        eval_status: Optional[str] = Query(None),
    ) -> List[TraceSummary]:
        items, total = await asyncio.to_thread(
            storage.list_traces_with_count,
            limit=limit,
            offset=offset,
            status=status,
            search=search,
            environment=environment,
            project=project,
            from_time=from_time,
            to_time=to_time,
            eval_status=eval_status,
        )
        response.headers["X-Total-Count"] = str(total)
        response.headers["X-Has-More"] = "true" if (offset + len(items) < total) else "false"
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return items

    @router.get("/api/v1/traces-count", dependencies=[Depends(verify_ingestion_key)])
    @router.get("/api/traces/count", include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def count_traces_endpoint(
        response: Response,
        status: Optional[str] = Query(None),
        search: Optional[str] = Query(None),
        environment: Optional[str] = Query(None),
        project: Optional[str] = Query(None),
        from_time: Optional[str] = Query(None),
        to_time: Optional[str] = Query(None),
        eval_status: Optional[str] = Query(None),
    ) -> Dict[str, int]:
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        total = await asyncio.to_thread(
            storage.count_traces,
            status=status,
            search=search,
            environment=environment,
            project=project,
            from_time=from_time,
            to_time=to_time,
            eval_status=eval_status,
        )
        return {"count": total}

    @router.get("/api/v1/traces/{trace_id}", response_model=Trace, dependencies=[Depends(verify_ingestion_key)])
    @router.get("/api/traces/{trace_id}", response_model=Trace, include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def get_trace(trace_id: str, response: Response) -> Trace:
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        trace = await asyncio.to_thread(storage.get_trace, trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace

    @router.delete("/api/v1/traces/{trace_id}", dependencies=[Depends(verify_ingestion_key)])
    @router.delete("/api/traces/{trace_id}", include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def delete_trace(trace_id: str) -> Dict[str, Any]:
        success = await asyncio.to_thread(storage.delete_trace, trace_id)
        if not success:
            raise HTTPException(status_code=404, detail="Trace not found")
        return {"deleted": True, "id": trace_id}

    @router.get("/api/v1/stats", response_model=StatsSummary, dependencies=[Depends(verify_ingestion_key)])
    @router.get("/api/stats", response_model=StatsSummary, include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def get_stats(response: Response) -> StatsSummary:
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return await asyncio.to_thread(storage.get_stats)

    @router.get("/api/v1/analytics/timeseries", dependencies=[Depends(verify_ingestion_key)])
    @router.get("/api/analytics/timeseries", include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def get_timeseries_analytics(
        response: Response,
        time_window: str = Query("24h", description="Time window for bucketing: 1h, 6h, 24h, 7d, 30d"),
        environment: Optional[str] = Query(None),
        project: Optional[str] = Query(None),
        from_time: Optional[float] = Query(None),
        to_time: Optional[float] = Query(None),
    ) -> Dict[str, Any]:
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return await asyncio.to_thread(
            storage.get_timeseries_analytics,
            time_window=time_window,
            environment=environment,
            project=project,
            from_time=from_time,
            to_time=to_time,
        )

    @router.get("/api/v1/analytics/nodes", dependencies=[Depends(verify_ingestion_key)])
    @router.get("/api/analytics/nodes", include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def get_node_analytics(
        response: Response,
        time_window: str = Query("24h", description="Time window for analysis: 1h, 6h, 24h, 7d, 30d"),
        environment: Optional[str] = Query(None),
        project: Optional[str] = Query(None),
    ) -> Dict[str, Any]:
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return await asyncio.to_thread(
            storage.get_node_analytics,
            time_window=time_window,
            environment=environment,
            project=project,
        )

    @router.delete("/api/v1/traces", dependencies=[Depends(verify_ingestion_key)])
    @router.delete("/api/traces", include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def clear_traces() -> Dict[str, Any]:
        await asyncio.to_thread(storage.clear)
        return {"cleared": True}

    @router.get("/api/v1/export/{trace_id}", include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    @router.get("/api/export/{trace_id}", include_in_schema=False, dependencies=[Depends(verify_ingestion_key)])
    async def export_trace(trace_id: str) -> JSONResponse:
        trace = await asyncio.to_thread(storage.get_trace, trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return JSONResponse(
            content=trace.model_dump(),
            headers={"Content-Disposition": f'attachment; filename="trace_{trace_id}.json"'},
        )

    return router



def create_server_app(
    db_path: str = "observe.db",
    db_url: Optional[str] = None,
    storage: Optional[BaseStorage] = None,
    api_key: Optional[str] = None,
    rate_limit_per_minute: Optional[int] = None,
) -> FastAPI:
    """Create and configure the standalone LangGraph Observability Server & Dashboard."""
    if storage is not None:
        store = storage
    elif db_url is not None:
        try:
            store = SQLAlchemyStorage(db_url)
        except Exception as e:
            logger.warning(f"Could not connect to {db_url}: {e}. Falling back to SQLite {db_path}")
            store = SQLiteStorage(db_path=db_path)
    elif os.getenv("DATABASE_URL") or os.getenv("OBSERVE_DATABASE_URL"):
        try:
            store = create_storage_from_url()
        except Exception as e:
            logger.warning(f"Could not connect to DATABASE_URL: {e}. Falling back to SQLite {db_path}")
            store = SQLiteStorage(db_path=db_path)
    else:
        store = SQLiteStorage(db_path=db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if hasattr(store, "close") and callable(store.close):
            store.close()

    app = FastAPI(
        title="LangGraph Observability Server",
        description="Standalone service for ingesting, querying, and visualizing LangGraph workflow traces.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Enforce request payload size limit (default 10MB)
    app.add_middleware(RequestSizeLimitMiddleware)

    # Enforce rate limiting per client IP (default 1200 req/min)
    app.add_middleware(RateLimiterMiddleware, requests_per_minute=rate_limit_per_minute)

    # Enable CORS so any backend or browser client can interact with server
    cors_origins_env = os.getenv("OBSERVE_CORS_ORIGINS")
    if cors_origins_env:
        cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
    else:
        cors_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=(cors_origins != ["*"]),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = get_server_router(store, api_key=api_key)
    app.include_router(router)
    return app


def mount_observability(
    app: FastAPI,
    path: str = "/observe",
    storage: Optional[BaseStorage] = None,
    exclude_paths: Optional[List[str]] = None,
    include_middleware: bool = True,
    api_key: Optional[str] = None,
    environment: Optional[str] = None,
    project: Optional[str] = None,
    mask_pii: bool = False,
    sample_rate: float = 1.0,
) -> FastAPI:
    """Mount the observability dashboard, API routes, and middleware onto an existing FastAPI app in one line.

    Example:
        app = FastAPI()
        mount_observability(app, path="/observe", api_key="secret-123", environment="production")
    """
    from langgraph_observe.client.collector import set_default_storage
    from langgraph_observe.client.middleware import LangGraphObserveMiddleware

    if storage is not None:
        store = storage
    elif os.getenv("DATABASE_URL") or os.getenv("OBSERVE_DATABASE_URL"):
        try:
            store = create_storage_from_url()
        except Exception as e:
            logger.warning(f"Could not connect to DATABASE_URL: {e}. Falling back to SQLite.")
            store = SQLiteStorage()
    else:
        store = SQLiteStorage()
    set_default_storage(store)

    clean_path = "/" + path.strip("/")

    # Mount API and UI router
    router = get_server_router(storage=store, api_key=api_key)
    app.include_router(router, prefix=clean_path, tags=["observability"])

    @app.get(clean_path, include_in_schema=False)
    async def dashboard_mount_view(request: Request) -> Response:
        return _make_dashboard_response(request)


    # Add tracing middleware
    if include_middleware:
        exclusions = list(exclude_paths or [])
        exclusions.extend([clean_path, "/docs", "/openapi.json", "/redoc", "/favicon.ico"])
        app.add_middleware(
            LangGraphObserveMiddleware,
            storage=store,
            exclude_paths=exclusions,
            environment=environment,
            project=project,
            mask_pii=mask_pii,
            sample_rate=sample_rate,
        )

    return app
