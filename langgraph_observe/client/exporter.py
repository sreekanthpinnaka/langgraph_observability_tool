from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx

from langgraph_observe.core.models import SpanStatus, StatsSummary, Trace, TraceSummary
from langgraph_observe.server.storage.base import BaseStorage

logger = logging.getLogger("langgraph_observe.client.exporter")


class RemoteStorage(BaseStorage):
    """Remote storage client that dispatches traces asynchronously to a standalone

    langgraph-observe server via HTTP. Completely decoupled from backend execution.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 20,
        flush_interval_seconds: float = 0.5,
        timeout: float = 4.0,
        client: Optional[httpx.Client] = None,
    ) -> None:
        raw_url = server_url or os.getenv("OBSERVE_SERVER_URL", "http://localhost:8765")
        self.server_url = raw_url.rstrip("/")
        self.api_key = api_key or os.getenv("OBSERVE_API_KEY")
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self.timeout = timeout
        self._custom_client = client

        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        self._client_lock = threading.Lock()
        self._client: Optional[httpx.Client] = client
        self._start_worker()

    def _start_worker(self) -> None:
        self._worker_thread = threading.Thread(
            target=self._background_sender,
            name="langgraph-observe-remote-sender",
            daemon=True,
        )
        self._worker_thread.start()

    def _get_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _get_client(self) -> httpx.Client:
        """Get or initialize the shared persistent HTTP client for read queries."""
        if self._custom_client is not None:
            return self._custom_client
        with self._client_lock:
            if self._client is None or self._client.is_closed:
                self._client = httpx.Client(timeout=self.timeout)
            return self._client

    def _get_http_client(self) -> httpx.Client:
        return self._get_client()

    def _get_sync_client(self) -> httpx.Client:
        return self._get_client()

    def _background_sender(self) -> None:
        """Background daemon thread that continuously batches and ships traces over HTTP."""
        client = httpx.Client(timeout=self.timeout) if self._custom_client is None else self._custom_client
        should_close = self._custom_client is None
        try:
            while not self._stop_event.is_set():
                items: List[Dict[str, Any]] = []
                try:
                    item = self._queue.get(timeout=self.flush_interval)
                    items.append(item)
                    while len(items) < self.batch_size:
                        try:
                            items.append(self._queue.get_nowait())
                        except queue.Empty:
                            break
                except queue.Empty:
                    continue

                if not items:
                    continue

                self._send_batch(client, items)
                for _ in items:
                    self._queue.task_done()
        finally:
            if should_close:
                try:
                    client.close()
                except Exception:
                    pass

    def _send_batch(self, client: httpx.Client, items: List[Dict[str, Any]], max_retries: int = 3) -> bool:
        """Post a batch of trace payloads to the remote server with exponential backoff retry.

        Returns True on successful delivery, False if permanently dropped.
        """
        headers = self._get_headers()
        url = f"{self.server_url}/api/v1/traces" if len(items) == 1 else f"{self.server_url}/api/v1/traces/batch"
        payload = items[0] if len(items) == 1 else items

        for attempt in range(max_retries):
            try:
                resp = client.post(url, json=payload, headers=headers)
                if resp.status_code < 400:
                    return True
                elif resp.status_code < 500:
                    # 4xx client errors (e.g. 401, 422) should not be retried
                    logger.warning(
                        f"Remote observability server rejected traces with status {resp.status_code}: {resp.text}"
                    )
                    return False
                else:
                    logger.warning(
                        f"Remote server returned {resp.status_code} (attempt {attempt + 1}/{max_retries})"
                    )
            except Exception as exc:
                logger.debug(f"Unable to deliver trace batch (attempt {attempt + 1}/{max_retries}): {exc}")

            if attempt < max_retries - 1 and not self._stop_event.is_set():
                time.sleep(0.15 * (2 ** attempt))

        logger.error(f"Permanently dropping {len(items)} traces after {max_retries} failed delivery attempts")
        return False

    def save_trace(self, trace: Trace) -> None:
        """Enqueue a trace for asynchronous remote delivery.

        Metrics are only computed if the trace has completed (status != RUNNING)
        to prevent prematurely locking in metric calculation on partially completed traces.
        """
        if not self._worker_thread.is_alive():
            logger.error("langgraph-observe: background sender thread died; restarting worker.")
            self._start_worker()

        if trace.status != SpanStatus.RUNNING:
            trace.compute_metrics()
        payload = trace.model_dump(mode="json")
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            logger.warning("Observability buffer queue full. Dropping trace.")

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        try:
            client = self._get_sync_client()
            res = client.get(f"{self.server_url}/api/v1/traces/{trace_id}", headers=self._get_headers())
            if res.status_code == 200:
                return Trace.model_validate(res.json())
        except Exception as e:
            logger.debug(f"Failed fetching trace {trace_id} from remote: {e}")
        return None

    def list_traces(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        search: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[TraceSummary]:
        items, _ = self.list_traces_with_count(
            limit=limit,
            offset=offset,
            status=status,
            search=search,
            environment=environment,
            project=project,
        )
        return items

    def list_traces_with_count(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        search: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Tuple[List[TraceSummary], int]:
        try:
            params: Dict[str, Any] = {"limit": limit, "offset": offset}
            if status:
                params["status"] = status
            if search:
                params["search"] = search
            if environment:
                params["environment"] = environment
            if project:
                params["project"] = project
            client = self._get_client()
            res = client.get(f"{self.server_url}/api/v1/traces", params=params, headers=self._get_headers())
            if res.status_code == 200:
                items = [TraceSummary.model_validate(item) for item in res.json()]
                total_header = res.headers.get("X-Total-Count")
                total = int(total_header) if total_header and total_header.isdigit() else len(items)
                return items, total
        except Exception as e:
            logger.debug(f"Failed listing traces with count from remote: {e}")
        return [], 0

    def count_traces(
        self,
        status: Optional[str] = None,
        search: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
    ) -> int:
        params: Dict[str, Any] = {}
        if status:
            params["status"] = status
        if search:
            params["search"] = search
        if environment:
            params["environment"] = environment
        if project:
            params["project"] = project
        try:
            client = self._get_client()
            res = client.get(
                f"{self.server_url}/api/v1/traces-count",
                params=params,
                headers=self._get_headers(),
            )
            if res.status_code == 200:
                data = res.json()
                if "count" in data:
                    return int(data["count"])
        except Exception:
            pass

        _, total = self.list_traces_with_count(
            limit=1,
            offset=0,
            status=status,
            search=search,
            environment=environment,
            project=project,
        )
        return total

    def get_stats(self) -> StatsSummary:
        try:
            client = self._get_client()
            res = client.get(f"{self.server_url}/api/v1/stats", headers=self._get_headers())
            if res.status_code == 200:
                return StatsSummary.model_validate(res.json())
        except Exception as e:
            logger.debug(f"Failed fetching stats from remote: {e}")
        return StatsSummary()

    def delete_trace(self, trace_id: str) -> bool:
        try:
            client = self._get_client()
            res = client.delete(f"{self.server_url}/api/v1/traces/{trace_id}", headers=self._get_headers())
            return res.status_code == 200
        except Exception:
            return False

    def clear(self) -> None:
        try:
            client = self._get_client()
            client.delete(f"{self.server_url}/api/v1/traces", headers=self._get_headers())
        except Exception:
            pass

    def get_timeseries_analytics(
        self,
        time_window: str = "24h",
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[float] = None,
        to_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            client = self._get_client()
            params: Dict[str, Any] = {"time_window": time_window}
            if environment:
                params["environment"] = environment
            if project:
                params["project"] = project
            if from_time is not None:
                params["from_time"] = from_time
            if to_time is not None:
                params["to_time"] = to_time
            res = client.get(
                f"{self.server_url}/api/v1/analytics/timeseries",
                params=params,
                headers=self._get_headers(),
            )
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            logger.debug(f"Failed fetching timeseries analytics from remote: {e}")
        return {"time_window": time_window, "buckets": [], "total_traces": 0}

    def get_node_analytics(
        self,
        time_window: str = "24h",
        environment: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            client = self._get_client()
            params: Dict[str, Any] = {"time_window": time_window}
            if environment:
                params["environment"] = environment
            if project:
                params["project"] = project
            res = client.get(
                f"{self.server_url}/api/v1/analytics/nodes",
                params=params,
                headers=self._get_headers(),
            )
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            logger.debug(f"Failed fetching node analytics from remote: {e}")
        return {"time_window": time_window, "nodes": [], "total_traces": 0}

    def drain(self, timeout: float = 3.0) -> bool:
        """Block until all queued traces have been dispatched and processed, or timeout.

        Returns True if the queue was cleanly drained, False if items remain.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self._queue.empty() and getattr(self._queue, "unfinished_tasks", 0) == 0:
                return True
            time.sleep(0.02)
        return self._queue.empty() and getattr(self._queue, "unfinished_tasks", 0) == 0

    def flush(self, timeout: float = 3.0) -> None:
        """Wait for pending traces to be dispatched, up to timeout seconds."""
        self.drain(timeout=timeout)

    def close(self) -> None:
        self.drain(timeout=1.0)
        self._stop_event.set()
        with self._client_lock:
            if self._client is not None and not self._client.is_closed:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._client = None


# Alias for clarity
AsyncRemoteExporter = RemoteStorage
