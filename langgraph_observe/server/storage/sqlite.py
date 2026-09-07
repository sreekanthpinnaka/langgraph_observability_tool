from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from langgraph_observe.core.models import (
    SpanStatus,
    SpanType,
    StatsSummary,
    Trace,
    TraceSummary,
)
from langgraph_observe.server.storage.base import BaseStorage


class SQLiteStorage(BaseStorage):
    """Production-ready SQLite storage backend for LangGraph traces and metrics."""

    def __init__(self, db_path: str = "observe.db") -> None:
        self.db_path = str(Path(db_path).resolve())
        self._local = threading.local()
        self._lock = threading.RLock()
        self._all_connections: Set[sqlite3.Connection] = set()
        self._closed = False
        self._stats_cache: Optional[StatsSummary] = None
        self._stats_cache_ts: float = 0.0
        self._stats_cache_ttl: float = 3.0
        self._trace_lru: OrderedDict[str, Trace] = OrderedDict()
        self._trace_lru_max: int = 128
        self._init_db()


    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        with self._lock:
            self._all_connections.add(conn)
        return conn

    def _get_connection(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("SQLiteStorage is closed.")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._create_connection()
            self._local.conn = conn
        else:
            try:
                conn.execute("SELECT 1;")
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                with self._lock:
                    self._all_connections.discard(conn)
                if self._closed:
                    raise RuntimeError("SQLiteStorage is closed.")
                conn = self._create_connection()
                self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_ms REAL,
                    http_method TEXT,
                    http_path TEXT,
                    http_status_code INTEGER,
                    total_spans INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    llm_calls INTEGER DEFAULT 0,
                    estimated_cost REAL DEFAULT 0.0,
                    has_error INTEGER DEFAULT 0,
                    environment TEXT DEFAULT 'production',
                    project TEXT,
                    node_counts TEXT,
                    eval_status TEXT,
                    eval_count INTEGER DEFAULT 0,
                    has_tool_loop INTEGER DEFAULT 0,
                    model_counts TEXT,
                    node_metrics TEXT,
                    data_json TEXT NOT NULL
                );
            """)
            # Check for column migration on existing SQLite databases
            cur = conn.execute("PRAGMA table_info(traces);")
            existing_cols = {row[1] for row in cur.fetchall()}
            if "estimated_cost" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE traces ADD COLUMN estimated_cost REAL DEFAULT 0.0;")
                except Exception:
                    pass
            if "environment" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE traces ADD COLUMN environment TEXT DEFAULT 'production';")
                except Exception:
                    pass
            if "project" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE traces ADD COLUMN project TEXT;")
                except Exception:
                    pass
            if "node_counts" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE traces ADD COLUMN node_counts TEXT;")
                except Exception:
                    pass
            if "eval_status" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE traces ADD COLUMN eval_status TEXT;")
                except Exception:
                    pass
            if "eval_count" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE traces ADD COLUMN eval_count INTEGER DEFAULT 0;")
                except Exception:
                    pass
            if "has_tool_loop" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE traces ADD COLUMN has_tool_loop INTEGER DEFAULT 0;")
                except Exception:
                    pass
            if "model_counts" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE traces ADD COLUMN model_counts TEXT;")
                except Exception:
                    pass
            if "node_metrics" not in existing_cols:
                try:
                    conn.execute("ALTER TABLE traces ADD COLUMN node_metrics TEXT;")
                except Exception:
                    pass

            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_start_time ON traces(start_time DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_name ON traces(name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_environment ON traces(environment);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_project ON traces(project);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_eval_status ON traces(eval_status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_has_tool_loop ON traces(has_tool_loop);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_duration ON traces(duration_ms);")
            conn.commit()


    def _prepare_trace_row(self, trace: Trace) -> tuple:
        trace.compute_metrics()
        data_json = trace.model_dump_json()

        http_method = trace.http.method if trace.http else None
        http_path = trace.http.path if trace.http else None
        http_status_code = trace.http.status_code if trace.http else None
        has_error = 1 if (trace.error or any(s.status == SpanStatus.FAILED for s in trace.spans)) else 0
        environment = trace.environment or "production"
        project = trace.project

        node_counts_map: Dict[str, int] = {}
        node_metrics_list: List[Dict[str, Any]] = []
        for s in trace.spans:
            if s.span_type == SpanType.NODE and s.name:
                node_counts_map[s.name] = node_counts_map.get(s.name, 0) + 1
                node_metrics_list.append({
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "status": s.status.value if hasattr(s.status, "value") else str(s.status),
                })
        node_counts_json = json.dumps(node_counts_map)
        node_metrics_json = json.dumps(node_metrics_list)

        model_counts_map: Dict[str, int] = {}
        if trace.llm_metrics and trace.llm_metrics.model_usage:
            for m_name, m_data in trace.llm_metrics.model_usage.items():
                calls = m_data.calls if hasattr(m_data, "calls") else (m_data.get("calls", 1) if isinstance(m_data, dict) else 1)
                model_counts_map[m_name] = model_counts_map.get(m_name, 0) + int(calls)
        else:
            for s in trace.spans:
                if s.span_type == SpanType.LLM:
                    usage_dict = s.metadata.get("usage") if isinstance(s.metadata.get("usage"), dict) else {}
                    m = (s.metadata or {}).get("model") or usage_dict.get("model") or s.name.replace("LLM: ", "").replace("Chat: ", "")
                    if m:
                        model_counts_map[m] = model_counts_map.get(m, 0) + 1
        model_counts_json = json.dumps(model_counts_map)

        # Extract eval status and count
        evals = []
        if isinstance(trace.metadata.get("evals"), list):
            evals.extend(trace.metadata["evals"])
        for s in trace.spans:
            if s.span_type == SpanType.EVAL:
                eval_meta = s.metadata.get("eval")
                if eval_meta and eval_meta not in evals:
                    evals.append(eval_meta)
                elif not eval_meta:
                    evals.append({"name": s.name, "passed": s.status == SpanStatus.COMPLETED})

        eval_count = len(evals)
        eval_status = None
        if eval_count > 0:
            all_passed = all(e.get("passed", True) for e in evals)
            eval_status = "passed" if all_passed else "failed"

        has_tool_loop = 1 if getattr(trace, "has_tool_loop", False) else 0

        return (
            trace.id,
            trace.name,
            trace.status.value,
            trace.start_time,
            trace.end_time,
            trace.duration_ms,
            http_method,
            http_path,
            http_status_code,
            len(trace.spans),
            trace.llm_metrics.total_tokens,
            trace.llm_metrics.llm_calls,
            trace.estimated_cost,
            has_error,
            environment,
            project,
            node_counts_json,
            eval_status,
            eval_count,
            has_tool_loop,
            model_counts_json,
            node_metrics_json,
            data_json,
        )

    def save_trace(self, trace: Trace) -> None:
        row = self._prepare_trace_row(trace)
        conn = self._get_connection()
        with self._lock:
            conn.execute(
                """
                INSERT INTO traces (
                    id, name, status, start_time, end_time, duration_ms,
                    http_method, http_path, http_status_code,
                    total_spans, total_tokens, llm_calls, estimated_cost, has_error,
                    environment, project, node_counts, eval_status, eval_count, has_tool_loop,
                    model_counts, node_metrics, data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    status=excluded.status,
                    end_time=excluded.end_time,
                    duration_ms=excluded.duration_ms,
                    http_method=excluded.http_method,
                    http_path=excluded.http_path,
                    http_status_code=excluded.http_status_code,
                    total_spans=excluded.total_spans,
                    total_tokens=excluded.total_tokens,
                    llm_calls=excluded.llm_calls,
                    estimated_cost=excluded.estimated_cost,
                    has_error=excluded.has_error,
                    environment=excluded.environment,
                    project=excluded.project,
                    node_counts=excluded.node_counts,
                    eval_status=excluded.eval_status,
                    eval_count=excluded.eval_count,
                    has_tool_loop=excluded.has_tool_loop,
                    model_counts=excluded.model_counts,
                    node_metrics=excluded.node_metrics,
                    data_json=excluded.data_json;
                """,
                row,
            )
            conn.commit()
            self._stats_cache = None
            self._trace_lru[trace.id] = trace
            if len(self._trace_lru) > self._trace_lru_max:
                self._trace_lru.popitem(last=False)

    def save_traces_batch(self, traces: List[Trace]) -> int:
        """Atomically persist a batch of traces in a single SQLite transaction."""
        if not traces:
            return 0
        rows = [self._prepare_trace_row(t) for t in traces]
        conn = self._get_connection()
        with self._lock:
            conn.executemany(
                """
                INSERT INTO traces (
                    id, name, status, start_time, end_time, duration_ms,
                    http_method, http_path, http_status_code,
                    total_spans, total_tokens, llm_calls, estimated_cost, has_error,
                    environment, project, node_counts, eval_status, eval_count, has_tool_loop,
                    model_counts, node_metrics, data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    status=excluded.status,
                    end_time=excluded.end_time,
                    duration_ms=excluded.duration_ms,
                    http_method=excluded.http_method,
                    http_path=excluded.http_path,
                    http_status_code=excluded.http_status_code,
                    total_spans=excluded.total_spans,
                    total_tokens=excluded.total_tokens,
                    llm_calls=excluded.llm_calls,
                    estimated_cost=excluded.estimated_cost,
                    has_error=excluded.has_error,
                    environment=excluded.environment,
                    project=excluded.project,
                    node_counts=excluded.node_counts,
                    eval_status=excluded.eval_status,
                    eval_count=excluded.eval_count,
                    has_tool_loop=excluded.has_tool_loop,
                    model_counts=excluded.model_counts,
                    node_metrics=excluded.node_metrics,
                    data_json=excluded.data_json;
                """,
                rows,
            )
            conn.commit()
            self._stats_cache = None
            for t in traces:
                self._trace_lru[t.id] = t
                if len(self._trace_lru) > self._trace_lru_max:
                    self._trace_lru.popitem(last=False)
        return len(rows)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        with self._lock:
            if trace_id in self._trace_lru:
                self._trace_lru.move_to_end(trace_id)
                return self._trace_lru[trace_id]

        conn = self._get_connection()
        cur = conn.execute("SELECT data_json FROM traces WHERE id = ?", (trace_id,))
        row = cur.fetchone()
        if not row:
            return None
        data = json.loads(row["data_json"])
        trace = Trace.model_validate(data)
        with self._lock:
            self._trace_lru[trace_id] = trace
            if len(self._trace_lru) > self._trace_lru_max:
                self._trace_lru.popitem(last=False)
        return trace


    def _build_filter_clause(
        self,
        status: Optional[str] = None,
        search: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        eval_status: Optional[str] = None,
    ) -> Tuple[str, List[Any]]:
        clause = ""
        params: List[Any] = []

        if status:
            clause += " AND LOWER(status) = LOWER(?)"
            params.append(status)

        if environment:
            clause += " AND LOWER(environment) = LOWER(?)"
            params.append(environment)

        if project:
            clause += " AND LOWER(project) = LOWER(?)"
            params.append(project)

        if from_time:
            clause += " AND start_time >= ?"
            params.append(from_time)

        if to_time:
            clause += " AND start_time <= ?"
            params.append(to_time)

        if eval_status:
            clause += " AND LOWER(eval_status) = LOWER(?)"
            params.append(eval_status)

        if search:
            escaped = (
                search.lower()
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            clause += (
                " AND (LOWER(name) LIKE ? ESCAPE '\\'"
                " OR LOWER(http_path) LIKE ? ESCAPE '\\'"
                " OR LOWER(id) LIKE ? ESCAPE '\\'"
                " OR LOWER(project) LIKE ? ESCAPE '\\')"
            )
            term = f"%{escaped}%"
            params.extend([term, term, term, term])

        return clause, params

    def count_traces(
        self,
        status: Optional[str] = None,
        search: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        eval_status: Optional[str] = None,
    ) -> int:
        conn = self._get_connection()
        where_clause, params = self._build_filter_clause(
            status, search, environment, project, from_time, to_time, eval_status
        )
        query = f"SELECT COUNT(*) as cnt FROM traces WHERE 1=1{where_clause}"
        cur = conn.execute(query, params)
        row = cur.fetchone()
        return int(row["cnt"]) if row else 0

    def list_traces_with_count(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        search: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        eval_status: Optional[str] = None,
    ) -> Tuple[List[TraceSummary], int]:
        conn = self._get_connection()
        where_clause, params = self._build_filter_clause(
            status, search, environment, project, from_time, to_time, eval_status
        )
        query = (
            "SELECT id, name, status, start_time, end_time, duration_ms, "
            "http_method, http_path, http_status_code, total_spans, total_tokens, "
            "llm_calls, estimated_cost, has_error, environment, project, "
            "eval_status, eval_count, has_tool_loop, COUNT(*) OVER() AS full_count "
            f"FROM traces WHERE 1=1{where_clause} "
            "ORDER BY start_time DESC LIMIT ? OFFSET ?"
        )
        params_with_paging = list(params)
        params_with_paging.extend([limit, offset])

        try:
            cur = conn.execute(query, params_with_paging)
            rows = cur.fetchall()

            if rows:
                total = int(rows[0]["full_count"])
            elif offset == 0:
                total = 0
            else:
                total = self.count_traces(
                    status=status,
                    search=search,
                    environment=environment,
                    project=project,
                    from_time=from_time,
                    to_time=to_time,
                    eval_status=eval_status,
                )
        except sqlite3.OperationalError:
            # Fallback for older SQLite versions (< 3.25.0) lacking window function support
            fallback_query = (
                "SELECT id, name, status, start_time, end_time, duration_ms, "
                "http_method, http_path, http_status_code, total_spans, total_tokens, "
                "llm_calls, estimated_cost, has_error, environment, project, "
                "eval_status, eval_count, has_tool_loop "
                f"FROM traces WHERE 1=1{where_clause} "
                "ORDER BY start_time DESC LIMIT ? OFFSET ?"
            )
            cur = conn.execute(fallback_query, params_with_paging)
            rows = cur.fetchall()
            total = self.count_traces(
                status=status,
                search=search,
                environment=environment,
                project=project,
                from_time=from_time,
                to_time=to_time,
                eval_status=eval_status,
            )

        summaries = [
            TraceSummary(
                id=r["id"],
                name=r["name"],
                status=SpanStatus(r["status"]),
                start_time=r["start_time"],
                end_time=r["end_time"],
                duration_ms=r["duration_ms"],
                http_method=r["http_method"],
                http_path=r["http_path"],
                http_status_code=r["http_status_code"],
                total_spans=r["total_spans"],
                total_tokens=r["total_tokens"],
                llm_calls=r["llm_calls"],
                estimated_cost=float(r["estimated_cost"] or 0.0),
                has_error=bool(r["has_error"]),
                environment=r["environment"] or "production",
                project=r["project"],
                eval_status=r["eval_status"],
                eval_count=int(r["eval_count"] or 0),
                has_tool_loop=bool(r["has_tool_loop"]) if "has_tool_loop" in r.keys() else False,
            )
            for r in rows
        ]
        return summaries, total

    def list_traces(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        search: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        eval_status: Optional[str] = None,
    ) -> List[TraceSummary]:
        items, _ = self.list_traces_with_count(
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
        return items

    def get_stats(self) -> StatsSummary:
        now_ts = time.monotonic()
        with self._lock:
            if self._stats_cache is not None and (now_ts - self._stats_cache_ts < self._stats_cache_ttl):
                return self._stats_cache

        conn = self._get_connection()
        cur = conn.execute("""
            SELECT
                COUNT(*) as total_traces,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_traces,
                SUM(CASE WHEN status = 'failed' OR has_error = 1 THEN 1 ELSE 0 END) as failed_traces,
                AVG(duration_ms) as avg_duration_ms,
                SUM(total_tokens) as total_tokens,
                SUM(llm_calls) as total_llm_calls,
                SUM(estimated_cost) as total_cost
            FROM traces
        """)
        row = cur.fetchone()
        if not row or row["total_traces"] == 0:
            res = StatsSummary()
            with self._lock:
                self._stats_cache = res
                self._stats_cache_ts = now_ts
            return res

        total = row["total_traces"] or 0
        completed = row["completed_traces"] or 0
        failed = row["failed_traces"] or 0
        avg_duration = row["avg_duration_ms"] or 0.0
        total_tokens = row["total_tokens"] or 0
        total_llm_calls = row["total_llm_calls"] or 0
        total_cost = row["total_cost"] or 0.0
        success_rate = (completed / total * 100.0) if total > 0 else 0.0

        cur_p95 = conn.execute("""
            SELECT duration_ms FROM traces
            WHERE duration_ms IS NOT NULL
            ORDER BY duration_ms ASC
            LIMIT 1 OFFSET (
                SELECT CAST(0.95 * (COUNT(*) - 1) AS INT)
                FROM traces
                WHERE duration_ms IS NOT NULL
            );
        """)
        row_p95 = cur_p95.fetchone()
        p95 = float(row_p95[0]) if (row_p95 and row_p95[0] is not None) else 0.0

        cur_nodes = conn.execute("SELECT node_counts FROM traces WHERE node_counts IS NOT NULL ORDER BY start_time DESC LIMIT 100")
        node_counts: Dict[str, int] = {}
        rows = cur_nodes.fetchall()
        if rows:
            for r in rows:
                try:
                    if r["node_counts"]:
                        counts = json.loads(r["node_counts"])
                        for k, v in counts.items():
                            node_counts[k] = node_counts.get(k, 0) + int(v)
                except Exception:
                    pass
        else:
            cur_fallback = conn.execute("SELECT data_json FROM traces ORDER BY start_time DESC LIMIT 100")
            for r in cur_fallback.fetchall():
                try:
                    t_dict = json.loads(r["data_json"])
                    for s in t_dict.get("spans", []):
                        if s.get("span_type") == SpanType.NODE.value:
                            n = s.get("name")
                            if n:
                                node_counts[n] = node_counts.get(n, 0) + 1
                except Exception:
                    pass

        stats = StatsSummary(
            total_traces=total,
            completed_traces=completed,
            failed_traces=failed,
            success_rate=round(success_rate, 1),
            avg_duration_ms=round(avg_duration, 2),
            p95_duration_ms=round(p95, 2),
            total_tokens=total_tokens,
            total_llm_calls=total_llm_calls,
            total_cost=round(float(total_cost), 6),
            node_execution_counts=node_counts,
        )
        with self._lock:
            self._stats_cache = stats
            self._stats_cache_ts = now_ts
        return stats


    def get_timeseries_analytics(
        self,
        time_window: str = "24h",
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        conn = self._get_connection()
        now_dt = datetime.now(timezone.utc)

        if from_time:
            try:
                start_dt = datetime.fromisoformat(from_time.replace("Z", "+00:00"))
            except Exception:
                start_dt = now_dt - timedelta(hours=24)
        else:
            window_lower = (time_window or "24h").lower()
            if window_lower == "15m":
                start_dt = now_dt - timedelta(minutes=15)
            elif window_lower == "1h":
                start_dt = now_dt - timedelta(hours=1)
            elif window_lower == "7d":
                start_dt = now_dt - timedelta(days=7)
            elif window_lower == "30d":
                start_dt = now_dt - timedelta(days=30)
            else:
                start_dt = now_dt - timedelta(hours=24)

        end_dt = now_dt
        if to_time:
            try:
                end_dt = datetime.fromisoformat(to_time.replace("Z", "+00:00"))
            except Exception:
                end_dt = now_dt

        total_seconds = max(60, int((end_dt - start_dt).total_seconds()))

        if total_seconds <= 15 * 60:
            num_buckets = 15
            bucket_sec = 60
        elif total_seconds <= 3600:
            num_buckets = 12
            bucket_sec = 300
        elif total_seconds <= 86400:
            num_buckets = 24
            bucket_sec = 3600
        elif total_seconds <= 7 * 86400:
            num_buckets = 14
            bucket_sec = 12 * 3600
        else:
            num_buckets = 30
            bucket_sec = 86400

        clause = " WHERE start_time >= ? AND start_time <= ?"
        params = [start_dt.isoformat(), end_dt.isoformat()]
        if environment:
            clause += " AND LOWER(environment) = LOWER(?)"
            params.append(environment)
        if project:
            clause += " AND LOWER(project) = LOWER(?)"
            params.append(project)

        query = (
            "SELECT id, name, status, start_time, duration_ms, estimated_cost, "
            "total_tokens, has_error, has_tool_loop, model_counts, data_json "
            f"FROM traces{clause} ORDER BY start_time ASC"
        )
        cur = conn.execute(query, params)
        rows = cur.fetchall()

        buckets = []
        for i in range(num_buckets):
            b_start = start_dt + timedelta(seconds=i * bucket_sec)
            buckets.append({
                "timestamp": b_start.isoformat(),
                "trace_count": 0,
                "error_count": 0,
                "error_rate": 0.0,
                "durations": [],
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "models": {},
            })

        for r in rows:
            try:
                row_dt = datetime.fromisoformat(r["start_time"].replace("Z", "+00:00"))
                offset_sec = (row_dt - start_dt).total_seconds()
                b_idx = int(offset_sec // bucket_sec)
                if 0 <= b_idx < num_buckets:
                    b = buckets[b_idx]
                    b["trace_count"] += 1
                    if r["has_error"] or r["status"] == "failed":
                        b["error_count"] += 1
                    dur = r["duration_ms"]
                    if dur is not None:
                        b["durations"].append(float(dur))
                    b["total_cost"] += float(r["estimated_cost"] or 0.0)
                    b["total_tokens"] += int(r["total_tokens"] or 0)

                    if r["model_counts"]:
                        try:
                            mc = json.loads(r["model_counts"])
                            if isinstance(mc, dict):
                                for m_name, calls in mc.items():
                                    b["models"][m_name] = b["models"].get(m_name, 0) + int(calls)
                        except Exception:
                            pass
                    elif r["data_json"]:
                        try:
                            dj = json.loads(r["data_json"])
                            lm = dj.get("llm_metrics", {})
                            mu = lm.get("model_usage", {})
                            if isinstance(mu, dict):
                                for m_name, m_data in mu.items():
                                    calls = m_data.get("calls", 1) if isinstance(m_data, dict) else 1
                                    b["models"][m_name] = b["models"].get(m_name, 0) + calls
                            elif dj.get("spans"):
                                for sp in dj["spans"]:
                                    if sp.get("span_type") == "llm":
                                        m = (sp.get("metadata") or {}).get("model") or sp.get("name", "").replace("LLM: ", "").replace("Chat: ", "")
                                        if m:
                                            b["models"][m] = b["models"].get(m, 0) + 1
                        except Exception:
                            pass
            except Exception:
                continue


        for b in buckets:
            tc = b["trace_count"]
            b["total_requests"] = tc
            b["success_count"] = max(0, tc - b["error_count"])
            b["error_rate"] = round((b["error_count"] / tc * 100.0), 1) if tc > 0 else 0.0
            durs = sorted(b.pop("durations"))
            if durs:
                avg_val = round(sum(durs) / len(durs), 2)
                p95_idx = min(len(durs) - 1, max(0, int(math.ceil(0.95 * len(durs))) - 1))
                p95_val = round(durs[p95_idx], 2)
                if p95_val < avg_val:
                    p95_val = avg_val
                b["avg_latency_ms"] = avg_val
                b["avg_duration_ms"] = avg_val
                b["p95_latency_ms"] = p95_val
                b["p95_duration_ms"] = p95_val
            else:
                b["avg_duration_ms"] = 0.0
                b["p95_duration_ms"] = 0.0
            b["total_cost"] = round(b["total_cost"], 4)
            try:
                b["label"] = datetime.fromisoformat(b["timestamp"]).strftime("%H:%M")
            except Exception:
                b["label"] = ""

        return {
            "time_window": time_window,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "total_traces": len(rows),
            "buckets": buckets,
        }

    def get_node_analytics(
        self,
        time_window: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        clause = " WHERE 1=1"
        params = []
        if time_window:
            now_dt = datetime.now(timezone.utc)
            window_lower = time_window.lower()
            if window_lower == "15m":
                s_dt = now_dt - timedelta(minutes=15)
            elif window_lower == "1h":
                s_dt = now_dt - timedelta(hours=1)
            elif window_lower == "7d":
                s_dt = now_dt - timedelta(days=7)
            else:
                s_dt = now_dt - timedelta(hours=24)
            clause += " AND start_time >= ?"
            params.append(s_dt.isoformat())

        if environment:
            clause += " AND LOWER(environment) = LOWER(?)"
            params.append(environment)
        if project:
            clause += " AND LOWER(project) = LOWER(?)"
            params.append(project)

        query = f"SELECT id, node_metrics, data_json FROM traces{clause} ORDER BY start_time DESC LIMIT 500"
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        total_traces = len(rows)
        if total_traces == 0:
            return {
                "time_window": time_window,
                "total_traces": 0,
                "nodes": [],
            }

        node_stats: Dict[str, Dict[str, Any]] = {}

        for r in rows:
            if r["node_metrics"]:
                try:
                    nm_list = json.loads(r["node_metrics"])
                    visited_in_trace: Set[str] = set()
                    for nm in nm_list:
                        n_name = nm.get("name")
                        if not n_name:
                            continue
                        visited_in_trace.add(n_name)
                        if n_name not in node_stats:
                            node_stats[n_name] = {
                                "node_name": n_name,
                                "execution_count": 0,
                                "traces_executed": 0,
                                "error_count": 0,
                                "durations": [],
                            }
                        ns = node_stats[n_name]
                        ns["execution_count"] += 1
                        if nm.get("status") == "failed":
                            ns["error_count"] += 1
                        dur = nm.get("duration_ms")
                        if dur is not None:
                            ns["durations"].append(float(dur))

                    for n_name in visited_in_trace:
                        if n_name in node_stats:
                            node_stats[n_name]["traces_executed"] += 1
                    continue
                except Exception:
                    pass

            if not r["data_json"]:
                continue
            try:
                t_dict = json.loads(r["data_json"])
                spans = t_dict.get("spans", [])
                visited_in_trace: Set[str] = set()

                for s in spans:
                    if s.get("span_type") == "node" and s.get("name"):
                        n_name = s["name"]
                        visited_in_trace.add(n_name)
                        if n_name not in node_stats:
                            node_stats[n_name] = {
                                "node_name": n_name,
                                "execution_count": 0,
                                "traces_executed": 0,
                                "error_count": 0,
                                "durations": [],
                            }
                        ns = node_stats[n_name]
                        ns["execution_count"] += 1
                        if s.get("status") == "failed":
                            ns["error_count"] += 1
                        dur = s.get("duration_ms")
                        if dur is not None:
                            ns["durations"].append(float(dur))

                for n_name in visited_in_trace:
                    if n_name in node_stats:
                        node_stats[n_name]["traces_executed"] += 1
            except Exception:
                continue


        result = []
        for n_name, ns in node_stats.items():
            exec_count = ns["execution_count"]
            durs = sorted(ns.pop("durations"))
            avg_dur = round(sum(durs) / len(durs), 2) if durs else 0.0
            p95_dur = round(durs[int(0.95 * (len(durs) - 1))], 2) if durs else 0.0
            min_dur = round(durs[0], 2) if durs else 0.0
            max_dur = round(durs[-1], 2) if durs else 0.0
            traces_exec = ns["traces_executed"]
            freq_pct = round((traces_exec / total_traces * 100.0), 1) if total_traces > 0 else 0.0
            err_count = ns["error_count"]
            err_rate = round((err_count / exec_count * 100.0), 1) if exec_count > 0 else 0.0

            result.append({
                "node_name": n_name,
                "executions": exec_count,
                "execution_count": exec_count,
                "traces_executed": traces_exec,
                "frequency_pct": freq_pct,
                "error_count": err_count,
                "error_rate": err_rate,
                "avg_duration_ms": avg_dur,
                "p95_duration_ms": p95_dur,
                "min_duration_ms": min_dur,
                "max_duration_ms": max_dur,
            })

        result.sort(key=lambda x: x["frequency_pct"], reverse=True)
        return {
            "time_window": time_window,
            "total_traces": total_traces,
            "nodes": result,
        }

    def delete_trace(self, trace_id: str) -> bool:
        conn = self._get_connection()
        with self._lock:
            self._stats_cache = None
            self._trace_lru.pop(trace_id, None)
            cur = conn.execute("DELETE FROM traces WHERE id = ?", (trace_id,))
            conn.commit()
            return cur.rowcount > 0

    def clear(self) -> None:
        conn = self._get_connection()
        with self._lock:
            self._stats_cache = None
            self._trace_lru.clear()
            conn.execute("DELETE FROM traces;")
            conn.commit()


    def close(self) -> None:
        self._closed = True
        with self._lock:
            for conn in list(self._all_connections):
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_connections.clear()
        if hasattr(self._local, "conn"):
            self._local.conn = None
