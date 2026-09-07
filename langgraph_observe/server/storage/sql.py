from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from langgraph_observe.core.models import (
    SpanStatus,
    SpanType,
    StatsSummary,
    Trace,
    TraceSummary,
)
from langgraph_observe.server.storage.base import BaseStorage

logger = logging.getLogger("langgraph_observe.server.storage.sql")

# Load environment variables from .env if present
load_dotenv()


class SQLAlchemyStorage(BaseStorage):
    """Universal production storage backend supporting MySQL (e.g. Google Cloud SQL),

    PostgreSQL, and SQLite via SQLAlchemy.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._stats_cache: Optional[StatsSummary] = None
        self._stats_cache_ts: float = 0.0
        self._stats_cache_ttl: float = 3.0
        self._trace_lru: OrderedDict[str, Trace] = OrderedDict()
        self._trace_lru_max: int = 128

        raw_url = (
            database_url
            or os.getenv("DATABASE_URL")
            or os.getenv("OBSERVE_DATABASE_URL")
            or "sqlite:///observe.db"
        )

        # Normalize mysql:// to mysql+pymysql:// if driver not explicitly set
        if raw_url.startswith("mysql://"):
            raw_url = raw_url.replace("mysql://", "mysql+pymysql://", 1)

        self.database_url = raw_url
        self.is_sqlite = "sqlite" in self.database_url.lower()
        self.is_mysql = "mysql" in self.database_url.lower()

        engine_kwargs: Dict[str, Any] = {}
        if self.is_mysql:
            # Cloud SQL optimizations: recycle connections before Cloud SQL timeout, pre-ping
            engine_kwargs.update(
                {
                    "pool_size": 10,
                    "max_overflow": 20,
                    "pool_recycle": 1800,  # 30 min recycle prevents Cloud SQL idle drop
                    "pool_pre_ping": True,  # Test connection liveness before query
                }
            )
        elif self.is_sqlite:
            engine_kwargs.update(
                {
                    "connect_args": {"check_same_thread": False},
                }
            )

        self.engine: Engine = create_engine(self.database_url, **engine_kwargs)
        try:
            self._init_db()
        except Exception as e:
            if self.is_mysql and ("1049" in str(e) or "Unknown database" in str(e)):
                self._ensure_mysql_database_exists()

                self._init_db()
            else:
                raise

    def _ensure_mysql_database_exists(self) -> None:
        """Auto-create MySQL database if it does not already exist."""
        try:
            from sqlalchemy.engine.url import make_url
            url = make_url(self.database_url)
            db_name = url.database
            if not db_name:
                return
            base_url = url.set(database="")
            temp_engine = create_engine(base_url)
            with temp_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
            temp_engine.dispose()
        except Exception as ex:
            logger.warning(f"Could not auto-create database: {ex}")

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        data_type = sa.Text(length=16777215) if self.is_mysql else sa.Text()

        metadata = sa.MetaData()
        self.traces_table = sa.Table(
            "traces",
            metadata,
            sa.Column("id", sa.String(64), primary_key=True),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("status", sa.String(32), nullable=False, index=True),
            sa.Column("start_time", sa.String(64), nullable=False, index=True),
            sa.Column("end_time", sa.String(64), nullable=True),
            sa.Column("duration_ms", sa.Float(), nullable=True, index=True),
            sa.Column("http_method", sa.String(16), nullable=True),
            sa.Column("http_path", sa.String(512), nullable=True),
            sa.Column("http_status_code", sa.Integer(), nullable=True),
            sa.Column("total_spans", sa.Integer(), default=0),
            sa.Column("total_tokens", sa.Integer(), default=0),
            sa.Column("llm_calls", sa.Integer(), default=0),
            sa.Column("estimated_cost", sa.Float(), default=0.0),
            sa.Column("has_error", sa.Integer(), default=0),
            sa.Column("environment", sa.String(64), default="production", index=True),
            sa.Column("project", sa.String(128), nullable=True, index=True),
            sa.Column("node_counts", sa.Text(), nullable=True),
            sa.Column("eval_status", sa.String(32), nullable=True, index=True),
            sa.Column("eval_count", sa.Integer(), default=0),
            sa.Column("has_tool_loop", sa.Integer(), default=0, index=True),
            sa.Column("model_counts", sa.Text(), nullable=True),
            sa.Column("node_metrics", sa.Text(), nullable=True),
            sa.Column("data_json", data_type, nullable=False),
        )

        metadata.create_all(self.engine)

        # Auto-migration for existing databases lacking estimated_cost / environment / project / node_counts / eval / has_tool_loop / model_counts / node_metrics columns
        try:
            insp = sa.inspect(self.engine)
            existing_cols = {c["name"] for c in insp.get_columns("traces")}
            if "estimated_cost" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(text("ALTER TABLE traces ADD COLUMN estimated_cost FLOAT DEFAULT 0.0"))
            if "environment" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(text("ALTER TABLE traces ADD COLUMN environment VARCHAR(64) DEFAULT 'production'"))
            if "project" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(text("ALTER TABLE traces ADD COLUMN project VARCHAR(128)"))
            if "node_counts" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(text("ALTER TABLE traces ADD COLUMN node_counts TEXT"))
            if "eval_status" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(text("ALTER TABLE traces ADD COLUMN eval_status VARCHAR(32)"))
            if "eval_count" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(text("ALTER TABLE traces ADD COLUMN eval_count INTEGER DEFAULT 0"))
            if "has_tool_loop" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(text("ALTER TABLE traces ADD COLUMN has_tool_loop INTEGER DEFAULT 0"))
            if "model_counts" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(text("ALTER TABLE traces ADD COLUMN model_counts TEXT"))
            if "node_metrics" not in existing_cols:
                with self.engine.begin() as conn:
                    conn.execute(text("ALTER TABLE traces ADD COLUMN node_metrics TEXT"))
        except Exception as ex:
            logger.debug(f"SQLAlchemy column migration notice: {ex}")

        if self.is_sqlite:
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL;"))
                conn.execute(text("PRAGMA synchronous=NORMAL;"))

    def _prepare_trace_dict(self, trace: Trace) -> Dict[str, Any]:
        trace.compute_metrics()
        data_json = trace.model_dump_json()

        http_method = trace.http.method if trace.http else None
        http_path = trace.http.path if trace.http else None
        http_status_code = trace.http.status_code if trace.http else None
        has_error = (
            1
            if (trace.error or any(s.status == SpanStatus.FAILED for s in trace.spans))
            else 0
        )

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

        return {
            "id": trace.id,
            "name": trace.name,
            "status": trace.status.value,
            "start_time": trace.start_time,
            "end_time": trace.end_time,
            "duration_ms": trace.duration_ms,
            "http_method": http_method,
            "http_path": http_path,
            "http_status_code": http_status_code,
            "total_spans": len(trace.spans),
            "total_tokens": trace.llm_metrics.total_tokens,
            "llm_calls": trace.llm_metrics.llm_calls,
            "estimated_cost": float(trace.estimated_cost or 0.0),
            "has_error": has_error,
            "environment": trace.environment or "production",
            "project": trace.project,
            "node_counts": node_counts_json,
            "eval_status": eval_status,
            "eval_count": eval_count,
            "has_tool_loop": has_tool_loop,
            "model_counts": model_counts_json,
            "node_metrics": node_metrics_json,
            "data_json": data_json,
        }


    def _execute_upsert(self, conn: Any, trace_data: Dict[str, Any]) -> None:
        """Atomic upsert without SELECT race condition across SQLite, MySQL, and PostgreSQL."""
        update_cols = {k: v for k, v in trace_data.items() if k != "id"}
        if self.is_sqlite:
            from sqlalchemy.dialects.sqlite import insert as sqlite_insert
            stmt = sqlite_insert(self.traces_table).values(**trace_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=[self.traces_table.c.id],
                set_=update_cols,
            )
            conn.execute(stmt)
        elif self.is_mysql:
            from sqlalchemy.dialects.mysql import insert as mysql_insert
            stmt = mysql_insert(self.traces_table).values(**trace_data)
            stmt = stmt.on_duplicate_key_update(**update_cols)
            conn.execute(stmt)
        elif "postgres" in self.database_url.lower():
            from sqlalchemy.dialects.postgresql import insert as pg_insert
            stmt = pg_insert(self.traces_table).values(**trace_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=[self.traces_table.c.id],
                set_=update_cols,
            )
            conn.execute(stmt)
        else:
            try:
                conn.execute(self.traces_table.insert().values(**trace_data))
            except Exception:
                conn.execute(
                    self.traces_table.update()
                    .where(self.traces_table.c.id == trace_data["id"])
                    .values(**trace_data)
                )

    def save_trace(self, trace: Trace) -> None:
        trace_data = self._prepare_trace_dict(trace)
        with self.engine.begin() as conn:
            self._execute_upsert(conn, trace_data)
        with self._lock:
            self._stats_cache = None
            self._trace_lru[trace.id] = trace
            if len(self._trace_lru) > self._trace_lru_max:
                self._trace_lru.popitem(last=False)

    def save_traces_batch(self, traces: List[Trace]) -> int:
        """Atomically persist a batch of traces in a single database transaction."""
        if not traces:
            return 0
        trace_dicts = [self._prepare_trace_dict(t) for t in traces]
        with self.engine.begin() as conn:
            for td in trace_dicts:
                self._execute_upsert(conn, td)
        with self._lock:
            self._stats_cache = None
            for t in traces:
                self._trace_lru[t.id] = t
                if len(self._trace_lru) > self._trace_lru_max:
                    self._trace_lru.popitem(last=False)
        return len(trace_dicts)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        with self._lock:
            if trace_id in self._trace_lru:
                self._trace_lru.move_to_end(trace_id)
                return self._trace_lru[trace_id]

        with self.engine.connect() as conn:
            row = conn.execute(
                sa.select(self.traces_table.c.data_json).where(
                    self.traces_table.c.id == trace_id
                )
            ).fetchone()
            if not row:
                return None
            data = json.loads(row[0])
            trace = Trace.model_validate(data)
            with self._lock:
                self._trace_lru[trace_id] = trace
                if len(self._trace_lru) > self._trace_lru_max:
                    self._trace_lru.popitem(last=False)
            return trace


    def _build_filter_conditions(
        self,
        status: Optional[str] = None,
        search: Optional[str] = None,
        environment: Optional[str] = None,
        project: Optional[str] = None,
        from_time: Optional[str] = None,
        to_time: Optional[str] = None,
        eval_status: Optional[str] = None,
    ) -> List[Any]:
        conditions: List[Any] = []
        if status:
            conditions.append(sa.func.lower(self.traces_table.c.status) == status.lower())
        if environment:
            conditions.append(sa.func.lower(self.traces_table.c.environment) == environment.lower())
        if project:
            conditions.append(sa.func.lower(self.traces_table.c.project) == project.lower())
        if from_time:
            conditions.append(self.traces_table.c.start_time >= from_time)
        if to_time:
            conditions.append(self.traces_table.c.start_time <= to_time)
        if eval_status:
            conditions.append(sa.func.lower(self.traces_table.c.eval_status) == eval_status.lower())
        if search:
            escaped = (
                search.lower()
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            search_term = f"%{escaped}%"
            conditions.append(
                sa.or_(
                    sa.func.lower(self.traces_table.c.name).like(search_term, escape="\\"),
                    sa.func.lower(self.traces_table.c.http_path).like(search_term, escape="\\"),
                    sa.func.lower(self.traces_table.c.id).like(search_term, escape="\\"),
                    sa.func.lower(self.traces_table.c.project).like(search_term, escape="\\"),
                )
            )
        return conditions

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
        conditions = self._build_filter_conditions(
            status, search, environment, project, from_time, to_time, eval_status
        )
        stmt = sa.select(sa.func.count()).select_from(self.traces_table)
        if conditions:
            stmt = stmt.where(sa.and_(*conditions))
        with self.engine.connect() as conn:
            cnt = conn.execute(stmt).scalar()
            return int(cnt or 0)

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
        conditions = self._build_filter_conditions(
            status, search, environment, project, from_time, to_time, eval_status
        )
        cols = [
            self.traces_table.c.id,
            self.traces_table.c.name,
            self.traces_table.c.status,
            self.traces_table.c.start_time,
            self.traces_table.c.end_time,
            self.traces_table.c.duration_ms,
            self.traces_table.c.http_method,
            self.traces_table.c.http_path,
            self.traces_table.c.http_status_code,
            self.traces_table.c.total_spans,
            self.traces_table.c.total_tokens,
            self.traces_table.c.llm_calls,
            self.traces_table.c.estimated_cost,
            self.traces_table.c.has_error,
            self.traces_table.c.environment,
            self.traces_table.c.project,
            self.traces_table.c.eval_status,
            self.traces_table.c.eval_count,
            self.traces_table.c.has_tool_loop,
            sa.func.count().over().label("full_count"),
        ]
        stmt = sa.select(*cols)
        if conditions:
            stmt = stmt.where(sa.and_(*conditions))

        stmt = stmt.order_by(self.traces_table.c.start_time.desc()).limit(limit).offset(offset)

        with self.engine.connect() as conn:
            try:
                rows = conn.execute(stmt).fetchall()
                if rows:
                    total = int(rows[0][19])
                elif offset == 0:
                    total = 0
                else:
                    count_stmt = sa.select(sa.func.count()).select_from(self.traces_table)
                    if conditions:
                        count_stmt = count_stmt.where(sa.and_(*conditions))
                    total = int(conn.execute(count_stmt).scalar() or 0)
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                # Fallback for SQL dialects without window function support (using clean connection)
                with self.engine.connect() as fallback_conn:
                    count_stmt = sa.select(sa.func.count()).select_from(self.traces_table)
                    if conditions:
                        count_stmt = count_stmt.where(sa.and_(*conditions))
                    total = int(fallback_conn.execute(count_stmt).scalar() or 0)

                    fallback_stmt = sa.select(*cols[:-1])
                    if conditions:
                        fallback_stmt = fallback_stmt.where(sa.and_(*conditions))
                    fallback_stmt = fallback_stmt.order_by(self.traces_table.c.start_time.desc()).limit(limit).offset(offset)
                    rows = fallback_conn.execute(fallback_stmt).fetchall()

        summaries: List[TraceSummary] = []
        for r in rows:
            summaries.append(
                TraceSummary(
                    id=str(r[0]),
                    name=str(r[1]),
                    status=SpanStatus(r[2]),
                    start_time=str(r[3]),
                    end_time=str(r[4]) if r[4] is not None else None,
                    duration_ms=float(r[5]) if r[5] is not None else None,
                    http_method=str(r[6]) if r[6] is not None else None,
                    http_path=str(r[7]) if r[7] is not None else None,
                    http_status_code=int(r[8]) if r[8] is not None else None,
                    total_spans=int(r[9] or 0),
                    total_tokens=int(r[10] or 0),
                    llm_calls=int(r[11] or 0),
                    estimated_cost=float(r[12] or 0.0),
                    has_error=bool(r[13]),
                    environment=str(r[14]) if r[14] is not None else "production",
                    project=str(r[15]) if r[15] is not None else None,
                    eval_status=str(r[16]) if r[16] is not None else None,
                    eval_count=int(r[17] or 0),
                    has_tool_loop=bool(r[18]) if len(r) > 18 and r[18] is not None else False,
                )
            )
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

        agg_stmt = sa.select(
            sa.func.count().label("total_traces"),
            sa.func.sum(
                sa.case((self.traces_table.c.status == "completed", 1), else_=0)
            ).label("completed_traces"),
            sa.func.sum(
                sa.case(
                    (
                        sa.or_(
                            self.traces_table.c.status == "failed",
                            self.traces_table.c.has_error == 1,
                        ),
                        1,
                    ),
                    else_=0,
                )
            ).label("failed_traces"),
            sa.func.avg(self.traces_table.c.duration_ms).label("avg_duration_ms"),
            sa.func.sum(self.traces_table.c.total_tokens).label("total_tokens"),
            sa.func.sum(self.traces_table.c.llm_calls).label("total_llm_calls"),
            sa.func.sum(self.traces_table.c.estimated_cost).label("total_cost"),
        )

        with self.engine.connect() as conn:
            agg_row = conn.execute(agg_stmt).fetchone()
            if not agg_row or not agg_row[0]:
                res = StatsSummary()
                with self._lock:
                    self._stats_cache = res
                    self._stats_cache_ts = now_ts
                return res

            total = int(agg_row[0] or 0)
            completed = int(agg_row[1] or 0)
            failed = int(agg_row[2] or 0)
            avg_duration = float(agg_row[3] or 0.0)
            total_tokens = int(agg_row[4] or 0)
            total_llm_calls = int(agg_row[5] or 0)
            total_cost = float(agg_row[6] or 0.0)
            success_rate = (float(completed) / float(total) * 100.0) if total > 0 else 0.0

            count_dur_stmt = (
                sa.select(sa.func.count())
                .select_from(self.traces_table)
                .where(self.traces_table.c.duration_ms.isnot(None))
            )
            dur_count = conn.execute(count_dur_stmt).scalar() or 0
            if dur_count > 0:
                p95_offset = max(0, int(0.95 * (dur_count - 1)))
                p95_stmt = (
                    sa.select(self.traces_table.c.duration_ms)
                    .where(self.traces_table.c.duration_ms.isnot(None))
                    .order_by(self.traces_table.c.duration_ms.asc())
                    .limit(1)
                    .offset(p95_offset)
                )
                p95_row = conn.execute(p95_stmt).fetchone()
                p95 = float(p95_row[0]) if p95_row and p95_row[0] is not None else 0.0
            else:
                p95 = 0.0

            nodes_stmt = (
                sa.select(self.traces_table.c.node_counts)
                .where(self.traces_table.c.node_counts.isnot(None))
                .order_by(self.traces_table.c.start_time.desc())
                .limit(100)
            )
            node_counts: Dict[str, int] = {}
            rows = conn.execute(nodes_stmt).fetchall()
            if rows:
                for r in rows:
                    try:
                        if r[0]:
                            counts = json.loads(r[0])
                            for k, v in counts.items():
                                node_counts[k] = node_counts.get(k, 0) + int(v)
                    except Exception:
                        pass
            else:
                fallback_stmt = (
                    sa.select(self.traces_table.c.data_json)
                    .order_by(self.traces_table.c.start_time.desc())
                    .limit(100)
                )
                for r in conn.execute(fallback_stmt).fetchall():
                    try:
                        t_dict = json.loads(r[0])
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
            total_cost=round(total_cost, 6),
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

        conditions = [
            self.traces_table.c.start_time >= start_dt.isoformat(),
            self.traces_table.c.start_time <= end_dt.isoformat(),
        ]
        if environment:
            conditions.append(sa.func.lower(self.traces_table.c.environment) == environment.lower())
        if project:
            conditions.append(sa.func.lower(self.traces_table.c.project) == project.lower())

        stmt = (
            sa.select(
                self.traces_table.c.id,
                self.traces_table.c.name,
                self.traces_table.c.status,
                self.traces_table.c.start_time,
                self.traces_table.c.duration_ms,
                self.traces_table.c.estimated_cost,
                self.traces_table.c.total_tokens,
                self.traces_table.c.has_error,
                self.traces_table.c.has_tool_loop,
                self.traces_table.c.model_counts,
                self.traces_table.c.data_json,
            )
            .where(sa.and_(*conditions))
            .order_by(self.traces_table.c.start_time.asc())
        )

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

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
                row_dt = datetime.fromisoformat(str(r[3]).replace("Z", "+00:00"))
                offset_sec = (row_dt - start_dt).total_seconds()
                b_idx = int(offset_sec // bucket_sec)
                if 0 <= b_idx < num_buckets:
                    b = buckets[b_idx]
                    b["trace_count"] += 1
                    if r[7] or str(r[2]) == "failed":
                        b["error_count"] += 1
                    dur = r[4]
                    if dur is not None:
                        b["durations"].append(float(dur))
                    b["total_cost"] += float(r[5] or 0.0)
                    b["total_tokens"] += int(r[6] or 0)

                    model_counts_raw = r[9]
                    data_json_raw = r[10]
                    if model_counts_raw:
                        try:
                            mc = json.loads(model_counts_raw)
                            if isinstance(mc, dict):
                                for m_name, calls in mc.items():
                                    b["models"][m_name] = b["models"].get(m_name, 0) + int(calls)
                        except Exception:
                            pass
                    elif data_json_raw:
                        try:
                            dj = json.loads(data_json_raw)
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
        conditions = []
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
            conditions.append(self.traces_table.c.start_time >= s_dt.isoformat())

        if environment:
            conditions.append(sa.func.lower(self.traces_table.c.environment) == environment.lower())
        if project:
            conditions.append(sa.func.lower(self.traces_table.c.project) == project.lower())

        stmt = (
            sa.select(self.traces_table.c.id, self.traces_table.c.node_metrics, self.traces_table.c.data_json)
            .order_by(self.traces_table.c.start_time.desc())
            .limit(500)
        )
        if conditions:
            stmt = stmt.where(sa.and_(*conditions))

        with self.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

        total_traces = len(rows)
        if total_traces == 0:
            return {
                "time_window": time_window,
                "total_traces": 0,
                "nodes": [],
            }

        node_stats: Dict[str, Dict[str, Any]] = {}

        for r in rows:
            node_metrics_raw = r[1]
            data_json_raw = r[2]
            if node_metrics_raw:
                try:
                    nm_list = json.loads(node_metrics_raw)
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

            if not data_json_raw:
                continue
            try:
                t_dict = json.loads(data_json_raw)
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
        with self.engine.begin() as conn:
            res = conn.execute(
                self.traces_table.delete().where(self.traces_table.c.id == trace_id)
            )
            with self._lock:
                self._stats_cache = None
                self._trace_lru.pop(trace_id, None)
            return res.rowcount > 0

    def clear(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(self.traces_table.delete())
            with self._lock:
                self._stats_cache = None
                self._trace_lru.clear()


    def close(self) -> None:
        try:
            self.engine.dispose()
        except Exception:
            pass


def create_storage_from_url(database_url: Optional[str] = None) -> BaseStorage:
    """Create a storage backend from DATABASE_URL or .env file."""
    load_dotenv()
    url = (
        database_url
        or os.getenv("DATABASE_URL")
        or os.getenv("OBSERVE_DATABASE_URL")
    )
    if url:
        return SQLAlchemyStorage(url)
    return SQLAlchemyStorage("sqlite:///observe.db")
