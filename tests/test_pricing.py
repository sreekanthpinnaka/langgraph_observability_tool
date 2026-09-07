import json
import logging
import os
import tempfile

import pytest

from langgraph_observe.core.models import (
    Span,
    SpanStatus,
    SpanType,
    Trace,
)
from langgraph_observe.core.pricing import (
    calculate_cost,
    find_pricing,
    load_pricing_from_dict,
    load_pricing_from_file,
    normalize_model_name,
    register_model_pricing,
)
from langgraph_observe.server.storage.memory import MemoryStorage
from langgraph_observe.server.storage.sql import SQLAlchemyStorage
from langgraph_observe.server.storage.sqlite import SQLiteStorage


def test_normalize_model_name():
    assert normalize_model_name("openai/gpt-4o") == "gpt-4o"
    assert normalize_model_name("models/gemini-1.5-pro") == "gemini-1.5-pro"
    assert normalize_model_name("anthropic/claude-3-5-sonnet:latest") == "claude-3-5-sonnet"
    assert normalize_model_name("gpt_4o_mini") == "gpt-4o-mini"
    assert normalize_model_name(None) == "unknown"


def test_find_pricing_standard_models():
    # OpenAI
    assert find_pricing("gpt-4o") == (2.50, 10.00)
    assert find_pricing("gpt-4o-mini") == (0.15, 0.60)
    assert find_pricing("gpt-4o-mini-2024-07-18") == (0.15, 0.60)
    assert find_pricing("openai/gpt-4o-2024-08-06") == (2.50, 10.00)

    # Anthropic
    assert find_pricing("claude-3-5-sonnet-20241022") == (3.00, 15.00)
    assert find_pricing("claude-3-5-haiku") == (0.80, 4.00)

    # Gemini
    assert find_pricing("gemini-1.5-pro") == (3.50, 10.50)
    assert find_pricing("gemini-1.5-flash") == (0.075, 0.30)

    # DeepSeek
    assert find_pricing("deepseek-chat") == (0.14, 0.28)
    assert find_pricing("deepseek-r1") == (0.55, 2.19)


def test_calculate_cost_exact():
    # 1,000 prompt tokens and 500 completion tokens on gpt-4o
    # prompt: (1000 / 1e6) * 2.50 = 0.0025
    # comp:   (500 / 1e6) * 10.00 = 0.0050
    # total:  0.0075
    res = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    assert res["model"] == "gpt-4o"
    assert res["prompt_tokens"] == 1000
    assert res["completion_tokens"] == 500
    assert res["total_tokens"] == 1500
    assert res["prompt_cost"] == 0.0025
    assert res["completion_cost"] == 0.005
    assert res["total_cost"] == 0.0075
    assert res["is_estimated"] is True


def test_custom_pricing_registration():
    register_model_pricing("internal-fine-tuned-llm", prompt_per_million=1.20, completion_per_million=3.40)
    cost = calculate_cost("internal-fine-tuned-llm", prompt_tokens=100_000, completion_tokens=50_000)
    # prompt: (100k / 1M) * 1.20 = 0.12
    # comp:   (50k / 1M) * 3.40  = 0.17
    # total:  0.29
    assert cost["prompt_cost"] == 0.12
    assert cost["completion_cost"] == 0.17
    assert cost["total_cost"] == 0.29
    assert cost["is_estimated"] is True


def test_trace_metrics_aggregation_with_cost():
    t = Trace(id="t_cost", name="agent_run")
    s1 = Span(
        id="s_llm1",
        trace_id="t_cost",
        name="LLM: gpt-4o",
        span_type=SpanType.LLM,
        metadata={
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
            }
        },
    )
    s1.finish(status=SpanStatus.COMPLETED)

    s2 = Span(
        id="s_llm2",
        trace_id="t_cost",
        name="LLM: claude-3-5-haiku",
        span_type=SpanType.LLM,
        metadata={
            "usage": {
                "prompt_tokens": 10_000,
                "completion_tokens": 2_000,
                "total_tokens": 12_000,
            }
        },
    )
    s2.finish(status=SpanStatus.COMPLETED)

    t.spans = [s1, s2]
    t.finish(status=SpanStatus.COMPLETED)
    t.compute_metrics()

    assert t.llm_metrics.total_tokens == 13500
    assert t.llm_metrics.prompt_tokens == 11000
    assert t.llm_metrics.completion_tokens == 2500
    assert t.llm_metrics.llm_calls == 2

    # s1 cost: 0.0075
    # s2 cost: prompt (10k/1M)*0.80 = 0.0080; comp (2k/1M)*4.00 = 0.0080 -> total 0.0160
    # trace cost: 0.0075 + 0.0160 = 0.0235
    assert t.estimated_cost == pytest.approx(0.0235, abs=1e-5)
    assert t.llm_metrics.total_cost == pytest.approx(0.0235, abs=1e-5)
    assert "gpt-4o" in t.llm_metrics.model_usage
    assert "claude-3-5-haiku" in t.llm_metrics.model_usage


def test_storage_cost_persistence_and_stats():
    # 1. Memory Storage
    mem = MemoryStorage()
    t = Trace(id="tr_1", name="llm_agent")
    s = Span(
        id="s1",
        trace_id="tr_1",
        name="LLM: gpt-4o-mini",
        span_type=SpanType.LLM,
        metadata={"usage": {"prompt_tokens": 2000, "completion_tokens": 1000, "total_tokens": 3000}},
    )
    s.finish(status=SpanStatus.COMPLETED)
    t.spans = [s]
    t.finish(status=SpanStatus.COMPLETED)

    mem.save_trace(t)
    summaries = mem.list_traces()
    assert len(summaries) == 1
    assert summaries[0].estimated_cost > 0.0
    stats = mem.get_stats()
    assert stats.total_cost > 0.0

    # 2. SQLite Storage
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_cost.db")
        sqlite_store = SQLiteStorage(db_path=db_path)
        sqlite_store.save_trace(t)

        sq_summaries = sqlite_store.list_traces()
        assert len(sq_summaries) == 1
        assert sq_summaries[0].estimated_cost > 0.0
        sq_stats = sqlite_store.get_stats()
        assert sq_stats.total_cost > 0.0
        sqlite_store.close()

    # 3. SQLAlchemy Storage
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = os.path.join(tmp_dir, "test_sql_cost.db")
        sql_store = SQLAlchemyStorage(database_url=f"sqlite:///{db_file}")
        sql_store.save_trace(t)

        sql_summaries = sql_store.list_traces()
        assert len(sql_summaries) == 1
        assert sql_summaries[0].estimated_cost > 0.0
        sql_stats = sql_store.get_stats()
        assert sql_stats.total_cost > 0.0
        sql_store.close()


def test_dynamic_pricing_and_custom_loaders(caplog):
    """Verify runtime registry updates, dictionary loaders, JSON file loaders, and unknown model fallback."""
    # 1. Latest models are recognized
    models_to_test = [
        ("claude-3-7-sonnet", 3.0, 15.0),
        ("gpt-4.5", 75.0, 150.0),
        ("gemini-2.5-flash", 0.15, 0.60),
        ("o4-mini", 1.10, 4.40),
        ("deepseek-v3", 0.14, 0.28),
    ]
    for m, p, c in models_to_test:
        pricing = find_pricing(m)
        assert pricing is not None, f"Model {m} not found in pricing"
        assert pricing[0] == p
        assert pricing[1] == c

    # 2. Dynamic dictionary loader
    loaded = load_pricing_from_dict({
        "my-custom-llm-1": [1.25, 5.00],
        "my-custom-llm-2": {"prompt": 0.50, "completion": 2.00},
    })
    assert loaded == 2
    assert find_pricing("my-custom-llm-1") == (1.25, 5.00)
    assert find_pricing("my-custom-llm-2") == (0.50, 2.00)

    # 3. Dynamic JSON file loader
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump({"file-loaded-model": [0.42, 1.42]}, tf)
        tmp_name = tf.name

    try:
        file_loaded = load_pricing_from_file(tmp_name)
        assert file_loaded == 1
        assert find_pricing("file-loaded-model") == (0.42, 1.42)
    finally:
        os.remove(tmp_name)

    # 4. Unknown model warning logged once
    with caplog.at_level(logging.WARNING):
        cost_info = calculate_cost("some-totally-unknown-model-xyz", 1000, 500)
        assert cost_info["total_cost"] == 0.0
        assert cost_info["is_estimated"] is False
        assert any("Unknown model 'some-totally-unknown-model-xyz'" in record.message for record in caplog.records)

