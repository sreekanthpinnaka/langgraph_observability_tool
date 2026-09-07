from __future__ import annotations

from uuid import uuid4
import pytest
from langchain_core.outputs import LLMResult, Generation, ChatGeneration
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END

from langgraph_observe.client.collector import TraceCollector, set_default_storage
from langgraph_observe.client.langgraph import observe_graph
from langgraph_observe.client.langgraph.callback import (
    LangGraphTraceCallbackHandler,
    _resolve_model_name,
)
from langgraph_observe.core.models import SpanStatus, SpanType
from langgraph_observe.server.storage.memory import MemoryStorage


def test_resolve_model_name_priority():
    """Verify priority order when resolving model names from metadata, params, and outputs."""
    # 1. llm_output wins
    assert _resolve_model_name(
        invocation_params={"model": "param_model"},
        metadata={"ls_model_name": "meta_model"},
        serialized={"name": "ser_model"},
        llm_output={"model_name": "output_model"},
    ) == "output_model"

    # 2. invocation_params wins over metadata
    assert _resolve_model_name(
        invocation_params={"model_name": "param_model"},
        metadata={"ls_model_name": "meta_model"},
        serialized={"name": "ser_model"},
    ) == "param_model"

    # 3. metadata wins over serialized
    assert _resolve_model_name(
        metadata={"ls_model_name": "meta_model"},
        serialized={"name": "ser_model"},
    ) == "meta_model"

    # 4. serialized wins over default
    assert _resolve_model_name(
        serialized={"name": "ser_model"},
        default="fallback",
    ) == "ser_model"

    # 5. default fallback
    assert _resolve_model_name(default="fallback") == "fallback"


def test_callback_lifecycle_and_memory_cleanup():
    """Verify callback properly tracks runs, records state updates, and cleans up internal maps."""
    storage = MemoryStorage()
    collector = TraceCollector(workflow_name="test_wf", storage=storage)
    handler = LangGraphTraceCallbackHandler(collector=collector)

    root_run_id = uuid4()
    child_node_id = uuid4()
    llm_run_id = uuid4()
    tool_run_id = uuid4()

    # 1. Start root workflow chain
    handler.on_chain_start(
        serialized={"name": "Workflow"},
        inputs={"question": "hello"},
        run_id=root_run_id,
    )
    assert len(handler._run_to_span_id) == 1
    assert str(root_run_id) in handler._run_to_span_id

    # 2. Start node chain (child of root)
    handler.on_chain_start(
        serialized=None,
        inputs={"question": "hello"},
        run_id=child_node_id,
        parent_run_id=root_run_id,
        metadata={"langgraph_node": "agent_node"},
    )
    assert len(handler._run_to_span_id) == 2

    # 3. Start LLM call (child of node)
    handler.on_chat_model_start(
        serialized=None,
        messages=[[AIMessage(content="hi")]],
        run_id=llm_run_id,
        parent_run_id=child_node_id,
        invocation_params={"model": "gpt-4o"},
    )
    assert len(handler._run_to_span_id) == 3

    # Verify parenting of LLM span
    llm_span = collector._spans_by_id[str(llm_run_id)]
    assert llm_span.name == "Chat: gpt-4o"
    assert llm_span.parent_id == str(child_node_id)

    # 4. End LLM call with usage metadata
    msg = AIMessage(content="I am here to help.")
    msg.usage_metadata = {"input_tokens": 15, "output_tokens": 25, "total_tokens": 40}
    generation = ChatGeneration(message=msg, text="I am here to help.")
    llm_result = LLMResult(generations=[[generation]])

    handler.on_llm_end(response=llm_result, run_id=llm_run_id)
    assert str(llm_run_id) not in handler._run_to_span_id
    assert llm_span.status == SpanStatus.COMPLETED
    assert llm_span.metadata.get("usage", {}).get("total_tokens") == 40
    assert llm_span.metadata.get("usage", {}).get("total_cost", 0.0) > 0.0

    # 5. Tool start and end
    handler.on_tool_start(
        serialized={"name": "calculator"},
        input_str="2+2",
        run_id=tool_run_id,
        parent_run_id=child_node_id,
    )
    assert str(tool_run_id) in handler._run_to_span_id

    handler.on_tool_end(output="4", run_id=tool_run_id)
    assert str(tool_run_id) not in handler._run_to_span_id

    # 6. End node chain with state update
    handler.on_chain_end(
        outputs={"answer": "4", "steps": 1},
        run_id=child_node_id,
    )
    assert str(child_node_id) not in handler._run_to_span_id

    # Verify node state update recorded
    node_updates = collector.trace.metadata.get("node_state_updates")
    assert node_updates is not None
    assert len(node_updates) == 1
    assert node_updates[0]["node"] == "agent_node"
    assert node_updates[0]["update"] == {"answer": "4", "steps": 1}

    # 7. End root chain
    handler.on_chain_end(outputs={"result": "done"}, run_id=root_run_id)
    assert str(root_run_id) not in handler._run_to_span_id

    # Clean map with zero memory leaks
    assert len(handler._run_to_span_id) == 0


def test_callback_error_handling_and_cleanup():
    """Verify error callbacks mark spans FAILED and pop active run IDs from memory."""
    storage = MemoryStorage()
    collector = TraceCollector(workflow_name="err_wf", storage=storage)
    handler = LangGraphTraceCallbackHandler(collector=collector)

    chain_id = uuid4()
    llm_id = uuid4()
    tool_id = uuid4()

    handler.on_chain_start(serialized={"name": "root"}, inputs={}, run_id=chain_id)
    handler.on_llm_start(serialized=None, prompts=["hello"], run_id=llm_id, parent_run_id=chain_id)
    handler.on_tool_start(serialized={"name": "search"}, input_str="query", run_id=tool_id, parent_run_id=chain_id)

    assert len(handler._run_to_span_id) == 3

    # Trigger errors on all
    handler.on_tool_error(ValueError("Tool failed"), run_id=tool_id)
    handler.on_llm_error(RuntimeError("API error"), run_id=llm_id)
    handler.on_chain_error(Exception("Chain failed"), run_id=chain_id)

    # All popped cleanly
    assert len(handler._run_to_span_id) == 0

    tool_span = collector._spans_by_id[str(tool_id)]
    assert tool_span.status == SpanStatus.FAILED
    assert tool_span.error["message"] == "Tool failed"

    llm_span = collector._spans_by_id[str(llm_id)]
    assert llm_span.status == SpanStatus.FAILED
    assert llm_span.error["message"] == "API error"


def test_on_llm_error_clean_handling():
    """Verify on_llm_error handles ConnectionError and records type and message."""
    collector = TraceCollector(name="error-test")
    handler = LangGraphTraceCallbackHandler(collector)

    run_id = uuid4()
    handler.on_llm_start(serialized=None, prompts=["Tell me a joke"], run_id=run_id)
    assert str(run_id) in handler._run_to_span_id

    exc = ConnectionError("OpenAI API rate limit exceeded")
    handler.on_llm_error(exc, run_id=run_id)

    # Span must be removed from tracking map
    assert str(run_id) not in handler._run_to_span_id

    # Span must be marked FAILED with complete error details
    span = collector._spans_by_id[str(run_id)]
    assert span.status == SpanStatus.FAILED
    assert span.error is not None
    assert span.error["type"] == "ConnectionError"
    assert "OpenAI API rate limit exceeded" in span.error["message"]


def test_callback_sampling_rate_zero():
    """Verify that setting sample_rate=0.0 bypasses trace collection while executing normally."""
    storage = MemoryStorage()
    set_default_storage(storage)

    builder = StateGraph(dict)
    builder.add_node("step", lambda s: {"out": "ok"})
    builder.add_edge(START, "step")
    builder.add_edge("step", END)

    graph = observe_graph(builder.compile(), name="SampledGraph", sample_rate=0.0)
    res = graph.invoke({"in": 1})
    assert res == {"out": "ok"}
    assert len(storage.list_traces()) == 0


def test_callback_llm_and_tool_pii_masking():
    """Verify that LangGraphTraceCallbackHandler masks LLM prompts, completions, and tool I/O when mask_pii=True."""
    storage = MemoryStorage()
    collector = TraceCollector(name="pii-masked-wf", storage=storage, mask_pii=True)
    handler = LangGraphTraceCallbackHandler(collector)

    root_id = uuid4()
    handler.on_chain_start(
        serialized={"name": "root_chain"},
        inputs={"user_message": "My email is secret_user@example.com"},
        run_id=root_id,
    )

    # 1. LLM Start & End with sensitive data
    llm_id = uuid4()
    handler.on_llm_start(
        serialized={"name": "gpt-4o"},
        prompts=["User SSN is 123-45-6789 and Card is 4111-2222-3333-4444"],
        run_id=llm_id,
        parent_run_id=root_id,
    )

    llm_result = LLMResult(
        generations=[[Generation(text="The user api_key is sk-proj-1234567890abcdef12345678 and email is test@company.com")]],
        llm_output={"token_usage": {"total_tokens": 50}},
    )
    handler.on_llm_end(llm_result, run_id=llm_id)

    # 2. Tool Start & End with sensitive data
    tool_id = uuid4()
    handler.on_tool_start(
        serialized={"name": "fetch_user_profile"},
        input_str="Fetch data for user with password secretpass123 and SSN 987-65-4321",
        run_id=tool_id,
        parent_run_id=root_id,
    )
    handler.on_tool_end(
        output={"status": "ok", "api_key": "sk-secret-xyz", "email": "customer@service.org"},
        run_id=tool_id,
    )

    # 3. Chain End
    handler.on_chain_end(
        outputs={"final_result": "Processed user with card 5555-4444-3333-2222"},
        run_id=root_id,
    )

    collector.finish_trace(status=SpanStatus.COMPLETED)
    trace = storage.get_trace(collector.trace.id)
    assert trace is not None

    # Verify root chain output was masked
    assert "5555-4444-3333-2222" not in str(trace.output)
    assert "[CARD_REDACTED]" in str(trace.output)

    # Verify LLM Span inputs & outputs
    llm_span = next(s for s in trace.spans if s.span_type == SpanType.LLM)
    assert "123-45-6789" not in str(llm_span.inputs)
    assert "4111-2222-3333-4444" not in str(llm_span.inputs)
    assert "[SSN_REDACTED]" in str(llm_span.inputs)
    assert "[CARD_REDACTED]" in str(llm_span.inputs)

    assert "sk-proj-1234567890abcdef12345678" not in str(llm_span.outputs)
    assert "test@company.com" not in str(llm_span.outputs)
    assert "[KEY_REDACTED]" in str(llm_span.outputs)
    assert "[EMAIL_REDACTED]" in str(llm_span.outputs)

    # Verify Tool Span inputs & outputs
    tool_span = next(s for s in trace.spans if s.span_type == SpanType.TOOL)
    assert "987-65-4321" not in str(tool_span.inputs)
    assert "[SSN_REDACTED]" in str(tool_span.inputs)

    assert "sk-secret-xyz" not in str(tool_span.outputs)
    assert "customer@service.org" not in str(tool_span.outputs)
    output_dict = tool_span.outputs.get("output") if isinstance(tool_span.outputs, dict) else {}
    assert output_dict.get("api_key") == "[REDACTED]"
    assert "[EMAIL_REDACTED]" in str(output_dict.get("email"))
