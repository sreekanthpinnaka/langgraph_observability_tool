from __future__ import annotations

import logging
import threading
import traceback
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from langgraph_observe.client.collector import (
    TraceCollector,
    get_default_storage,
)
from langgraph_observe.core.context import (
    get_current_collector,
    get_current_span_id,
    get_current_trace,
)
from langgraph_observe.core.masking import get_default_masker
from langgraph_observe.core.models import SpanStatus, SpanType
from langgraph_observe.core.pricing import calculate_cost
from langgraph_observe.core.serializer import safe_serialize

logger = logging.getLogger("langgraph_observe.client.langgraph")


def _resolve_model_name(
    invocation_params: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    serialized: Optional[Dict[str, Any]] = None,
    llm_output: Optional[Dict[str, Any]] = None,
    default: str = "unknown",
) -> str:
    """Deterministically resolve model name across invocation parameters, metadata, serialized configs, and LLM output."""
    if llm_output:
        name = llm_output.get("model_name") or llm_output.get("model")
        if name:
            return str(name)
    if invocation_params:
        name = invocation_params.get("model_name") or invocation_params.get("model")
        if name:
            return str(name)
    if metadata:
        name = metadata.get("ls_model_name") or metadata.get("model")
        if name:
            return str(name)
    if serialized and "name" in serialized and serialized["name"]:
        return str(serialized["name"])
    return default


class LangGraphTraceCallbackHandler(BaseCallbackHandler):
    """LangChain / LangGraph callback handler that deterministically extracts granular

    trace spans, node transitions, state diffs, LLM prompts/completions, and tool invocations.

    Guarantees concurrency safety across parallel branches, asynchronous fan-out, and nested subgraphs
    by strictly mapping LangChain's unique run_id and parent_run_id trees instead of fragile LIFO stacks.
    """

    def __init__(
        self,
        collector: Optional[TraceCollector] = None,
        workflow_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._collector = collector
        self._workflow_name = workflow_name
        self._run_to_span_id: Dict[str, str] = {}
        self._lock = threading.Lock()

    def _get_collector(self) -> TraceCollector:
        if self._collector is not None:
            return self._collector
        ctx_collector = get_current_collector()
        if ctx_collector is not None:
            return ctx_collector
        current_trace = get_current_trace()
        if current_trace is not None:
            c = TraceCollector(trace_id=current_trace.id, storage=get_default_storage())
            c.trace = current_trace
            return c
        c = TraceCollector(workflow_name=self._workflow_name or "langgraph_run")
        return c

    def _map_parent_id(self, parent_run_id: Optional[UUID]) -> Optional[str]:
        """Deterministically resolve the parent span ID from LangChain's parent_run_id tree."""
        if parent_run_id is not None:
            parent_str = str(parent_run_id)
            with self._lock:
                if parent_str in self._run_to_span_id:
                    return self._run_to_span_id[parent_str]
        return get_current_span_id()

    # --- Chains (Workflow & Nodes) ---

    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        try:
            collector = self._get_collector()
            run_id_str = str(run_id)
            parent_id = self._map_parent_id(parent_run_id)

            metadata_dict = dict(metadata or {})
            if tags:
                metadata_dict["tags"] = tags

            langgraph_node = metadata_dict.get("langgraph_node")
            if langgraph_node:
                parent_span = collector._spans_by_id.get(parent_id) if parent_id else None
                if parent_span and parent_span.name == langgraph_node:
                    span_name = f"{langgraph_node}:exec"
                    span_type = SpanType.CUSTOM
                else:
                    span_name = langgraph_node
                    span_type = SpanType.NODE
            else:
                span_name = name or (serialized.get("name") if serialized else "chain") or "workflow"
                span_type = SpanType.WORKFLOW if parent_id is None else SpanType.CUSTOM

            span = collector.start_span(
                span_id=run_id_str,
                name=span_name,
                span_type=span_type,
                parent_id=parent_id,
                inputs=inputs,
                metadata=metadata_dict,
            )

            with self._lock:
                self._run_to_span_id[run_id_str] = span.id

            if span_type == SpanType.WORKFLOW and collector.trace.input is None:
                collector.set_trace_input(inputs)

        except Exception as e:
            logger.debug(f"Error in on_chain_start: {e}")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            collector = self._get_collector()
            run_id_str = str(run_id)
            with self._lock:
                span_id = self._run_to_span_id.pop(run_id_str, run_id_str)

            span = collector.finish_span(
                span_id=span_id,
                status=SpanStatus.COMPLETED,
                outputs=outputs,
            )

            if span and span.span_type == SpanType.WORKFLOW and collector.trace.output is None:
                collector.trace.output = safe_serialize(
                    outputs,
                    mask_pii=collector.mask_pii,
                    pii_masker=collector.pii_masker,
                )

            node_name = None
            if span and span.span_type == SpanType.NODE:
                node_name = span.name
            elif span and span.metadata.get("langgraph_node"):
                node_name = span.metadata["langgraph_node"]

            if node_name and isinstance(outputs, dict):
                collector.apply_node_state_update(node_name=node_name, state_update=outputs)

        except Exception as e:
            logger.debug(f"Error in on_chain_end: {e}")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            collector = self._get_collector()
            run_id_str = str(run_id)
            with self._lock:
                span_id = self._run_to_span_id.pop(run_id_str, run_id_str)

            err_dict = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            span = collector.finish_span(
                span_id=span_id,
                status=SpanStatus.FAILED,
                error=err_dict,
            )
            if span and span.span_type == SpanType.WORKFLOW and collector.trace.error is None:
                collector.trace.error = err_dict
        except Exception as e:
            logger.debug(f"Error in on_chain_error: {e}")

    # --- LLM Invocations ---

    def _start_llm_span(
        self,
        prefix: str,
        inputs: Dict[str, Any],
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        serialized: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        default_model: str = "llm",
    ) -> None:
        try:
            collector = self._get_collector()
            run_id_str = str(run_id)
            parent_id = self._map_parent_id(parent_run_id)

            model_name = _resolve_model_name(
                invocation_params=invocation_params,
                metadata=metadata,
                serialized=serialized,
                default=default_model,
            )

            meta = dict(metadata or {})
            meta["model"] = model_name
            if invocation_params:
                meta["invocation_params"] = invocation_params
            if tags:
                meta["tags"] = tags

            span = collector.start_span(
                span_id=run_id_str,
                name=f"{prefix}: {model_name}",
                span_type=SpanType.LLM,
                parent_id=parent_id,
                inputs=inputs,
                metadata=meta,
            )
            with self._lock:
                self._run_to_span_id[run_id_str] = span.id
        except Exception as e:
            logger.debug(f"Error in _start_llm_span ({prefix}): {e}")

    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        collector = self._get_collector()
        clean_prompts = prompts
        if collector and collector.mask_pii:
            masker = collector.pii_masker or get_default_masker()
            clean_prompts = [masker.mask_text(str(p)) for p in prompts]
        self._start_llm_span(
            prefix="LLM",
            inputs={"prompts": clean_prompts},
            run_id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            metadata=metadata,
            invocation_params=invocation_params,
            tags=tags,
            default_model="llm",
        )

    def on_chat_model_start(
        self,
        serialized: Optional[Dict[str, Any]],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        collector = self._get_collector()
        m_pii = collector.mask_pii if collector else False
        p_masker = collector.pii_masker if collector else None
        flat_messages = [safe_serialize(m, mask_pii=m_pii, pii_masker=p_masker) for sub in messages for m in sub]
        self._start_llm_span(
            prefix="Chat",
            inputs={"messages": flat_messages},
            run_id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            metadata=metadata,
            invocation_params=invocation_params,
            tags=tags,
            default_model="chat_model",
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            collector = self._get_collector()
            run_id_str = str(run_id)
            with self._lock:
                span_id = self._run_to_span_id.pop(run_id_str, run_id_str)

            generations_data = []
            m_pii = collector.mask_pii if collector else False
            p_masker = collector.pii_masker if collector else None
            masker = (p_masker or get_default_masker()) if m_pii else None

            for gen_list in response.generations:
                for gen in gen_list:
                    gen_text = gen.text
                    if masker and gen_text:
                        gen_text = masker.mask_text(gen_text)
                    gen_item: Dict[str, Any] = {"text": gen_text}
                    if hasattr(gen, "message"):
                        gen_item["message"] = safe_serialize(gen.message, mask_pii=m_pii, pii_masker=p_masker)
                    generations_data.append(gen_item)

            meta_update = {}
            usage_data = {}
            if response.llm_output:
                meta_update["llm_output"] = safe_serialize(response.llm_output, mask_pii=m_pii, pii_masker=p_masker)
                for key in ("token_usage", "usage"):
                    if key in response.llm_output and isinstance(response.llm_output[key], dict):
                        toks = response.llm_output[key]
                        usage_data["total_tokens"] = toks.get("total_tokens", 0)
                        usage_data["prompt_tokens"] = (
                            toks.get("prompt_tokens", 0) or toks.get("input_tokens", 0)
                        )
                        usage_data["completion_tokens"] = (
                            toks.get("completion_tokens", 0) or toks.get("output_tokens", 0)
                        )
                        break

            # Modern LangChain chat models put usage in message.usage_metadata or response_metadata
            if not usage_data:
                for gen_list in response.generations:
                    for gen in gen_list:
                        msg = getattr(gen, "message", None)
                        if msg:
                            um = getattr(msg, "usage_metadata", None)
                            if isinstance(um, dict):
                                in_tok = um.get("input_tokens", 0)
                                out_tok = um.get("output_tokens", 0)
                                tot_tok = um.get("total_tokens", in_tok + out_tok)
                                usage_data["prompt_tokens"] = in_tok
                                usage_data["completion_tokens"] = out_tok
                                usage_data["total_tokens"] = tot_tok
                                break
                            rm = getattr(msg, "response_metadata", None)
                            if isinstance(rm, dict):
                                toks = rm.get("token_usage") or rm.get("usage")
                                if isinstance(toks, dict):
                                    in_tok = toks.get("prompt_tokens", 0) or toks.get("input_tokens", 0)
                                    out_tok = toks.get("completion_tokens", 0) or toks.get("output_tokens", 0)
                                    tot_tok = toks.get("total_tokens", in_tok + out_tok)
                                    usage_data["prompt_tokens"] = in_tok
                                    usage_data["completion_tokens"] = out_tok
                                    usage_data["total_tokens"] = tot_tok
                                    break
                    if usage_data:
                        break

            # Resolve model name
            model_name = _resolve_model_name(llm_output=response.llm_output, default="")
            if not model_name:
                span = collector._spans_by_id.get(span_id)
                if span:
                    model_name = span.metadata.get("model") or span.name.replace("LLM: ", "").replace("Chat: ", "")
            if not model_name:
                model_name = "unknown"

            if usage_data:
                p_tok = usage_data.get("prompt_tokens", 0)
                c_tok = usage_data.get("completion_tokens", 0)
                cost_info = calculate_cost(model_name, p_tok, c_tok)
                usage_data["model"] = cost_info["model"]
                usage_data["prompt_cost"] = cost_info["prompt_cost"]
                usage_data["completion_cost"] = cost_info["completion_cost"]
                usage_data["total_cost"] = cost_info["total_cost"]
                usage_data["currency"] = "USD"
                meta_update["usage"] = usage_data
                meta_update["model"] = cost_info["model"]
            elif model_name:
                meta_update["model"] = model_name

            collector.finish_span(
                span_id=span_id,
                status=SpanStatus.COMPLETED,
                outputs={"generations": generations_data},
                metadata=meta_update,
            )
        except Exception as e:
            logger.debug(f"Error in on_llm_end: {e}")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            collector = self._get_collector()
            run_id_str = str(run_id)
            with self._lock:
                span_id = self._run_to_span_id.pop(run_id_str, run_id_str)

            err_dict = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            collector.finish_span(
                span_id=span_id,
                status=SpanStatus.FAILED,
                error=err_dict,
            )
        except Exception as e:
            logger.debug(f"Error in on_llm_error: {e}")

    # --- Tool Execution ---

    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            collector = self._get_collector()
            run_id_str = str(run_id)
            parent_id = self._map_parent_id(parent_run_id)

            tool_name = (serialized.get("name") if serialized else "tool") or "tool"
            meta = dict(metadata or {})
            if tags:
                meta["tags"] = tags

            clean_input = input_str
            if collector and collector.mask_pii:
                masker = collector.pii_masker or get_default_masker()
                clean_input = masker.mask_text(str(input_str))

            span = collector.start_span(
                span_id=run_id_str,
                name=f"Tool: {tool_name}",
                span_type=SpanType.TOOL,
                parent_id=parent_id,
                inputs={"input": clean_input},
                metadata=meta,
            )
            with self._lock:
                self._run_to_span_id[run_id_str] = span.id
        except Exception as e:
            logger.debug(f"Error in on_tool_start: {e}")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            collector = self._get_collector()
            run_id_str = str(run_id)
            with self._lock:
                span_id = self._run_to_span_id.pop(run_id_str, run_id_str)

            m_pii = collector.mask_pii if collector else False
            p_masker = collector.pii_masker if collector else None
            collector.finish_span(
                span_id=span_id,
                status=SpanStatus.COMPLETED,
                outputs={"output": safe_serialize(output, mask_pii=m_pii, pii_masker=p_masker)},
            )
        except Exception as e:
            logger.debug(f"Error in on_tool_end: {e}")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            collector = self._get_collector()
            run_id_str = str(run_id)
            with self._lock:
                span_id = self._run_to_span_id.pop(run_id_str, run_id_str)

            err_dict = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
            collector.finish_span(
                span_id=span_id,
                status=SpanStatus.FAILED,
                error=err_dict,
            )
        except Exception as e:
            logger.debug(f"Error in on_tool_error: {e}")
