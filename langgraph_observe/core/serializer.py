from __future__ import annotations

import dataclasses
from datetime import date, datetime
import json
from typing import Any, Dict, Optional
import uuid


from langgraph_observe.core.masking import PIIMasker, get_default_masker


def safe_serialize(
    obj: Any,
    max_depth: int = 8,
    _current_depth: int = 0,
    mask_pii: bool = False,
    pii_masker: Optional[PIIMasker] = None,
) -> Any:
    """Recursively convert arbitrary Python objects (LangChain messages, Pydantic models,

    dataclasses, etc.) into JSON-serializable primitives without raising exceptions.
    Optionally scrubs sensitive credentials and PII when mask_pii=True.
    """
    res = _safe_serialize_core(obj, max_depth, _current_depth)
    if _current_depth == 0 and mask_pii:
        masker = pii_masker or get_default_masker()
        return masker.mask(res)
    return res


def _safe_serialize_core(obj: Any, max_depth: int = 8, _current_depth: int = 0) -> Any:
    if _current_depth >= max_depth:
        return f"<Truncated: max depth {max_depth} reached>"

    if obj is None or isinstance(obj, (bool, int, float)):
        return obj

    if isinstance(obj, str):
        if len(obj) > 10000:
            return obj[:10000] + f"... <Truncated: {len(obj) - 10000} chars omitted>"
        return obj

    # Guard against serializing SQLAlchemy Sessions, Engines, or internal structures
    type_name = type(obj).__name__
    if type_name in ("Session", "scoped_session", "AsyncSession", "Engine", "Connection", "IdentityMap"):
        return f"<{type_name}>"

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, uuid.UUID):
        return str(obj)

    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return f"<bytes: {len(obj)} bytes>"

    # Check for LangChain BaseMessage / message-like objects
    if hasattr(obj, "type") and hasattr(obj, "content"):
        msg_dict: Dict[str, Any] = {
            "type": getattr(obj, "type", "message"),
            "content": _safe_serialize_core(getattr(obj, "content", ""), max_depth, _current_depth + 1),
        }
        if hasattr(obj, "name") and getattr(obj, "name"):
            msg_dict["name"] = getattr(obj, "name")
        if hasattr(obj, "tool_calls") and getattr(obj, "tool_calls"):
            msg_dict["tool_calls"] = _safe_serialize_core(getattr(obj, "tool_calls"), max_depth, _current_depth + 1)
        if hasattr(obj, "additional_kwargs") and getattr(obj, "additional_kwargs"):
            msg_dict["additional_kwargs"] = _safe_serialize_core(getattr(obj, "additional_kwargs"), max_depth, _current_depth + 1)
        if hasattr(obj, "response_metadata") and getattr(obj, "response_metadata"):
            msg_dict["response_metadata"] = _safe_serialize_core(getattr(obj, "response_metadata"), max_depth, _current_depth + 1)
        return msg_dict

    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        try:
            return _safe_serialize_core(obj.model_dump(), max_depth, _current_depth + 1)
        except Exception:
            pass

    # Pydantic v1
    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return _safe_serialize_core(obj.dict(), max_depth, _current_depth + 1)
        except Exception:
            pass

    # Dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return _safe_serialize_core(dataclasses.asdict(obj), max_depth, _current_depth + 1)
        except Exception:
            pass

    # Dicts
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            result[str(k)] = _safe_serialize_core(v, max_depth, _current_depth + 1)
        return result

    # Lists, tuples, sets (cap at 100 items for trace efficiency)
    if isinstance(obj, (list, tuple, set)):
        items = list(obj)
        MAX_ITEMS = 100
        if len(items) > MAX_ITEMS:
            serialized = [_safe_serialize_core(item, max_depth, _current_depth + 1) for item in items[:MAX_ITEMS]]
            serialized.append(f"<Truncated: {len(items) - MAX_ITEMS} items omitted for trace efficiency>")
            return serialized
        return [_safe_serialize_core(item, max_depth, _current_depth + 1) for item in items]

    # Exceptions
    if isinstance(obj, BaseException):
        return {
            "type": type(obj).__name__,
            "message": str(obj),
        }

    # Fallback to string representation
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        s = str(obj)
        if len(s) > 10000:
            return s[:10000] + f"... <Truncated: {len(s) - 10000} chars omitted>"
        return s


def calculate_state_diff(
    before: Optional[Any],
    after: Optional[Any],
    mask_pii: bool = False,
    pii_masker: Optional[PIIMasker] = None,
) -> Dict[str, Any]:
    """Calculate the difference between LangGraph state before and after node execution."""
    diff: Dict[str, Any] = {}

    before_dict = safe_serialize(before, mask_pii=mask_pii, pii_masker=pii_masker) if before is not None else {}
    after_dict = safe_serialize(after, mask_pii=mask_pii, pii_masker=pii_masker) if after is not None else {}

    if not isinstance(before_dict, dict) or not isinstance(after_dict, dict):
        if before_dict != after_dict:
            return {"_state_": {"action": "replaced", "old": before_dict, "new": after_dict}}
        return {}

    # Check keys in after
    for k, new_val in after_dict.items():
        if k not in before_dict:
            diff[k] = {"action": "added", "new": new_val}
        else:
            old_val = before_dict[k]
            if old_val != new_val:
                # Special handling for appended lists (like messages)
                if isinstance(old_val, list) and isinstance(new_val, list) and len(new_val) > len(old_val):
                    diff[k] = {
                        "action": "appended",
                        "appended_count": len(new_val) - len(old_val),
                        "items": new_val[len(old_val):],
                    }
                else:
                    diff[k] = {"action": "modified", "old": old_val, "new": new_val}

    return diff
