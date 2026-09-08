from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, Set, Tuple
import urllib.request

logger = logging.getLogger("langgraph_observe.pricing")

# Standard Pricing in USD per 1,000,000 tokens (Prompt / Completion)
DEFAULT_MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI Next-Gen & Current Models
    "gpt-6-astra": (10.00, 50.00),
    "gpt-6": (10.00, 50.00),
    "gpt-5.6-sol": (4.00, 20.00),
    "gpt-5.6": (4.00, 20.00),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.5-pro": (30.00, 180.00),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4-pro": (15.00, 90.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.25, 2.00),
    "gpt-5.4-nano": (0.05, 0.40),
    "gpt-5.3-codex": (2.00, 12.00),
    "gpt-5.2-pro": (10.50, 84.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.1": (1.50, 12.00),
    "gpt-5-pro": (30.00, 180.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5": (5.00, 30.00),
    "o3-pro": (10.00, 40.00),
    "o3": (1.00, 4.00),
    "o4-mini": (1.10, 4.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    # OpenAI Standard & Realtime Models
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-11-20": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-2024-05-13": (5.00, 15.00),
    "chatgpt-4o-latest": (5.00, 15.00),
    "gpt-4o-realtime-preview": (5.00, 20.00),
    "gpt-4o-realtime-preview-2024-12-17": (5.00, 20.00),
    "gpt-4o-realtime-preview-2024-10-01": (5.00, 20.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4o-mini-realtime-preview": (0.60, 2.40),
    "gpt-4o-mini-realtime-preview-2024-12-17": (0.60, 2.40),
    "gpt-4.5": (75.00, 150.00),
    "gpt-4.5-preview": (75.00, 150.00),
    "gpt-4.5-preview-2025-02-27": (75.00, 150.00),
    "o1": (15.00, 60.00),
    "o1-2024-12-17": (15.00, 60.00),
    "o1-preview": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o1-mini-2024-09-12": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    "o3-mini-2025-01-31": (1.10, 4.40),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4-turbo-2024-04-09": (10.00, 30.00),
    "gpt-4-turbo-preview": (10.00, 30.00),
    "gpt-4-0125-preview": (10.00, 30.00),
    "gpt-4-1106-preview": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    "gpt-3.5-turbo-instruct": (1.50, 2.00),
    "text-embedding-3-small": (0.02, 0.00),
    "text-embedding-3-large": (0.13, 0.00),
    "text-embedding-ada-002": (0.10, 0.00),
    "babbage-002": (0.40, 0.40),
    "davinci-002": (2.00, 2.00),
    # Anthropic Models
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-3-7-sonnet-20250219": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet-20240620": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    # Google Gemini Models
    "gemini-1.5-pro": (3.50, 10.50),
    "gemini-1.5-pro-latest": (3.50, 10.50),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-latest": (0.075, 0.30),
    "gemini-1.5-flash-8b": (0.0375, 0.15),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-exp": (0.10, 0.40),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (3.50, 10.50),
    # DeepSeek Models
    "deepseek-chat": (0.14, 0.28),
    "deepseek-v3": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
    "deepseek-r1": (0.55, 2.19),
    # Meta Llama (Standard Hosted Rates)
    "llama-3.3-70b": (0.59, 0.79),
    "llama-3.1-70b": (0.59, 0.79),
    "llama-3.1-8b": (0.05, 0.08),
    "llama-3-70b": (0.70, 0.90),
    "llama-3-8b": (0.10, 0.10),
    "mixtral-8x7b": (0.24, 0.24),
    "qwen-2.5-72b": (0.35, 0.40),
}

# User-registered custom overrides
CUSTOM_MODEL_PRICING: Dict[str, Tuple[float, float]] = {}
_WARNED_UNKNOWN_MODELS: Set[str] = set()

# Standard Cached Input Pricing in USD per 1,000,000 tokens
DEFAULT_CACHED_INPUT_PRICING: Dict[str, float] = {
    "gpt-6-astra": 1.00,
    "gpt-6": 1.00,
    "gpt-5.6-sol": 0.40,
    "gpt-5.6": 0.40,
    "gpt-5.6-terra": 0.20,
    "gpt-5.6-luna": 0.02,
    "gpt-5.5-pro": 3.00,
    "gpt-5.5": 0.50,
    "gpt-5.4-pro": 1.50,
    "gpt-5.4": 0.25,
    "gpt-5.4-mini": 0.025,
    "gpt-5.4-nano": 0.005,
    "gpt-5.3-codex": 0.20,
    "gpt-5.2-pro": 1.05,
    "gpt-5.2": 0.175,
    "gpt-5.1": 0.15,
    "gpt-5-pro": 3.00,
    "gpt-5-mini": 0.025,
    "gpt-5-nano": 0.005,
    "gpt-5": 0.50,
    "o3-pro": 2.50,
    "o3": 0.25,
    "o4-mini": 0.275,
    "gpt-4.1-mini": 0.10,
    "gpt-4.1": 0.50,
    "gpt-4o": 1.25,
}

CUSTOM_CACHED_INPUT_PRICING: Dict[str, float] = {}


def register_cached_input_pricing(model_name: str, cached_per_million: float) -> None:
    """Register or override cached input pricing for a specific model (USD per 1M tokens)."""
    norm = normalize_model_name(model_name)
    CUSTOM_CACHED_INPUT_PRICING[norm] = float(cached_per_million)


def find_cached_input_pricing(model_name: Optional[str]) -> Optional[float]:
    """Look up cached input pricing per 1M tokens for a given model name."""
    norm = normalize_model_name(model_name)
    if norm == "unknown":
        return None

    if norm in CUSTOM_CACHED_INPUT_PRICING:
        return CUSTOM_CACHED_INPUT_PRICING[norm]

    if norm in DEFAULT_CACHED_INPUT_PRICING:
        return DEFAULT_CACHED_INPUT_PRICING[norm]

    sorted_keys = sorted(DEFAULT_CACHED_INPUT_PRICING.keys(), key=lambda k: len(k), reverse=True)
    for key in sorted_keys:
        if norm.startswith(key):
            return DEFAULT_CACHED_INPUT_PRICING[key]

    return None


def register_model_pricing(
    model_name: str,
    prompt_per_million: float,
    completion_per_million: float,
) -> None:
    """Register or override pricing for a specific model (USD per 1M tokens)."""
    norm = normalize_model_name(model_name)
    CUSTOM_MODEL_PRICING[norm] = (float(prompt_per_million), float(completion_per_million))


def normalize_model_name(model_name: Optional[str]) -> str:
    """Normalize model identifier by stripping provider prefixes and formatting."""
    if not model_name:
        return "unknown"

    name = str(model_name).strip().lower()
    # Remove provider prefixes (e.g., openai/gpt-4o -> gpt-4o, models/gemini-1.5-pro -> gemini-1.5-pro)
    if "/" in name:
        name = name.split("/")[-1]
    if ":" in name:
        name = name.split(":")[0]

    name = name.replace("_", "-")
    return name


def find_pricing(model_name: Optional[str]) -> Optional[Tuple[float, float]]:
    """Look up prompt & completion pricing per 1M tokens for a given model name.

    Returns (prompt_per_million, completion_per_million) or None.
    """
    norm = normalize_model_name(model_name)
    if norm == "unknown":
        return None

    # 1. Check custom pricing
    if norm in CUSTOM_MODEL_PRICING:
        return CUSTOM_MODEL_PRICING[norm]

    # 2. Check exact default pricing
    if norm in DEFAULT_MODEL_PRICING:
        return DEFAULT_MODEL_PRICING[norm]

    # 3. Fuzzy matching on custom prefixes
    for key, price in CUSTOM_MODEL_PRICING.items():
        if norm.startswith(key) or key.startswith(norm):
            return price

    # 4. Fuzzy matching on standard model families
    sorted_keys = sorted(DEFAULT_MODEL_PRICING.keys(), key=lambda k: len(k), reverse=True)
    for key in sorted_keys:
        if norm.startswith(key):
            return DEFAULT_MODEL_PRICING[key]

    # 5. Generic family fallback matching
    if "gpt-6" in norm:
        return DEFAULT_MODEL_PRICING["gpt-6-astra"]
    if "gpt-5.6-terra" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.6-terra"]
    if "gpt-5.6-luna" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.6-luna"]
    if "gpt-5.6-sol" in norm or "gpt-5.6" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.6"]
    if "gpt-5.5-pro" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.5-pro"]
    if "gpt-5.5" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.5"]
    if "gpt-5.4-pro" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.4-pro"]
    if "gpt-5.4-mini" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.4-mini"]
    if "gpt-5.4-nano" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.4-nano"]
    if "gpt-5.4" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.4"]
    if "gpt-5.3-codex" in norm or "gpt-5.3" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.3-codex"]
    if "gpt-5.2-pro" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.2-pro"]
    if "gpt-5.2" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.2"]
    if "gpt-5.1" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5.1"]
    if "gpt-5-pro" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5-pro"]
    if "gpt-5-nano" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5-nano"]
    if "gpt-5-mini" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5-mini"]
    if "gpt-5" in norm:
        return DEFAULT_MODEL_PRICING["gpt-5"]
    if "gpt-4.5" in norm:
        return DEFAULT_MODEL_PRICING["gpt-4.5"]
    if "gpt-4.1-mini" in norm:
        return DEFAULT_MODEL_PRICING["gpt-4.1-mini"]
    if "gpt-4.1" in norm:
        return DEFAULT_MODEL_PRICING["gpt-4.1"]
    if "gpt-4o-mini-realtime" in norm or ("realtime" in norm and "mini" in norm):
        return DEFAULT_MODEL_PRICING["gpt-4o-mini-realtime-preview"]
    if "gpt-4o-realtime" in norm or "realtime" in norm:
        return DEFAULT_MODEL_PRICING["gpt-4o-realtime-preview"]
    if "gpt-4o-mini" in norm:
        return DEFAULT_MODEL_PRICING["gpt-4o-mini"]
    if "gpt-4o" in norm:
        return DEFAULT_MODEL_PRICING["gpt-4o"]
    if "o4-mini" in norm:
        return DEFAULT_MODEL_PRICING["o4-mini"]
    if "o3-pro" in norm:
        return DEFAULT_MODEL_PRICING["o3-pro"]
    if "o3-mini" in norm:
        return DEFAULT_MODEL_PRICING["o3-mini"]
    if "o3" in norm:
        return DEFAULT_MODEL_PRICING["o3"]
    if "o1-mini" in norm:
        return DEFAULT_MODEL_PRICING["o1-mini"]
    if "o1" in norm:
        return DEFAULT_MODEL_PRICING["o1"]
    if "text-embedding-3-small" in norm:
        return DEFAULT_MODEL_PRICING["text-embedding-3-small"]
    if "text-embedding-3-large" in norm:
        return DEFAULT_MODEL_PRICING["text-embedding-3-large"]
    if "text-embedding-ada-002" in norm:
        return DEFAULT_MODEL_PRICING["text-embedding-ada-002"]
    if "gpt-4" in norm:
        return DEFAULT_MODEL_PRICING["gpt-4-turbo"]
    if "gpt-3.5" in norm:
        return DEFAULT_MODEL_PRICING["gpt-3.5-turbo"]
    if "claude-3-7-sonnet" in norm:
        return DEFAULT_MODEL_PRICING["claude-3-7-sonnet"]
    if "claude-3-5-sonnet" in norm:
        return DEFAULT_MODEL_PRICING["claude-3-5-sonnet"]
    if "claude-3-5-haiku" in norm:
        return DEFAULT_MODEL_PRICING["claude-3-5-haiku"]
    if "claude-3-opus" in norm:
        return DEFAULT_MODEL_PRICING["claude-3-opus"]
    if "claude-3-haiku" in norm:
        return DEFAULT_MODEL_PRICING["claude-3-haiku"]
    if "gemini-2.5-flash" in norm:
        return DEFAULT_MODEL_PRICING["gemini-2.5-flash"]
    if "gemini-2.0-flash" in norm:
        return DEFAULT_MODEL_PRICING["gemini-2.0-flash"]
    if "gemini-1.5-flash" in norm:
        return DEFAULT_MODEL_PRICING["gemini-1.5-flash"]
    if "gemini-1.5-pro" in norm:
        return DEFAULT_MODEL_PRICING["gemini-1.5-pro"]
    if "deepseek-reasoner" in norm or "deepseek-r1" in norm:
        return DEFAULT_MODEL_PRICING["deepseek-reasoner"]
    if "deepseek-chat" in norm or "deepseek-v3" in norm:
        return DEFAULT_MODEL_PRICING["deepseek-chat"]

    return None


def calculate_cost(
    model_name: Optional[str],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
) -> Dict[str, Any]:
    """Calculate the estimated USD cost for a given model and token counts."""
    p_tok = max(0, int(prompt_tokens or 0))
    c_tok = max(0, int(completion_tokens or 0))
    cached_tok = max(0, int(cached_tokens or 0))
    total_tok = p_tok + c_tok

    norm_name = normalize_model_name(model_name)
    pricing = find_pricing(norm_name)

    if pricing is not None:
        p_rate, c_rate = pricing
        cached_rate = find_cached_input_pricing(norm_name)
        if cached_tok > 0 and cached_rate is not None:
            uncached_p_tok = max(0, p_tok - cached_tok)
            p_cost = (uncached_p_tok / 1_000_000.0) * p_rate + (cached_tok / 1_000_000.0) * cached_rate
        else:
            p_cost = (p_tok / 1_000_000.0) * p_rate
        c_cost = (c_tok / 1_000_000.0) * c_rate
        total_cost = p_cost + c_cost
        is_estimated = True
    else:
        p_cost = 0.0
        c_cost = 0.0
        total_cost = 0.0
        is_estimated = False
        if norm_name not in ("unknown", "") and norm_name not in _WARNED_UNKNOWN_MODELS:
            _WARNED_UNKNOWN_MODELS.add(norm_name)
            logger.warning(
                f"langgraph-observe: Unknown model '{norm_name}', cannot estimate cost. "
                f"Use register_model_pricing() or OBSERVE_PRICING_FILE to configure."
            )

    res: Dict[str, Any] = {
        "model": norm_name,
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "total_tokens": total_tok,
        "prompt_cost": round(p_cost, 6),
        "completion_cost": round(c_cost, 6),
        "total_cost": round(total_cost, 6),
        "currency": "USD",
        "is_estimated": is_estimated,
    }
    if cached_tok > 0:
        res["cached_tokens"] = cached_tok
    return res


def load_pricing_from_dict(pricing_data: Dict[str, Any]) -> int:
    """Load model pricing overrides from a dictionary.

    Format can be:
        {"my-model": [prompt_cost_per_m, completion_cost_per_m]}
        or
        {"my-model": {"prompt": prompt_cost_per_m, "completion": completion_cost_per_m}}
    """
    loaded = 0
    for model, rates in pricing_data.items():
        if isinstance(rates, (list, tuple)) and len(rates) >= 2:
            register_model_pricing(model, float(rates[0]), float(rates[1]))
            loaded += 1
        elif isinstance(rates, dict):
            p = rates.get("prompt") or rates.get("input") or rates.get("prompt_per_million") or 0.0
            c = rates.get("completion") or rates.get("output") or rates.get("completion_per_million") or 0.0
            register_model_pricing(model, float(p), float(c))
            loaded += 1
    return loaded


def load_pricing_from_file(file_path: str) -> int:
    """Load pricing overrides from a JSON file."""
    path = Path(file_path)
    if not path.is_file():
        logger.warning(f"Pricing file '{file_path}' does not exist.")
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return load_pricing_from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load pricing from '{file_path}': {e}")
    return 0


def load_pricing_from_url(url: str, timeout: float = 5.0) -> int:
    """Load pricing overrides from an HTTP/HTTPS URL returning JSON."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "langgraph-observe"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        if isinstance(data, dict):
            return load_pricing_from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load pricing from URL '{url}': {e}")
    return 0


def init_pricing() -> None:
    """Initialize pricing from environment variables or local pricing.json."""
    pricing_file = os.getenv("OBSERVE_PRICING_FILE")
    pricing_url = os.getenv("OBSERVE_PRICING_URL")

    if pricing_file:
        load_pricing_from_file(pricing_file)
    elif os.path.exists("pricing.json"):
        load_pricing_from_file("pricing.json")

    if pricing_url:
        load_pricing_from_url(pricing_url)


# Auto-initialize pricing on import
init_pricing()
