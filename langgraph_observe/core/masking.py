from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple


DEFAULT_SENSITIVE_KEYS: Set[str] = {
    "password",
    "passwd",
    "secret",
    "token",
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "set-cookie",
    "private_key",
    "client_secret",
    "refresh_token",
    "ssn",
    "session",
    "session_id",
}


DEFAULT_REGEX_RULES: List[Tuple[str, str]] = [
    # Email addresses
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL_REDACTED]"),
    # Credit card numbers (16 digits separated by spaces or hyphens)
    (r"\b(?:\d{4}[ -]?){3}\d{4}\b", "[CARD_REDACTED]"),
    # US Social Security Numbers
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]"),
    # API Keys / Bearer tokens (e.g. sk-..., bearer ey...)
    (r"(?i)\b(?:sk-[a-zA-Z0-9_-]{20,}|bearer\s+[a-zA-Z0-9._~+/-]{15,}=*)\b", "[KEY_REDACTED]"),
    # Phone numbers (US and standard international formats)
    (r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]"),
]


class PIIMasker:
    """High-performance client-side PII and sensitive data scrubber.

    Sanitizes strings, dictionary keys, and nested structures before transmission
    to prevent sensitive data leaks in observability logs.
    """

    def __init__(
        self,
        sensitive_keys: Optional[Set[str]] = None,
        regex_rules: Optional[List[Tuple[str, str]]] = None,
        custom_patterns: Optional[List[Tuple[str, str]]] = None,
        extra_sensitive_keys: Optional[Set[str]] = None,
    ) -> None:
        self.sensitive_keys = set(sensitive_keys if sensitive_keys is not None else DEFAULT_SENSITIVE_KEYS)
        if extra_sensitive_keys:
            self.sensitive_keys.update(k.lower() for k in extra_sensitive_keys)

        rules = list(regex_rules if regex_rules is not None else DEFAULT_REGEX_RULES)
        if custom_patterns:
            rules.extend(custom_patterns)

        self._compiled_rules: List[Tuple[Pattern[str], str]] = [
            (re.compile(pattern), repl) for pattern, repl in rules
        ]

    def add_rule(self, pattern: str, replacement: str) -> None:
        """Add a custom regex redaction rule."""
        self._compiled_rules.append((re.compile(pattern), replacement))

    def add_sensitive_keys(self, keys: List[str]) -> None:
        """Add additional dictionary key names that should be automatically redacted."""
        for k in keys:
            self.sensitive_keys.add(k.lower())

    def mask_string(self, text: str) -> str:
        """Apply all compiled regex rules to sanitize a string."""
        if not text:
            return text
        result = text
        for pattern, replacement in self._compiled_rules:
            result = pattern.sub(replacement, result)
        return result

    def mask_text(self, text: str) -> str:
        """Alias for mask_string."""
        return self.mask_string(text)

    def is_sensitive_key(self, key: str) -> bool:
        """Check if a dictionary key name indicates sensitive credential data."""
        k = key.lower().strip()
        if k in self.sensitive_keys:
            return True
        tokens = set(re.split(r"[_\-.]+", k))
        return bool(tokens & self.sensitive_keys)

    def mask(self, data: Any, max_depth: int = 15) -> Any:
        """Recursively sanitize data structures (dicts, lists, primitives)."""
        if max_depth <= 0:
            return data

        if isinstance(data, str):
            return self.mask_string(data)

        if isinstance(data, dict):
            masked_dict: Dict[str, Any] = {}
            for k, v in data.items():
                k_str = str(k)
                if self.is_sensitive_key(k_str):
                    masked_dict[k] = "[REDACTED]"
                else:
                    masked_dict[k] = self.mask(v, max_depth - 1)
            return masked_dict

        if isinstance(data, (list, tuple, set)):
            masked_seq = [self.mask(item, max_depth - 1) for item in data]
            if isinstance(data, tuple):
                return tuple(masked_seq)
            if isinstance(data, set):
                try:
                    return set(masked_seq)
                except TypeError:
                    return masked_seq
            return masked_seq

        return data


_default_masker: Optional[PIIMasker] = None


def get_default_masker() -> PIIMasker:
    """Get or initialize the global default PIIMasker instance."""
    global _default_masker
    if _default_masker is None:
        _default_masker = PIIMasker()
    return _default_masker


def mask_pii(data: Any) -> Any:
    """Convenience helper to scrub PII from data using the default masker."""
    return get_default_masker().mask(data)
