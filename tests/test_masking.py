from __future__ import annotations

import pytest
from langgraph_observe.core.masking import PIIMasker, get_default_masker, mask_pii
from langgraph_observe.core.serializer import safe_serialize, calculate_state_diff
from langgraph_observe.client.collector import TraceCollector
from langgraph_observe.server.storage.memory import MemoryStorage
from langgraph_observe.core.models import SpanType, SpanStatus


def test_pii_masker_regex_patterns():
    masker = PIIMasker()

    # Email
    text = "Contact user at alice.smith@example.co.uk or bob@test.com"
    masked = masker.mask_text(text)
    assert "alice.smith@example.co.uk" not in masked
    assert "bob@test.com" not in masked
    assert "[EMAIL_REDACTED]" in masked

    # Credit card
    text_cc = "Payment with 4111-2222-3333-4444 or 4111222233334444"
    masked_cc = masker.mask_text(text_cc)
    assert "4111-2222-3333-4444" not in masked_cc
    assert "[CARD_REDACTED]" in masked_cc

    # SSN
    text_ssn = "Customer SSN is 123-45-6789"
    masked_ssn = masker.mask_text(text_ssn)
    assert "123-45-6789" not in masked_ssn
    assert "[SSN_REDACTED]" in masked_ssn

    # API Keys / Bearer
    text_key = "Bearer sk-proj-1234567890abcdef12345678 and key sk-ant-api03-abcdef12345678"
    masked_key = masker.mask_text(text_key)
    assert "sk-proj-1234567890abcdef12345678" not in masked_key
    assert "[KEY_REDACTED]" in masked_key


def test_sensitive_dictionary_keys():
    masker = PIIMasker()
    payload = {
        "username": "john_doe",
        "password": "SuperSecretPassword123!",
        "api_key": "sk-secret-token",
        "credentials": {
            "bearer_token": "token-xyz",
            "refresh_token": "refresh-123",
            "session_id": "sess-abc",
        },
        "details": [
            {"cookie": "sessionId=123", "public_info": "safe to view"},
        ],
    }

    cleaned = masker.mask(payload)
    assert cleaned["username"] == "john_doe"
    assert cleaned["password"] == "[REDACTED]"
    assert cleaned["api_key"] == "[REDACTED]"
    assert cleaned["credentials"]["bearer_token"] == "[REDACTED]"
    assert cleaned["credentials"]["refresh_token"] == "[REDACTED]"
    assert cleaned["credentials"]["session_id"] == "[REDACTED]"
    assert cleaned["details"][0]["cookie"] == "[REDACTED]"
    assert cleaned["details"][0]["public_info"] == "safe to view"


def test_custom_masker_patterns():
    custom_masker = PIIMasker(
        custom_patterns=[(r"\bEMP-\d{4}\b", "[EMPLOYEE_ID_REDACTED]")],
        extra_sensitive_keys={"internal_code", "secret_sauce"},
    )

    data = {
        "employee": "John Doe EMP-9876",
        "internal_code": "ALPHA-99",
        "secret_sauce": "Ketchup + Mayo",
        "role": "Engineer",
    }

    masked = custom_masker.mask(data)
    assert "[EMPLOYEE_ID_REDACTED]" in masked["employee"]
    assert masked["internal_code"] == "[REDACTED]"
    assert masked["secret_sauce"] == "[REDACTED]"
    assert masked["role"] == "Engineer"


def test_safe_serialize_with_masking():
    data = {
        "user_email": "contact@secure.org",
        "credentials": {
            "password": "hidden_password",
        },
        "query": "Call me at +1 555-123-4567",
    }

    serialized = safe_serialize(data, mask_pii=True)
    assert "contact@secure.org" not in str(serialized)
    assert "hidden_password" not in str(serialized)
    assert serialized["credentials"]["password"] == "[REDACTED]"
    assert "[EMAIL_REDACTED]" in str(serialized)


def test_state_diff_with_masking():
    old_state = {"tokens": ["a"], "user": {"name": "Alice", "email": "old@test.com", "secret": "old_sec"}}
    new_state = {
        "tokens": ["a", "b"],
        "user": {"name": "Bob", "email": "new@test.com", "secret": "new_sec"},
    }

    diff = calculate_state_diff(old_state, new_state, mask_pii=True)
    assert "user" in diff
    user_diff = diff["user"]
    assert "old_sec" not in str(user_diff)
    assert "new_sec" not in str(user_diff)
    assert "old@test.com" not in str(user_diff)
    assert "new@test.com" not in str(user_diff)


def test_trace_collector_masking_integration():
    storage = MemoryStorage()
    collector = TraceCollector(
        name="secure_workflow",
        storage=storage,
        mask_pii=True,
        environment="production",
        project="payments",
    )

    collector.set_trace_input({"customer_email": "vip@bank.com", "password": "plaintext_pwd"})

    span_id = collector.start_span(
        name="charge_card",
        span_type=SpanType.TOOL,
        inputs={"card_num": "4111-2222-3333-4444", "api_key": "sk-charge-key"},
    )

    collector.finish_span(
        span_id=span_id,
        status=SpanStatus.COMPLETED,
        outputs={"receipt": "Receipt sent to vip@bank.com", "auth_token": "auth123"},
        state_diff={"balance": {"action": "modified", "old": "100", "new": "50"}},
    )

    collector.finish_trace(
        status=SpanStatus.COMPLETED,
        output={"status": "success", "user": "vip@bank.com"},
    )

    trace = storage.get_trace(collector.trace.id)
    assert trace is not None
    assert trace.environment == "production"
    assert trace.project == "payments"

    # Verify root trace input & output
    assert trace.input["customer_email"] == "[EMAIL_REDACTED]"
    assert trace.input["password"] == "[REDACTED]"
    assert trace.output["user"] == "[EMAIL_REDACTED]"

    # Verify span inputs & outputs
    span = trace.spans[0]
    assert "[CARD_REDACTED]" in span.inputs["card_num"]
    assert span.inputs["api_key"] == "[REDACTED]"
    assert "[EMAIL_REDACTED]" in span.outputs["receipt"]
    assert span.outputs["auth_token"] == "[REDACTED]"


def test_pii_masker_safe_set_handling():
    """Verify that masking sets with unhashable masked objects converts safely to lists without errors."""
    masker = PIIMasker()
    # A set with simple strings: returns set
    s1 = {"normal", "hello@example.com"}
    masked_s1 = masker.mask(s1)
    assert isinstance(masked_s1, set)
    assert "[EMAIL_REDACTED]" in masked_s1

    # Simulate an object whose mask returns an unhashable dict
    class UnhashableOnMask:
        def __hash__(self):
            return 42

        def __eq__(self, other):
            return self is other

    class CustomMasker(PIIMasker):
        def mask(self, data, max_depth=10):
            if isinstance(data, UnhashableOnMask):
                return {"unhashable": "dict"}
            return super().mask(data, max_depth)

    c_masker = CustomMasker()
    mixed_set = {UnhashableOnMask()}
    result = c_masker.mask(mixed_set)
    # Must successfully return list without raising TypeError
    assert isinstance(result, list)
    assert result == [{"unhashable": "dict"}]

