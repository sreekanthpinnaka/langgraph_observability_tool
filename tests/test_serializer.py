from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel

from langgraph_observe.core.serializer import calculate_state_diff, safe_serialize


class SamplePydantic(BaseModel):
    name: str
    count: int


@dataclass
class SampleDataClass:
    title: str
    tags: list


class MockLangChainMessage:
    def __init__(self, content: str, msg_type: str = "human") -> None:
        self.content = content
        self.type = msg_type
        self.additional_kwargs = {"token_est": 12}


def test_safe_serialize_primitives():
    assert safe_serialize(123) == 123
    assert safe_serialize("hello") == "hello"
    assert safe_serialize(True) is True
    assert safe_serialize(None) is None


def test_safe_serialize_complex_objects():
    now = datetime.now(timezone.utc)
    uid = uuid.uuid4()
    data = {
        "dt": now,
        "id": uid,
        "pydantic": SamplePydantic(name="test", count=42),
        "dataclass": SampleDataClass(title="doc", tags=["a", "b"]),
        "msg": MockLangChainMessage("Hello world", "ai"),
        "set_val": {1, 2, 3},
    }

    serialized = safe_serialize(data)
    assert serialized["dt"] == now.isoformat()
    assert serialized["id"] == str(uid)
    assert serialized["pydantic"] == {"name": "test", "count": 42}
    assert serialized["dataclass"] == {"title": "doc", "tags": ["a", "b"]}
    assert serialized["msg"]["type"] == "ai"
    assert serialized["msg"]["content"] == "Hello world"
    assert set(serialized["set_val"]) == {1, 2, 3}


def test_calculate_state_diff_added_modified():
    before = {"a": 1, "b": "old"}
    after = {"a": 1, "b": "new", "c": [1, 2]}

    diff = calculate_state_diff(before, after)
    assert diff["b"] == {"action": "modified", "old": "old", "new": "new"}
    assert diff["c"] == {"action": "added", "new": [1, 2]}
    assert "a" not in diff


def test_calculate_state_diff_appended_list():
    before = {"messages": ["hello"]}
    after = {"messages": ["hello", "how are you?"]}

    diff = calculate_state_diff(before, after)
    assert diff["messages"]["action"] == "appended"
    assert diff["messages"]["appended_count"] == 1
    assert diff["messages"]["items"] == ["how are you?"]


def test_safe_serialize_large_payload_protection():
    # Test list truncation at 100
    big_list = list(range(250))
    res_list = safe_serialize(big_list)
    assert len(res_list) == 101
    assert "<Truncated: 150 items omitted for trace efficiency>" in res_list[-1]

    # Test string truncation at 10,000 chars
    big_str = "x" * 20000
    res_str = safe_serialize(big_str)
    assert len(res_str) < 15000
    assert "<Truncated: 10000 chars omitted>" in res_str

    # Test SQLAlchemy Session protection
    class Session:
        def __init__(self):
            self.identity_map = "internal"

    sess = Session()
    assert safe_serialize(sess) == "<Session>"

