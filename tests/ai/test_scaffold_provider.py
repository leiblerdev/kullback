"""The offline models: TestModel is scripted, RecordedModel replays a stored Run, neither goes near a network."""

from __future__ import annotations

import pytest

from kullback.ai import provider as provider_module
from kullback.ai.provider import (
    Model,
    ModelConfig,
    ModelReply,
    RecordedModel,
    require_live_calls_enabled,
)


def test_live_calls_are_off_by_default():
    """conftest's autouse fixture forces the flag False in this process, so the module's own
    source is what says whether the shipped default is off."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(provider_module.__file__).read_text(encoding="utf-8"))
    assigned = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", "") == "ALLOW_MODEL_REQUESTS" for target in node.targets)
        and isinstance(node.value, ast.Constant)
    ]
    assert assigned == [False], assigned
    with pytest.raises(RuntimeError):
        require_live_calls_enabled()


def test_model_interface_is_abstract():
    """The base refuses, and every model the package ships is a Model that answers instead of
    inheriting that refusal."""
    with pytest.raises(NotImplementedError):
        Model().query([{"role": "user", "content": "hi"}])
    for cls in (provider_module.TestModel, RecordedModel, provider_module.AnthropicModel,
                provider_module.OpenAIModel, provider_module.OpenAICompatibleModel):
        assert issubclass(cls, Model), cls.__name__
        assert cls.query is not Model.query, cls.__name__
    assert isinstance(provider_module.TestModel(["ok"]).query([]), ModelReply)


def test_test_model_returns_scripted_replies_in_order(make_test_model):
    model = make_test_model(
        [
            "hello",
            {"tool_calls": [{"id": "c1", "name": "get_order_details", "arguments": {"order_id": "#W1"}}]},
            ModelReply(content="done", usage={"input": 10, "output": 2}),
        ]
    )
    first = model.query([{"role": "user", "content": "hi"}], tools=[{"name": "get_order_details"}])
    assert first.content == "hello" and first.tool_calls == []

    second = model.query([], config=ModelConfig(temperature=0.0, seed=1))
    assert second.tool_calls[0].name == "get_order_details"
    assert second.tool_calls[0].arguments == {"order_id": "#W1"}

    third = model.query([])
    assert third.usage.input == 10 and third.usage.output == 2

    assert len(model.calls) == 3
    assert model.calls[0]["tools"] == [{"name": "get_order_details"}]
    with pytest.raises(IndexError):
        model.query([])


def test_test_model_parses_string_json_arguments(make_test_model):
    model = make_test_model([{"tool_calls": [{"id": "c1", "name": "t", "arguments": '{"a": 1}'}]}])
    assert model.query([]).tool_calls[0].arguments == {"a": 1}


def test_recorded_model_replays_model_call_events(make_recorded_model):
    model = make_recorded_model(
        [
            {"idx": 0, "type": "user_turn", "payload": {"content": "cancel #W1"}},
            {
                "idx": 1,
                "type": "model_call",
                "payload": {
                    "reply": {
                        "content": None,
                        "tool_calls": [{"id": "c1", "name": "cancel_pending_order", "arguments": {"order_id": "#W1"}}],
                        "usage": {"input": 100, "output": 20},
                    }
                },
            },
            {"idx": 2, "type": "tool_result", "payload": {"content": "cancelled"}},
            {"idx": 3, "type": "model_call", "payload": {"reply": {"content": "Your order is cancelled."}}},
        ]
    )
    first = model.query([])
    assert first.tool_calls[0].name == "cancel_pending_order"
    assert first.usage.input == 100
    assert model.query([]).content == "Your order is cancelled."
    with pytest.raises(IndexError):
        model.query([])


def test_recorded_model_reads_plain_assistant_messages(make_recorded_model):
    model = make_recorded_model(
        [
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": "cancel #W1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "c1", "name": "find_user_id_by_name_zip", "arguments": {"zip": "28236"}}],
                "usage": {"prompt_tokens": 5344, "completion_tokens": 103},
            },
        ],
        name="trace.jsonl",
    )
    assert model.query([]).content.startswith("Hi!")
    reply = model.query([])
    assert reply.tool_calls[0].arguments == {"zip": "28236"}
    assert (reply.usage.input, reply.usage.output) == (5344, 103)


def test_replies_are_copies_so_a_caller_cannot_mutate_the_script(make_test_model):
    model = make_test_model([ModelReply(content="x")], loop=True)
    first = model.query([])
    first.content = "changed"
    assert model.query([]).content == "x"


def test_recorded_model_is_a_model(make_recorded_model):
    """It is a Model by type and by behaviour: the interface's arguments are accepted and the
    stored reply comes back, with no network and no live-call flag involved."""
    model = make_recorded_model([{"role": "assistant", "content": "a"}])
    assert isinstance(model, Model) and issubclass(RecordedModel, Model)
    reply = model.query([{"role": "user", "content": "hi"}], tools=[{"name": "t"}],
                        config=ModelConfig(temperature=0.0))
    assert isinstance(reply, ModelReply) and reply.content == "a"
    assert model.calls[0]["tools"] == [{"name": "t"}]
