import json

import pytest

from kullback.runner import loop

pytestmark = pytest.mark.skipif(not hasattr(loop, "advance"), reason="needs the step-split patch")


def test_episode_uses_the_shared_strip_on_nonheld_evidence(tmp_path):
    from kullback.episode import BuiltEnvironment, Episode
    from kullback.runner.records import RawPtr, ToolCall, Trace
    from tests.episode.invented import write_env

    root = write_env(tmp_path / "env")
    ptr = RawPtr(file_hash="invented")
    trace = Trace(trace_id="rec1", raw_hash="invented", ingest_version="1", source="invented", raw_ptr=ptr,
                  tool_calls=[ToolCall(name="describe_widget", args={"widget_id": "w1"},
                                       result={"label": "company-secret"}, raw_ptr=ptr)])
    (root / "traces").mkdir()
    (root / "traces" / "rec1.json").write_text(trace.model_dump_json(), encoding="utf-8")
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    episode.reset("widget_task", seed=7)
    strip = episode._state.user.answer_strip
    assert strip is not None
    assert strip("company-secret")[1]


def test_episode_does_not_build_strip_from_held_out_trace(tmp_path):
    from kullback.episode import BuiltEnvironment, Episode
    from kullback.runner.records import RawPtr, Trace
    from tests.episode.invented import write_env

    root = write_env(tmp_path / "env")
    trace = Trace(trace_id="rec1", raw_hash="invented", ingest_version="1", source="invented",
                  raw_ptr=RawPtr(file_hash="invented"))
    (root / "traces").mkdir()
    (root / "traces" / "rec1.json").write_text(trace.model_dump_json(), encoding="utf-8")
    (root / "anchor.json").write_text(json.dumps({"held_out": {"widget_task": ["rec1"]}}), encoding="utf-8")
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    episode.reset("widget_task", seed=7)
    assert episode._state.user is None


def test_builder_and_episode_share_the_strip_implementation():
    from kullback.builder import intent
    from kullback.user import value_strip

    assert intent.value_strip is value_strip.value_strip
    assert intent.strip_intent is value_strip.strip_intent


@pytest.mark.parametrize("owner_name,dependency_name", [
    ("kullback.builder.compile_env", "kullback.episode.loading"),
    ("kullback.builder.sandbox", "kullback.episode.loading"),
    ("kullback.builder.intent", "kullback.user.value_strip"),
])
def test_closure_hash_of_owner_follows_the_moved_module(owner_name, dependency_name):
    """The G1 move put the implementation in episode.loading and user.value_strip; the closure
    hash carries those bytes inside the owning module's hash, so an edit there moves it."""
    import importlib

    from kullback.builder import cache_reach as reach
    assert dependency_name in reach.import_closure(owner_name)
    owner = importlib.import_module(owner_name)
    assert reach.closure_hash(owner) != reach.closure_hash(owner, exempt=frozenset({dependency_name}))


def test_unrelated_module_hash_does_not_follow_loader(monkeypatch):
    from kullback.builder import build
    from kullback.episode import loading

    original = build._module_hash
    before = original(build.provider)
    monkeypatch.setattr(build, "_module_hash", lambda module: "changed" if module is loading else original(module))
    assert build._module_hash(build.provider) == before
