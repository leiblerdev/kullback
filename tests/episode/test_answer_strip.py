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


def test_stage_hash_tracks_the_moved_strip_module(monkeypatch):
    from kullback.builder import build, intent
    from kullback.user import value_strip

    original = build._module_hash
    before = build._version("strip-probe", test_stage_hash_tracks_the_moved_strip_module, intent)
    monkeypatch.setattr(build, "_module_hash", lambda module: "changed" if module is value_strip else original(module))
    after = build._version("strip-probe", test_stage_hash_tracks_the_moved_strip_module, intent)
    assert before != after


@pytest.mark.parametrize("parent_name", ["compile_env", "sandbox", "intent"])
def test_direct_module_hash_tracks_moved_implementation(monkeypatch, parent_name):
    from kullback.builder import build
    from kullback.episode import loading
    from kullback.user import value_strip

    parent = getattr(build, parent_name)
    dependency = value_strip if parent_name == "intent" else loading
    original = build._module_hash
    before = original(parent)
    monkeypatch.setattr(build, "_module_hash", lambda module: "changed" if module is dependency else original(module))
    assert build._module_hash(parent) != before


def test_unrelated_module_hash_does_not_follow_loader(monkeypatch):
    from kullback.builder import build
    from kullback.episode import loading

    original = build._module_hash
    before = original(build.provider)
    monkeypatch.setattr(build, "_module_hash", lambda module: "changed" if module is loading else original(module))
    assert build._module_hash(build.provider) == before
