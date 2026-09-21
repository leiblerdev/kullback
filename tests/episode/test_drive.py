"""Driving a Run from outside: reset, scripted steps, a reward by code (G1).

Needs the step-split patch (docs/frozen-patches/step-split.patch): the interface steps the world
through `advance`, so this module skips unless the patch is applied.
"""

from __future__ import annotations

import pytest

from kullback.runner import loop

pytestmark = pytest.mark.skipif(
    not (hasattr(loop, "ask") and hasattr(loop, "advance")),
    reason="needs the step-split patch",
)

from kullback.episode import BuiltEnvironment, Episode, EpisodeError  # noqa: E402
from tests.episode.invented import SCRIPTED_RENAME, write_env  # noqa: E402


def drive(env_root, outdir, messages) -> Episode:
    env = BuiltEnvironment(env_root)
    episode = Episode(env, outdir=outdir)
    episode.reset("widget_task", seed=7)
    for message in messages:
        out = episode.step(message)
        if out.done:
            break
    return episode


def test_reset_then_scripted_steps_runs_the_task_to_its_stop():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root, out = Path(tmp) / "env", Path(tmp) / "out"
        write_env(root)
        episode = drive(root, out, SCRIPTED_RENAME)
        assert episode._state.stopped
        assert episode._state.run.termination_reason in ("user_stop", "agent_stop")
        names = [e.payload.get("name") for e in episode._state.run.events if e.type == "tool_call"]
        assert names == ["describe_widget", "rename_widget"]
        routes = {e.payload.get("name") for e in episode._state.run.events
                  if e.type == "tool_result"}
        assert routes == {"describe_widget", "rename_widget"}
        assert (out / "widget_task").is_dir()


def test_reset_hands_the_policy_what_it_needs_to_act():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "env"
        write_env(root)
        env = BuiltEnvironment(root)
        episode = Episode(env, outdir=Path(tmp) / "out")
        info = episode.reset("widget_task", seed=7)
        assert info.run_id == "widget_task-7"
        assert {spec["name"] for spec in info.tools} == {"describe_widget", "rename_widget"}
        assert info.system_prompt is not None
        assert info.opening is not None


def test_same_seed_same_task_same_messages_gives_the_same_run():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "env"
        write_env(root)
        first = drive(root, Path(tmp) / "out1", SCRIPTED_RENAME).run_record()
        second = drive(root, Path(tmp) / "out2", SCRIPTED_RENAME).run_record()
        assert first["events"] == second["events"]
        assert first["seed"] == second["seed"]


def test_finished_episode_refuses_further_steps():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "env"
        write_env(root)
        episode = drive(root, Path(tmp) / "out", SCRIPTED_RENAME)
        with pytest.raises(EpisodeError):
            episode.step({"content": "hello", "tool_calls": []})


def test_step_before_reset_is_refused():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "env"
        write_env(root)
        episode = Episode(BuiltEnvironment(root), outdir=Path(tmp) / "out")
        with pytest.raises(EpisodeError):
            episode.step({"content": "hello", "tool_calls": []})


def test_reward_scores_a_finished_run_by_code():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "env"
        write_env(root)
        episode = drive(root, Path(tmp) / "out", SCRIPTED_RENAME)
        reward = episode.reward()
        assert reward.score == 1
        assert reward.class_ == "pass"


def test_reward_withholds_a_number_where_a_judge_must_hold():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "env"
        write_env(root, verifier_atoms=[
            {"id": "j1", "kind": "hard", "judge": True,
             "predicate_src": "wrote('rename_widget')", "target": {"kind": "write", "tool": "rename_widget"}},
        ])
        episode = drive(root, Path(tmp) / "out", SCRIPTED_RENAME)
        reward = episode.reward()
        assert reward.score is None
        assert reward.class_ == "not_verdicted"
        assert reward.reason != ""


def test_reset_preserves_the_supplied_seed(tmp_path):
    root = write_env(tmp_path / "env")
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    assert episode.reset("widget_task", seed=123).seed == 123
    assert episode.run_record()["seed"] == 123


def test_repeated_seed_keeps_each_completed_run_file(tmp_path):
    root = write_env(tmp_path / "env")
    episode = drive(root, tmp_path / "out", SCRIPTED_RENAME)
    first_path = episode.run_path()
    first_bytes = first_path.read_bytes()
    episode.reset("widget_task", seed=7)
    assert episode.run_path() != first_path
    assert first_path.read_bytes() == first_bytes


def test_reset_refuses_to_abandon_an_unfinished_run(tmp_path):
    root = write_env(tmp_path / "env")
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    episode.reset("widget_task", seed=7)
    before = episode.run_path().read_bytes()
    with pytest.raises(EpisodeError, match="finish the current Run"):
        episode.reset("widget_task", seed=8)
    assert episode.run_path().read_bytes() == before


def test_a_package_without_user_rules_opens_with_task_intent(tmp_path):
    root = write_env(tmp_path / "env", with_rules=False)
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    reset = episode.reset("widget_task", seed=7)
    assert reset.opening == "give widget w1 the label striped"
    assert episode.transcript()[-1] == {"role": "user", "content": reset.opening}


def test_returned_transcript_cannot_mutate_recorded_tool_calls(tmp_path):
    root = write_env(tmp_path / "env")
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    episode.reset("widget_task", seed=7)
    episode.step(SCRIPTED_RENAME[0])
    transcript = episode.transcript()
    assistant = next(m for m in transcript if m.get("tool_calls"))
    assistant["tool_calls"][0]["name"] = "changed_by_caller"
    assert next(m for m in episode.transcript() if m.get("tool_calls"))["tool_calls"][0]["name"] == "describe_widget"


def test_reset_tool_specs_are_detached_from_recording(tmp_path):
    root = write_env(tmp_path / "env")
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    reset = episode.reset("widget_task", seed=7)
    reset.tools[0]["parameters"]["properties"].clear()
    assert episode._state.tools[0]["parameters"]["properties"]
    assert episode._tools[0]["parameters"]["properties"]


def test_step_does_not_retain_callers_argument_objects(tmp_path):
    import copy
    root = write_env(tmp_path / "env")
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    episode.reset("widget_task", seed=7)
    message = copy.deepcopy(SCRIPTED_RENAME[0])
    episode.step(message)
    message["tool_calls"][0]["arguments"]["widget_id"] = "changed"
    recorded = next(m for m in episode.transcript() if m.get("tool_calls"))
    assert recorded["tool_calls"][0]["arguments"]["widget_id"] == "w1"
    call = next(e for e in episode._state.run.events if e.type == "tool_call")
    assert call.payload["args"]["widget_id"] == "w1"


def test_step_results_cannot_mutate_world_or_recording(tmp_path):
    root = write_env(tmp_path / "env")
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    episode.reset("widget_task", seed=7)
    row = episode._router.tools.db.widgets["w1"]
    episode._router.tools.describe_widget = lambda widget_id: row.__dict__
    result = episode.step(SCRIPTED_RENAME[0])
    result.results[0]["result"]["label"] = "changed"
    assert row.label == "plain"
    recorded = next(e for e in episode._state.run.events if e.type == "tool_result")
    assert recorded.payload["result"]["label"] == "plain"
