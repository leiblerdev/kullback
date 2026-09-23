"""Birth checks for a Task, by code alone: a walk passes, do-nothing fails, args trace.

The driver is a fake episode over widgets: describe_widget reads a label, rename_widget
writes it. A plain scoring function stands in for the Verifier.
"""

from __future__ import annotations

import pytest

from kullback.runner import loop
from kullback.runner.world.birth import (
    birth_checks,
    check_arguments_trace,
    check_do_nothing_fails,
    check_forbidden_step_trips,
    check_walk_passes,
)
from kullback.runner.world.episode import EpisodeError

INTENT = "give widget w1 the label striped"
FACTS = [{"field": "widget_id", "value": "w1"}, {"field": "label", "value": "striped"}]
GOOD_WALK = [
    {"name": "describe_widget", "arguments": {"widget_id": "w1"}},
    {"name": "rename_widget", "arguments": {"widget_id": "w1", "label": "striped"}},
]


def passing_scorer(calls):
    """The verifier stand-in: 1 when a rename ran, else 0."""
    if any(call["name"] == "rename_widget" for call in calls):
        return (1, "pass", "")
    return (0, "fail", "no rename ran")


class _Reset:
    def __init__(self, opening):
        self.tools = [{"name": "describe_widget"}, {"name": "rename_widget"}]
        self.opening = opening


class _StepOut:
    def __init__(self, results, done):
        self.results = results
        self.user_message = "noted"
        self.done = done


class _Reward:
    def __init__(self, score, class_, reason):
        self.score = score
        self.class_ = class_
        self.reason = reason


class FakeEpisode:
    """In-memory driver over one widget: describe reads, rename writes, scorer judges."""

    def __init__(self, *, scorer=None, finish=True, fault=None, omit_first_result=False):
        self.db = {"w1": "plain"}
        self.calls = []
        self.steps = 0
        self.resets = 0
        self.task_id = None
        self._scorer = scorer or passing_scorer
        self._finish = finish
        self._fault = fault
        self._omit_first_result = omit_first_result

    def reset(self, task_id, seed=0):
        if self._fault == "reset":
            raise EpisodeError("reset blew up")
        self.calls = []
        self.steps = 0
        self.resets += 1
        self.task_id = task_id
        return _Reset(INTENT)

    def step(self, message):
        if self._fault == "step":
            raise EpisodeError("step blew up")
        self.steps += 1
        results = []
        for position, call in enumerate(message.get("tool_calls", []) or []):
            entry = self._run(call)
            entry["id"] = call.get("id")
            if position == 0 and self.steps == 1 and self._omit_first_result:
                continue
            results.append(entry)
        return _StepOut(results, bool(self._finish))

    def _run(self, call):
        name = call.get("name")
        args = call.get("arguments", {}) or {}
        self.calls.append({"name": name, "arguments": args})
        if name == "describe_widget":
            return self._describe(name, args)
        if name == "rename_widget":
            return self._rename(name, args)
        return {"name": name, "result": None, "error": f"no such tool: {name}"}

    def _describe(self, name, args):
        label = self.db.get(args.get("widget_id"))
        if label is None:
            return {"name": name, "result": None, "error": "unknown widget"}
        return {"name": name, "result": {"widget_id": args.get("widget_id"), "label": label},
                "error": None}

    def _rename(self, name, args):
        if args.get("widget_id") not in self.db:
            return {"name": name, "result": None, "error": "unknown widget"}
        self.db[args.get("widget_id")] = args.get("label")
        return {"name": name, "result": {"widget_id": args.get("widget_id"),
                                        "label": args.get("label")}, "error": None}

    def reward(self):
        if self._fault == "reward":
            raise EpisodeError("reward blew up")
        score, class_, reason = self._scorer(self.calls)
        return _Reward(score, class_, reason)

    def run_file(self):
        slug = "+".join(_call_slug(call) for call in self.calls)
        return "fake-run/" + (slug or "empty")


def _call_slug(call):
    pairs = ",".join(key + "=" + str(value) for key, value in call["arguments"].items())
    return call["name"] + "(" + pairs + ")"


FLAT_WALK = [
    {"name": "describe_widget", "arguments": {"widget_id": "w1"}},
    {"name": "rename_widget", "arguments": {"widget_id": "w1", "label": "flat"}},
]


def flags_flat_run(path):
    """Test-local predicate: trips only on a run that renamed to the flat label."""
    return ["forbidden rename"] if "label=flat" in path else []


def test_good_walk_is_born():
    report = birth_checks(FakeEpisode(), "widget_task", GOOD_WALK, intent=INTENT,
                          user_facts=FACTS)
    assert report.born
    assert report.walk_passes.outcome == "held"
    assert report.do_nothing_fails.outcome == "held"
    assert report.arguments_trace.outcome == "held"
    assert report.forbidden_step.outcome == "not_run"
    guarded = birth_checks(FakeEpisode(), "widget_task", GOOD_WALK, intent=INTENT,
                           user_facts=FACTS, forbidden_walk=FLAT_WALK,
                           violations_of=flags_flat_run)
    assert guarded.forbidden_step.outcome == "held"
    assert guarded.born


def test_verifier_that_passes_do_nothing_is_not_born_and_says_why():
    episode = FakeEpisode(scorer=lambda calls: (1, "pass", ""))
    report = birth_checks(episode, "widget_task", GOOD_WALK, intent=INTENT, user_facts=FACTS)
    assert not report.born
    assert report.do_nothing_fails.outcome == "failed"
    assert "did nothing" in report.do_nothing_fails.reason


def test_do_nothing_without_signal_is_no_signal_and_not_born():
    def scorer(calls):
        if calls:
            return (1, "pass", "")
        return (None, "not_verdicted", "a judge must hold")

    out = check_do_nothing_fails(FakeEpisode(scorer=scorer), "widget_task")
    assert out.outcome == "no_signal"
    report = birth_checks(FakeEpisode(scorer=scorer), "widget_task", GOOD_WALK, intent=INTENT,
                          user_facts=FACTS)
    assert not report.born


def test_argument_from_nowhere_is_reported_with_index_path_and_value():
    walk = [{"name": "rename_widget", "arguments": {"widget_id": "w1", "label": "mango"}}]
    traced = check_arguments_trace(walk, [], intent=INTENT, user_facts=FACTS)
    assert traced.outcome == "failed"
    assert [(u.call_index, u.path, u.value) for u in traced.untraced] == [(0, "label", "mango")]


def test_result_of_a_call_is_not_earlier_for_that_same_call():
    walk = [
        {"name": "describe_widget", "arguments": {"widget_id": "w1"}},
        {"name": "rename_widget", "arguments": {"widget_id": "w1", "label": "mango"}},
    ]
    episode = FakeEpisode(omit_first_result=True)
    walked = check_walk_passes(episode, "widget_task", walk)
    assert [entry["id"] for entry in walked.results] == ["call-2"]
    assert walked.results[0]["result"]["label"] == "mango"
    traced = check_arguments_trace(walk, walked.results, intent=INTENT, user_facts=FACTS)
    assert traced.outcome == "failed"
    assert [(u.call_index, u.path, u.value) for u in traced.untraced] == [(1, "label", "mango")]


def test_value_only_in_a_later_result_is_still_untraced():
    walk = [
        {"name": "rename_widget", "arguments": {"widget_id": "w1", "label": "mango"}},
        {"name": "describe_widget", "arguments": {"widget_id": "w1"}},
    ]
    episode = FakeEpisode()
    walked = check_walk_passes(episode, "widget_task", walk)
    assert walked.results[1]["result"]["label"] == "mango"
    traced = check_arguments_trace(walk, walked.results, intent=INTENT, user_facts=FACTS)
    assert traced.outcome == "failed"
    assert [(u.call_index, u.path, u.value) for u in traced.untraced] == [(0, "label", "mango")]


@pytest.mark.parametrize(
    "walk,intent,facts",
    [
        pytest.param(
            [{"name": "describe_widget", "arguments": {"widget_id": "w1"}}],
            INTENT,
            [],
            id="in_intent",
        ),
        pytest.param(
            [{"name": "describe_widget", "arguments": {"widget_id": "w1"}}],
            "do the chore",
            [{"field": "widget_id", "value": "w1"}],
            id="in_user_fact",
        ),
        pytest.param(
            [
                {"name": "describe_widget", "arguments": {"widget_id": "w1"}},
                {"name": "rename_widget", "arguments": {"widget_id": "w1", "label": "plain"}},
            ],
            "do the chore",
            [{"field": "widget_id", "value": "w1"}],
            id="in_earlier_result",
        ),
        pytest.param(
            [{"name": "rename_widget", "arguments": {"widget_id": "w1", "label": "pale blue"}}],
            "give widget w1 the label pale blue",
            [],
            id="two_words_in_intent",
        ),
        pytest.param(
            [{"name": "describe_widget", "arguments": {"widget_id": ["w1"]}}],
            INTENT,
            FACTS,
            id="inside_a_list",
        ),
    ],
)
def test_argument_found_in_each_allowed_source_traces(walk, intent, facts):
    drivable = all(not isinstance(value, list) for call in walk for value in call["arguments"].values())
    results = check_walk_passes(FakeEpisode(), "widget_task", walk).results if drivable else []
    traced = check_arguments_trace(walk, results, intent=intent, user_facts=facts)
    assert traced.outcome == "held"


@pytest.mark.parametrize(
    "value,intent,untraced",
    [
        pytest.param("1", INTENT, [(0, "widget_id", "1")], id="single_character_inside_a_token"),
        pytest.param("w1", "give widget w1 the label striped.", [], id="full_stop"),
        pytest.param("w1", "give widget w1, then stop", [], id="comma"),
    ],
)
def test_value_traces_only_as_a_whole_token_even_against_punctuation(value, intent, untraced):
    walk = [{"name": "describe_widget", "arguments": {"widget_id": value}}]
    traced = check_arguments_trace(walk, [], intent=intent, user_facts=[])
    assert traced.outcome == ("failed" if untraced else "held")
    assert [(u.call_index, u.path, u.value) for u in traced.untraced] == untraced


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"dry_run": True, "note": None}, id="booleans_and_none"),
        pytest.param({"label": "", "note": ""}, id="empty_strings"),
    ],
)
def test_booleans_none_and_empty_strings_carry_no_trace_burden(extra):
    walk = [{"name": "rename_widget", "arguments": {"widget_id": "w1", "label": "striped", **extra}}]
    traced = check_arguments_trace(walk, [], intent=INTENT, user_facts=FACTS)
    assert traced.outcome == "held"


def test_predicate_tripping_on_the_passing_walk_too_fails_without_driving_forbidden():
    episode = FakeEpisode()
    out = check_forbidden_step_trips(episode, "widget_task", FLAT_WALK,
                                     lambda path: ["trips on everything"],
                                     control_walk=GOOD_WALK)
    assert out.outcome == "failed"
    assert "trips on the passing walk too" in out.reason
    assert episode.resets == 1
    report = birth_checks(FakeEpisode(), "widget_task", GOOD_WALK, intent=INTENT,
                          user_facts=FACTS, forbidden_walk=FLAT_WALK,
                          violations_of=lambda path: ["trips on everything"])
    assert report.forbidden_step.outcome == "failed"
    assert "trips on the passing walk too" in report.forbidden_step.reason
    assert not report.born


def test_predicate_reading_the_run_holds_forbidden_and_clears_control():
    episode = FakeEpisode()
    seen = []

    def violations_of(path):
        seen.append(path)
        return flags_flat_run(path)

    out = check_forbidden_step_trips(episode, "widget_task", FLAT_WALK, violations_of,
                                     control_walk=GOOD_WALK)
    assert out.outcome == "held"
    assert episode.resets == 2
    assert all(path.startswith("fake-run/") for path in seen)
    assert any("rename_widget" in path and "label=flat" in path for path in seen)


def test_forbidden_walk_without_violations_fails():
    out = check_forbidden_step_trips(FakeEpisode(), "widget_task", GOOD_WALK, lambda path: [])
    assert out.outcome == "failed"


@pytest.mark.parametrize(
    "walk,predicate",
    [
        pytest.param(GOOD_WALK, None, id="no_predicate"),
        pytest.param(None, lambda path: ["late"], id="no_walk"),
    ],
)
def test_forbidden_check_without_its_inputs_is_not_run(walk, predicate):
    out = check_forbidden_step_trips(FakeEpisode(), "widget_task", walk, predicate)
    assert out.outcome == "not_run"


def test_run_that_never_finishes_fails_within_max_steps():
    episode = FakeEpisode(finish=False)
    out = check_walk_passes(episode, "widget_task", GOOD_WALK, max_steps=3)
    assert out.outcome == "failed"
    assert episode.steps == 3


def _walk(episode):
    return check_walk_passes(episode, "widget_task", GOOD_WALK)


def _do_nothing(episode):
    return check_do_nothing_fails(episode, "widget_task")


def _forbidden(episode):
    return check_forbidden_step_trips(episode, "widget_task", GOOD_WALK, lambda path: [])


@pytest.mark.parametrize(
    "check,fault",
    [
        pytest.param(_walk, "reset", id="walk_reset_fault"),
        pytest.param(_walk, "step", id="walk_step_fault"),
        pytest.param(_walk, "reward", id="walk_reward_fault"),
        pytest.param(_do_nothing, "reset", id="do_nothing_reset_fault"),
        pytest.param(_forbidden, "reset", id="forbidden_reset_fault"),
    ],
)
def test_driver_episode_error_is_a_failed_check(check, fault):
    out = check(FakeEpisode(fault=fault))
    assert out.outcome == "failed"
    assert "blew up" in out.reason


needs_step_split = pytest.mark.skipif(
    not (hasattr(loop, "ask") and hasattr(loop, "advance")),
    reason="needs the step-split patch",
)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param("records", id="user_fact_records"),
        pytest.param("dict", id="plain_dict"),
        pytest.param("values", id="plain_values"),
    ],
)
def test_user_facts_trace_through_their_values_in_every_shape(shape):
    from kullback.runner.records import UserFact

    facts = {
        "records": [UserFact(field="widget_id", value="w1"), UserFact(field="label", value="striped")],
        "dict": {"widget_id": "w1", "label": "striped"},
        "values": ["w1", "striped"],
    }[shape]
    traced = check_arguments_trace(GOOD_WALK, [], intent="do the chore", user_facts=facts)
    assert traced.outcome == "held"


def test_calls_drive_and_trace_as_recorded_objects_plain_objects_and_short_key_dicts():
    from types import SimpleNamespace

    from kullback.runner.records import RawPtr, ToolCall

    call = ToolCall(name="rename_widget", args={"widget_id": "w1", "label": "mango"},
                    raw_ptr=RawPtr(file_hash="test"))
    episode = FakeEpisode()
    walked = check_walk_passes(episode, "widget_task", [call])
    assert episode.calls == [{"name": "rename_widget",
                              "arguments": {"widget_id": "w1", "label": "mango"}}]
    traced = check_arguments_trace([call], walked.results, intent=INTENT, user_facts=FACTS)
    assert traced.outcome == "failed"
    assert [(u.call_index, u.path, u.value) for u in traced.untraced] == [(0, "label", "mango")]
    plain = [SimpleNamespace(name="describe_widget", arguments={"widget_id": "w1"})]
    walked = check_walk_passes(FakeEpisode(), "widget_task", plain)
    assert [entry["name"] for entry in walked.results] == ["describe_widget"]
    short = [{"name": "describe_widget", "args": {"widget_id": "w1"}}]
    assert check_arguments_trace(short, [], intent=INTENT, user_facts=FACTS).outcome == "held"


def test_walk_without_signal_is_no_signal():
    def scorer(calls):
        if calls:
            return (None, "not_verdicted", "a judge must hold")
        return (0, "fail", "no rename ran")

    out = check_walk_passes(FakeEpisode(scorer=scorer), "widget_task", GOOD_WALK)
    assert out.outcome == "no_signal"


def test_birth_without_intent_or_facts_leaves_trace_not_run_but_born():
    report = birth_checks(FakeEpisode(), "widget_task", GOOD_WALK)
    assert report.arguments_trace.outcome == "not_run"
    assert report.born


@needs_step_split
def test_real_episode_births_the_scripted_walk_and_one_tracing_through_an_earlier_result(tmp_path):
    from kullback.runner.world import BuiltEnvironment, Episode
    from tests.episode.invented import write_env

    root = write_env(tmp_path / "env")
    probe = Episode(BuiltEnvironment(root), outdir=tmp_path / "probe")
    probed = check_walk_passes(probe, "widget_task", [GOOD_WALK[0]])
    label = probed.results[0]["result"]["label"]
    walk = [GOOD_WALK[0], {"name": "rename_widget",
                           "arguments": {"widget_id": "w1", "label": label}}]
    report = birth_checks(Episode(BuiltEnvironment(root), outdir=tmp_path / "out"),
                          "widget_task", walk, intent="do the chore",
                          user_facts=[{"field": "widget_id", "value": "w1"}])
    assert report.arguments_trace.outcome == "held"
    assert report.born
    scripted = birth_checks(Episode(BuiltEnvironment(root), outdir=tmp_path / "scripted"),
                            "widget_task", GOOD_WALK, intent=INTENT, user_facts=FACTS)
    assert scripted.born
    idle = check_do_nothing_fails(Episode(BuiltEnvironment(root), outdir=tmp_path / "idle"),
                                  "widget_task")
    assert idle.outcome == "held"


@needs_step_split
def test_forbidden_walk_over_the_real_episode_reads_the_run_file(tmp_path):
    from pathlib import Path

    from kullback.runner.world import BuiltEnvironment, Episode
    from tests.episode.invented import write_env

    root = write_env(tmp_path / "env")
    seen = {}

    def violations_of(path):
        seen["path"] = str(path)
        text = Path(path).read_text(encoding="utf-8")
        if "rename_widget" in text and "flat" in text:
            return ["forbidden rename"]
        return []

    out = check_forbidden_step_trips(Episode(BuiltEnvironment(root), outdir=tmp_path / "out"),
                                     "widget_task", FLAT_WALK, violations_of,
                                     control_walk=GOOD_WALK)
    assert out.outcome == "held"
    assert Path(seen["path"]).is_file()