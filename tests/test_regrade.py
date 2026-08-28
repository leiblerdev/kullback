"""regrade.py re-scores stored Runs against a new Verifier or Environment version, and never serves a stale Verdict."""

from __future__ import annotations

import json

import pytest

from harness.runner.regrade import cache_key, regrade, regrade_run, verdict_path
from harness.shared.records import Atom, Environment, Verdict, Verifier

CANCEL = "cancel_pending_order"
READ = "get_order_details"
WRITE_TOOLS = {CANCEL, "delete_order", "modify_order"}


def simple_canon(value):
    """The same stand-in canonicalizer test_verdict.py uses."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        return " ".join(value.split()).lower()
    if isinstance(value, (int, float)):
        number = float(value)
        return int(number) if number.is_integer() else number
    return value


def oracle_lines(run_id="r1", order_id="W123"):
    return [
        {"run_id": run_id, "env_id": "e1", "task_id": "t1"},
        {"idx": 0, "type": "user_turn", "payload": {"content": "I want to stop order W123"}},
        {"idx": 1, "type": "model_call", "payload": {"content": "Why?"}},
        {"idx": 2, "type": "user_turn", "payload": {"content": "Yes, go ahead."}},
        {"idx": 3, "type": "tool_call", "payload": {"id": "c3", "name": READ, "args": {"order_id": "W123"}}},
        {"idx": 4, "type": "tool_result", "payload": {"id": "c3", "result": {"status": "pending"}}},
        {"idx": 5, "type": "tool_call",
         "payload": {"id": "c5", "name": CANCEL, "args": {"order_id": order_id, "reason": "late"}}},
        {"idx": 6, "type": "tool_result", "payload": {"id": "c5", "result": {"status": "cancelled"}}},
        {"idx": 7, "type": "model_call", "payload": {"content": "Refunded to your original card."}},
        {"idx": 8, "type": "stop", "payload": {"termination_reason": "done"}},
    ]


def _verifier(version="v1", order_id="W123"):
    return Verifier(
        task_id="t1",
        verifier_version=version,
        atoms=[
            Atom(
                id="a_cancel",
                kind="required",
                provenance="user_stated",
                predicate_src=f'wrote("cancel_pending_order", order_id="{order_id}")',
            )
        ],
    )


@pytest.fixture
def runs(tmp_path):
    """Two stored Runs: one cancels W123, one cancels W999."""
    paths = []
    for name, order_id in (("r1", "W123"), ("r2", "W999")):
        lines = oracle_lines(run_id=name, order_id=order_id)
        path = tmp_path / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(json.dumps(line) + "\n")
        paths.append(path)
    return paths


def test_regrade_scores_stored_runs_without_re_executing(runs, tmp_path):
    out = regrade(runs, _verifier(), simple_canon, out_dir=tmp_path / "verdicts", write_tools=WRITE_TOOLS)
    assert [v.run_id for v in out] == ["r1", "r2"]
    assert [v.passed for v in out] == [True, False]
    assert out[1].failing_atom == "a_cancel"


def test_every_version_lands_on_the_regraded_verdict(runs, tmp_path):
    env = Environment(env_id="env-9", schema_version="s1", tools_version="t2", policy_version="p3")
    out = regrade(
        runs[:1],
        _verifier("v4"),
        simple_canon,
        out_dir=tmp_path / "verdicts",
        environment=env,
        runner_version="runner-7",
        write_tools=WRITE_TOOLS,
    )[0]
    assert out.env_id == "env-9"
    assert (out.schema_version, out.tools_version, out.policy_version) == ("s1", "t2", "p3")
    assert out.verifier_version == "v4"
    assert out.runner_version == "runner-7"
    assert out.verdict_version


def test_verdicts_are_written_to_disk(runs, tmp_path):
    out_dir = tmp_path / "verdicts"
    verifier = _verifier()
    written = regrade(runs, verifier, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS)
    for record in written:
        path = verdict_path(out_dir, record.run_id,
                            cache_key(record.run_id, verifier, canon=simple_canon,
                                      write_tools=WRITE_TOOLS))
        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == record.run_id


def test_a_cached_verdict_is_reused_when_no_version_changed(runs, tmp_path):
    """Tamper with the stored Verdict: an unchanged key must serve the file, not recompute."""
    out_dir = tmp_path / "verdicts"
    verifier = _verifier()
    regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS)
    path = verdict_path(out_dir, "r1", cache_key("r1", verifier, canon=simple_canon,
                                                 write_tools=WRITE_TOOLS))
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored.update({"pass": False, "class": "fail", "failing_atom": "tampered"})
    path.write_text(json.dumps(stored), encoding="utf-8")

    again = regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS)[0]
    assert again.passed is False
    assert again.failing_atom == "tampered"


def test_a_new_verifier_version_does_not_serve_the_stale_verdict(runs, tmp_path):
    """The stale-cache test: bump verifier_version and the old Verdict is neither read nor returned."""
    out_dir = tmp_path / "verdicts"
    old = _verifier("v1")
    regrade(runs[:1], old, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS)
    stale_path = verdict_path(out_dir, "r1", cache_key("r1", old, canon=simple_canon,
                                                       write_tools=WRITE_TOOLS))
    stored = json.loads(stale_path.read_text(encoding="utf-8"))
    stored["pass"] = False
    stored["failing_atom"] = "tampered"
    stale_path.write_text(json.dumps(stored), encoding="utf-8")

    new = _verifier("v2")
    fresh = regrade(runs[:1], new, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS)[0]
    assert fresh.passed is True
    assert fresh.failing_atom is None
    assert fresh.verifier_version == "v2"
    assert verdict_path(out_dir, "r1", cache_key("r1", new, canon=simple_canon,
                                                 write_tools=WRITE_TOOLS)) != stale_path
    assert stale_path.is_file()  # the old version's Verdict stays on disk beside the new one


def test_a_new_environment_version_invalidates_the_cache(runs, tmp_path):
    first = Environment(env_id="env-1", schema_version="s1", tools_version="t1", policy_version="p1")
    second = first.model_copy(update={"tools_version": "t2"})
    verifier = _verifier()
    assert cache_key("r1", verifier, first) != cache_key("r1", verifier, second)

    out_dir = tmp_path / "verdicts"
    regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, environment=first, write_tools=WRITE_TOOLS)
    regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, environment=second, write_tools=WRITE_TOOLS)
    keys = [cache_key("r1", verifier, env, canon=simple_canon, write_tools=WRITE_TOOLS)
            for env in (first, second)]
    assert all(verdict_path(out_dir, "r1", key).is_file() for key in keys)


def test_a_new_runner_version_invalidates_the_cache(runs):
    verifier = _verifier()
    assert cache_key("r1", verifier, runner_version="a") != cache_key("r1", verifier, runner_version="b")


def test_a_new_judge_version_invalidates_the_cache(runs):
    """D76: a new judge version retires every Verdict that rested on the old one."""
    verifier = _verifier()
    assert cache_key("r1", verifier, judge_version="j1") != cache_key("r1", verifier, judge_version="j2")


def test_changed_judge_answers_invalidate_the_cache_at_the_same_judge_version(runs):
    """D84: a person who overturns a judge atom moves no version, so the answers are in the key."""
    verifier = _verifier()
    held = cache_key("r1", verifier, judge_version="j1", judge_results={"a_polite": {"verdict": "pass"}})
    turned = cache_key("r1", verifier, judge_version="j1", judge_results={"a_polite": {"verdict": "fail"}})
    assert held != turned


def test_a_changed_verifier_body_at_the_same_version_still_changes_the_key(runs):
    """A Verifier edited without a version bump must not be served from the old key."""
    assert cache_key("r1", _verifier("v1", "W123")) != cache_key("r1", _verifier("v1", "W999"))


def test_judge_results_are_looked_up_per_run(runs, tmp_path):
    verifier = Verifier(
        task_id="t1",
        verifier_version="v1",
        atoms=[
            Atom(id="a_cancel", kind="allowed", predicate_src='wrote("cancel_pending_order")'),
            Atom(id="a_polite", kind="required", judge=True),
        ],
    )
    out = regrade(
        runs,
        verifier,
        simple_canon,
        out_dir=tmp_path / "verdicts",
        judge_results={"r1": {"a_polite": True}, "r2": {"a_polite": False}},
        judge_version="j1",
        write_tools=WRITE_TOOLS,
    )
    assert [v.passed for v in out] == [True, False]
    assert all(v.judge_used for v in out)


def test_regrade_run_returns_one_verdict(runs, tmp_path):
    out = regrade_run(runs[0], _verifier(), simple_canon, out_dir=tmp_path, write_tools=WRITE_TOOLS)
    assert isinstance(out, Verdict)
    assert out.run_id == "r1"


def test_regrade_works_without_an_out_dir(runs):
    out = regrade(runs, _verifier(), simple_canon, write_tools=WRITE_TOOLS)
    assert [v.passed for v in out] == [True, False]


def test_regrade_never_calls_a_model():
    """A grep for the word passes on any module that imports judge.py, so walk the import graph."""
    from test_verdict import import_closure

    closure = import_closure("harness.runner.regrade")
    assert "harness.runner.regrade" in closure
    assert "harness.shared.provider" not in closure
    assert not [name for name in closure if name.startswith("harness.builder")]


# --- D84: a Run in canon.py's regrade queue is re-scored even under the same versions ---

def test_a_queued_run_is_rescored_and_the_queue_is_emptied(runs, tmp_path):
    """An overturned equivalence entry moves no version, so only the queue can force the re-score."""
    from harness.shared.canon import queue_regrade, queued_regrades

    out_dir, verifier = tmp_path / "verdicts", _verifier()
    regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS)
    path = verdict_path(out_dir, "r1", cache_key("r1", verifier, canon=simple_canon,
                                                 write_tools=WRITE_TOOLS))
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["failing_atom"] = "tampered"
    path.write_text(json.dumps(stored), encoding="utf-8")

    (tmp_path / "equivalence_uses.jsonl").write_text(
        json.dumps({"key": "k1", "run_id": "r1", "route": "cache"}) + "\n", encoding="utf-8")
    assert queue_regrade(tmp_path, "k1", "a person overturned reason") == ["r1"]
    again = regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS,
                    queue_dir=tmp_path)[0]
    assert again.failing_atom is None
    assert queued_regrades(tmp_path) == []


def test_a_run_that_is_not_queued_still_serves_its_cached_verdict(runs, tmp_path):
    out_dir, verifier = tmp_path / "verdicts", _verifier()
    regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS)
    path = verdict_path(out_dir, "r1", cache_key("r1", verifier, canon=simple_canon,
                                                 write_tools=WRITE_TOOLS))
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["failing_atom"] = "tampered"
    path.write_text(json.dumps(stored), encoding="utf-8")
    again = regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS,
                    queue_dir=tmp_path)[0]
    assert again.failing_atom == "tampered"


def test_regrade_takes_refresh_and_re_scores_every_run(runs, tmp_path):
    """A person who wants the batch re-scored has to be able to say so without a queue file."""
    out_dir, verifier = tmp_path / "verdicts", _verifier()
    regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS)
    path = verdict_path(out_dir, "r1", cache_key("r1", verifier, canon=simple_canon,
                                                 write_tools=WRITE_TOOLS))
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["failing_atom"] = "tampered"
    path.write_text(json.dumps(stored), encoding="utf-8")
    again = regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS,
                    refresh=True)[0]
    assert again.failing_atom is None


def test_the_queue_keeps_the_runs_this_batch_did_not_re_score(runs, tmp_path):
    """D84 queues every Run that used an overturned pair; a batch of one must not drop the others."""
    from harness.shared.canon import queue_regrade, queued_regrades

    (tmp_path / "equivalence_uses.jsonl").write_text(
        json.dumps({"key": "k1", "run_id": "r1", "route": "cache"}) + "\n"
        + json.dumps({"key": "k1", "run_id": "r2", "route": "cache"}) + "\n", encoding="utf-8")
    assert queue_regrade(tmp_path, "k1", "a person overturned reason") == ["r1", "r2"]
    regrade(runs[:1], _verifier(), simple_canon, out_dir=tmp_path / "verdicts",
            write_tools=WRITE_TOOLS, queue_dir=tmp_path)
    assert queued_regrades(tmp_path) == ["r2"]
    regrade(runs[1:], _verifier(), simple_canon, out_dir=tmp_path / "verdicts",
            write_tools=WRITE_TOOLS, queue_dir=tmp_path)
    assert queued_regrades(tmp_path) == []


def test_the_write_tool_set_is_part_of_the_cache_key(tmp_path):
    """The Verdict reads write_tools, so a Verdict scored without it must not be served with it."""
    lines = oracle_lines()
    lines.insert(-1, {"idx": 7, "type": "tool_call",
                      "payload": {"id": "c7", "name": "modify_order", "args": {"order_id": "W900"}}})
    lines.insert(-1, {"idx": 8, "type": "tool_result", "payload": {"id": "c7", "result": {"ok": True}}})
    path = tmp_path / "r1.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")

    out_dir, verifier = tmp_path / "verdicts", _verifier()
    assert cache_key("r1", verifier) != cache_key("r1", verifier, write_tools=WRITE_TOOLS)
    loose = regrade([path], verifier, simple_canon, out_dir=out_dir)[0]
    strict = regrade([path], verifier, simple_canon, out_dir=out_dir, write_tools=WRITE_TOOLS)[0]
    assert loose.passed is True
    assert strict.passed is False
    assert strict.failing_atom == "extra_write:modify_order"


def test_the_canonicalizer_is_part_of_the_cache_key(runs):
    """D39: the canon rules are data, so two Verdicts under different rules are different Verdicts."""
    verifier = _verifier()
    assert cache_key("r1", verifier, canon=simple_canon) != cache_key("r1", verifier, canon=None)


def test_changed_judge_results_are_not_served_from_the_cache(runs, tmp_path):
    verifier = Verifier(task_id="t1", verifier_version="v1", atoms=[
        Atom(id="a_cancel", kind="allowed", predicate_src='wrote("cancel_pending_order")'),
        Atom(id="a_polite", kind="required", judge=True)])
    out_dir = tmp_path / "verdicts"
    held = regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, judge_version="j1",
                   judge_results={"r1": {"a_polite": {"verdict": "pass"}}}, write_tools=WRITE_TOOLS)[0]
    assert held.passed is True
    turned = regrade(runs[:1], verifier, simple_canon, out_dir=out_dir, judge_version="j1",
                     judge_results={"r1": {"a_polite": {"verdict": "fail"}}}, write_tools=WRITE_TOOLS)[0]
    assert turned.passed is False
    assert turned.failing_atom == "a_polite"
