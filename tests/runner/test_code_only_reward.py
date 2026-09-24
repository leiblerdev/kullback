from __future__ import annotations

import json
from typing import get_args

import pytest

from kullback.gates import verifier_suite as S
from kullback.runner import target as T
from kullback.runner.records import Atom, AtomKind, Column, EntitySchema, Event, Run, Verdict, Verifier, as_dict
from kullback.runner.regrade import cache_key, regrade_run, verdict_path
from kullback.runner.verdict import MUST_HOLD, verdict

WRITE_TOOLS = {"archive_entry"}


def _ev(type_: str, **payload) -> dict:
    return {"type": type_, "payload": payload}


def _user(text: str) -> dict:
    return _ev("user_turn", content=text)


def _assistant(text: str) -> dict:
    return _ev("model_call", reply={"content": text})


def _call(name: str, args: dict, kind: str = "write", cid: str = "c1") -> dict:
    return _ev("tool_call", id=cid, name=name, args=args, kind=kind)


def _result(data, cid: str = "c1") -> dict:
    return _ev("tool_result", id=cid, result=data)


def _make_run(run_id: str, events: list[dict]) -> Run:
    return Run(
        run_id=run_id,
        task_id="t9",
        termination_reason="success",
        events=[Event(idx=i, type=e["type"], payload=e["payload"]) for i, e in enumerate(events)],
    )


def _passing_run() -> Run:
    return _make_run(
        "r1",
        [
            _user("Please archive entry E-100."),
            _call("fetch_entry", {"entry_id": "E-100"}, kind="read", cid="c0"),
            _result({"entry_id": "E-100", "title": "ledger"}, cid="c0"),
            _call("archive_entry", {"entry_id": "E-100"}, cid="c1"),
            _result({"entry_id": "E-100", "archived": True}, cid="c1"),
            _assistant("Done, entry E-100 is archived."),
        ],
    )


def _write_lines() -> list[dict]:
    return [
        {"run_id": "r1", "env_id": "e1", "task_id": "t9", "model": "test/model"},
        {"idx": 0, "type": "user_turn", "payload": {"content": "Please archive entry E-100."}},
        {"idx": 1, "type": "tool_call",
         "payload": {"id": "c1", "name": "archive_entry", "args": {"entry_id": "E-100"}}},
        {"idx": 2, "type": "tool_result",
         "payload": {"id": "c1", "result": {"entry_id": "E-100", "archived": True}}},
        {"idx": 3, "type": "model_call", "payload": {"content": "Done."}},
        {"idx": 4, "type": "stop", "payload": {"termination_reason": "done"}},
    ]


def _write_path(tmp_path, lines, name="run.jsonl"):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")
    return path


def test_kind_definitions_match_the_scoreability_guard():
    assert set(get_args(AtomKind)) == {"required", "allowed", "forbidden", "question", "communicate", "hard"}
    assert MUST_HOLD == {"required", "question", "communicate", "hard"}
    assert set(T.JUDGE_MUST_HOLD) == set(MUST_HOLD)


@pytest.mark.parametrize("judge_kind", ["required", "hard", "question", "communicate"])
@pytest.mark.parametrize("judge_value", [True, False, {"verdict": "abstain"}, None],
                         ids=["true", "false", "abstain", "missing"])
def test_must_hold_judge_is_unscoreable_for_every_judge_value(tmp_path, judge_kind, judge_value):
    verifier = Verifier(
        task_id="t9",
        verifier_version="v1",
        atoms=[
            Atom(id="w0", kind="required",
                 predicate_src='wrote("archive_entry", entry_id="E-100")'),
            Atom(id="j0", kind=judge_kind, judge=True, description="policy tone"),
        ],
    )
    results = {} if judge_value is None else {"j0": judge_value}
    out = verdict(_write_path(tmp_path, _write_lines()), verifier, write_tools=WRITE_TOOLS,
                  judge_results=results)
    assert out.passed is False
    assert out.class_ == "not_verdicted"
    assert out.failing_atom == "j0"

    judge_write = S.make_atom("j0", judge_kind, {"kind": "write", "tool": "archive_entry",
                                                 "entity": _entity("E-100")}, judge=True)
    no_write = Verifier(task_id="t9", verifier_version="v1", atoms=[judge_write])
    malformed = Verifier(task_id="t9", verifier_version="v1", atoms=[
        S.make_atom("w0", "required", {"kind": "write", "tool": "archive_entry", "entity": _entity("E-100")}),
        S.make_atom("j0", judge_kind, {"kind": "write", "tool": ["archive_entry"],
                                       "entity": _entity("E-100")}, judge=True),
    ])
    for unscoreable, run in ((no_write, _quiet_run()), (malformed, _unmarked_write_run())):
        assert S.check_run(unscoreable, run) == (False, "j0")
        out = verdict(run, unscoreable, judge_results=results)
        assert out.passed is False
        assert out.class_ == "not_verdicted"
        assert out.failing_atom == "j0"


def _predicate_write(entity: str) -> Atom:
    return Atom(id="w0", kind="required", predicate_src=f'wrote("archive_entry", entry_id="{entity}")')


def _judge(kind: str = "required") -> Atom:
    return Atom(id="j0", kind=kind, judge=True, description="policy tone")


def _lines_run(tmp_path):
    return _write_path(tmp_path, _write_lines())


_EXTRA_TOOLS = {"archive_entry", "purge_entry"}
_JUDGE_VALUES = [(True, "true"), (False, "false"), ({"verdict": "abstain"}, "abstain"), (None, "missing")]
_CODE_WINS = [
    *[pytest.param(lambda: [_predicate_write("E-999"), _judge()], _lines_run, WRITE_TOOLS,
                   value, False, "w0", id=f"predicate_write_over_{name}_judge")
      for value, name in _JUDGE_VALUES],
    pytest.param(lambda: [S.make_atom("w0", "required", {"kind": "write", "tool": "archive_entry", "entity": "E-999"}),
                          _judge("hard")],
                 lambda _: _passing_run(), WRITE_TOOLS, True, True, "w0", id="gate_keeps_first_code_failure"),
    *[pytest.param(lambda kind=kind: [_structured_judge(kind),
                                      S.make_atom("w0", "required",
                                                  {"kind": "write", "tool": "archive_entry", "entity": "E-888"})],
                   lambda _: _passing_run(), WRITE_TOOLS, True, True, "w0", id=f"structured_{kind}_judge")
      for kind in ("required", "hard", "question", "communicate", "allowed", "forbidden")],
    pytest.param(lambda: [S.make_atom("w0", "required",
                                      {"kind": "write", "tool": "archive_entry", "entity": _entity("E-100")}),
                          _judge()],
                 lambda _: _extra_run(), _EXTRA_TOOLS, True, True, "extra_write:purge_entry", id="extra_write"),
    pytest.param(lambda: [S.make_atom("w0", "required", {"kind": "write", "tool": "archive_entry", "entity": "E-888"}),
                          S.make_atom("j0", "required", {"kind": "write", "tool": ["archive_entry"],
                                                         "entity": _entity("E-100")}, judge=True)],
                 lambda _: _unmarked_write_run(), None, True, True, "w0", id="malformed_judge_tool"),
]


@pytest.mark.parametrize("atoms, run, tools, judge_value, gate, failing", _CODE_WINS)
def test_a_code_failure_fails_the_verdict_whatever_the_judge_says(tmp_path, atoms, run, tools, judge_value,
                                                                     gate, failing):
    """A failing code atom or an extra write decides the Verdict before any judge atom, must-hold,
    structured or malformed, and the build-time gate names the same failure."""
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=atoms())
    if gate:
        assert S.check_run(verifier, run(tmp_path), write_tools=tools) == (False, failing)
    results = {} if judge_value is None else {"j0": judge_value}
    out = verdict(run(tmp_path), verifier, write_tools=tools, judge_results=results)
    assert out.passed is False
    assert out.class_ == "fail"
    assert out.failing_atom == failing


@pytest.mark.parametrize("code_passes", [True, False], ids=["code_pass", "code_fail"])
@pytest.mark.parametrize("structured", [False, True], ids=["predicate", "structured"])
@pytest.mark.parametrize("judge_kind", ["allowed", "forbidden"])
@pytest.mark.parametrize("judge_value", [True, False, {"verdict": "abstain"}, None],
                         ids=["true", "false", "abstain", "missing"])
def test_an_optional_judge_never_changes_the_code_verdict(tmp_path, judge_value, judge_kind, structured, code_passes):
    """An allowed or forbidden judge atom stays diagnostic: the code atoms decide pass or fail alone."""
    if structured:
        write = S.make_atom("w0", "required", {"kind": "write", "tool": "archive_entry",
                                               "entity": _entity("E-100") if code_passes else "E-888"})
        run = _passing_run()
    else:
        write = _predicate_write("E-100" if code_passes else "E-999")
        run = _write_path(tmp_path, _write_lines())
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[write, _judge(judge_kind)])
    results = {} if judge_value is None else {"j0": judge_value}
    if structured:
        assert S.check_run(verifier, run, write_tools=WRITE_TOOLS) == ((True, None) if code_passes else (False, "w0"))
    out = verdict(run, verifier, write_tools=WRITE_TOOLS, judge_results=results)
    assert out.passed is code_passes
    assert out.class_ == ("pass" if code_passes else "fail")
    assert out.failing_atom == (None if code_passes else "w0")


@pytest.mark.parametrize("judge_value", [True, False, {"verdict": "abstain"}, None],
                         ids=["true", "false", "abstain", "missing"])
def test_judge_only_verifier_never_passes(tmp_path, judge_value):
    verifier = Verifier(
        task_id="t9",
        verifier_version="v1",
        atoms=[Atom(id="j0", kind="required", judge=True, description="policy tone")],
    )
    results = {} if judge_value is None else {"j0": judge_value}
    assert S.check_run(verifier, _quiet_run(), write_tools=WRITE_TOOLS) == (False, "j0")
    quiet = verdict(_quiet_run(), verifier, write_tools=WRITE_TOOLS, judge_results=results)
    assert quiet.passed is False
    assert quiet.class_ == "not_verdicted"
    assert quiet.failing_atom == "j0"


def test_supplied_judge_opinions_are_reported_without_scoring(tmp_path):
    verifier = Verifier(
        task_id="t9",
        verifier_version="v1",
        atoms=[
            Atom(id="w0", kind="required",
                 predicate_src='wrote("archive_entry", entry_id="E-100")'),
            Atom(id="j0", kind="required", judge=True, description="policy tone"),
        ],
    )
    path = _write_path(tmp_path, _write_lines())
    passing = verdict(path, verifier, write_tools=WRITE_TOOLS, judge_results={"j0": True})
    failing = verdict(path, verifier, write_tools=WRITE_TOOLS, judge_results={"j0": False})
    for out in (passing, failing):
        assert out.passed is False
        assert out.class_ == "not_verdicted"
        assert out.failing_atom == "j0"
        assert out.judge_used is True
    assert "judge_reported:j0:pass" in passing.notes
    assert "judge_reported:j0:fail" in failing.notes
    missing = verdict(path, verifier, write_tools=WRITE_TOOLS)
    assert missing.judge_used is False
    assert "judge_atom_unevaluated:j0" in missing.notes
    abstained = verdict(path, verifier, write_tools=WRITE_TOOLS,
                        judge_results={"j0": {"verdict": "abstain"}})
    assert abstained.judge_used is True
    assert any(note.startswith("judge_abstained:j0") for note in abstained.notes)
    assert abstained.passed is False and abstained.class_ == "not_verdicted"


def _entity(value: str) -> str:
    return T.text_of(T.canon_fn(None)(value))


def _quiet_run() -> Run:
    return _make_run(
        "r0",
        [
            _user("Please archive entry E-100."),
            _assistant("Archived."),
        ],
    )


def test_gate_scoreability_refuses_a_judge_only_must_hold():
    verifier = Verifier(
        task_id="t9",
        verifier_version="v1",
        atoms=[
            S.make_atom("w0", "required",
                        {"kind": "write", "tool": "archive_entry", "entity": _entity("E-100")}),
            Atom(id="j0", kind="hard", judge=True, description="policy tone"),
        ],
    )
    assert S.check_run(verifier, _passing_run(), write_tools=WRITE_TOOLS) == (False, "j0")


def test_grounded_communicate_still_scores_without_any_judge(tmp_path):
    key = T._key(T.canon_fn(None), "E-999")
    atom = S.make_atom("c0", "communicate", {"kind": "communicate", "value": key, "text": "E-999"})
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[atom])
    run = _make_run(
        "r1",
        [
            _user("Please archive entry E-100."),
            _call("fetch_entry", {"entry_id": "E-100"}, kind="read", cid="c0"),
            _result({"entry_id": "E-100", "title": "ledger"}, cid="c0"),
            _call("archive_entry", {"entry_id": "E-100"}, cid="c1"),
            _result({"entry_id": "E-100", "archived": True}, cid="c1"),
            _assistant("Done, entry E-999 is archived."),
        ],
    )
    assert T.communicate_values(run, T.canon_fn(None)) == {}
    assert S.check_run(verifier, run, write_tools=WRITE_TOOLS) == (False, "c0")
    out = verdict(run, verifier, write_tools=WRITE_TOOLS)
    assert out.passed is False and out.failing_atom == "c0"


def test_semantic_and_judge_diagnostics_are_both_recorded(tmp_path):
    lines = _write_lines()
    lines[-1] = {"idx": 4, "type": "stop",
                 "payload": {"termination_reason": "done",
                             "start_state": {"entries": {"E-1": {"status": "open", "note": "filed by the user"}}},
                             "end_state": {"entries": {"E-1": {"status": "filed", "note": "filed by user"}}}}}
    schema = EntitySchema(
        tables=["entries"],
        columns=[
            Column(table="entries", name="status", **{"class": "hard"}),
            Column(table="entries", name="note", **{"class": "semantic"}),
        ],
    )
    semantic = Atom(id="s0", kind="required",
                    predicate_src='"note" not in diff()["entries.E-1"]["fields"]')
    judge_atom = Atom(id="j0", kind="required", judge=True, description="policy tone")
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=[semantic, judge_atom])
    out = verdict(_write_path(tmp_path, lines), verifier, schema=schema, workdir=tmp_path,
                  write_tools=WRITE_TOOLS, judge_results={"j0": True})
    assert out.judge_used is True
    assert any(note.startswith("semantic_unresolved:") for note in out.notes)
    assert "judge_reported:j0:pass" in out.notes
    uses = (tmp_path / "equivalence_uses.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(uses) == 1 and json.loads(uses[0])["run_id"] == "r1"


def _structured_judge(kind):
    if kind == "required":
        return S.make_atom("j0", "required",
                           {"kind": "write", "tool": "archive_entry", "entity": "E-999"},
                           judge=True)
    if kind == "hard":
        return Atom(id="j0", kind="hard", judge=True,
                    predicate_src="def check():\n    return False")
    if kind == "question":
        return S.make_atom("j0", "question",
                           {"kind": "question", "key": "confirm:archive_entry",
                            "tool": "archive_entry", "field": None}, judge=True)
    if kind == "communicate":
        key = T._key(T.canon_fn(None), "E-999")
        return S.make_atom("j0", "communicate",
                           {"kind": "communicate", "value": key, "text": "E-999"},
                           judge=True)
    if kind == "allowed":
        return S.make_atom("j0", "allowed",
                           {"kind": "question", "key": "confirm:archive_entry",
                            "tool": "archive_entry", "field": None}, judge=True)
    return S.make_atom("j0", "forbidden",
                       {"kind": "write", "tool": "archive_entry",
                        "entity": _entity("E-100")}, judge=True)


def test_judge_write_target_present_does_not_authorize_the_write():
    verifier = Verifier(
        task_id="t9",
        verifier_version="v1",
        atoms=[S.make_atom("j0", "required",
                           {"kind": "write", "tool": "archive_entry",
                            "entity": _entity("E-100")}, judge=True)],
    )
    assert S.check_run(verifier, _passing_run(), write_tools=WRITE_TOOLS) == (
        False, "extra_write:archive_entry")
    out = verdict(_passing_run(), verifier, write_tools=WRITE_TOOLS,
                  judge_results={"j0": True})
    assert out.passed is False
    assert out.class_ == "fail"
    assert out.failing_atom == "extra_write:archive_entry"
    tools = {"archive_entry", "purge_entry"}
    verifier = Verifier(
        task_id="t9",
        verifier_version="v1",
        atoms=[
            S.make_atom("w0", "required",
                        {"kind": "write", "tool": "archive_entry",
                         "entity": _entity("E-100")}),
            S.make_atom("j0", "allowed",
                        {"kind": "write", "tool": "purge_entry",
                         "entity": _entity("E-100")}, judge=True),
        ],
    )
    assert S.check_run(verifier, _extra_run(), write_tools=tools) == (
        False, "extra_write:purge_entry")
    out = verdict(_extra_run(), verifier, write_tools=tools, judge_results={"j0": True})
    assert out.passed is False
    assert out.class_ == "fail"
    assert out.failing_atom == "extra_write:purge_entry"


def _malformed_run() -> Run:
    return Run(run_id="review-invalid", task_id="t", events=[])


def _malformed_atom() -> Atom:
    return Atom(id="bad", kind="required",
                target={"kind": "write", "tool": "write_item", "entity": ["invalid"]})


def test_malformed_write_entity_stays_unverdicted():
    verifier = Verifier(task_id="t", verifier_version="v", atoms=[_malformed_atom()])
    out = verdict(_malformed_run(), verifier, write_tools={"write_item"})
    assert out.passed is False
    assert out.class_ == "not_verdicted"
    assert out.failing_atom == "bad"
    assert "atom_error:bad:TypeError" in out.notes
    verifier = Verifier(task_id="t", verifier_version="v", atoms=[
        Atom(id="j0", kind="required", judge=True, description="policy tone"),
        _malformed_atom(),
    ])
    out = verdict(_malformed_run(), verifier, write_tools={"write_item"},
                  judge_results={"j0": True})
    assert out.passed is False
    assert out.class_ == "not_verdicted"
    assert out.failing_atom == "j0"
    assert "judge_reported:j0:pass" in out.notes
    assert "atom_error:bad:TypeError" in out.notes


def test_legacy_judge_pass_cache_is_recomputed_under_new_semantics(tmp_path):
    verifier = Verifier(task_id="t", verifier_version="v",
                        atoms=[Atom(id="j", kind="required", judge=True)])
    run = Run(run_id="r", task_id="t", events=[])
    legacy = Verdict(run_id="r", verifier_version="v", verdict_version="2",
                     **{"pass": True, "class": "pass"},
                     judge_used=True, notes=["judge_reported:j:pass"])
    legacy_key = cache_key("r", verifier, None, None, None, "2",
                           judge_results={"j": True}, write_tools={"write_item"})
    legacy_path = verdict_path(tmp_path, "r", legacy_key)
    legacy_path.write_text(json.dumps(as_dict(legacy), indent=2, sort_keys=True),
                           encoding="utf-8")
    before = legacy_path.read_bytes()
    out = regrade_run(run, verifier, judge_results={"j": True}, write_tools={"write_item"},
                      out_dir=tmp_path)
    assert out.passed is False
    assert out.class_ == "not_verdicted"
    assert out.failing_atom == "j"
    assert out.verdict_version == "3"
    assert "judge_reported:j:pass" in out.notes
    assert legacy_path.read_bytes() == before
    new_key = cache_key("r", verifier, None, None, None, "3",
                        judge_results={"j": True}, write_tools={"write_item"})
    assert verdict_path(tmp_path, "r", new_key) != legacy_path
    assert sorted(tmp_path.glob("r.*.json")) != [legacy_path]
    again = regrade_run(run, verifier, judge_results={"j": True}, write_tools={"write_item"},
                        out_dir=tmp_path)
    assert again.notes == out.notes
    assert again.verdict_version == "3"
    turned = regrade_run(run, verifier, judge_results={"j": False},
                         write_tools={"write_item"}, out_dir=tmp_path)
    assert turned.passed is False
    assert "judge_reported:j:fail" in turned.notes


def test_broken_predicate_keeps_unproven_coverage_unverdicted():
    verifier = Verifier(
        task_id="t9",
        verifier_version="v1",
        atoms=[
            Atom(id="j0", kind="required", judge=True, description="policy tone"),
            Atom(id="b0", kind="required"),
        ],
    )
    assert S.check_run(verifier, _passing_run(), write_tools=WRITE_TOOLS) == (
        False, "atom b0: kind None is not scored")
    out = verdict(_passing_run(), verifier, write_tools=WRITE_TOOLS,
                  judge_results={"j0": True})
    assert out.passed is False
    assert out.class_ == "not_verdicted"
    assert out.failing_atom == "j0"


def _extra_run() -> Run:
    return _make_run(
        "r2",
        [
            _user("Please archive entry E-100 and purge entry E-100."),
            _call("archive_entry", {"entry_id": "E-100"}, cid="c1"),
            _result({"entry_id": "E-100", "archived": True}, cid="c1"),
            _call("purge_entry", {"entry_id": "E-100"}, cid="c2"),
            _result({"entry_id": "E-100", "purged": True}, cid="c2"),
            _assistant("Done."),
        ],
    )


def _unmarked_write_run() -> Run:
    return _make_run(
        "r1",
        [
            _user("Please archive entry E-100."),
            _ev("tool_call", id="c1", name="archive_entry",
                args={"entry_id": "E-100"}),
            _result({"entry_id": "E-100", "archived": True}, cid="c1"),
            _assistant("Done, entry E-100 is archived."),
        ],
    )


def _unmarked_purge_run() -> Run:
    return _make_run(
        "r2",
        [
            _user("Please purge entry E-100."),
            _ev("tool_call", id="c1", name="purge_entry",
                args={"entry_id": "E-100"}),
            _result({"entry_id": "E-100", "purged": True}, cid="c1"),
            _assistant("Done."),
        ],
    )


def test_required_judge_write_detected_without_metadata_or_kind():
    verifier = Verifier(
        task_id="t9",
        verifier_version="v1",
        atoms=[S.make_atom("j0", "required",
                           {"kind": "write", "tool": "archive_entry",
                            "entity": _entity("E-100")}, judge=True)],
    )
    assert S.check_run(verifier, _unmarked_write_run()) == (
        False, "extra_write:archive_entry")
    out = verdict(_unmarked_write_run(), verifier, judge_results={"j0": True})
    assert out.passed is False
    assert out.class_ == "fail"
    assert out.failing_atom == "extra_write:archive_entry"
    verifier = Verifier(
        task_id="t9",
        verifier_version="v1",
        atoms=[S.make_atom("j0", "allowed",
                           {"kind": "write", "tool": "purge_entry",
                            "entity": _entity("E-100")}, judge=True)],
    )
    assert S.check_run(verifier, _unmarked_purge_run()) == (
        False, "extra_write:purge_entry")
    out = verdict(_unmarked_purge_run(), verifier, judge_results={"j0": True})
    assert out.passed is False
    assert out.class_ == "fail"
    assert out.failing_atom == "extra_write:purge_entry"
    control = Verifier(
        task_id="t9",
        verifier_version="v1",
        atoms=[
            S.make_atom("a0", "allowed",
                        {"kind": "write", "tool": "purge_entry",
                         "entity": _entity("E-100")}),
            S.make_atom("j0", "allowed",
                        {"kind": "write", "tool": "purge_entry",
                         "entity": _entity("E-100")}, judge=True),
        ],
    )
    assert S.check_run(control, _unmarked_purge_run()) == (True, None)
    cout = verdict(_unmarked_purge_run(), control, judge_results={"j0": True})
    assert cout.passed is True
    assert cout.class_ == "pass"
    assert cout.failing_atom is None


_MISSING = object()


def _malformed_judge_verifier(tool):
    payload = {"kind": "write", "entity": _entity("E-100")}
    if tool is not _MISSING:
        payload["tool"] = tool
    return Verifier(
        task_id="t9",
        verifier_version="v1",
        atoms=[Atom(id="j0", kind="required", judge=True, target=payload)],
    )


@pytest.mark.parametrize("bad_tool", [["archive_entry"], {"tool": "archive_entry"}, 42, "", None,
                                         _MISSING],
                         ids=["list", "dict", "numeric", "empty", "none", "missing"])
def test_malformed_judge_tool_grants_no_detection_and_never_crashes(bad_tool):
    verifier = _malformed_judge_verifier(bad_tool)
    assert S.check_run(verifier, _quiet_run()) == (False, "j0")
    quiet = verdict(_quiet_run(), verifier, judge_results={"j0": True})
    assert quiet.passed is False
    assert quiet.class_ == "not_verdicted"
    assert quiet.failing_atom == "j0"
    assert S.check_run(verifier, _unmarked_write_run()) == (False, "j0")
    out = verdict(_unmarked_write_run(), verifier, judge_results={"j0": True})
    assert out.passed is False
    assert out.class_ == "not_verdicted"
    assert out.failing_atom == "j0"
    assert S.check_run(verifier, _quiet_run(), write_tools=WRITE_TOOLS) == (False, "j0")
    explicit = verdict(_unmarked_write_run(), verifier, write_tools=WRITE_TOOLS,
                       judge_results={"j0": True})
    assert explicit.passed is False
    assert explicit.class_ == "fail"
    assert explicit.failing_atom == "extra_write:archive_entry"


def _side_effects(out) -> str:
    return next(note for note in out.notes if note.startswith("side_effects="))


def _unmarked_double_write_run() -> Run:
    return _make_run(
        "r3",
        [
            _user("Please archive entry E-100."),
            _ev("tool_call", id="c1", name="archive_entry",
                args={"entry_id": "E-100"}),
            _result({"entry_id": "E-100", "archived": True}, cid="c1"),
            _ev("tool_call", id="c2", name="archive_entry",
                args={"entry_id": "E-100"}),
            _result({"entry_id": "E-100", "archived": True}, cid="c2"),
            _assistant("Done."),
        ],
    )


def _unmarked_mixed_write_run() -> Run:
    return _make_run(
        "r4",
        [
            _user("Please archive entry E-100."),
            _ev("tool_call", id="c1", name="archive_entry",
                args={"entry_id": "E-100"}),
            _ev("tool_result", id="c1", result=None, error={"class": "body_fault"}),
            _ev("tool_call", id="c2", name="archive_entry",
                args={"entry_id": "E-100"}),
            _result({"entry_id": "E-100", "archived": True}, cid="c2"),
            _assistant("Done."),
        ],
    )


def _judge_write(kind: str) -> Atom:
    return S.make_atom("j0", kind, {"kind": "write", "tool": "archive_entry", "entity": _entity("E-100")}, judge=True)


@pytest.mark.parametrize("run, covered, count", [
    (_unmarked_write_run, False, 1),
    (_unmarked_double_write_run, False, 2),
    (_unmarked_mixed_write_run, False, 1),
    (_unmarked_write_run, True, 1),
], ids=["one_call", "repeated_calls", "failed_call_skipped", "covered_write"])
def test_each_extra_write_call_that_succeeded_counts_one_side_effect(run, covered, count):
    """Every write call that succeeded counts once, a failed call not at all, and a write an allowed
    atom covers still counts though it is no extra write."""
    if covered:
        atoms = [S.make_atom("a0", "allowed", {"kind": "write", "tool": "archive_entry", "entity": _entity("E-100")}),
                 _judge_write("allowed")]
    else:
        atoms = [_judge_write("required")]
    verifier = Verifier(task_id="t9", verifier_version="v1", atoms=atoms)
    if covered:
        assert S.check_run(verifier, run()) == (True, None)
    out = verdict(run(), verifier, judge_results={"j0": True})
    assert out.passed is covered
    assert out.class_ == ("pass" if covered else "fail")
    assert out.failing_atom == (None if covered else "extra_write:archive_entry")
    assert _side_effects(out) == f"side_effects={count}"


