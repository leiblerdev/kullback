"""Tests for the transition gate: a body is accepted on its answers and its effects (G6).

The world here is an invented kiln shed: pots standing on shelves, each pot at some growing
stage, and a write tool that moves one pot to a new stage. The evidence the gate rules on is
handed in as data, the way the compile stage hands what the traces witness; no test reads a
corpus and no name here is a customer name.
"""

from __future__ import annotations

import copy

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.builder import sandbox as sb
from kullback.runner.canon import CanonRules
from kullback.runner.records import Column, EntitySchema, FieldStat, ToolCall, ToolSig

KILN = {
    "shelves": {
        "SH-1": {"shelf_id": "SH-1", "side": "east"},
        "SH-2": {"shelf_id": "SH-2", "side": "west"},
    },
    "pots": {
        "POT-01": {"pot_id": "POT-01", "shelf_id": "SH-1", "stage": "seedling"},
        "POT-02": {"pot_id": "POT-02", "shelf_id": "SH-2", "stage": "seedling"},
    },
}

MOVES_RIGHT = ("pot = self.db.pots[pot_id]\n"
               "pot.stage = new_stage\n"
               "return {\"pot_id\": pot.pot_id, \"stage\": pot.stage}\n")
ANSWERS_RIGHT_WRITES_WRONG = ("pot = self.db.pots[pot_id]\n"
                              "pot.stage = \"seedling\"\n"
                              "return {\"pot_id\": pot.pot_id, \"stage\": new_stage}\n")
ANSWERS_RIGHT_WRITES_NOTHING = "return {\"pot_id\": pot_id, \"stage\": new_stage}\n"
MOVES_AN_EXTRA_POT = ("pot = self.db.pots[pot_id]\n"
                      "pot.stage = new_stage\n"
                      "for key in self.db.pots:\n"
                      "    self.db.pots[key].stage = new_stage\n"
                      "return {\"pot_id\": pot.pot_id, \"stage\": pot.stage}\n")


def _schema() -> EntitySchema:
    columns = [Column(table=table, name=name, **{"class": "hard"})
               for table, rows in KILN.items()
               for name in sorted({key for row in rows.values() for key in row})]
    return EntitySchema(tables=sorted(KILN), columns=columns,
                        id_patterns={"pots.pot_id": r"^POT-\d+$", "shelves.shelf_id": r"^SH-\d$"})


def _sig() -> ToolSig:
    return ToolSig(name="move_pot", description="Move one pot to a new stage.",
                   args_fields=[FieldStat(name="pot_id", types=["str"], optional=False),
                                FieldStat(name="new_stage", types=["str"], optional=False)],
                   kind="write", unclassified=False)


def _call(call_id: str, pot: str, stage: str) -> ToolCall:
    return ToolCall(id=call_id, name="move_pot", args={"pot_id": pot, "new_stage": stage},
                    result={"pot_id": pot, "stage": stage}, raw_ptr=PTR)


def _evidence(*calls_with_stages: tuple[str, str, str]) -> dict:
    """Per call id, the witnessed rows: (call id, row id, after stage) triples."""
    out = {}
    for call_id, row, stage in calls_with_stages:
        out[call_id] = [{"tool": "move_pot", "table": "pots", "row": row, "path": "stage",
                          "before": "seedling", "after": stage, "named": True,
                          "ambiguous": False, "unattributed": False, "formulas": []}]
    return out


EVIDENCE_BOTH = _evidence(("c1", "POT-01", "sprout"), ("c2", "POT-02", "bloom"))


def _gates(body: str, tmp_path, calls, evidence, rules=None, where="box"):
    schema, sig = _schema(), _sig()
    source = ce.module_source(schema, [sig], {sig.name: body})
    box = sb.Sandbox(source, KILN, tmp_path / where)
    return sb.run_gates(source, box, calls, [], schema, rules=rules, sig=sig,
                        transition_evidence=evidence)


CALLS = [_call("c1", "POT-01", "sprout"), _call("c2", "POT-02", "bloom")]


def test_a_body_that_answers_right_and_writes_the_wrong_value_is_refused_with_the_column_named(
        tmp_path):
    gates = _gates(ANSWERS_RIGHT_WRITES_WRONG, tmp_path, CALLS, EVIDENCE_BOTH)
    failed = [gate for gate in gates if not gate.passed]
    assert [gate.stage for gate in failed] == [sb.TRANSITION_STAGE]
    assert "stage" in failed[0].failures[0]
    assert failed[0].metrics["transition_disagrees"] == 2


def test_a_body_that_answers_right_and_writes_nothing_is_refused(tmp_path):
    gates = _gates(ANSWERS_RIGHT_WRITES_NOTHING, tmp_path, CALLS, EVIDENCE_BOTH)
    failed = [gate for gate in gates if not gate.passed]
    assert [gate.stage for gate in failed] == [sb.TRANSITION_STAGE]
    assert "did not move" in failed[0].failures[0]


def test_a_body_whose_write_no_later_read_witnesses_is_accepted_and_counted_unwitnessed(tmp_path):
    gates = _gates(ANSWERS_RIGHT_WRITES_NOTHING, tmp_path, CALLS, {})
    assert all(gate.passed for gate in gates), [g.failures for g in gates if not g.passed]
    transition = next(gate for gate in gates if gate.stage == sb.TRANSITION_STAGE)
    assert transition.metrics["transition_unwitnessed"] == 2
    assert transition.metrics["transition_disagrees"] == 0


def test_a_body_that_changes_an_extra_row_the_recording_never_shows_is_refused(tmp_path):
    """An extra row fails while an extra column on a witnessed row does not (see next test).

    The recording is the only statement of what the write does, and a row no sighting
    associates with the call has no partial-read cover: nothing the traces display can explain
    the body touching it. A side effect no later read owns is what the replay blames a read
    for, so the repair loop has to hear of it at build time rather than downstream at replay."""
    gates = _gates(MOVES_AN_EXTRA_POT, tmp_path, CALLS[:1], _evidence(("c1", "POT-01", "sprout")))
    failed = [gate for gate in gates if not gate.passed]
    assert [gate.stage for gate in failed] == [sb.TRANSITION_STAGE]
    assert "POT-02" in failed[0].failures[0]


MOVES_AN_EXTRA_COLUMN = ("pot = self.db.pots[pot_id]\n"
                         "pot.stage = new_stage\n"
                         "pot.shelf_id = \"SH-2\"\n"
                         "return {\"pot_id\": pot.pot_id, \"stage\": pot.stage}\n")


def test_an_extra_column_on_a_row_the_recording_shows_the_call_moving_is_accepted(tmp_path):
    """Sightings are partial: readers home only some columns of each result, so a column no
    later read displays is usually a column nothing displayed rather than one that did not move.
    On the stored workdirs every extra on an evidenced call sat on an already witnessed row,
    while the genuinely missed columns failed; failing the former refused whole tools' worth of
    agreeing bodies."""
    schema, sig = _schema(), _sig()
    source = ce.module_source(schema, [sig], {sig.name: MOVES_AN_EXTRA_COLUMN})
    box = sb.Sandbox(source, KILN, tmp_path)
    evidence = _evidence(("c1", "POT-01", "sprout"))
    ruling = sb.gate_transition(box, CALLS[:1], schema, evidence)
    assert ruling.passed is True, ruling.failures
    assert ruling.metrics["transition_agrees"] == 1
    assert ruling.metrics["transition_extras"] == 1


def test_canon_rules_make_two_spellings_of_one_value_agree(tmp_path):
    rules = CanonRules(lowercase=True)
    calls = [_call("c1", "POT-01", "Sprout")]
    evidence = _evidence(("c1", "POT-01", "Sprout"))
    gates = _gates(MOVES_RIGHT, tmp_path, calls, evidence, rules=rules)
    assert all(gate.passed for gate in gates), [g.failures for g in gates if not g.passed]
    transition = next(gate for gate in gates if gate.stage == sb.TRANSITION_STAGE)
    assert transition.metrics["transition_agrees"] == 1


def test_a_body_that_moves_what_was_witnessed_passes_with_the_share_published(tmp_path):
    gates = _gates(MOVES_RIGHT, tmp_path, CALLS, EVIDENCE_BOTH)
    assert all(gate.passed for gate in gates), [g.failures for g in gates if not g.passed]
    transition = next(gate for gate in gates if gate.stage == sb.TRANSITION_STAGE)
    assert transition.metrics["transition_agrees"] == 2
    assert transition.metrics["transition_unwitnessed"] == 0


def test_a_body_that_leaves_a_witnessed_column_alone_because_the_world_held_it_agrees(tmp_path):
    """No body can move a column to the value it already holds: where the world after the
    call holds the witnessed value, an untouched column is agreement, not a miss."""
    world = copy.deepcopy(KILN)
    world["pots"]["POT-01"]["stage"] = "sprout"
    schema, sig = _schema(), _sig()
    source = ce.module_source(schema, [sig], {sig.name: ANSWERS_RIGHT_WRITES_NOTHING})
    box = sb.Sandbox(source, world, tmp_path)
    ruling = sb.gate_transition(box, [_call("c1", "POT-01", "sprout")], schema,
                                _evidence(("c1", "POT-01", "sprout")))
    assert ruling.passed is True, ruling.failures
    assert ruling.metrics["transition_agrees"] == 1


def test_a_body_that_leaves_a_column_alone_while_the_world_holds_another_is_refused(tmp_path):
    schema, sig = _schema(), _sig()
    source = ce.module_source(schema, [sig], {sig.name: ANSWERS_RIGHT_WRITES_NOTHING})
    box = sb.Sandbox(source, KILN, tmp_path)
    ruling = sb.gate_transition(box, [_call("c1", "POT-01", "sprout")], schema,
                                _evidence(("c1", "POT-01", "sprout")))
    assert ruling.passed is False
    assert "stage" in ruling.failures[0]


DELETES_THE_ROW = ("pot = self.db.pots[pot_id]\n"
                   "del self.db.pots[pot_id]\n"
                   "return {\"pot_id\": pot_id, \"stage\": new_stage}\n")


def test_a_witnessed_row_the_world_does_not_have_after_the_call_is_still_missing(tmp_path):
    schema, sig = _schema(), _sig()
    source = ce.module_source(schema, [sig], {sig.name: DELETES_THE_ROW})
    box = sb.Sandbox(source, KILN, tmp_path)
    ruling = sb.gate_transition(box, [_call("c1", "POT-01", "sprout")], schema,
                                _evidence(("c1", "POT-01", "sprout")))
    assert ruling.passed is False
    assert "did not move" in ruling.failures[0]


def test_a_short_diff_run_that_answers_for_fewer_calls_than_given_fails_closed(tmp_path):
    """Ruling on the partial set would pass calls nothing compared, so the gate refuses to
    rule at all, the way it does on a truncated diff."""

    class ShortBox:
        def run_diff(self, calls, want=()):
            return []

    ruling = sb.gate_transition(ShortBox(), CALLS[:1], _schema(),
                                _evidence(("c1", "POT-01", "sprout")))
    assert ruling.passed is False
    assert "came back for 0 of 1" in ruling.failures[0]


def test_a_call_that_changes_more_rows_than_the_gate_reads_is_refused(tmp_path):
    """A truncated diff fails closed: the unlisted rows were never compared, so agreement on
    the listed ones proves nothing about them."""

    class TruncatingBox:
        def run_diff(self, calls, want=()):
            return [{"changed": {}, "truncated": True, "wanted": {}} for _ in calls]

    ruling = sb.gate_transition(TruncatingBox(), CALLS[:1], _schema(),
                                _evidence(("c1", "POT-01", "sprout")))
    assert ruling.passed is False
    assert "more rows than the gate reads" in ruling.failures[0]


def test_the_chain_without_evidence_answers_exactly_as_before(tmp_path):
    """Behaviour-preserving proof for every existing caller: no evidence, no new failure.

    The chain carries one more passing row than before, and every ruling before it is what it
    was: with no witnessed change each call rules unwitnessed, which never fails a body."""
    schema, sig = _schema(), _sig()
    source = ce.module_source(schema, [sig], {sig.name: MOVES_RIGHT})
    box = sb.Sandbox(source, KILN, tmp_path)
    gates = sb.run_gates(source, box, CALLS, [], schema, sig=sig)
    stages = [(gate.stage, gate.passed) for gate in gates]
    assert stages == [("parses", True), ("confined", True),
                      ("compile_tools.memorised_values", True), ("executes_on_s0", True),
                      ("deterministic", True), ("non_trivial", True), ("sensitivity", True),
                      ("replay_fidelity", True), (sb.TRANSITION_STAGE, True)]
    assert gates[-1].metrics["transition_unwitnessed"] == 2
