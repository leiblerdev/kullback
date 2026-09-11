"""Failed world-invariance stops the compile chain so replay is not the deciding score (D247)."""
from __future__ import annotations

from conftest import PTR
from kullback.builder import compile_env as ce
from kullback.builder import sandbox as sb
from kullback.runner.records import Column, EntitySchema, FieldStat, ToolCall, ToolSig

WORKSHOP = {
    "looms": {
        "LM-10": {"loom_id": "LM-10", "phase": "warp"},
        "LM-20": {"loom_id": "LM-20", "phase": "warp"},
    },
}
DAWN = WORKSHOP
DUSK = {
    "looms": {
        "LM-10": {"loom_id": "LM-10", "phase": "weft"},
        "LM-20": {"loom_id": "LM-20", "phase": "weft"},
    },
}

READS_THE_PHASE = ('loom = self.db.looms[loom_id]\n'
                   'return {"phase": loom.phase}\n')
WRITES_A_CONSTANT = 'return {"phase": "warp"}\n'


def _schema() -> EntitySchema:
    columns = [Column(table="looms", name=name, **{"class": "hard"})
               for name in ("loom_id", "phase")]
    return EntitySchema(tables=["looms"], columns=columns,
                        id_patterns={"looms.loom_id": r"^LM-\d{2}$"})


def _sig() -> ToolSig:
    return ToolSig(name="loom_pass", description="Report one loom.",
                   args_fields=[FieldStat(name="loom_id", types=["str"], optional=False)],
                   kind="read", unclassified=False)


def _call(call_id: str, args: dict, result) -> ToolCall:
    return ToolCall(id=call_id, name="loom_pass", args=args, result=result, raw_ptr=PTR)


def test_failed_invariance_stops_the_chain_so_replay_is_not_the_deciding_score(tmp_path):
    """(d) A world-blind body stops before replay, so shown matches cannot keep it."""
    calls = [_call("c1", {"loom_id": "LM-10"}, {"phase": "warp"}),
             _call("c2", {"loom_id": "LM-10"}, {"phase": "weft"})]
    states = {"c1": DAWN, "c2": DUSK}
    schema, sig = _schema(), _sig()
    source = ce.module_source(schema, [sig], {sig.name: WRITES_A_CONSTANT})
    box = sb.Sandbox(source, WORKSHOP, tmp_path, call_states=states,
                     call_tasks={"c1": "task-dawn", "c2": "task-dusk"})
    gates = sb.run_gates(source, box, calls, [], schema, sig=sig)
    stages = [gate.stage for gate in gates]
    assert stages[-1] == "world_invariance"
    assert "replay_fidelity" not in stages
    assert "sensitivity" not in stages
    assert gates[-1].passed is False
    passed, matched = ce.attempt_score(gates)
    assert matched == 0
    reading_source = ce.module_source(schema, [sig], {sig.name: READS_THE_PHASE})
    reading_box = sb.Sandbox(reading_source, WORKSHOP, tmp_path / "reads", call_states=states,
                             call_tasks={"c1": "task-dawn", "c2": "task-dusk"})
    reading = sb.run_gates(reading_source, reading_box, calls, [], schema, sig=sig)
    assert ce.attempt_score(reading) > (passed, matched)
    assert "replay_fidelity" in [gate.stage for gate in reading]
