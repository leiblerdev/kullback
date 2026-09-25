"""The whole harness over the invented library world: ingest, mine, cluster, compile, replay, verdict, regrade.

No model is called. The tool bodies come from a TestModel scripted with the bodies below, which is
what a correct Builder LLM would have written for these six library tools.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from kullback import report
from kullback.ai.provider import RecordedModel, TestModel
from kullback.builder import cluster, compile_env, ingest, mine, policy, user_sim
from kullback.examiner import derive as verifier_mod
from kullback.runner import canon, loop, regrade, route
from kullback.runner import verdict as verdict_mod
from kullback.runner.canon import CanonRules
from kullback.runner.records import (
    Environment,
    GateResult,
    RunnerVersion,
    Trace,
    UserFact,
    UserRules,
    Verifier,
    as_dict,
)

TOOL_BODIES = {
    "find_member_id_by_name_zip": """
for member_id, member in self.db.members.items():
    name = member.name or {}
    address = member.address or {}
    if (name.get("first_name") == first_name and name.get("last_name") == last_name
            and address.get("zip") == zip):
        return member_id
raise ValueError("Member not found")
""",
    "find_member_id_by_email": """
for member_id, member in self.db.members.items():
    if (member.email or "") == email:
        return member_id
raise ValueError("No member with email " + email)
""",
    "get_loan_details": """
loan = self.db.loans.get(loan_id)
if loan is None:
    raise ValueError("Loan not found")
return loan
""",
    "get_hold_details": """
hold = self.db.holds.get(hold_id)
if hold is None:
    raise ValueError("Hold not found")
return hold
""",
    "update_loan": """
loan = self.db.loans.get(loan_id)
if loan is None:
    raise ValueError("Loan not found")
# Renewals land on the third day after the end of the due month: the date both fixture
# recordings share, computed from the row rather than written as a constant.
year, month, day = [int(part) for part in loan.due.split("-")]
month_lengths = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31,
                 11: 30, 12: 31}
last = month_lengths[month]
renewed = last + 3
if renewed > last:
    renewed = renewed - last
    month = month + 1
loan.due = "%04d-%02d-%02d" % (year, month, renewed)
return loan
""",
    "cancel_hold": """
hold = self.db.holds.get(hold_id)
if hold is None:
    raise ValueError("Hold not found")
hold.status = "cancelled"
return hold
""",
}

REFERENCE_TRACE = "libsim01"  # the loan renewal on loan_01
RENEWED_LOAN = "loan_01"


class RecordedUser:
    """The Reference's own user turns, in order, then the stop marker the recorded user gave."""

    def __init__(self, trace: Trace) -> None:
        self.said = [t.content for t in trace.turns if t.role == "user" and t.content]
        self.index = 0

    def reply(self, transcript) -> str:
        if self.index >= len(self.said):
            return "###STOP###"
        said = self.said[self.index]
        self.index += 1
        return said


def _reference_jsonl(trace: Trace, path: Path) -> Path:
    """The Reference's assistant messages as a JSONL, which is what RecordedModel replays."""
    calls = {call.id: call for call in trace.tool_calls}
    lines = []
    for turn in trace.turns:
        if turn.role != "assistant":
            continue
        lines.append({"role": "assistant", "content": turn.content, "tool_calls": [
            {"id": i, "name": calls[i].name, "arguments": calls[i].args}
            for i in turn.tool_call_ids if i in calls]})
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def _write(path: Path, body) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, default=str) + "\n", encoding="utf-8")


def _environment(build: dict) -> Environment:
    return Environment(env_id="e2e", schema_version="1", tools_version="1", policy_version="1",
                       assisted_tools=[n for n, b in build["builds"].items() if b.assisted])


def _gates(build: dict) -> list:
    """Every gate this build ran, in stage order."""
    gates = [GateResult.model_validate(build["summary"]["gate"]), build["tools_gate"]]
    return gates + [g for b in build["builds"].values() for g in b.gates]


@pytest.fixture(scope="module")
def build(tmp_path_factory, world_a_small_path) -> dict:
    """One whole build over the fixture, from the raw file to a regraded Verdict."""
    workdir = tmp_path_factory.mktemp("e2e")
    fixture = world_a_small_path

    summary = ingest.ingest_file(fixture, workdir)
    traces = [Trace.model_validate(json.loads(p.read_text(encoding="utf-8")))
              for p in sorted((workdir / "traces").glob("*.json"))]

    sigs = mine.mine_tools(traces)
    schema = mine.mine_schema(traces)
    tools_gate = mine.gate_tools(sigs)
    categories, tasks = cluster.cluster_runs(traces, sigs)
    write_tools = {sig.name for sig in sigs if sig.kind == "write"}

    state = compile_env.build_starting_state(traces, schema, workdir, tasks, sigs)

    calls_by_tool: dict[str, list] = {}
    for trace in traces:
        for call in trace.tool_calls:
            calls_by_tool.setdefault(call.name, []).append(call)

    builds = {}
    for sig in sigs:
        builds[sig.name] = compile_env.grade_body(
            sig, TOOL_BODIES[sig.name], calls_by_tool[sig.name], schema, state.db,
            workdir / "tools" / sig.name)
    bodies = {name: build.body for name, build in builds.items()}

    policy_text = next((t.system_prompt for t in traces if t.system_prompt), "")
    sentences = policy.split_policy(policy_text)

    reference = next(t for t in traces if t.trace_id == REFERENCE_TRACE)
    task = next(t for t in tasks if reference.trace_id in t.run_ids)

    environment = Environment(env_id="e2e", schema_version="1", tools_version="1", policy_version="1",
                              assisted_tools=[n for n, b in builds.items() if b.assisted])
    bundle = compile_env.EnvBundle(
        environment=environment, schema=schema, tools=sigs, bodies=bodies, db=state.db,
        overlays=state.overlays, overlay_values=compile_env.overlay_values(workdir),
        policy_text=policy_text, tasks=tasks, verifiers=[], assumptions=state.assumptions,
        domain="library")
    emitted = compile_env.emit_tau2_shape(bundle, workdir / "env")

    # the oracle replay: the Reference's own model replies and user turns, over the compiled tools
    source = compile_env.module_source(schema, sigs, bodies)
    overlay, overlay_rows = compile_env.load_overlay(workdir, task.id)
    router = route.Router(env_tools_module=compile_env.load_toolkit(source, state.db), starting_state=state.db,
                          overlay=overlay, overlay_rows=overlay_rows, tool_sigs=sigs)
    model = RecordedModel(_reference_jsonl(reference, workdir / "reference.jsonl"))
    run_state = loop.new_run_state("oracle", workdir=workdir / "runs" / task.id, env_id="e2e",
                                   task_id=task.id, trace_id=reference.trace_id, model="recorded",
                                   user=RecordedUser(reference), max_turns=40)
    loop.run(run_state, model, router=router)

    # a second Run beside the oracle replay: a Candidate that goes straight to the write, on its own
    # copy of the Starting state, so the report has a Run batch to count and not just the replay.
    candidate_router = route.Router(env_tools_module=compile_env.load_toolkit(source, deepcopy(state.db)),
                                    starting_state=deepcopy(state.db),
                                    overlay=overlay, overlay_rows=overlay_rows, tool_sigs=sigs)
    write_call = next(c for c in reference.tool_calls if c.name == "update_loan")
    candidate_model = TestModel([
        {"tool_calls": [{"id": "x1", "name": write_call.name, "arguments": write_call.args}]},
        {"content": "Your renewal is done. ###STOP###"},
    ], loop=True)
    candidate_state = loop.new_run_state("candidate-1", workdir=workdir / "runs" / task.id, env_id="e2e",
                                         task_id=task.id, model="candidate/scripted", seed=1, max_turns=4)
    loop.run(candidate_state, candidate_model, router=candidate_router)

    rules = user_sim.derive_user_rules(reference)
    simulated = user_sim.SimulatedUser(rules, starting_state_reader=router.state)

    verifier = verifier_mod.derive_verifier(task, run_state.path, write_tools=write_tools, canon=CanonRules())
    verdict = verdict_mod.verdict(run_state.path, verifier, canon, environment=environment,
                                  write_tools=write_tools, schema=schema)
    regraded = regrade.regrade([run_state.path], verifier, canon, out_dir=workdir / "verdicts",
                               environment=environment, write_tools=write_tools, schema=schema)
    again = regrade.regrade([run_state.path], verifier, canon, out_dir=workdir / "verdicts",
                            environment=environment, write_tools=write_tools, schema=schema)

    return {
        "workdir": workdir, "candidate_state": candidate_state, "summary": summary, "traces": traces,
        "sigs": sigs, "schema": schema,
        "tools_gate": tools_gate, "categories": categories, "tasks": tasks, "state": state,
        "builds": builds, "sentences": sentences, "emitted": emitted, "task": task,
        "reference": reference, "router": router, "run_state": run_state, "rules": rules,
        "simulated": simulated, "verifier": verifier, "verdict": verdict, "regraded": regraded,
        "again": again, "write_tools": write_tools,
    }


# --- ingest ---

def test_ingest_reads_the_fixture_and_keeps_the_grader_fields_out_of_the_traces(build):
    assert build["summary"]["format"] == "tau2_native"
    assert build["summary"]["runs"] == 3
    assert build["summary"]["gate"]["pass"] is True
    assert len(build["traces"]) == 3
    workdir = build["workdir"]
    sidecars = sorted((workdir / "grader").glob("*.json"))
    assert len(sidecars) == 3
    for path in sorted((workdir / "traces").glob("*.json")):
        text = path.read_text(encoding="utf-8")
        assert "reward_info" not in text and "evaluation_criteria" not in text


# --- mine ---

def test_mining_finds_the_six_tools_the_two_writes_and_the_three_tables(build):
    names = {sig.name: sig.kind for sig in build["sigs"]}
    assert set(names) == set(TOOL_BODIES)
    assert build["write_tools"] == {"update_loan", "cancel_hold"}
    assert build["schema"].tables == ["holds", "loans", "members"]
    assert build["schema"].id_patterns["loans.loan_id"] == r"^loan_0\d$"


def test_the_mine_gate_names_the_thin_tools_rather_than_passing_them(build):
    """Three Runs are thin evidence: the gate says which tools have fewer than three calls."""
    gate = build["tools_gate"]
    assert gate.passed is False
    thin = {failure.split(":")[0] for failure in gate.failures}
    assert thin == set(TOOL_BODIES)
    assert gate.metrics["writes"] == 2


# --- cluster ---

def test_each_simulation_of_the_fixture_becomes_its_own_task(build):
    """The three fixture simulations are libsim01, libsim02 and libsim03: three different Tasks.

    The two renewal Runs share the loan lookup and renewal calls, but the Category is still one
    per write-tool signature, so the renewal Category still holds both of them.
    """
    tasks = {t.id: t for t in build["tasks"]}
    assert len(tasks) == 3
    assert build["task"].run_ids == ["libsim01"]
    assert [c.write_tools for c in build["categories"]].count(["update_loan"]) == 1


# --- starting state ---

def test_the_starting_state_undoes_the_recorded_write(build):
    loan = build["state"].db["loans"][RENEWED_LOAN]
    assert loan["status"] == "checked_out"
    assert loan["due"] == "2026-03-20"
    assert build["state"].db["holds"]["hold_01"]["status"] == "held"
    assert (build["workdir"] / "db.json").exists()


def test_every_task_has_an_overlay_pinning_its_own_rows(build):
    for task in build["tasks"]:
        overlay, values = compile_env.load_overlay(build["workdir"], task.id)
        assert overlay.task_id == task.id
        assert overlay.rows and all(row.version_hash in values for row in overlay.rows)


# --- compiled tools ---

def test_four_of_the_six_tools_pass_every_gate(build):
    passed = {name for name, b in build["builds"].items() if not b.assisted}
    assert passed == set(TOOL_BODIES) - {"find_member_id_by_email", "find_member_id_by_name_zip"}
    for name in passed:
        assert all(gate.passed for gate in build["builds"][name].gates), name


def test_the_find_tools_are_assisted_because_the_traces_show_no_member_rows(build):
    """The world the traces show has no member rows, so the lookups cannot be replayed."""
    for name in ("find_member_id_by_email", "find_member_id_by_name_zip"):
        assisted = build["builds"][name]
        assert assisted.assisted is True
        failing = [gate.stage for gate in assisted.gates if not gate.passed]
        assert failing == ["replay_fidelity"]


def test_replay_fidelity_is_measured_on_the_held_out_split_too(build):
    splits = [gate.metrics.get("split") for gate in build["builds"]["get_loan_details"].gates
              if gate.stage == "replay_fidelity"]
    assert splits == ["shown", "held_out"]


# --- the emitted tau2 shape ---

def test_the_policy_splits_into_sentences_with_spans(build):
    """Every sentence carries the span it was cut from, so a Verifier can point back at the policy
    file rather than at a copy of the words."""
    sentences = build["sentences"]
    policy_text = next((t.system_prompt for t in build["traces"] if t.system_prompt), "")
    assert len(sentences) >= 10
    assert all(s.text for s in sentences)
    for sentence in sentences:
        assert sentence.end > sentence.start, sentence.text
        slice_ = policy_text[sentence.start:sentence.end]
        assert slice_.strip(), sentence.text
        assert " ".join(slice_.split()) == sentence.text, sentence.text


# --- the oracle replay ---

def test_the_replay_follows_the_reference_call_for_call_every_call_answered_by_code(build):
    run = build["run_state"].run
    replayed = [e.payload["name"] for e in run.events if e.type == "tool_call"]
    assert replayed == [c.name for c in build["reference"].tool_calls]
    assert run.route_counts == {"code": len(build["reference"].tool_calls)}
    assert run.assisted is False
    assert run.termination_reason == "user_stop"


def test_the_replay_reproduces_the_recorded_write(build):
    loan = build["router"].world()["loans"][RENEWED_LOAN]
    assert loan["status"] == "checked_out"
    assert loan["due"] == "2026-04-03"
    assert loan["member_id"] == "member_01"
    assert loan["copy_id"] == "copy_01"


def test_the_run_jsonl_carries_its_start_and_end_state(build):
    lines = [json.loads(line) for line in
             build["run_state"].path.read_text(encoding="utf-8").splitlines() if line.strip()]
    footer, stop = lines[-1], lines[-2]
    assert footer["run_id"] == "oracle" and "start_state" not in footer
    assert stop["type"] == "stop"
    assert stop["payload"]["start_state"]["loans"][RENEWED_LOAN]["due"] == "2026-03-20"
    assert stop["payload"]["end_state"]["loans"][RENEWED_LOAN]["due"] == "2026-04-03"


# --- the simulated user ---

def test_the_simulated_user_gives_the_recorded_facts_and_never_invents_one(build):
    facts = {fact.field: fact.value for fact in build["rules"].facts}
    assert facts.get("zip") == "11201"
    answer = build["simulated"].reply([{"role": "assistant", "content": "What is your zip code?"}])
    assert "11201" in answer


def test_an_unavailable_fact_reaches_the_verdict_as_an_environment_mark(tmp_path):
    """D77: the Simulated user says it has no such fact, the loop records the tag, the Verdict marks it."""
    rules = UserRules(facts=[UserFact(field="zip", value="11201")])
    user = user_sim.SimulatedUser(rules)
    model = TestModel([{"content": "What is your email address?"}], loop=True)
    state = loop.new_run_state("unavailable", workdir=tmp_path, user=user, max_turns=2)
    loop.run(state, model)
    turns = [e for e in state.run.events if e.type == "user_turn"]
    assert turns and "fact_unavailable" in turns[0].payload["tags"]
    assert "email" in turns[0].payload["text"]
    verdict = verdict_mod.verdict(state.path, Verifier(task_id="t"), canon)
    assert "env_mark:fact_unavailable" in verdict.notes


# --- verifier and verdict ---

def test_the_verifier_names_the_write_and_its_values(build):
    """Every argument of the write is covered, as a literal where a user said it and a shape where
    the agent read it off the world (D190); the atom ids carry the column either way."""
    atoms = {atom.id: atom for atom in build["verifier"].atoms}
    write = [a for a in atoms.values() if a.target.get("kind") == "write"]
    assert [a.target["tool"] for a in write] == ["update_loan"]
    covered = {name.split(".", 1)[1] for name in atoms if name.startswith("w0.")}
    assert {"loan_id"} <= covered
    assert all(a.predicate_src for a in atoms.values() if not a.judge)


def test_the_reference_passes_its_own_verifier(build):
    verdict = build["verdict"]
    assert verdict.passed is True, verdict.notes
    assert verdict.failing_atom is None
    assert verdict.class_ == "pass"
    assert verdict.environment_suspected is False
    assert verdict.env_id == "e2e"
    assert (verdict.schema_version, verdict.tools_version, verdict.policy_version) == ("1", "1", "1")
    assert verdict.verifier_version == build["verifier"].verifier_version
    assert verdict.verdict_version == verdict_mod.VERDICT_VERSION


def test_a_run_that_never_wrote_fails_the_same_verifier(build):
    """The Verifier discriminates: the same atoms on a Run with no write must fail (D79 check 3)."""
    empty = build["workdir"] / "empty.jsonl"
    empty.write_text(json.dumps({"run_id": "empty", "termination_reason": "user_stop"}) + "\n"
                     + json.dumps({"idx": 0, "type": "stop", "payload": {"reason": "user_stop"}}) + "\n",
                     encoding="utf-8")
    verdict = verdict_mod.verdict(empty, build["verifier"], canon, write_tools=build["write_tools"])
    assert verdict.passed is False


# --- regrade ---

def test_regrade_writes_one_verdict_per_run_and_is_stable(build):
    first, second = build["regraded"], build["again"]
    assert len(first) == 1 and len(second) == 1
    assert first[0].model_dump() == second[0].model_dump()
    assert first[0].passed is True
    written = list((build["workdir"] / "verdicts").glob("*.json"))
    assert len(written) == 1


# --- the report ---

def _lay_out_the_build(build) -> Path:
    """The records this build produced, in the workdir layout cli.py and report.py read (D85)."""
    workdir = build["workdir"]
    _write(workdir / "environment.json", as_dict(_environment(build)))
    # regrade_gate (D97) refuses a Verdict with no runner_version, so cli verdict needs one frozen.
    _write(workdir / "runner_version.json", as_dict(RunnerVersion(runner_version="rv-1")))
    _write(workdir / "gates.json", [as_dict(g) for g in _gates(build)])
    _write(workdir / "schema.json", as_dict(build["schema"]))
    _write(workdir / "tool_sigs.json", [as_dict(sig) for sig in build["sigs"]])
    for task in build["tasks"]:
        _write(workdir / "tasks" / f"{task.id}.json", as_dict(task))
    _write(workdir / "verifiers" / f"{build['task'].id}.json", as_dict(build["verifier"]))
    return workdir


def test_the_report_reads_the_build_off_disk(build):
    """cli verdict then cli report over that layout: the numbers are the ones the build produced."""
    from typer.testing import CliRunner

    from kullback import cli

    workdir = _lay_out_the_build(build)
    scored = CliRunner().invoke(cli.app, ["verdict", "--workdir", str(workdir)])
    assert scored.exit_code == 0, scored.output
    written = sorted((workdir / "verdicts" / build["task"].id).glob("*.json"))
    assert [p.name.split(".")[0] for p in written] == ["candidate-1", "oracle"]

    made = CliRunner().invoke(cli.app, ["report", "--workdir", str(workdir)])
    assert made.exit_code == 0, made.output
    assert not [line for line in made.output.splitlines() if line.startswith("not read")]

    data = report.load(workdir)
    text = (workdir / "report.md").read_text(encoding="utf-8")
    assert data.environment is not None and data.tasks and data.verdicts
    assert len(data.overlays) == len(build["tasks"])
    assert {r.run_id for r in data.runs} == {"oracle", "candidate-1"}
    assert "find_member_id_by_email" in text
    assert build["task"].id in text
    assert "The decision is yours." in text

    numbers = report.task_numbers(data, next(t for t in data.tasks if t.id == build["task"].id))
    block = text.split(f"### Task {build['task'].id}", 1)[1].split("### Task")[0]
    assert numbers["runs_graded"] == len(written), "every Verdict the build wrote is counted once"
    assert f"- Runs graded: {numbers['runs_graded']}" in block
    assert "  - oracle: pass" in block, "the Reference passes its own Verifier"
    assert "  - candidate-1: fail, failing atom q.confirm" in block, "and the Candidate that never asked fails"
    assert numbers["candidate_pass_rate"] == 0.5
    assert "- Candidate pass rate: 50% (2 Runs)" in block
    assert "- Margin: n/a" in block, "no frontier Run in this batch, so there is nothing to compare against"


def test_the_candidate_run_is_graded_beside_the_oracle_replay(build):
    """One Run batch beside the replay, and the Verifier tells the two apart.

    The scripted Candidate makes the same write with the same arguments and never asks the user
    first, which is the atom the Reference's own turn earned, so it fails where the replay passes.
    """
    candidate = build["candidate_state"].run
    assert candidate.model == "candidate/scripted" and candidate.assisted is False
    assert [e.payload["name"] for e in candidate.events if e.type == "tool_call"] \
        == ["update_loan"]
    verdict = verdict_mod.verdict(build["candidate_state"].path, build["verifier"], canon,
                                  environment=Environment(env_id="e2e", schema_version="1",
                                                          tools_version="1", policy_version="1"),
                                  write_tools=build["write_tools"], schema=build["schema"])
    assert verdict.passed is False
    assert verdict.failing_atom.startswith("q.confirm"), verdict.notes
    assert build["verdict"].passed is True, "the same Verifier passes the Reference replay"
