"""The Verifier writer: checks from the Intent, the recorded policy, the tool schemas and the Starting state.

Ported from the intentv experiment, prompt revision r1 (DESIGN.md, the Spec role). One model session
reads the Intent facts, the recorded agent policy, the tool list and, through two read-only lookups,
the Starting state; it writes demands, each with a because and the ids of the facts it answers. It
never sees a Run, a Reference, a Verifier or an End state: the inputs come from the Environment's own
files and `writer_tools.refuse_run_inputs` refuses anything else.

Code then decides what stands: a check whose because quotes neither the Intent nor a policy section
is dropped, a check resting only on facts that were accepted and never witnessed is dropped, a write
whose because quotes neither a witnessed fact nor the policy text is dropped, facts no check names
become gaps, and `must_not` adds the forbidden side. Demands refused on their because, and checks that
make the Spec unpassable (compile.UNSATISFIABLE), go back to the writer once, in one follow-up turn of
the same conversation, and the corrections that ground are kept. `repair` answers a ruling by changing
or defending the ruled checks only, under the same rules.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Optional

from kullback.runner.records import Record, Verifier
from kullback.spec import writer_tools
from kullback.spec.actions import action_tools_of
from kullback.spec.compile import (
    ATOM_KINDS,
    FORBIDDEN_EQUALS_REQUIRED,
    NO_WRITE_BESIDE_REQUIRED,
    WRITE_WITHOUT_ROW,
    Compiled,
    compile_spec,
)
from kullback.spec.end_state import Gates, called_tools, gates_of, reference_run, unseen_demand
from kullback.spec.ground import coverage, valid_because, valid_write_because
from kullback.spec.must_not import must_not
from kullback.spec.schema import (
    Check,
    FactSource,
    IntentFact,
    Spec,
    SpecIntent,
    intent_text,
    load_spec,
    save_spec,
    spec_verifier_path,
    tier_of,
)

# The one prompt of the r1 arm, with fact ids and unwitnessed facts added (DESIGN.md rule 5).
SYSTEM_PROMPT = """You write the end-state demands for one customer request. You have never seen \
any agent run, reference solution, or existing check for this task, and you must not ask for one: \
derive everything from the Intent, the policy, the tool list, and the Starting state reads below.

Rules:
- Demand only what the Intent asks for or the policy requires. Never demand a step just because \
it would be a sensible way to do the job.
- The recorded agent policy below is the standing policy. Reason from it, never from memory of \
how such systems work. Name its section in because when it decides.
- Spend at most 4 read rounds, then write the JSON with what you have. A truncated list is not a \
reason to keep reading; demand the shape instead of the literal you could not see.
- Say only what the Intent states word for word: the because must quote the Intent span that IS \
the fact. Never demand a computed, read, or rephrased figure as a say; if the answer must convey \
it some other way, leave it out.
- List every write an acceptable run may make: required where the Intent or policy demands it, \
allowed everywhere else it is acceptable. Set cap to the number of entities listed, so a run that \
stays inside the listed writes never fails on coverage.
- Every exact value (ids, names, amounts, codes) must come from a Starting state read in this \
session, never from memory or guesswork. If a value is not in the Starting state, demand the \
shape (which tool and field) rather than a literal, or leave it out.
- The Intent is a list of facts, each with an id. Every demand names in fact_ids the facts it \
answers. A fact marked unwitnessed was only accepted by the user, never volunteered: a demand may \
name it, but a demand resting on unwitnessed facts alone is dropped.
- Output one JSON object only, no prose: {"demands": [{"id": "d0", "kind": "required", \
"demand": "write" | "say" | "ask" | "no_write" | "cap" | "shape", ...fields..., \
"fact_ids": ["f1"], "because": "quote of the Intent span or the policy rule name"}]}.
- write needs tool, id_field, entity, values (an object of field to value read from the state).
- say needs values: list the values the agent must state (an amount, an id, a status word), one \
each, never a sentence.
- ask needs field (a field of the request the agent must ask the user about) or confirm_tool.
- no_write needs no fields: the end state is the starting state.
- cap needs count: the number of entities listed above.
- shape needs tool, field, id_field: the write must carry a value the run read for that column.
- refuse needs nothing, or tool: the policy or the user refuses the request, and the agent must say so.
- handoff needs tool: the user asks for a person, and the agent must hand the conversation over.
- A transfer or hand-off is the handoff demand, never a write.
- A cap counts data writes only, never a hand-off.
- The id_field of a write is an argument that names a row.
- Never write code, a predicate or a lambda: a write is a cell of the end state (tool, row, values).
- because must quote the Intent span word for word (at least 12 characters) or name the policy \
rule or policy section. A demand with no valid because is dropped.
- A write's because quotes the user's words asking for it, or names the policy section and \
quotes the sentence requiring it; a section alone fails.
"""

FOLLOWUP_PROMPT = """Code refused these demands of yours, each for the reason and rule named. Send a \
corrected version of each, under the same id, or leave it out to drop it. Send only these demands, \
as one JSON object, no prose: {"demands": [...]}.
"""

# Why a demand carrying code is refused (D320): what a Run must write is the end state's cells, read by code.
WRITE_PREDICATE = "write demands are end-state cells"
# The keys a demand may not carry: code the Runner would evaluate in place of the end state.
_PREDICATE_KEYS = ("predicate", "predicate_src", "code", "lambda")

# The rule each refusal a writer is told about breaks, in the words the follow-up turn shows it.
SENT_BACK_RULES = {
    "ungrounded": "because must quote the Intent word for word (at least 12 characters) or name a policy section",
    "unwitnessed": "a demand may not rest on unwitnessed facts alone; its because must quote a witnessed fact",
    "condition_unquoted": "a write's because must quote the user's own words asking for that action, or name "
                          "the policy section and quote the sentence of it that requires the action; naming a "
                          "section alone is not enough",
    NO_WRITE_BESIDE_REQUIRED: "a no_write forbids every write of the tools it covers, so it cannot stand beside a "
                              "required write on one of them; keep the one the Intent asks for",
    FORBIDDEN_EQUALS_REQUIRED: "one demand cannot be both required and forbidden; keep the one the Intent asks for",
    WRITE_WITHOUT_ROW: "a required write names one row: id_field, and entity as that single id value, not an object",
    "write_predicate": WRITE_PREDICATE,
    "unseen_conduct": "a hand-off or confirmation demand must be one the Reference shows; the Reference made no "
                      "such call",
    "value_not_argument": "a write's values name arguments of its tool only",
}

REPAIR_PROMPT = """A ruling says some checks you wrote for this customer request are wrong. You \
still have never seen any agent run, and you must not ask for one. For each ruled check, either \
change it or defend it; you may not touch any other check and you may not add checks.

Rules:
- Every rule of the writer still holds: reason from the recorded policy, never memory; quote the \
Intent word for word; take exact values only from Starting state reads; demand the shape when a \
literal is unseen; at most 4 read rounds.
- A changed check keeps its id and must still carry a because that quotes the Intent (at least 12 \
characters) or names a policy section, or code keeps the old check.
- Defend a check when the Intent or the policy still requires it; say why in because.
- Output one JSON object only, no prose: {"checks": [{"id": "<ruled check id>", "action": \
"change" | "defend", "kind": "required", "demand": "write" | "say" | "ask" | "no_write" | "cap" | \
"shape", ...fields..., "fact_ids": ["f1"], "because": "..."}]}.
"""

# The longest message body a session is sent, about 6000 tokens: it bounds one call's input cost
# whatever the policy or tool list holds (held-out bodies ran 12000 to 18000 characters).
BODY_CHARS = 24000
# The longest ruling excerpt a repair session is shown: a quote, never a transcript.
MAX_EXCERPT = 1000
_META = ("id", "kind", "because", "fact_ids", "action")


class WriterInputs(NamedTuple):
    """Everything one writer session may see; none of it comes from a Run, `reference_tools` aside.

    `reference_tools` is never shown: code alone reads it to send back a hand-off or confirmation the
    Reference never made (None: no Reference, nothing sent back).
    """
    intent: SpecIntent
    policy_text: str
    sections: list[str]
    tools: list[dict]
    write_tools: set[str]
    state: dict
    constraints: list[dict]
    canon: Callable[[Any], Any]
    # The tools the world records as actions (spec/actions.py): never in write_tools.
    action_tools: frozenset = frozenset()
    reference_tools: Optional[frozenset] = None


class Ruling(Record):
    """What a repair answers: the ruled check ids, the reason in words, and the Examiner's excerpt."""
    check_ids: list[str]
    reason: str
    excerpt: str = ""


class Written(NamedTuple):
    spec: Spec
    counts: dict
    session: dict


def load_inputs(workdir: Path, task_id: str, intent: Optional[SpecIntent] = None) -> WriterInputs:
    """The Environment's own files for one Task: policy, tool schemas, Starting state, constraints.

    The policy is the recorded agent's own system prompt, else the Environment's policy.md. The
    Starting state is the shared world with the Task's overlay laid over it, never a Run's record.
    An action tool (spec/actions.py) is kept out of the write tools.
    """
    from kullback.runner.state import StateView
    from kullback.runner.target import canon_fn
    from kullback.runner.world.environment import BuiltEnvironment

    env = BuiltEnvironment(workdir)
    if intent is None:
        spec = load_spec(workdir, task_id)
        if spec is None:
            raise FileNotFoundError(f"no Spec for Task {task_id}: mine its Intent first")
        intent = spec.intent
    policy = env.system_prompt(env.task(task_id)) or ""
    overlay, rows = env.overlay(task_id)
    tools = writer_tools.tool_rows(env.sigs)
    actions = frozenset(action_tools_of(env.schema))
    called = called_tools(reference_run(workdir, task_id))
    return WriterInputs(intent=intent, policy_text=policy, sections=writer_tools.policy_sections_of(workdir),
                        tools=tools, write_tools={row["name"] for row in tools if row["kind"] == "write"} - actions,
                        state=StateView(env.db, overlay, rows).shared, constraints=_constraints(workdir),
                        canon=canon_fn(env.canon_rules), action_tools=actions,
                        reference_tools=None if called is None else frozenset(called))


def _constraints(workdir: Path) -> list[dict]:
    """The compiled policy constraints the Examiner kept, or none."""
    path = Path(workdir) / "constraints_check.json"
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("constraints") or [])
    except (OSError, ValueError, AttributeError):
        return []


def unwitnessed(fact: IntentFact) -> bool:
    """An accepted fact no other recording volunteers (DESIGN.md rule 5)."""
    return fact.stance == "accepted" and not fact.witnesses


def _facts_view(intent: SpecIntent) -> list[dict]:
    return [dict({"id": fact.id, "text": fact.text, "stance": fact.stance},
                 **({"unwitnessed": True} if unwitnessed(fact) else {}))
            for fact in intent.facts]


def _body(inputs: WriterInputs) -> dict:
    return {"intent": _facts_view(inputs.intent), "tools": inputs.tools,
            "policy": inputs.sections or ["no policy sections; the Intent alone governs"],
            "policy_text": inputs.policy_text or "no recorded agent policy found"}


def _message(prompt: str, body: dict) -> tuple[list[dict], int]:
    """The one user message, its body cut at BODY_CHARS; the count of characters cut, 0 when whole."""
    text = json.dumps(body)
    cut = max(len(text) - BODY_CHARS, 0)
    tail = "\n(inputs truncated)" if cut else ""
    return [{"role": "user", "content": prompt + "\n" + text[:BODY_CHARS] + tail}], cut


def writer_messages(inputs: WriterInputs) -> tuple[list[dict], int]:
    return _message(SYSTEM_PROMPT, _body(inputs))


def _refusal(item: Any, fact_ids: list[str], inputs: WriterInputs) -> str:
    """Why code refuses a demand before it becomes a Check, or empty when it stands."""
    if not isinstance(item, dict) or item.get("kind") not in ATOM_KINDS:
        return "bad_kind"
    if any(key in item for key in _PREDICATE_KEYS):
        return "write_predicate"
    because = str(item.get("because") or "")
    if not valid_because(because, inputs.intent.text, inputs.sections):
        return "ungrounded"
    facts = {fact.id: fact for fact in inputs.intent.facts}
    witnessed = intent_text([fact for fact in inputs.intent.facts if not unwitnessed(fact)])
    if fact_ids and all(unwitnessed(facts[f]) for f in fact_ids):
        return "unwitnessed"
    if not valid_because(because, witnessed, inputs.sections):
        return "unwitnessed"
    if item.get("demand") == "write" and not valid_write_because(because, witnessed, inputs.policy_text):
        return "condition_unquoted"
    if item.get("demand") == "write" and _not_arguments(item, inputs):
        return "value_not_argument"
    return ""


def _arguments(tool: Any, inputs: WriterInputs) -> Optional[list[str]]:
    """The argument names of a listed tool, or None when the tool is not listed or lists none (unknown)."""
    return next((list(row["args"]) for row in inputs.tools if row.get("name") == tool and row.get("args")), None)


def _not_arguments(item: dict, inputs: WriterInputs) -> list[str]:
    """The keys of a write's values that are not arguments of its tool (none when the tool is not listed)."""
    args = _arguments(item.get("tool"), inputs)
    values = item.get("values")
    if args is None or not isinstance(values, dict):
        return []
    return sorted(str(key) for key in values if key not in args)


def _check_of(number: Any, item: Any, inputs: WriterInputs) -> tuple[Optional[Check], str]:
    """One demand as a Check, or the reason code refuses it."""
    known = {fact.id for fact in inputs.intent.facts}
    raw_ids = item.get("fact_ids") if isinstance(item, dict) else None
    fact_ids = [str(f) for f in raw_ids or [] if str(f) in known]
    refusal = _refusal(item, fact_ids, inputs)
    if refusal:
        return None, refusal
    demand = {key: value for key, value in item.items() if key not in _META}
    check_id = number if isinstance(number, str) else f"c{number}"
    return Check(id=check_id, kind=item["kind"], demand=demand, because=str(item["because"]),
                 tier=tier_of(item["kind"], demand), fact_ids=fact_ids), ""


def _keep(demands: Iterable[Any], inputs: WriterInputs) -> tuple[list[Check], dict, list[tuple[int, Any, str]]]:
    """The checks code keeps, how many of each refusal it made, and each refused demand with its code."""
    checks, refused = [], []
    counts = {"demands": 0, "kept": 0, "ungrounded": 0, "unwitnessed": 0, "condition_unquoted": 0, "bad_kind": 0,
              "write_predicate": 0, "value_not_argument": 0}
    for number, item in enumerate(demands):
        counts["demands"] += 1
        check, refusal = _check_of(number, item, inputs)
        if check is None:
            counts[refusal] += 1
            refused.append((number, item, refusal))
            continue
        counts["kept"] += 1
        checks.append(check)
    return checks, counts, refused


def _spec_of(task_id: str, checks: list[Check], counts: dict, inputs: WriterInputs) -> Spec:
    spec = with_gaps(Spec(task_id=task_id, intent=inputs.intent, checks=checks), inputs)
    counts["gaps"] = len(spec.gaps)
    return spec


def build_spec(task_id: str, demands: Iterable[Any], inputs: WriterInputs) -> tuple[Spec, dict]:
    """The Spec code keeps from the writer's demands, and how many of each refusal it made."""
    checks, counts, _ = _keep(demands, inputs)
    return _spec_of(task_id, checks, counts, inputs), counts


def _writer_id(number: int, item: Any) -> str:
    return str(item.get("id") or f"d{number}") if isinstance(item, dict) else f"d{number}"


def _rule(code: str, item: Any, inputs: Optional[WriterInputs]) -> str:
    """The rule line a refused demand is sent back with; a value_not_argument names its tool's arguments."""
    rule = SENT_BACK_RULES[code]
    if code == "value_not_argument" and inputs is not None and isinstance(item, dict):
        rule += f"; {item.get('tool')} takes {', '.join(_arguments(item.get('tool'), inputs) or [])}"
    return rule


def followup_message(refused: list[tuple[int, Any, str]], inputs: Optional[WriterInputs] = None) -> str:
    """The one follow-up turn: each refused demand's id, its reason code and the rule it broke."""
    rows = [{"id": _writer_id(number, item), "reason": code, "rule": _rule(code, item, inputs)}
            for number, item, code in refused]
    return FOLLOWUP_PROMPT + json.dumps({"refused": rows})


def _send_back(messages: list[dict], session: dict, refused: list[tuple[int, Any, str]], inputs: WriterInputs,
               model: Any, config: Any, price: Callable[[dict], float],
               ceiling_usd: Optional[float]) -> tuple[list[Check], dict]:
    """One follow-up turn in the same conversation; the corrected demands code keeps, and that turn's session.

    A corrected demand keeps the check id of the demand it corrects; one under an id never sent back is ignored.
    """
    numbers = {_writer_id(number, item): number for number, item, _ in refused}
    turn = messages + [{"role": "assistant", "content": session["text"]},
                       {"role": "user", "content": followup_message(refused, inputs)}]
    remaining = None if ceiling_usd is None else max(ceiling_usd - float(session.get("usd") or 0.0), 0.0)
    again = writer_tools.run_session(model, turn, inputs.state, _config(config), writer_tools.MAX_READ_ROUNDS,
                                     price, force_answer=True, ceiling_usd=remaining)
    corrected, _ = writer_tools.parse_list(again["text"], "demands")
    kept: dict[int, Check] = {}
    for item in corrected:
        number = numbers.get(str(item.get("id"))) if isinstance(item, dict) else None
        if number is None or number in kept:
            continue
        check, _ = _check_of(number, item, inputs)
        if check is not None:
            kept[number] = check
    return [kept[number] for number in sorted(kept)], again


def with_gaps(spec: Spec, inputs: WriterInputs) -> Spec:
    """The Spec with its gaps: facts no check covers, then one line per check compile names as dropped.

    A compile reason (a write with no entity, say) is a check the Verifier silently lacks, so it is shown
    where the Examiner and the trust report already look.
    """
    spec = spec.model_copy(update={"gaps": []})
    reasons = compile_spec(spec, inputs.write_tools, inputs.canon, inputs.sections, inputs.action_tools).reasons
    return spec.model_copy(update={"gaps": coverage(spec) + list(reasons)})


def _price(model_id: Optional[str]) -> Callable[[dict], float]:
    from kullback.ai.usage import Usage
    from kullback.runner.budget import call_cost

    return lambda usage: call_cost(Usage(**usage), model_id)


def _config(config: Any) -> Any:
    from kullback.ai.provider import ModelConfig

    return config or ModelConfig(thinking={"type": "adaptive", "display": "summarized"}, max_tokens=2000)


def write_spec(task_id: str, inputs: WriterInputs, model: Any, *, model_id: Optional[str] = None,
               config: Any = None, ceiling_usd: Optional[float] = None) -> Written:
    """One writer session for one Task, and the Spec code keeps from it; past `ceiling_usd` it answers at once."""
    writer_tools.refuse_run_inputs(inputs.intent, inputs.state)
    messages, cut = writer_messages(inputs)
    session = writer_tools.run_session(model, messages, inputs.state, _config(config),
                                       writer_tools.MAX_READ_ROUNDS, _price(model_id),
                                       force_answer=True, ceiling_usd=ceiling_usd)
    demands, error = writer_tools.parse_list(session["text"], "demands")
    checks, counts, refused = _keep(demands, inputs)
    refused = [row for row in refused if row[2] in SENT_BACK_RULES]
    clashing = _clashing(task_id, checks, demands, inputs)
    clashed = {f"c{number}" for number, _, _ in clashing}
    unseen = [(int(check.id[1:]), demands[int(check.id[1:])], "unseen_conduct") for check in checks
              if check.id not in clashed and unseen_demand(check, inputs.reference_tools)]
    out = clashed | {f"c{number}" for number, _, _ in unseen}
    checks = [check for check in checks if check.id not in out]
    counts.update(sent_back=len(refused), sent_back_kept=0, unsatisfiable_sent_back=len(clashing),
                  unseen_sent_back=len(unseen))
    refused = sorted(refused + clashing + unseen, key=lambda row: row[0])
    if refused:
        corrected, again = _send_back(messages, session, refused, inputs, model, config, _price(model_id),
                                      ceiling_usd)
        counts["sent_back_kept"] = len(corrected)
        checks = sorted(checks + corrected, key=lambda check: int(check.id[1:]))
        session = dict(again, usd=float(session.get("usd") or 0.0) + float(again.get("usd") or 0.0),
                       tool_rounds=session.get("tool_rounds", 0) + again.get("tool_rounds", 0),
                       tool_uses=session.get("tool_uses", []) + again.get("tool_uses", []))
    counts["unsatisfiable"] = sorted({code for _, code in _unpassable(task_id, checks, inputs)})
    spec = _spec_of(task_id, checks, counts, inputs)
    counts.update(parse_error=error, truncated_chars=cut)
    return Written(spec, counts, session)


def _unpassable(task_id: str, checks: list[Check], inputs: WriterInputs) -> tuple[tuple[str, str], ...]:
    spec = Spec(task_id=task_id, intent=inputs.intent, checks=checks)
    return compile_spec(spec, inputs.write_tools, inputs.canon, inputs.sections, inputs.action_tools).unsatisfiable


def _clashing(task_id: str, checks: list[Check], demands: list, inputs: WriterInputs) -> list[tuple[int, Any, str]]:
    """Each check that makes the Spec unpassable (compile.UNSATISFIABLE) as a demand to send back, with its reason."""
    first: dict[str, str] = {}
    for check_id, code in _unpassable(task_id, checks, inputs):
        first.setdefault(check_id, code)
    return [(int(check_id[1:]), demands[int(check_id[1:])], code) for check_id, code in first.items()]


# The demands that stay weighted atoms beside an expected end state: facts told and questions asked (D320).
ATOM_DEMANDS = ("say", "ask")


def verifier_of(spec: Spec, inputs: WriterInputs, gates: Optional[Gates] = None) -> Compiled:
    """The Spec's checks compiled, with its gates when it has an expected end state.

    With an expected end state the writes are its cells, so only fact and question checks compile to
    atoms and `must_not` adds nothing: the forbidden list and the collateral rule answer for it. With
    none (no Reference yet) every check compiles as before, with the must-not atoms. The Spec's
    review edits to its atoms (`atom_edits`) are applied last.
    """
    if gates is not None and gates.expected:
        kept = spec.model_copy(update={"checks": [c for c in spec.checks if c.demand.get("demand") in ATOM_DEMANDS]})
        compiled = compile_spec(kept, inputs.write_tools, inputs.canon, inputs.sections, inputs.action_tools)
        atoms = compiled.verifier.atoms
    else:
        compiled = compile_spec(spec, inputs.write_tools, inputs.canon, inputs.sections, inputs.action_tools)
        must = [atom for atom in compiled.verifier.atoms if atom.kind in ("required", "allowed")]
        atoms = compiled.verifier.atoms + must_not(must, spec.intent.text, inputs.constraints, sorted(inputs.write_tools))
    update: dict = {"atoms": edited_atoms(atoms, spec.atom_edits)}
    if gates is not None:
        update.update(expected=gates.expected, forbidden=gates.forbidden, conduct=gates.conduct)
    return compiled._replace(verifier=compiled.verifier.model_copy(update=update))


def edited_atoms(atoms: list, edits: Iterable[dict]) -> list:
    """The atoms after each review edit in order: its drop ids out, its added atoms in (ids replaced)."""
    from kullback.gates import verifier_suite

    out = list(atoms)
    for edit in edits or ():
        dropped = {str(atom_id) for atom_id in edit.get("drop") or []}
        added = [verifier_suite.make_atom(str(a["id"]), a["kind"], dict(a["payload"]),
                                          description=a.get("description")) for a in edit.get("add") or []]
        new_ids = {atom.id for atom in added}
        out = [atom for atom in out if atom.id not in dropped and atom.id not in new_ids] + added
    return out


def runner_verifier_path(workdir: Path, task_id: str) -> Path:
    """Where the Runner reads a Task's Verifier from (kullback score, BuiltEnvironment.verifier)."""
    return Path(workdir) / "verifiers" / f"{task_id}.json"


def end_state_counts(gates: Gates) -> dict:
    """What the Spec record keeps of its gates: the Reference, the cells, how many are unsupported."""
    cells = gates.expected[0].cells if gates.expected else []
    return {"reference": gates.reference, "end_states": len(gates.expected), "cells": len(cells),
            "unsupported": gates.unsupported, "forbidden": len(gates.forbidden), "conduct": len(gates.conduct),
            "unseen": gates.unseen}


def write_verifier(spec: Spec, workdir: Path, inputs: Optional[WriterInputs] = None,
                   gates: Optional[Gates] = None) -> list[Path]:
    """Saves the Spec and its whole Verifier, the gates derived by code from the Task's Reference.

    The Spec is the one writer of a Task's Verifier (D320): the Runner's file and the Spec's copy are
    the same bytes. The end-state counts ride on the saved Spec as `end_state`. The review's kept cell
    and conduct edits (D325) are applied after, so a router repair does not drop them.
    """
    from kullback.spec.review import apply_end_state_edits  # review imports this module
    inputs = inputs or load_inputs(workdir, spec.task_id, spec.intent)
    gates = gates if gates is not None else gates_of(workdir, spec, inputs.policy_text, inputs.canon,
                                                    write_tools=inputs.write_tools)
    verifier: Verifier = verifier_of(spec, inputs, gates).verifier
    text = json.dumps(verifier.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    spec = spec.model_copy(update={"end_state": end_state_counts(gates)})
    paths = [save_spec(workdir, spec), spec_verifier_path(workdir, spec.task_id),
             runner_verifier_path(workdir, spec.task_id)]
    for path in paths[1:]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    apply_end_state_edits(workdir, spec)
    return paths


def _ruled_view(spec: Spec, ruling: Ruling) -> dict:
    ruled = [check.model_dump(mode="json") for check in spec.checks if check.id in set(ruling.check_ids)]
    return {"ruled_checks": ruled, "ruling": {"reason": ruling.reason,
                                              "excerpt": str(ruling.excerpt)[:MAX_EXCERPT]}}


def repair_messages(spec: Spec, ruling: Ruling, inputs: WriterInputs) -> tuple[list[dict], int]:
    return _message(REPAIR_PROMPT, dict(_body(inputs), **_ruled_view(spec, ruling)))


def apply_repair(spec: Spec, ruling: Ruling, answers: Iterable[Any], inputs: WriterInputs) -> tuple[Spec, dict]:
    """The Spec after a repair: only ruled checks change, and only to grounded checks."""
    ruled = set(ruling.check_ids) & {check.id for check in spec.checks}
    replaced: dict[str, Check] = {}
    counts = {"changed": 0, "defended": 0, "refused": 0, "not_ruled": 0}
    for item in answers:
        check_id = str(item.get("id")) if isinstance(item, dict) else ""
        if check_id not in ruled:
            counts["not_ruled"] += 1
        elif item.get("action") == "defend":
            counts["defended"] += 1
        else:
            check, _ = _check_of(check_id, item, inputs)
            counts["changed" if check else "refused"] += 1
            if check is not None:
                replaced[check_id] = check
    checks = [replaced.get(check.id, check) for check in spec.checks]
    repaired = spec.model_copy(update={"checks": checks, "version": spec.version + 1})
    return with_gaps(repaired, inputs), counts


def repair(spec: Spec, ruling: Ruling, inputs: WriterInputs, model: Any, *,
           model_id: Optional[str] = None, config: Any = None,
           ceiling_usd: Optional[float] = None) -> Written:
    """One repair session answering a ruling; no Run is shown, only the ruling's excerpt."""
    ruling = Ruling.model_validate(ruling.model_dump() if isinstance(ruling, Record) else ruling)
    writer_tools.refuse_run_inputs(inputs.intent, inputs.state, ruling.excerpt)
    messages, cut = repair_messages(spec, ruling, inputs)
    session = writer_tools.run_session(model, messages, inputs.state, _config(config),
                                       writer_tools.MAX_READ_ROUNDS, _price(model_id),
                                       force_answer=True, ceiling_usd=ceiling_usd)
    answers, error = writer_tools.parse_list(session["text"], "checks")
    repaired, counts = apply_repair(spec, ruling, answers, inputs)
    counts.update(parse_error=error, truncated_chars=cut)
    return Written(repaired, counts, session)


def ruling_of(body: dict) -> Ruling:
    """A ruling as the router hands it over: its checks, its reason, and its quoted evidence.

    The evidence is the ruling's excerpt, else its quotes joined; a quote that is not text is left out.
    """
    quotes = body.get("quotes") or body.get("quoted") or []
    excerpt = body.get("excerpt") or "\n".join(str(q) for q in quotes if isinstance(q, str))
    return Ruling(check_ids=[str(c) for c in body.get("check_ids") or []], reason=str(body.get("reason") or ""),
                  excerpt=str(excerpt))


def repair_for_ruling(workdir: Path, task_id: str, ruling: dict, model: Any,
                      ceiling_usd: Optional[float] = None, *, model_id: Optional[str] = None) -> Spec:
    """The router's entry point: repair the Task's Spec from one ruling file and tell the bus.

    Loads the Spec and the writer's inputs, repairs the ruled checks, saves the Spec (version + 1),
    rewrites both Verifier files and publishes spec.repaired when any check changed,
    else spec.defended.
    """
    from kullback.agent.bus import Bus
    from kullback.spec import events

    spec = load_spec(workdir, task_id)
    if spec is None:
        raise FileNotFoundError(f"no Spec for Task {task_id}: nothing to repair")
    inputs = load_inputs(workdir, task_id, spec.intent)
    ruled = ruling_of(ruling)
    written = repair(spec, ruled, inputs, model, model_id=model_id or getattr(model, "name", None),
                     ceiling_usd=ceiling_usd)
    write_verifier(written.spec, workdir, inputs)
    bus = Bus(Path(workdir) / "bus.jsonl", agent="spec")
    before = {check.id: check for check in spec.checks}
    changed = [check.id for check in written.spec.checks if before.get(check.id) != check]
    if changed:
        events.spec_repaired(bus, task_id, written.spec.version, changed)
    else:
        events.spec_defended(bus, task_id, ruled.check_ids, str(ruling.get("code") or "other"))
    return written.spec


def spoken_intent(task_id: str, text: str) -> SpecIntent:
    """Spoken user text as one volunteered fact per sentence: the stand-in until a mined Intent exists."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]
    facts = [IntentFact(id=f"f{n}", text=s, stance="volunteered", source=FactSource(recording="spoken", turn=0))
             for n, s in enumerate(sentences)]
    return SpecIntent(task_id=task_id, facts=facts, text=intent_text(facts))

