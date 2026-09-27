"""The Verifier writer: checks from the Intent, the recorded policy, the tool schemas and the Starting state.

Ported from the intentv experiment, prompt revision r1 (DESIGN.md, the Spec role). One model session
reads the Intent facts, the recorded agent policy, the tool list and, through two read-only lookups,
the Starting state; it writes demands, each with a because and the ids of the facts it answers. It
never sees a Run, a Reference, a Verifier or an End state: the inputs come from the Environment's own
files and `writer_tools.refuse_run_inputs` refuses anything else.

Code then decides what stands: a check whose because quotes neither the Intent nor a policy section
is dropped, a check resting only on facts that were accepted and never witnessed is dropped, facts no
check names become gaps, and `must_not` adds the forbidden side. `repair` answers a ruling by changing
or defending the ruled checks only, under the same rules.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Optional

from kullback.runner.records import Record, Verifier
from kullback.spec import writer_tools
from kullback.spec.compile import ATOM_KINDS, Compiled, compile_spec
from kullback.spec.ground import coverage, valid_because
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
- say needs text (the Intent's own words for the fact).
- ask needs field (a field of the request the agent must ask the user about) or confirm_tool.
- no_write needs no fields: the end state is the starting state.
- cap needs count: the number of entities listed above.
- shape needs tool, field, id_field: the write must carry a value the run read for that column.
- because must quote the Intent span word for word (at least 12 characters) or name the policy \
rule or policy section. A demand with no valid because is dropped.
"""

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
    """Everything one writer session may see; none of it comes from a Run."""
    intent: SpecIntent
    policy_text: str
    sections: list[str]
    tools: list[dict]
    write_tools: set[str]
    state: dict
    constraints: list[dict]
    canon: Callable[[Any], Any]


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
    return WriterInputs(intent=intent, policy_text=policy, sections=writer_tools.policy_sections_of(workdir),
                        tools=tools, write_tools={row["name"] for row in tools if row["kind"] == "write"},
                        state=StateView(env.db, overlay, rows).shared, constraints=_constraints(workdir),
                        canon=canon_fn(env.canon_rules))


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
    because = str(item.get("because") or "")
    if not valid_because(because, inputs.intent.text, inputs.sections):
        return "ungrounded"
    facts = {fact.id: fact for fact in inputs.intent.facts}
    witnessed = intent_text([fact for fact in inputs.intent.facts if not unwitnessed(fact)])
    if fact_ids and all(unwitnessed(facts[f]) for f in fact_ids):
        return "unwitnessed"
    return "" if valid_because(because, witnessed, inputs.sections) else "unwitnessed"


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


def build_spec(task_id: str, demands: Iterable[Any], inputs: WriterInputs) -> tuple[Spec, dict]:
    """The Spec code keeps from the writer's demands, and how many of each refusal it made."""
    checks, counts = [], {"demands": 0, "kept": 0, "ungrounded": 0, "unwitnessed": 0, "bad_kind": 0}
    for number, item in enumerate(demands):
        counts["demands"] += 1
        check, refusal = _check_of(number, item, inputs)
        if check is None:
            counts[refusal] += 1
            continue
        counts["kept"] += 1
        checks.append(check)
    spec = with_gaps(Spec(task_id=task_id, intent=inputs.intent, checks=checks), inputs)
    counts["gaps"] = len(spec.gaps)
    return spec, counts


def with_gaps(spec: Spec, inputs: WriterInputs) -> Spec:
    """The Spec with its gaps: facts no check covers, then one line per check compile names as dropped.

    A compile reason (a write with no entity, say) is a check the Verifier silently lacks, so it is shown
    where the Examiner and the trust report already look.
    """
    spec = spec.model_copy(update={"gaps": []})
    reasons = compile_spec(spec, inputs.write_tools, inputs.canon, inputs.sections).reasons
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
    spec, counts = build_spec(task_id, demands, inputs)
    counts.update(parse_error=error, truncated_chars=cut)
    return Written(spec, counts, session)


def verifier_of(spec: Spec, inputs: WriterInputs) -> Compiled:
    """The Spec's checks compiled, with the must-not atoms `must_not` adds from them."""
    compiled = compile_spec(spec, inputs.write_tools, inputs.canon, inputs.sections)
    must = [atom for atom in compiled.verifier.atoms if atom.kind in ("required", "allowed")]
    forbid = must_not(must, spec.intent.text, inputs.constraints, sorted(inputs.write_tools))
    atoms = compiled.verifier.atoms + forbid
    return compiled._replace(verifier=compiled.verifier.model_copy(update={"atoms": atoms}))


def spec_verifier_path(workdir: Path, task_id: str) -> Path:
    return Path(workdir) / "spec" / "verifiers" / f"{task_id}.json"


def runner_verifier_path(workdir: Path, task_id: str) -> Path:
    """Where the Runner reads a Task's Verifier from (kullback score, BuiltEnvironment.verifier)."""
    return Path(workdir) / "verifiers" / f"{task_id}.json"


def write_verifier(spec: Spec, workdir: Path, inputs: Optional[WriterInputs] = None,
                   verifier_from: str = "runs") -> list[Path]:
    """Saves the Spec and its compiled Verifier; under `intent` the Runner's Verifier is replaced too."""
    inputs = inputs or load_inputs(workdir, spec.task_id, spec.intent)
    verifier: Verifier = verifier_of(spec, inputs).verifier
    text = json.dumps(verifier.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    paths = [save_spec(workdir, spec), spec_verifier_path(workdir, spec.task_id)]
    if verifier_from == "intent":
        paths.append(runner_verifier_path(workdir, spec.task_id))
    for path in paths[1:]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
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


def _runner_reads_spec(workdir: Path, task_id: str) -> bool:
    """Whether the Runner's Verifier for the Task was compiled from its Spec (compile_spec's version)."""
    path = runner_verifier_path(workdir, task_id)
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("verifier_version", "")).startswith("spec")
    except (OSError, ValueError, AttributeError):
        return False


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
    rewrites both Verifier files (the Runner's only when it was compiled from the Spec, so a
    workdir scoring on Runs keeps its Verifier) and publishes spec.repaired when any check changed,
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
    write_verifier(written.spec, workdir, inputs, "intent" if _runner_reads_spec(workdir, task_id) else "runs")
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

