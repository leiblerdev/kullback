"""Intent-first Verifier experiment (fx-0927/intentv).

Derives what must be true from the Intent and the policy, then judges Runs
against it, never the other way round. Runs are measurement only: the writer
model never sees a Run, the Reference, or the current Verifier.

Subcommands:
  sample  pick 40 tasks by the grader sidecar, fixed seed, both corpora
  write   one writer call per task with Intent, policy, tool list and
          read-only Starting state tools; stores demands plus thinking
  score   compile demands to Verifiers with the existing atom builders and
          score every recording with the existing scorer, both Verifiers

Only scripts/experiments and .work-scratch are touched. Original workdirs
are never written. No customer strings reach stdout; raw demands stay under
.work-scratch and are never committed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from kullback.spec import writer_tools

HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[2]
SCRATCH = WORKTREE / ".work-scratch"
MAIN_ENV = Path("/Users/krishuagarwal/Desktop/Programming/leibler/kullback/.env")

SEED = 7
MODEL_ID = "bedrock/global.anthropic.claude-opus-5-5"
PRICE_MODEL = "bedrock/anthropic.claude-opus-5-5"
MAX_TOOL_ROUNDS = 4
# corpus -> (wrong count, right count)
STRATA = {"retail": (16, 10), "airline": (9, 5)}

ATOM_KINDS = ("required", "allowed", "forbidden", "question", "communicate", "hard")


def _load_main_env() -> None:
    """Keys from the main .env into the environment, never overriding, never printed."""
    if not MAIN_ENV.is_file():
        return
    for line in MAIN_ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


# --- sidecar split ----------------------------------------------------------


def sidecar_rewards(clone: Path) -> dict:
    """trace_id -> reward off the grader sidecar, measurement only."""
    out = {}
    for path in (clone / "grader").glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out[data["trace_id"]] = data["fields"]["reward_info"]["reward"]
        except (KeyError, ValueError, TypeError):
            continue
    return out


def reference_traces(clone: Path, task_id: str) -> list[str]:
    """Trace ids behind this Task's Reference, off exam/references.json."""
    refs = _read_json(clone / "exam" / "references.json")
    entry = refs.get(task_id, {})
    return [r["trace_id"] for r in entry.get("references", []) if r.get("trace_id")]


def task_class(traces: list[str], rewards: dict) -> str:
    """right when every Reference trace scores 1, wrong when every one scores 0."""
    scored = [rewards[t] for t in traces if t in rewards]
    if not scored:
        return "unknown"
    if all(r == 1 for r in scored):
        return "right"
    if all(r == 0 for r in scored):
        return "wrong"
    return "mixed"


def confirmed_tasks(clone: Path) -> set[str]:
    status = _read_json(clone / "exam" / "task_status.json")
    return {t for t, row in status.items() if row.get("reference_confirmed")}


def sample_tasks(seed: int = SEED, exclude: frozenset = frozenset(),
                 allow: frozenset | None = None) -> dict:
    """40 confirmed tasks: 25 on a wrong Reference, 15 on a right one, both corpora."""
    rng = random.Random(seed)
    picked: list[dict] = []
    for corpus, (n_wrong, n_right) in STRATA.items():
        clone = SCRATCH / corpus
        rewards = sidecar_rewards(clone)
        confirmed = confirmed_tasks(clone)
        refs = _read_json(clone / "exam" / "references.json")
        pools: dict[str, list[str]] = {"wrong": [], "right": []}
        for task_id in refs:
            if task_id not in confirmed or task_id in exclude:
                continue
            if allow is not None and (corpus, task_id) not in allow:
                continue
            cls = task_class(reference_traces(clone, task_id), rewards)
            if cls in pools:
                pools[cls].append(task_id)
        for cls, want in (("wrong", n_wrong), ("right", n_right)):
            pool = sorted(pools[cls])
            take = min(want, len(pool))
            for task_id in rng.sample(pool, take):
                picked.append({"corpus": corpus, "task_id": task_id, "ref_class": cls,
                               "short": len(pool) < want})
    picked.sort(key=lambda r: (r["corpus"], r["task_id"]))
    return {"seed": seed, "tasks": picked}


# --- writer inputs (harness side; the writer never sees a Run) --------------


def intent_text(clone: Path, task_id: str) -> str:
    """The request in the customer's own words, off exam/spoken, else the user turns."""
    spoken = clone / "exam" / "spoken" / f"{task_id}.txt"
    if spoken.is_file():
        return spoken.read_text(encoding="utf-8").strip()
    runs = sorted((clone / "runs" / task_id).glob("*.jsonl"))
    if not runs:
        return ""
    turns = []
    for line in runs[0].read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "user_turn":
            turns.append(str(event.get("payload", {}).get("text", "")))
    return "\n\n".join(t for t in turns if t.strip())


def policy_rules(clone: Path, task_id: str) -> list[str]:
    """The policy the writer may name: disclosure conditions plus compiled constraints."""
    rules: list[str] = []
    try:
        check = _read_json(clone / "constraints_check.json")
        rules += [str(c.get("text", "")) for c in check.get("constraints", []) if c.get("text")]
    except (OSError, ValueError):
        pass
    seen = set()
    for trace_id in reference_traces(clone, task_id):
        rules_path = clone / "exam" / "user_rules" / f"{trace_id}.json"
        if not rules_path.is_file():
            continue
        try:
            user_rules = json.loads(rules_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for rule in user_rules.get("disclosure", []):
            name = f"disclosure:{rule.get('field')}:{rule.get('condition')}"
            if name not in seen:
                seen.add(name)
                rules.append(name)
    return rules


def tool_list(clone: Path) -> tuple[list[dict], set[str], set[str]]:
    """Name, kind and arg names per tool; the write set and the read set."""
    sigs = _read_json(clone / "tool_sigs.json")
    tools = [{"name": s["name"], "kind": s.get("kind", "read"),
              "args": sorted(a["name"] for a in s.get("args_fields", []))}
             for s in sigs if s.get("name")]
    tools.sort(key=lambda t: t["name"])
    writes = {t["name"] for t in tools if t["kind"] == "write"}
    reads = {t["name"] for t in tools if t["kind"] != "write"}
    return tools, writes, reads

def start_state(clone: Path, task_id: str) -> dict:
    """The Environment's Starting state for this Task, off the Reference file's stop event."""
    refs = _read_json(clone / "exam" / "references.json")
    entry = refs.get(task_id, {})
    run_ids = [r["run_id"] for r in entry.get("references", [])]
    search = [clone / "runs" / task_id / f"{r}.jsonl" for r in run_ids]
    search += [clone / "exam" / "runs" / task_id / f"{r}.jsonl" for r in run_ids]
    for path in search:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            state = (event.get("payload") or {}).get("start_state")
            if event.get("type") == "stop" and isinstance(state, dict):
                return state
    raise SystemExit(f"no Starting state for {task_id}")

_POLICY_CACHE: dict[str, str] = {}


def corpus_policy(clone: Path) -> str:
    """The recorded agent's own policy text, off the raw traces, constant per corpus."""
    key = str(clone)
    if key not in _POLICY_CACHE:
        text = ""
        raws = sorted((clone / "raw").glob("*.json"))
        if raws:
            try:
                sims = json.loads(raws[0].read_text(encoding="utf-8")).get("simulations", [])
                text = str((sims[0] or {}).get("policy") or "")
            except (ValueError, IndexError):
                text = ""
        _POLICY_CACHE[key] = text
    return _POLICY_CACHE[key]


def _words_of(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", _norm(text)) if len(w) >= 4}


def _result_keys(run, fn) -> set[str]:
    from kullback.runner.target import _key as canon_key

    keys: set[str] = set()

    def gather(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                gather(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                gather(value)
        elif node is not None:
            try:
                keys.add(canon_key(fn, node))
            except Exception:
                pass

    for event in run.events:
        if event.type == "tool_result":
            gather((event.payload or {}).get("result"))
    return keys


def diagnose_run(run, demands: list, verifier, intent: str, writes: set[str], fn) -> list[dict]:
    """Every failed atom of one recording, grouped by kind and reason. No values kept."""
    from kullback.runner import target as _target

    effects = _target.write_effects(run, writes, fn)
    asked = set(_target.question_keys(run, effects, fn))
    said = set(_target.communicate_values(run, fn))
    present = {(e["tool"], e["entity"]) for e in effects.values()}
    written = {(e["tool"], e["entity"], f): v
               for e in effects.values() for f, v in e["values"].items()}
    result_keys = _result_keys(run, fn)
    intent_words = _words_of(intent)
    out = []
    for atom in verifier.atoms:
        holds = _target.atom_holds(atom, run, fn, writes,
                                   effects=effects, asked=asked, said=said)
        if holds is not False:
            continue
        payload = _target.atom_payload(atom)
        kind, tool, field = payload.get("kind"), payload.get("tool"), payload.get("field")
        number = str(atom.id).lstrip("i").split(".")[0]
        demand = demands[int(number)] if number.isdigit() and int(number) < len(demands) else {}
        reason = "other"
        if kind == "write":
            reason = "write_not_made" if (tool, payload.get("entity")) not in present else "other"
        elif kind == "write_value":
            key = (tool, payload.get("entity"), field)
            if key not in written and (tool, payload.get("entity")) not in present:
                reason = "write_not_made"
            elif payload.get("value") in result_keys:
                reason = "value_differs_read"
            else:
                reason = "value_differs_unread"
        elif kind == "communicate":
            text = str(demand.get("text") if isinstance(demand, dict) else payload.get("text"))
            reason = ("fact_not_said_named" if _words_of(text) & intent_words
                      else "fact_not_said_unnamed")
        elif kind == "question":
            reason = "question_not_asked"
        elif kind == "entity_count":
            reason = "cap_exceeded"
        elif kind == "hard":
            reason = "hard_rule"
        out.append({"atom": atom.id, "kind": atom.kind, "target": kind, "tool": tool,
                    "field": field, "reason": reason})
    return out

READ_TOOLS = writer_tools.READ_TOOLS
_read_impl = writer_tools.read_tools


SYSTEM_PROMPT_R0 = """You write the end-state demands for one customer request. You have never seen \
any agent run, reference solution, or existing check for this task, and you must not ask for one: \
derive everything from the Intent, the policy, the tool list, and the Starting state reads below.

Rules:
- Demand only what the Intent asks for or the policy requires. Never demand a step just because \
it would be a sensible way to do the job.
- Every exact value (ids, names, amounts, codes) must come from a Starting state read in this \
session, never from memory or guesswork. If a value is not in the Starting state, demand the \
shape (which tool and field) rather than a literal, or leave it out.
- Several acceptable end states may be listed: mark alternatives with kind "allowed".
- Output one JSON object only, no prose: {"demands": [{"id": "d0", "kind": "required", \
"demand": "write" | "say" | "ask" | "no_write" | "cap" | "shape", ...fields..., \
"because": "quote of the Intent span or the policy rule name"}]}.
- write needs tool, id_field, entity, values (an object of field to value read from the state).
- say needs text (a fact the final answer must state, read from the state or the Intent).
- ask needs field (a field of the request the agent must ask the user about) or confirm_tool.
- no_write needs no fields: the end state is the starting state.
- cap needs count: how many entities may be written at most.
- shape needs tool, field, id_field: the write must carry a value the run read for that column.
- because must quote the Intent span word for word (at least 12 characters) or name the policy \
rule. A demand with no valid because is dropped.
"""


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
- Output one JSON object only, no prose: {"demands": [{"id": "d0", "kind": "required", \
"demand": "write" | "say" | "ask" | "no_write" | "cap" | "shape", ...fields..., \
"because": "quote of the Intent span or the policy rule name"}]}.
- write needs tool, id_field, entity, values (an object of field to value read from the state).
- say needs text (the Intent's own words for the fact).
- ask needs field (a field of the request the agent must ask the user about) or confirm_tool.
- no_write needs no fields: the end state is the starting state.
- cap needs count: the number of entities listed above.
- shape needs tool, field, id_field: the write must carry a value the run read for that column.
- because must quote the Intent span word for word (at least 12 characters) or name the policy \
rule or policy section. A demand with no valid because is dropped.
"""


def policy_sections(clone: Path) -> list[str]:
    """Section headers of the recorded agent policy, for because naming."""
    out = []
    for line in corpus_policy(clone).splitlines():
        clean = line.strip().strip("#").replace("**", "").strip()
        if line.startswith("#") and len(clean.split()) >= 2:
            out.append(f"section:{_norm(clean)}")
    return out


def because_rules(clone: Path, task_id: str) -> list[str]:
    """Every rule a because may name: disclosure conditions plus policy sections."""
    return policy_rules(clone, task_id) + policy_sections(clone)


def writer_messages(intent: str, policy: list[str], tools: list[dict],
                    policy_text: str = "", prompt: str = "r1") -> list[dict]:
    """The writer prompt: r0 is the round-0 text and body exactly, r1 adds the policy text."""
    if prompt == "r0":
        body = {"intent": intent,
                "policy": policy or ["no compiled policy rules; the Intent alone governs"],
                "tools": tools}
        return [{"role": "user", "content": SYSTEM_PROMPT_R0 + "\n" + json.dumps(body)[:12000]}]
    body = {"intent": intent,
            "tools": tools,
            "policy": policy or ["no compiled disclosure rules; the Intent alone governs"],
            "policy_text": policy_text or "no recorded agent policy found"}
    return [{"role": "user", "content": SYSTEM_PROMPT + "\n" + json.dumps(body)[:24000]}]

def _price(usage: dict) -> float:
    from kullback.runner import budget

    rates = budget.PRICES.get(PRICE_MODEL, {})
    total = 0.0
    for key in ("input", "output", "cache_read", "cache_write"):
        total += usage.get(key, 0) / 1_000_000 * rates.get(key, 0.0)
    return total


def run_writer(model: object, messages: list[dict], state: dict, config: object,
               max_rounds: int = MAX_TOOL_ROUNDS) -> dict:
    """One bounded writer session: read-only state tools, then a final JSON object."""
    from kullback.ai.provider import ModelConfig

    return writer_tools.run_session(model, messages, state, config or ModelConfig(), max_rounds, _price)


def parse_demands(text: str) -> tuple[list[dict], str]:
    """The demand list off a final answer, or the reason it holds none."""
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except (ValueError, IndexError):
        return [], "no JSON object"
    demands = data.get("demands") if isinstance(data, dict) else None
    if not isinstance(demands, list):
        return [], "no demands list"
    return demands, ""


# --- compile demands to a Verifier with the existing builders ----------------


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def valid_because(demand: dict, intent: str, policy: list[str]) -> bool:
    """A because that quotes an Intent span or names a policy rule or section."""
    because = _norm(demand.get("because"))
    if not because:
        return False
    flat = _norm(intent)
    for span in re.split(r"[.!?\n]+", because):
        span = span.strip("\"' ").strip()
        if len(span) >= 12 and span in flat:
            return True
    words = set(re.findall(r"[a-z0-9_]+", because))
    for rule in policy:
        if rule.startswith("section:"):
            phrase = rule[len("section:"):]
            if phrase and phrase in because:
                return True
            continue
        for token in re.findall(r"[a-z0-9_]+", _norm(rule)):
            if len(token) >= 3 and token in words:
                return True
    return False


def compile_verifier(demands: list[dict], intent: str, policy: list[str],
                     write_tools: set[str], fn) -> tuple[object, int]:
    """Demands to a Verifier, reusing derive.py atom kinds and the gate builders. No new kinds."""
    from kullback import derive
    from kullback.gates import verifier_suite
    from kullback.runner.records import Verifier
    from kullback.runner.target import _key as canon_key

    atoms = []
    invalid = 0
    for number, demand in enumerate(demands):
        if not isinstance(demand, dict) or demand.get("kind") not in ATOM_KINDS:
            invalid += 1
            continue
        if not valid_because(demand, intent, policy):
            invalid += 1
            continue
        want = demand.get("demand")
        atom_id = f"i{number}"
        kind = demand["kind"]
        if want == "write" and demand.get("tool") in write_tools and demand.get("id_field"):
            entity = str(demand.get("entity", ""))
            base = {"tool": demand["tool"], "entity": entity,
                    "entity_raw": entity, "id_field": demand["id_field"], "at": 0}
            atoms.append(verifier_suite.make_atom(
                atom_id, kind, dict(base, kind="write"),
                description=f"{demand['tool']} writes {entity or 'an entity'}"))
            values = demand.get("values") or {}
            if isinstance(values, dict):
                for field in sorted(values):
                    atoms.append(verifier_suite.make_atom(
                        f"{atom_id}.{field}", kind,
                        dict(base, kind="write_value", field=field,
                             value=canon_key(fn, values[field]), raw=values[field]),
                        description=f"{demand['tool']} {field} is {values[field]}"))
        elif want == "say" and demand.get("text"):
            text = str(demand["text"])
            atoms.append(verifier_suite.make_atom(
                atom_id, "communicate",
                {"kind": "communicate", "value": canon_key(fn, text),
                 "text": text, "field": None, "source_tool": None},
                description=f"the final answer states {text}"))
        elif want == "ask" and (demand.get("field") or demand.get("confirm_tool")):
            if demand.get("field"):
                key, tool, field = f"field:{demand['field']}", demand.get("tool"), demand["field"]
            else:
                key = f"confirm:{demand['confirm_tool']}"
                tool, field = demand["confirm_tool"], None
            atoms.append(verifier_suite.make_atom(
                atom_id, "question", {"kind": "question", "key": key, "tool": tool, "field": field},
                description=f"the agent asks the user about {key.split(':', 1)[-1]}"))
        elif want == "no_write":
            atoms.append(derive.no_write_atom(write_tools))
            atoms[-1] = atoms[-1].model_copy(update={"id": atom_id})
        elif want == "cap" and isinstance(demand.get("count"), int):
            atoms.append(verifier_suite.make_atom(
                atom_id, "required", {"kind": "entity_count", "count": demand["count"]},
                description=f"the Run makes at most {demand['count']} write calls"))
        elif (want == "shape" and demand.get("tool") in write_tools
              and demand.get("field") and demand.get("id_field")):
            atom = derive.shape_atom(atom_id, demand["tool"], demand["field"],
                                     demand["id_field"], write_tools)
            atoms.append(atom)
        else:
            invalid += 1
    return Verifier(task_id="", atoms=atoms, verifier_version="intentv1", seed_run_ids=[]), invalid


def steering_leak(clone: Path, task_id: str, demands: list, fn) -> dict:
    """How far the recorded agent steered the Intent source (offline proxy, counts only).

    exam/spoken holds the user turns of the Reference runs, so a value the recorded
    agent elicited or stated and the user repeated lands in the Intent. A result value
    is steered when its token is spoken anywhere but in the opening user turn. A demand
    rests on steering when its because quotes such a span. Counts only, no values kept.
    """
    from kullback.runner.records import load_task_run

    refs = _read_json(clone / "exam" / "references.json").get(task_id, {})
    run_ids = [r["run_id"] for r in refs.get("references", [])]
    paths = [clone / "runs" / task_id / f"{r}.jsonl" for r in run_ids]
    paths += [clone / "exam" / "runs" / task_id / f"{r}.jsonl" for r in run_ids]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        return {"steered_values": 0, "steered_demand_becauses": 0, "user_turns": 0}
    try:
        run = load_task_run(path, task_id)
    except Exception:
        return {"steered_values": 0, "steered_demand_becauses": 0, "user_turns": 0}
    turns = [str(e.payload.get("text", "")) for e in run.events if e.type == "user_turn"]
    if not turns:
        return {"steered_values": 0, "steered_demand_becauses": 0, "user_turns": 0}
    first = _norm(turns[0])
    later = _norm(" ".join(turns[1:]))
    tokens: set[str] = set()

    def gather(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                gather(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                gather(value)
        elif isinstance(node, str):
            for word in re.findall(r"[a-z0-9]+", node.casefold()):
                if len(word) >= 4:
                    tokens.add(word)

    for event in run.events:
        if event.type == "tool_result":
            gather((event.payload or {}).get("result"))
    steered = {t for t in tokens if t in later and t not in first}
    quoted = 0
    for demand in demands if isinstance(demands, list) else []:
        if not isinstance(demand, dict):
            continue
        for span in re.split(r"[.!?\n]+", _norm(demand.get("because"))):
            span = span.strip("\"' ").strip()
            if len(span) >= 12 and span in later and span not in first:
                quoted += 1
                break
    _ = fn
    return {"steered_values": len(steered), "steered_demand_becauses": quoted,
            "user_turns": len(turns)}

# --- scoring -----------------------------------------------------------------


def recording_paths(clone: Path, task_id: str) -> dict[str, Path]:
    """Every recording of the Task on disk, by run id."""
    out = {}
    for base in (clone / "runs" / task_id, clone / "exam" / "runs" / task_id):
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.jsonl")):
            out.setdefault(path.stem, path)
    return out


def current_verifier(clone: Path, task_id: str) -> object:
    from kullback.runner.records import Verifier

    for base in (clone / "exam" / "derived", clone / "exam" / "verifiers"):
        path = base / f"{task_id}.json"
        if path.is_file():
            return Verifier.model_validate(json.loads(path.read_text(encoding="utf-8")))
    raise SystemExit(f"no current Verifier for {task_id}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--seed", type=int, default=SEED)
    p_sample.add_argument("--out", type=Path, required=True)
    p_sample.add_argument("--exclude", type=Path, default=None)
    p_sample.add_argument("--allow", type=Path, default=None)
    p_write = sub.add_parser("write")
    p_write.add_argument("--sample", type=Path, required=True)
    p_write.add_argument("--model", default=MODEL_ID)
    p_write.add_argument("--ceiling-usd", type=float, default=12.0)
    p_write.add_argument("--out-dir", type=Path, required=True)
    p_write.add_argument("--only", nargs="*", default=None)
    p_write.add_argument("--prompt", choices=("r0", "r1"), default="r1")
    p_score = sub.add_parser("score")
    p_score.add_argument("--sample", type=Path, required=True)
    p_score.add_argument("--demands-dir", type=Path, required=True)
    p_score.add_argument("--out", type=Path, required=True)
    p_diag = sub.add_parser("diagnose")
    p_diag.add_argument("--sample", type=Path, required=True)
    p_diag.add_argument("--demands-dir", type=Path, required=True)
    p_diag.add_argument("--score", type=Path, required=True)
    p_diag.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "sample":
        excluded = frozenset(r["task_id"] for r in
                             json.loads(args.exclude.read_text(encoding="utf-8"))["tasks"]
                             ) if args.exclude else frozenset()
        allowed = (frozenset(tuple(entry.split("/", 1)) for entry in
                             json.loads(args.allow.read_text(encoding="utf-8"))
                             ) if args.allow else None)
        body = sample_tasks(args.seed, excluded, allowed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        counts: dict[str, int] = {}
        for row in body["tasks"]:
            key = f"{row['corpus']}/{row['ref_class']}"
            counts[key] = counts.get(key, 0) + 1
        print(json.dumps({"tasks": len(body["tasks"]), "strata": counts}))
        return 0

    if args.cmd == "write":
        _load_main_env()
        from kullback.ai.provider import ModelConfig, live_model

        if os.environ.get("HARNESS_ALLOW_MODEL_REQUESTS", "") not in ("1", "true", "yes", "on"):
            raise SystemExit("live model requests are off")
        sample = json.loads(args.sample.read_text(encoding="utf-8"))["tasks"]
        if args.only:
            sample = [r for r in sample if r["task_id"] in args.only]
        model = live_model(args.model)
        prompt = args.prompt
        config = ModelConfig(thinking={"type": "adaptive", "display": "summarized"},
                             max_tokens=3000 if prompt == "r0" else 2000)
        rounds = 6 if prompt == "r0" else MAX_TOOL_ROUNDS
        args.out_dir.mkdir(parents=True, exist_ok=True)
        spent = 0.0
        for row in sample:
            if spent >= args.ceiling_usd:
                print(json.dumps({"task": row["task_id"], "status": "skipped_on_ceiling"}),
                      flush=True)
                continue
            clone = SCRATCH / row["corpus"]
            intent = intent_text(clone, row["task_id"])
            policy = policy_rules(clone, row["task_id"]) if prompt == "r0" else because_rules(
                clone, row["task_id"])
            tools, _, _ = tool_list(clone)
            state = start_state(clone, row["task_id"])
            result = run_writer(
                model, writer_messages(intent, policy, tools,
                                       corpus_policy(clone) if prompt == "r1" else "",
                                       prompt), state, config, rounds)
            spent += result["usd"]
            demands, error = parse_demands(result["text"])
            out = {"corpus": row["corpus"], "task_id": row["task_id"],
                   "ref_class": row["ref_class"], "prompt": prompt, "demands": demands,
                   "parse_error": error, "usd": round(result["usd"], 4),
                   "usage": result["usage"], "tool_rounds": result["tool_rounds"],
                   "tool_uses": result.get("tool_uses", []),
                   "thinking": result["thinking"]}
            (args.out_dir / f"{row['task_id']}.json").write_text(
                json.dumps(out, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"task": row["task_id"], "demands": len(demands),
                              "error": error or None, "usd": round(result["usd"], 4),
                              "spent": round(spent, 4)}), flush=True)
        print(json.dumps({"spent_usd": round(spent, 4)}))
        return 0

    if args.cmd == "score":
        from kullback.runner import canon
        from kullback.runner.target import canon_fn
        from kullback.runner.verdict import verdict

        sample = {r["task_id"]: r for r in
                  json.loads(args.sample.read_text(encoding="utf-8"))["tasks"]}
        rows = []
        for task_id, row in sorted(sample.items()):
            clone = SCRATCH / row["corpus"]
            demand_file = args.demands_dir / f"{task_id}.json"
            if not demand_file.is_file():
                rows.append({"task_id": task_id, **row, "status": "no_demands"})
                continue
            stored = json.loads(demand_file.read_text(encoding="utf-8"))
            rules = canon.load_rules(clone / "canon-rules.json")
            fn = canon_fn(rules)
            _, writes, _ = tool_list(clone)
            intent = intent_text(clone, task_id)
            policy = because_rules(clone, task_id)
            verifier, invalid = compile_verifier(stored["demands"], intent, policy, writes, fn)
            verifier = verifier.model_copy(update={"task_id": task_id})
            current = current_verifier(clone, task_id)
            rewards = sidecar_rewards(clone)
            id_of = {r["run_id"]: r.get("trace_id") for r in
                     _read_json(clone / "exam" / "references.json")
                     .get(task_id, {}).get("references", [])}
            recs = []
            for run_id, path in sorted(recording_paths(clone, task_id).items()):
                scored = {}
                for name, check in (("intent", verifier), ("current", current)):
                    try:
                        result = verdict(str(path), check, canon=rules, rules=rules,
                                         write_tools=writes)
                        scored[name] = {"pass": bool(result.passed),
                                        "failing": getattr(result, "failing_atom", None)}
                    except Exception as error:
                        scored[name] = {"pass": None, "error": type(error).__name__}
                grade = rewards.get(id_of.get(run_id))
                recs.append({"run_id": run_id,
                             "is_reference": run_id in id_of,
                             "grade": grade, **scored})
            rows.append({"task_id": task_id, **row, "status": "scored",
                         "invalid_because": invalid, "atoms": len(verifier.atoms),
                         "kinds": sorted(a.kind for a in verifier.atoms),
                         "leak": steering_leak(clone, task_id, stored["demands"], fn),
                         "recordings": recs})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"tasks": len(rows),
                          "scored": sum(1 for r in rows if r["status"] == "scored")}))
        return 0
    if args.cmd == "diagnose":
        from kullback.runner import canon
        from kullback.runner.records import load_task_run
        from kullback.runner.target import canon_fn

        sample = {r["task_id"]: r for r in
                  json.loads(args.sample.read_text(encoding="utf-8"))["tasks"]}
        scored = {r["task_id"]: r for r in
                  json.loads(args.score.read_text(encoding="utf-8"))["rows"]}
        rows = []
        for task_id, row in sorted(sample.items()):
            clone = SCRATCH / row["corpus"]
            stored = json.loads((args.demands_dir / f"{task_id}.json").read_text(
                encoding="utf-8"))
            rules = canon.load_rules(clone / "canon-rules.json")
            fn = canon_fn(rules)
            _, writes, _ = tool_list(clone)
            intent = intent_text(clone, task_id)
            policy = because_rules(clone, task_id)
            verifier, _ = compile_verifier(stored["demands"], intent, policy, writes, fn)
            verifier = verifier.model_copy(update={"task_id": task_id})
            for rec in scored.get(task_id, {}).get("recordings", []):
                if not rec["is_reference"] or rec["grade"] != 1 or rec["intent"]["pass"]:
                    continue
                path = recording_paths(clone, task_id)[rec["run_id"]]
                run = load_task_run(path, task_id)
                rows.append({"task_id": task_id, "corpus": row["corpus"],
                             "run_id": rec["run_id"],
                             "verdict_failing": rec["intent"]["failing"],
                             "failures": diagnose_run(run, stored["demands"], verifier,
                                                      intent, writes, fn)})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
        groups: dict[str, int] = {}
        for row in rows:
            for failure in row["failures"]:
                key = f"{failure['target']}/{failure['reason']}"
                groups[key] = groups.get(key, 0) + 1
        print(json.dumps({"recordings": len(rows), "groups": groups}))
        return 0
    raise SystemExit("unknown command")


if __name__ == "__main__":
    sys.exit(main())
