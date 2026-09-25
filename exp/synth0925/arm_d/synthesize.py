"""Step 4: per cluster, scenarios and blueprints on other rows, all levels.

A blueprint is a walk that visits the cluster's write call, bound with the
coherent binder, run in a fresh world. Birth checks are code: runs,
do-nothing fails, every strict prefix fails, D79 suite. Up to 3 revisions
with the failure reasons fed back; then kept as rejected. Nothing is deleted.
"""
import json
from pathlib import Path

from kullback import graph as graph_mod
from kullback import synthesise as S
from kullback.examiner import derive as derive_mod
from kullback.examiner import reference as reference_mod
from kullback.runner import budget as budget_mod
from kullback.runner.records import Task, as_dict

SYNTH_ID = "opencode-go/muse-spark-1.3-contributor"

LEVEL_BAND = {"easy": "w1t2p1", "medium": "w2t3p1", "hard": "w3t3p1"}
LEVEL_UNITS = {"easy": 1, "medium": 2, "hard": 3}

SCENARIO_PROMPT = """A support agent can fix customer problems by performing operations in
their account. One ability needs practice: {skill}.
Evidence from past failures: {evidence}
Write a short customer situation that needs exactly {units} fix(es), as JSON:
{{"situation": "who the customer is and what happened, no names or ids",
"goal": "what fixed looks like, in plain words",
"tell": ["facts the customer must be told at the end, as field names"]}}.
No tool, system or table names. No ids, no values. Answer JSON only."""

REQUEST_PROMPT = """A person is writing to a support agent about a problem they want solved.
Situation: {situation}
Goal: {goal}
Write their first message in one or two sentences, in their own words. Do not
name any tool, any system, any table or any step, and do not say how the agent
should do it. Do not include any value that is not in the list below.
The only values this person knows and may state:
{facts}
Write the message and nothing else."""

REVISE_NOTE = """The last draft failed ({reasons}). Write a new message that avoids that."""


def make_synth(workdir: Path):
    from exp.synth0925.arm_d.judge import ENV_PATH
    from kullback.ai import provider
    provider.load_dotenv(ENV_PATH)
    provider.enable_live_calls_from_env()
    return provider.live_model(SYNTH_ID)


def cost_of(reply, model_id: str = SYNTH_ID) -> float:
    return budget_mod.call_cost(reply.usage, model_id)


def askable(workdir) -> set:
    """Fields some recorded user stated: the artifact first, else the stored rules."""
    from kullback.runner.records import read_json
    fields = S.askable_fields(workdir)
    if fields:
        return fields
    out = set()
    rules = Path(workdir) / "user_rules"
    if rules.is_dir():
        for path in sorted(rules.glob("*.json")):
            try:
                body = read_json(path, {}) or {}
            except (OSError, ValueError):
                continue
            for row in body.get("facts") or []:
                if row.get("field"):
                    out.add(str(row["field"]))
    return out


def cluster_write(cluster: dict, evidence: list[dict]) -> str:
    """The write call the cluster's failures circled, read off member references."""
    ranked = cluster_writes(cluster, evidence)
    return ranked[0] if ranked else ""


def cluster_writes(cluster: dict, evidence: list[dict], top: int = 3) -> list:
    """The write calls the cluster's failures circled, most common first."""
    from collections import Counter
    members = set(cluster.get("member_task_ids") or ())
    counts: Counter = Counter()
    for item in evidence:
        if item.get("task_id") in members:
            for write in (item.get("reference") or {}).get("writes") or []:
                counts[write.get("call")] += 1
    if not counts:
        for item in evidence:
            if item.get("task_id") in members:
                for write in (item.get("run") or {}).get("writes") or []:
                    counts[write.get("call")] += 1
    return [call for call, _ in counts.most_common(top)]


def member_rows(cluster: dict, evidence: list[dict]) -> set[str]:
    rows = set()
    members = set(cluster.get("member_task_ids") or ())
    for item in evidence:
        if item.get("task_id") in members:
            for source in ("reference", "run"):
                for write in (item.get(source) or {}).get("writes") or []:
                    if write.get("row"):
                        rows.add(str(write["row"]))
    return rows


def scenario_for(model, skill: str, evidence_text: str, level: str) -> tuple[dict, float]:
    prompt = SCENARIO_PROMPT.format(skill=skill, evidence=evidence_text or "none",
                                    units=LEVEL_UNITS[level])
    try:
        reply = model.query([{"role": "user", "content": prompt}])
    except Exception:
        return {}, 0.0
    spent = cost_of(reply)
    try:
        text = reply.content or ""
        body = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return {"situation": str(body.get("situation") or "")[:600],
                "goal": str(body.get("goal") or "")[:400],
                "tell": [str(t) for t in body.get("tell") or ""][:6]}, spent
    except (ValueError, AttributeError):
        return {}, spent


def request_for(model, scenario: dict, said: list[dict], held: list[dict],
                fallback: str, reasons: str = "") -> tuple[str, str, float]:
    lines = "\n".join(f"- {row['field']}: {row['value']}" for row in said) or "- nothing"
    prompt = REQUEST_PROMPT.format(situation=scenario.get("situation", ""),
                                   goal=scenario.get("goal", ""), facts=lines)
    if reasons:
        prompt += "\n" + REVISE_NOTE.format(reasons=reasons)
    try:
        reply = model.query([{"role": "user", "content": prompt}])
    except Exception as error:
        return fallback, f"the request writer failed: {type(error).__name__}", 0.0
    spent = cost_of(reply)
    text = str(getattr(reply, "content", "") or "").strip()
    if not text:
        return fallback, "the request writer answered nothing", spent
    leaked = [row["field"] for row in held if str(row["value"]) and str(row["value"]) in text]
    if leaked:
        return fallback, "the draft stated a value only the world knows", spent
    return text, "", spent


def birth_checks(world, calls: list[dict], request: str, task_id: str,
                 ident: str, fn) -> tuple[bool, dict, dict]:
    """runs, do-nothing fails, strict prefixes fail, suite. Returns verdict, details, costs."""
    transcript, spoken = S.spoken_for(request, calls)
    run = S.record_run(world, task_id, spoken, f"synthd-{task_id}", transcript)
    checks = {"runs": run is not None, "do_nothing_fails": False,
              "prefix_fails": False, "suite_passed": False, "suite_failures": []}
    if run is None:
        return False, {"checks": checks, "stage": "runs"}, {}
    try:
        reference = reference_mod.load(run["path"], reference_mod.REROLL,
                                       run_id=run["run_id"],
                                       write_tools=world.write_tools, fn=fn, atoms=())
    except (OSError, ValueError, TypeError) as error:
        checks["runs"] = f"no reference: {type(error).__name__}"
        return False, {"checks": checks, "stage": "runs"}, {}
    alternates = S.second_paths(world, task_id, spoken, transcript, reference,
                                f"synthd-{task_id}")
    task = Task(id=task_id, run_ids=[run["run_id"]], intent=request, name=None)
    verifier = derive_mod.derive_verifier(task, run["path"],
                                          rerun_paths=[row["path"] for row in alternates],
                                          canon=world.canon_rules,
                                          write_tools=world.write_tools, intent=request)
    from kullback.runner import target as target_mod
    from kullback.runner import verdict as verdict_mod
    empty_run = S.record_run(world, task_id, [], f"synthd-{task_id}-empty",
                             S.spoken_for(request, [])[0])
    if empty_run is not None:
        try:
            passed, _ = target_mod.check_run(
                verifier, verdict_mod.load_run(empty_run["path"]), world.canon_rules)
            checks["do_nothing_fails"] = not passed
        except (OSError, ValueError, TypeError):
            checks["do_nothing_fails"] = False
    write_pos = [pos for pos, call in enumerate(calls)
                 if call["name"] in world.write_tools]
    prefixes_ok = True
    for stop in range(len(write_pos) - 1):
        prefix = calls[:write_pos[stop] + 1]
        prow = S.record_run(world, task_id, prefix, f"synthd-{task_id}-p{stop}",
                            S.spoken_for(request, prefix)[0])
        if prow is None:
            prefixes_ok = False
            break
        try:
            passed, _ = target_mod.check_run(
                verifier, verdict_mod.load_run(prow["path"]), world.canon_rules)
            if passed:
                prefixes_ok = False
                break
        except (OSError, ValueError, TypeError):
            prefixes_ok = False
            break
    checks["prefix_fails"] = prefixes_ok and len(write_pos) > 0
    passed, failures = S._suite(world, verifier, run, alternates, request)
    checks["suite_passed"] = passed
    checks["suite_failures"] = failures
    store = S.store_dir(world.workdir)
    store_tasks = store / "tasks"
    store_tasks.mkdir(parents=True, exist_ok=True)
    (store / "verifiers").mkdir(parents=True, exist_ok=True)
    ok = all(checks[key] is True for key in
             ("runs", "do_nothing_fails", "prefix_fails", "suite_passed"))
    details = {"checks": checks, "writes": len(write_pos),
               "run_id": run["run_id"], "alternates": len(alternates)}
    if ok:
        fields = askable(world.workdir)
        said, held = S.facts_of(calls, fields)
        from kullback.runner.records import as_dict as _as_dict
        S.write_json(store_tasks / f"{task_id}.json",
                     {"format": S.FORMAT, "synthetic": True,
                      "task": _as_dict(task), "seed": ident,
                      "walk": [{"tool": call["name"], "args": call["args"]}
                               for call in calls],
                      "user_rules": _as_dict(S.user_rules_for(said)),
                      "record_facts": [row["field"] for row in held]})
        S.write_json(store / "verifiers" / f"{task_id}.json", as_dict(verifier))
    return ok, details, {}


def build_task(world, body: dict, observed: dict, run_walk, cluster: dict,
               write_calls: list, level: str, scenario: dict, model, ident: str,
               fn, seed: str, index: int) -> dict:
    """One task line: up to 4 tries (birth plus 3 revisions)."""
    task_id = f"synthd_{abs(hash(ident)) % (16 ** 10):010d}"
    line = {"task_id": task_id, "arm": "d", "kind": "do", "level": level,
            "asked_level": level,
            "units": LEVEL_UNITS[level], "scenario_id": cluster.get("skill"),
            "cluster": cluster.get("skill"), "revisions": 0, "cost_usd": 0.0}
    reasons = ""
    for revision in range(4):
        calls, why, attempt = [], "no draw", f"{ident}:rev{revision}"
        for draw in range(8):
            sub = f"{attempt}:d{draw}"
            must = [write_calls[0]] if write_calls and draw < 3 else ()
            steps = graph_mod.walk(body, LEVEL_BAND[level], sub, must_visit=must)
            if not steps:
                why = "no walk visits the cluster write"
                continue
            calls, why = run_walk(world, steps, observed, sub)
            if why or not calls:
                continue
            reached = {call["name"] for call in calls
                       if call["name"] in world.write_tools}
            if write_calls and not (reached & set(write_calls)):
                why = "walk missed the cluster write"
                calls = []
                continue
            attempt = sub
            break
        if why or not calls:
            reasons = f"blueprint {why or 'empty'}"
            line["revisions"] = revision
            continue
        fields = askable(world.workdir)
        said, held = S.facts_of(calls, fields)
        fallback = S.written_intent(calls, world.write_tools)
        request, refusal, spent = request_for(model, scenario, said, held,
                                              fallback, reasons)
        line["cost_usd"] += spent
        if refusal and not request:
            request = fallback
        ok, details, _ = birth_checks(world, calls, request, task_id, attempt, fn)
        line.update({"request": request, "user_facts": said,
                     "blueprint_writes": [{"name": call["name"], "args": call["args"]}
                                          for call in calls
                                          if call["name"] in world.write_tools],
                     "must_tell": [row for row in scenario.get("tell", [])],
                     "checks": details["checks"]})
        line["revisions"] = revision
        if ok:
            line["status"] = "admitted"
            line["reasons"] = refusal or ""
            line["units"] = details["writes"]
            line["level"] = ("easy" if details["writes"] <= 1 else
                             "medium" if details["writes"] == 2 else "hard")
            return line
        reasons = failed_reasons(details["checks"])
    line.setdefault("request", "")
    line.setdefault("user_facts", [])
    line.setdefault("blueprint_writes", [])
    line.setdefault("must_tell", [])
    line.setdefault("checks", {"runs": False, "do_nothing_fails": False,
                               "prefix_fails": False, "suite_passed": False,
                               "suite_failures": []})
    line["status"] = "rejected"
    line["reasons"] = reasons
    return line


def failed_reasons(checks: dict) -> str:
    missed = [key for key in ("runs", "do_nothing_fails", "prefix_fails",
                              "suite_passed") if checks.get(key) is not True]
    extra = "; suite: " + ",".join(checks.get("suite_failures") or []) if \
        checks.get("suite_failures") else ""
    return "failed " + ",".join(missed) + extra
