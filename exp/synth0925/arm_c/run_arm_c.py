"""Arm C orchestrator: search agent, scenarios, blueprints, birth checks.

Usage:
  PYTHONPATH=$PWD uv run python exp/synth0925/arm_c/run_arm_c.py --phase search
  PYTHONPATH=$PWD uv run python exp/synth0925/arm_c/run_arm_c.py --phase pilot
  PYTHONPATH=$PWD uv run python exp/synth0925/arm_c/run_arm_c.py --phase full
  PYTHONPATH=$PWD uv run python exp/synth0925/arm_c/run_arm_c.py --phase report

Phases are resumable: existing scenarios.jsonl and tasks.jsonl rows are kept.
Keyed seed synth0925-c drives every random draw. Ceiling 20 USD total.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from exp.synth0925.arm_c import birth as B
from exp.synth0925.arm_c import coherent as C
from exp.synth0925.arm_c import webtools as W
from exp.synth0925.arm_c.cost import JUDGE_ID, SYNTH_ID, Priced, Share

R = Path("/Users/krishuagarwal/Desktop/Programming/leibler/kullback/.claude/herdr/research-0925/rs-synth")
OUT = R / "exp" / "arm-c"
WORK = OUT / "retail"
SEED = "synth0925-c"
CEIL = 20.0
ARM = "c"

WANT_SCENARIOS = 40
WANT_TASKS = 60
LEVEL_QUOTA = {"easy": 20, "medium": 20, "hard": 20}
MAX_READ_ITERS = 6
MAX_REVISIONS = 3

ATTEMPTS_PATH = None
ID_SALT = ""


def note_attempt(obj: dict) -> None:
    """One attempt record: counts and names only, never argument values."""
    if ATTEMPTS_PATH is None:
        return
    try:
        with open(ATTEMPTS_PATH, "a") as fh:
            fh.write(json.dumps(obj) + "\n")
    except OSError:
        pass

def log(*parts) -> None:
    print(*parts, flush=True)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def capability_summary() -> dict:
    sigs = read_json(WORK / "tool_sigs.json", [])
    schema = read_json(WORK / "schema.json", {})
    db = read_json(WORK / "db.json", {})
    tools = [{"name": s.get("name"), "kind": s.get("kind"),
              "args": sorted((s.get("args_schema") or {}).get("properties", {}).keys()),
              "required": sorted((s.get("args_schema") or {}).get("required", []) or [])}
             for s in sigs]
    tables = {}
    for name, rows in (db.items() if isinstance(db, dict) else []):
        tables[str(name)] = len(rows) if hasattr(rows, "__len__") else 0
    return {"tools": tools, "tables": tables,
            "columns": [str(c.get("table")) + "." + str(c.get("name"))
                        for c in (schema.get("columns") or [])][:40]}


def clean(text: str) -> str:
    return B.scrub(str(text or "")).strip()


def extract_json(text: str):
    import re as _re
    decoder = json.JSONDecoder()
    text = str(text or "")
    text = _re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    best = None
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except Exception:
            continue
        if isinstance(obj, (dict, list)):
            if best is None or len(str(obj)) > len(str(best)):
                best = obj
    if best is not None:
        return best
    return None


def query_model(model: Priced, prompt: str, share: Share) -> tuple:
    reply = model.query([{"role": "user", "content": prompt}], share=share)
    return clean(getattr(reply, "content", "") or ""), model.cost_of(reply)


# --- search agent ------------------------------------------------------------

QUERY_PROMPT = """You are learning what customers of a service ask for. Below is what the service can do: its tools with argument names, and its tables with row counts. No customer data is shown.

{capability}

Propose {n} web search queries that find real situations customers of this kind of service bring: forum threads, help pages, complaint threads. Queries must be generic to the kind of service implied by the tools, and must not name any tool, table, or system. One query per line, nothing else."""


def phase_search(share: Share, synth: Priced) -> dict:
    key = W.tinyfish_key()
    if not key:
        raise RuntimeError("no search key found")
    cap = capability_summary()
    cap_text = json.dumps(cap, indent=1)[:3000]
    text, _ = query_model(synth, QUERY_PROMPT.format(capability=cap_text, n=12), share)
    queries = [clean(line).lstrip("0123456789.-) ").strip() for line in text.splitlines()]
    queries = [q for q in queries if len(q) > 8][:12]
    if len(queries) < 6:
        raise RuntimeError("too few search queries proposed")
    log(f"queries proposed: {len(queries)}")
    searches, seen_urls = [], {}
    for q in queries:
        share.refuse_if_broke()
        try:
            rows = W.tinyfish_search(q, key)
        except Exception as error:
            log(f"search failed: {type(error).__name__}")
            continue
        share.searches += 1
        searches.append({"query": q, "hits": len(rows)})
        for row in rows:
            if row["url"] not in seen_urls:
                seen_urls[row["url"]] = row
    log(f"searches made: {share.searches}, urls seen: {len(seen_urls)}")
    urls = [u for u in seen_urls if not W.guard_ok(u)][:24]
    log(f"urls passing guard: {len(urls)} of {len(seen_urls)}")
    pages, errors = W.tinyfish_fetch(urls, key)
    share.fetches += len(urls)
    log(f"fetched pages: {len(pages)}, fetch errors: {len(errors)}")
    tried_fc, got_fc = 0, 0
    for row in errors[:4]:
        tried_fc += 1
        page = W.firecrawl_fetch(row["url"])
        if not page.get("error"):
            pages.append(page)
            got_fc += 1
    log(f"keyless scrape attempts: {tried_fc}, pages gained: {got_fc}")
    ev = {"searches": searches, "pages": pages, "errors": errors[:20],
          "keyless_attempts": tried_fc, "keyless_pages": got_fc}
    (OUT / "evidence.json").write_text(json.dumps(ev, indent=1))
    return ev


# --- scenarios ---------------------------------------------------------------

SCEN_PROMPT = """You write scenario records for a service whose tools and tables are listed below (counts only, no customer data). You also have short excerpts of real customer situations from the public web.

{capability}

Evidence excerpts:
{evidence}

Write {n} varied scenario records as one JSON array. Each record:
- persona: one line, who this customer is (age band, habits, no names)
- situation: two lines, what happened in their own terms
- goal: one line, what they want
- facts: list of field value pairs the customer holds and can state (use plain field names like email, name, zip, product, reason; values must be invented placeholders, never copied from the evidence)
- sources: list of the evidence urls that inspired it (from the list below, at most 2)
- hard: one line, what makes this tricky
- level_hint: easy, medium or hard (easy is one action, medium two, hard three or mixed asks)
- wrong_fact: true when the customer holds a wrong fact that must be corrected, else false

Allowed evidence urls:
{urls}

Rules: no tool, table or system names anywhere. Plain words only. Return only the JSON array, no prose before or after it. Keep each string short."""


def source_kind(url: str) -> str:
    host = url.lower()
    if any(w in host for w in ("forum", "community", "reddit", "thread", "discussion")):
        return "forum"
    if any(w in host for w in ("support", "help", "faq", "how-to", "guide", "docs")):
        return "help_page"
    if any(w in host for w in ("complaint", "review", "trustpilot", "bbb", "pissed")):
        return "complaint"
    if "news" in host:
        return "news"
    return "other"


def phase_scenarios(share: Share, synth: Priced) -> list:
    have = []
    path = OUT / "scenarios.jsonl"
    if path.exists():
        have = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    log(f"scenarios kept: {len(have)}")
    if len(have) >= WANT_SCENARIOS:
        return have
    ev = json.loads((OUT / "evidence.json").read_text())
    pages = ev.get("pages") or []
    if not pages:
        raise RuntimeError("no fetched pages to base scenarios on")
    cap_text = json.dumps(capability_summary()["tools"])[:1500]
    need = WANT_SCENARIOS - len(have)
    per_call = 5
    sid = len(have)
    while need > 0:
        share.refuse_if_broke()
        batch = pages[:5]
        pages = pages[5:] or (ev.get("pages") or [])[:5]
        ev_text = "\n\n".join(f"[{i}] {p.get('title', '')}: {(p.get('text') or '')[:500]}"
                              for i, p in enumerate(batch))
        urls = "\n".join(f"[{i}] {p.get('url', '')}" for i, p in enumerate(batch))
        n = min(per_call, need)
        text, _ = query_model(synth, SCEN_PROMPT.format(
            capability=cap_text, evidence=ev_text, n=n, urls=urls), share)
        log(f"batch reply chars: {len(text)}")
        data = extract_json(text)
        if isinstance(data, dict):
            for key in ("scenarios", "records", "items"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            log("scenario batch did not parse, retrying once")
            text, _ = query_model(synth, SCEN_PROMPT.format(
                capability=cap_text, evidence=ev_text, n=n, urls=urls)
                + "\nReturn only the JSON array.", share)
            data = extract_json(text)
        if not isinstance(data, list):
            log("scenario batch failed to parse, skipping")
            continue
        with path.open("a") as fh:
            for rec in data[:n]:
                if not isinstance(rec, dict):
                    continue
                sid += 1
                srcs = [str(u) for u in (rec.get("sources") or [])][:2]
                kinds = sorted({source_kind(u) for u in srcs} or {"other"})
                row = {"id": f"sc-{sid:03d}", "persona": clean(rec.get("persona"))[:300],
                       "situation": clean(rec.get("situation"))[:600],
                       "goal": clean(rec.get("goal"))[:300],
                       "facts": [{"field": clean(f.get("field"))[:40],
                                  "value": clean(f.get("value"))[:120]}
                                 for f in (rec.get("facts") or []) if isinstance(f, dict)][:8],
                       "sources": srcs, "source_kinds": kinds,
                       "hard": clean(rec.get("hard"))[:300],
                       "level_hint": str(rec.get("level_hint") or "").strip().lower(),
                       "wrong_fact": bool(rec.get("wrong_fact"))}
                if row["level_hint"] not in ("easy", "medium", "hard"):
                    row["level_hint"] = "medium"
                fh.write(json.dumps(row) + "\n")
                have.append(row)
        need = WANT_SCENARIOS - len(have)
        log(f"scenarios kept: {len(have)}")
    return have


# --- blueprint agent ---------------------------------------------------------

BP_PROMPT = """You map a customer scenario onto a service. The service tools (name, kind, argument names) are below. Read tools only read; write tools change state. No customer data is shown.

Tools:
{tools}

Scenario {sid} (requested level {level}, requested kind {kind}):
- persona: {persona}
- situation: {situation}
- goal: {goal}
- facts the customer holds: {facts}
- tricky part: {hard}

{feedback}You work in rounds. In each round reply with exactly one JSON object, no other text:
R) explore: {{"call_reads": [{{"name": read tool, "args": {{...}}}}]}} to run up to 4 read calls and see their answers next round. Start here: the facts above identify a real customer. Use them as read keys first, then follow what the answers return.
V) ask for keys: {{"need_values": {{"tool": write tool, "arg": argument}}}} to receive values past customers used for one argument, then use them as read keys.
A) final blueprint: {{"reads": [{{"name": tool, "args": {{}}}}], "writes": [{{"name": tool, "args": {{}}}}], "must_tell": ["..."], "kind": "do|decline|mixed", "level": "easy|medium|hard", "wrong_fact": false, "request": "the customer's opening message in their own words, one or two sentences, using only facts they hold", "unservable": ""}}.
   Reads come first. Row ids are rebound from the world before anything runs, so prefer ids from read answers but do not stall on them. Request text must not name any tool, table or system step. Keep args minimal and complete.
   Structure by level: easy is one write as do, medium two writes as do or a decline, hard three or more writes, or mixed, or a wrong fact the user holds.
B) {{"unservable": "one line reason", "kind": "decline"}} when no tool can serve this scenario at all.
No other text."""


def level_ok(level: str, kind: str, n_writes: int, wrong_fact: bool) -> bool:
    if level == "easy":
        return n_writes == 1 and kind == "do"
    if level == "medium":
        return (n_writes == 2) or (kind == "decline")
    return (n_writes >= 3) or (kind == "mixed") or bool(wrong_fact)


def judge_check(judge: Priced, share: Share, request: str, kind: str, level: str) -> str:
    prompt = ("A customer wrote this opening message to a support agent. Answer with one JSON "
              f"object {{\"kind_ok\": true|false, \"level_guess\": \"easy|medium|hard\"}} and nothing else. "
              f"Claimed kind: {kind}. Claimed level: {level}. Message: {request[:500]}")
    try:
        text, _ = query_model(judge, prompt, share)
        data = extract_json(text)
        if isinstance(data, dict):
            return ("judge_kind_ok" if data.get("kind_ok") else "judge_kind_doubt")
    except Exception:
        return "judge_skipped"
    return "judge_skipped"


def observed_values(workdir, tool: str, arg: str, n: int = 5) -> list:
    """Values the recordings showed for one argument: the binding fallback."""
    from kullback import synthesise as S
    from kullback.sampling import sample_key
    body = S.load_graph(workdir)
    pool = list((((body.get("args") or {}).get(tool) or {}).get(arg) or {}).get("values") or [])
    if not pool:
        return []
    start = sample_key("arm-c-obs", f"{SEED}:{tool}:{arg}", "") % max(len(pool), 1)
    return [str(pool[(start + i) % len(pool)]) for i in range(min(n, len(pool)))]


def exec_reads(world, toolkit, calls: list) -> list:
    """Run read calls against the scratch world. Answers truncated, never logged."""
    out = []
    for item in calls or []:
        name = str(item.get("name") or "")
        args = dict(item.get("args") or {})
        body = getattr(toolkit, name, None)
        if body is None:
            out.append({"name": name, "error": "unknown_tool"})
            continue
        try:
            answer = body(**args)
        except Exception as error:
            out.append({"name": name, "error": type(error).__name__})
            continue
        out.append({"name": name, "answer": str(answer)[:1500]})
    return out


def build_task(world, synth: Priced, judge: Priced, share: Share, scen: dict,
               level: str, kind_want: str, slot: int) -> dict:
    tools = capability_summary()["tools"]
    def show(tool: dict) -> str:
        req = set(tool.get("required") or [])
        args = ", ".join(a + ("*" if a in req else "") for a in tool["args"])
        return f"- {tool['name']} ({tool['kind']}): {args}"
    tool_text = "\n".join(show(t) for t in tools) + "\n(* marks a required argument)"
    anchor = C.draw_anchor(world, f"{SEED}:{slot}")
    user_facts = (anchor or {}).get("facts") or scen.get("facts") or []
    facts = "\n".join(f"- {f['field']}: {f['value']}" for f in user_facts)
    feedback = ""
    history = ""
    toolkit = world.toolkit(world.fresh_db())
    usd0 = share.model_usd
    max_calls = (MAX_REVISIONS + 1) * 3
    calls_made = 0
    for rev in range(MAX_REVISIONS + 1):
        for it in range(MAX_READ_ITERS):
            if calls_made >= max_calls:
                break
            share.refuse_if_broke()
            prompt = BP_PROMPT.format(tools=tool_text, sid=scen["id"], level=level,
                                      kind=kind_want, persona=scen.get("persona"),
                                      situation=scen.get("situation"), goal=scen.get("goal"),
                                      facts=facts or "- nothing stated",
                                      hard=scen.get("hard"), feedback=feedback + history)
            text, _ = query_model(synth, prompt, share)
            calls_made += 1
            bp = extract_json(text)
            log(f"slot {slot} rev {rev} it {it} reply: "
                f"{'unparsed' if not isinstance(bp, dict) else sorted(bp.keys())}")
            if not isinstance(bp, dict):
                feedback = "FEEDBACK: your reply did not parse as JSON. Send only the JSON object. "
                continue
            if isinstance(bp.get("call_reads"), list):
                answers = exec_reads(world, toolkit, bp["call_reads"][:4])
                history = (history + "\nREAD ANSWERS: " + json.dumps(answers)[:2500])[-3000:]
                if it >= MAX_READ_ITERS - 2:
                    feedback = ("FEEDBACK: answers above. Rounds are nearly out: send the final "
                                "blueprint or unservable next. ")
                else:
                    feedback = ("FEEDBACK: answers above. Continue: call more reads, ask for values, "
                                "or send the final blueprint. ")
                continue
            if isinstance(bp.get("need_values"), dict):
                nv = bp["need_values"]
                vals = observed_values(WORK, str(nv.get("tool") or ""), str(nv.get("arg") or ""))
                history = (history + "\nRECORDED VALUES: " + json.dumps(vals)[:800])[-3000:]
                feedback = ("FEEDBACK: recorded values above. Use them to fill read keys, then "
                            "call reads to confirm rows before writing. ")
                continue
            explored = "READ ANSWERS:" in history or "RECORDED VALUES:" in history
            done, outcome = settle_blueprint(world, synth, judge, share, scen, level, bp,
                                             slot, rev, explored, user_facts)
            log(f"slot {slot} rev {rev} settle: {outcome} "
                f"{str(done.get('reasons', done))[:120] if isinstance(done, dict) else str(done)[:120]}")
            if outcome == "soft":
                feedback = str(done)
                continue
            if outcome == "retry":
                feedback = str(done)
                break
            done["cost_usd"] = round(share.model_usd - usd0, 6)
            return done
        else:
            feedback = "FEEDBACK: round limit hit. Send the final blueprint or unservable now. "
    return {"status": "rejected", "revisions": MAX_REVISIONS,
            "reasons": ["revisions_exhausted"],
            "cost_usd": round(share.model_usd - usd0, 6),
            "scenario_id": scen["id"]}
def settle_blueprint(world, synth: Priced, judge: Priced, share: Share, scen: dict,
                     level: str, bp: dict, slot: int, rev: int, explored: bool = True,
                     user_facts: list | None = None):
    """One finished blueprint through the birth checks. Returns (result, action).

    action is done (final), retry (birth issue, next revision) or soft
    (format issue, keep exploring in this revision).
    """
    if str(bp.get("unservable") or "").strip():
        return {"status": "rejected", "revisions": rev,
                "reasons": ["unservable:" + clean(bp.get("unservable"))[:160]],
                "scenario_id": scen["id"]}, "done"
    if not explored:
        return ("FEEDBACK: you sent a final blueprint before calling any read. First round must "
                "be call_reads or need_values, then confirm rows, then the blueprint. "), "soft"
    reads = bp.get("reads") or []
    writes = bp.get("writes") or []
    kind = str(bp.get("kind") or "").strip().lower()
    lvl = str(bp.get("level") or "").strip().lower()
    if kind not in ("do", "decline", "mixed"):
        return "FEEDBACK: kind must be one of do, decline, mixed. ", "soft"
    if lvl != level:
        return f"FEEDBACK: requested level was {level}, you set {lvl}. Fix it. ", "soft"
    if not level_ok(level, kind, len(writes), bool(bp.get("wrong_fact")) or scen.get("wrong_fact")):
        return (f"FEEDBACK: level {level} needs the matching structure: easy is one write, "
                "medium two writes or a decline, hard three or more or mixed or a wrong fact. "
                f"You gave {len(writes)} writes as {kind}. Fix it. "), "soft"
    ident = f"{SEED}:{scen['id']}:{slot}:r{rev}"
    calls, why, info = C.run_blueprint(world, reads, writes, ident)
    info = info or {}
    note_attempt({"slot": slot, "rev": rev, "phase": "bind", "why": why or "ran",
                  "failed_tool": info.get("tool"), "n_reads": len(reads or []),
                  "n_writes": len(writes or []), "stats": info.get("stats") or {}})
    if (why or "").startswith("no_host_row"):
        return {"status": "rejected", "revisions": rev, "kind": kind,
                "reasons": [why], "scenario_id": scen["id"]}, "done"
    if why:
        if why.startswith(("crashed", "unknown_tool")):
            return {"status": "rejected", "revisions": rev, "kind": kind,
                    "reasons": [why], "scenario_id": scen["id"]}, "done"
        where = f"call {info.get('index')} ({info.get('tool')})" if info.get("tool") \
            else "a call"
        return (f"FEEDBACK: the environment refused {where} with class {why}. Row ids "
                "are rebound from the world automatically, so do not resend other ids: "
                "change what the request asks, call different reads, or mark "
                "unservable. "), "retry"
    request = clean(bp.get("request"))[:500]
    if not request:
        return "FEEDBACK: request text is empty. Write the customer's opening message. ", "soft"
    held_by_user = user_facts if user_facts is not None else (scen.get("facts") or [])
    said_vals = {str(f.get("value")) for f in held_by_user if f.get("value")}
    held_vals = set()
    for call in calls:
        for _path, val in _arg_leaves(call.get("args")):
            if str(val) and str(val) not in said_vals:
                held_vals.add(str(val))
    leaked = [v for v in held_vals if v and v in request]
    if leaked:
        return "FEEDBACK: the request states a value only the world knows. Rewrite it. ", "soft"
    must_tell = [clean(m)[:200] for m in (bp.get("must_tell") or [])][:5]
    said = [{"field": str(f.get("field")), "value": f.get("value")}
            for f in held_by_user]
    task_id = B.task_uid(f"{SEED}:{ID_SALT}:{scen['id']}:{slot}")
    _, checks = B.make_task(world, calls, request, said, task_id, SEED)
    note_attempt({"slot": slot, "rev": rev, "phase": "checks",
                  "why": "runs_failed" if not checks["runs"] else "ran",
                  "failed_tool": None, "n_reads": len(reads or []),
                  "n_writes": len(writes or [])})
    reasons = []
    if not checks["runs"]:
        reasons.append("runs_failed")
    if not checks["do_nothing_fails"]:
        reasons.append("do_nothing_passed")
    if not checks["prefix_fails"]:
        reasons.append("prefix_passed")
    if not checks["suite_passed"]:
        reasons += [f"suite:{f}" for f in checks["suite_failures"]]
    if reasons:
        if rev == MAX_REVISIONS:
            return {"status": "rejected", "revisions": rev, "task_id": task_id,
                    "reasons": reasons, "scenario_id": scen["id"],
                    "blueprint_writes": [{"name": c["name"], "args": c["args"]}
                                         for c in calls if c["name"] in world.write_tools]}, "done"
        return f"FEEDBACK: birth checks failed: {', '.join(reasons[:4])}. Revise the blueprint. ", "retry"
    check_note = judge_check(judge, share, request, kind, level)
    row = {"status": "admitted", "revisions": rev,
           "reasons": [check_note] if check_note else [],
           "scenario_id": scen["id"]}
    row.update({"kind": kind, "level": level,
                "units": len([c for c in calls if c["name"] in world.write_tools]),
                "request": request, "user_facts": said,
                "blueprint_writes": [{"name": c["name"], "args": c["args"]}
                                     for c in calls if c["name"] in world.write_tools],
                "must_tell": must_tell, "checks": checks, "task_id": task_id})
    return row, "done"


def _arg_leaves(args, prefix=""):
    out = []
    if isinstance(args, dict):
        for key, val in args.items():
            out += _arg_leaves(val, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(args, list):
        for val in args:
            out += _arg_leaves(val, prefix)
    elif isinstance(args, str) and args:
        out.append((prefix, args))
    return out


# --- driver ------------------------------------------------------------------

def slots(plan: int) -> list:
    order = (["easy"] * LEVEL_QUOTA["easy"] + ["medium"] * LEVEL_QUOTA["medium"]
             + ["hard"] * LEVEL_QUOTA["hard"])
    rnd = random.Random(SEED)
    rnd.shuffle(order)
    kinds = {"easy": "do", "medium": "do", "hard": "do"}
    return [{"level": lv, "kind": kinds[lv]} for lv in order[:plan]]


TASKS_PATH = OUT / "tasks.jsonl"


def existing_tasks() -> list:
    path = TASKS_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_slot(world, synth: Priced, judge: Priced, share: Share, slot: int) -> dict | None:
    """One slot through the blueprint loop, appended to TASKS_PATH. Returns the line."""

    scen_path = OUT / "scenarios.jsonl"
    scenarios = [json.loads(line) for line in scen_path.read_text().splitlines() if line.strip()]
    if not scenarios:
        raise RuntimeError("no scenarios found")
    plan = slots(WANT_TASKS)
    spec = plan[slot]
    scen = scenarios[slot % len(scenarios)]
    t0 = time.time()
    try:
        task = build_task(world, synth, judge, share, scen,
                          spec["level"], spec["kind"], slot)
    except RuntimeError as error:
        log(f"slot {slot} stopped: {error}")
        return None
    task["task_id"] = task.get("task_id") or B.task_uid(f"{SEED}:{ID_SALT}:rej:{slot}")
    task.setdefault("kind", "do" if spec["level"] == "easy" else spec["level"])
    task.setdefault("level", spec["level"])
    task.setdefault("units", 0)
    task.setdefault("scenario_id", scen["id"])
    task.setdefault("request", "")
    task.setdefault("user_facts", [])
    task.setdefault("blueprint_writes", [])
    task.setdefault("must_tell", [])
    task.setdefault("checks", {"runs": False, "do_nothing_fails": False,
                               "prefix_fails": False, "suite_passed": False,
                               "suite_failures": []})
    line = {"task_id": task["task_id"], "arm": ARM, "kind": task["kind"],
            "level": task["level"], "units": task["units"],
            "scenario_id": task.get("scenario_id"), "request": task["request"],
            "user_facts": task["user_facts"],
            "blueprint_writes": task["blueprint_writes"],
            "must_tell": task["must_tell"], "checks": task["checks"],
            "revisions": task.get("revisions", 0), "status": task["status"],
            "reasons": [B.scrub(r) for r in (task.get("reasons") or [])],
            "cost_usd": round(float(task.get("cost_usd", 0.0)), 6)}
    with TASKS_PATH.open("a") as fh:
        fh.write(json.dumps(line) + "\n")
    dt = time.time() - t0
    log(f"slot {slot} {spec['level']} {scen['id']} -> {line['status']} "
        f"rev {line['revisions']} ({dt:.0f}s)")
    return line


def phase_tasks(share: Share, synth: Priced, judge: Priced, limit: int) -> list:
    from kullback import synthesise as S

    rows = existing_tasks()
    log(f"tasks kept: {len(rows)}")
    if len(rows) >= limit:
        return rows
    world = S.World(WORK)
    for slot in range(len(rows), limit):
        share.refuse_if_broke()
        line = run_slot(world, synth, judge, share, slot)
        if line is None:
            break
        rows.append(line)
    return rows


def funnel(rows: list) -> str:
    lines = ["level admitted rejected"]
    for lv in ("easy", "medium", "hard"):
        a = sum(1 for r in rows if r["level"] == lv and r["status"] == "admitted")
        j = sum(1 for r in rows if r["level"] == lv and r["status"] != "admitted")
        lines.append(f"{lv} {a} {j}")
    return "\n".join(lines)


def phase_report(share: Share, wall_s: float) -> None:
    rows = existing_tasks()
    scenarios = []
    if (OUT / "scenarios.jsonl").exists():
        scenarios = [json.loads(line) for line in (OUT / "scenarios.jsonl").read_text().splitlines()
                     if line.strip()]
    ev = read_json(OUT / "evidence.json", {})
    by_level: dict = {}
    rev_hist: dict = {}
    check_counts = {"runs": 0, "do_nothing_fails": 0, "prefix_fails": 0, "suite_passed": 0}
    fail_reasons: dict = {}
    status = {"admitted": 0, "rejected": 0}
    for r in rows:
        status[r["status"]] = status.get(r["status"], 0) + 1
        by_level.setdefault(r["level"], {"admitted": 0, "rejected": 0})
        by_level[r["level"]][r["status"]] += 1
        rev_hist[str(r.get("revisions", 0))] = rev_hist.get(str(r.get("revisions", 0)), 0) + 1
        for key in check_counts:
            if (r.get("checks") or {}).get(key):
                check_counts[key] += 1
        for reason in r.get("reasons") or []:
            fail_reasons[str(reason)[:60]] = fail_reasons.get(str(reason)[:60], 0) + 1
    per_kind: dict = {}
    for s in scenarios:
        for k in s.get("source_kinds") or ["other"]:
            per_kind[k] = per_kind.get(k, 0) + 1
    summary = {"arm": ARM, "seed": SEED, "tasks": len(rows), "status": status,
               "by_level": by_level, "check_counts": check_counts,
               "revisions": rev_hist, "fail_reasons": fail_reasons,
               "model_usd": round(share.model_usd, 4),
               "search_usd": round(share.search_usd, 4),
               "total_usd": round(share.total(), 4),
               "model_calls": share.model_calls,
               "searches": share.searches, "fetches": share.fetches,
               "keyless_attempts": (ev.get("keyless_attempts", 0)),
               "scenarios": len(scenarios), "scenarios_per_source_kind": per_kind,
               "wall_s": round(wall_s, 1)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary, indent=1))
    log("--- funnel ---")
    log(funnel(rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["search", "scenarios", "pilot", "full", "report"])
    args = ap.parse_args()
    t0 = time.time()
    share = Share(CEIL)
    synth = Priced(SYNTH_ID, CEIL)
    judge = Priced(JUDGE_ID, CEIL)
    if args.phase == "search":
        phase_search(share, synth)
    elif args.phase == "scenarios":
        phase_scenarios(share, synth)
    elif args.phase == "pilot":
        phase_tasks(share, synth, judge, 10)
    elif args.phase == "full":
        phase_tasks(share, synth, judge, WANT_TASKS)
    elif args.phase == "report":
        phase_report(share, 0.0)
    wall = time.time() - t0
    state = {"model_usd": share.model_usd, "search_usd": share.search_usd,
             "model_calls": share.model_calls, "searches": share.searches,
             "fetches": share.fetches, "wall_s": wall}
    (OUT / f"spend-{args.phase}.json").write_text(json.dumps(state, indent=1))
    log(f"phase {args.phase} done in {wall:.0f}s, model {share.model_usd:.4f} USD, "
        f"search {share.search_usd:.4f} USD")


if __name__ == "__main__":
    main()
