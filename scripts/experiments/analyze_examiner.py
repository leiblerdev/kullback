"""Step 1 of fx-examwhy: observe Examiner sessions in bus.jsonl (no model).

Usage: python3 scripts/experiments/analyze_examiner.py
Reads build bus logs (read-only) + fx-examab2 clone bus logs, prints per-session
tool sequences, post-refusal behavior, session ends, and aggregate counts.
"""
import json
import re
from collections import Counter

BUILD = "/Users/krishuagarwal/.herdr/worktrees/kullback/overhaul-0922-smoke9"
SCRATCH = "/Users/krishuagarwal/.herdr/worktrees/kullback/fx-0925-examab2/.work-scratch"

READ_TOOLS = {"read", "grep", "inspect", "find", "ls"}
TASK_RE = re.compile(r"task_[0-9a-f]{12}")


def load_bus(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def sessions_of(events, agent="examiner"):
    """Split one agent's events into sessions on agent_start/agent_end."""
    ex = [e for e in events if e["agent"] == agent]
    sessions, cur = [], []
    for e in ex:
        t = e["event"].get("type")
        if t == "agent_start":
            cur = [e]
        elif t == "agent_end":
            cur.append(e)
            sessions.append(cur)
            cur = []
        else:
            cur.append(e)
    if cur:
        sessions.append(cur)  # unterminated (crashed/still running)
    return sessions


def tool_calls(session):
    """(tool_name, task_id-or-None, ok-or-None) per tool_execution_start, with outcome."""
    starts = {}
    out = []
    for e in session:
        ev = e["event"]
        if ev.get("type") == "tool_execution_start":
            args = ev.get("arguments") or {}
            tid = args.get("task_id")
            if not tid:
                m = TASK_RE.search(json.dumps(args))
                tid = m.group(0) if m else None
            starts[ev.get("tool_call_id")] = (ev.get("tool_name"), tid)
        elif ev.get("type") == "tool_execution_end":
            key = ev.get("tool_call_id")
            if key in starts:
                name, tid = starts.pop(key)
                content = ev.get("result", ev.get("content", ""))
                s = json.dumps(content) if not isinstance(content, str) else content
                ok = None
                if name in ("edit_verifier", "propose_verifier"):
                    ok = not ("refused" in s[:200].lower() or "refused" in s.lower()[:500])
                out.append((name, tid, ok, s))
    for _key, (name, tid) in starts.items():
        out.append((name, tid, None, "<no end event>"))
    return out


def turn_texts(session):
    """Assistant content text per turn (thinking blocks are encrypted/empty)."""
    texts = []
    for e in session:
        ev = e["event"]
        if ev.get("type") == "turn_end":
            msg = ev.get("message") or {}
            c = msg.get("content")
            if isinstance(c, list):
                c = " ".join(
                    b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
                )
            texts.append((ev.get("turn"), c or None, bool(msg.get("tool_calls"))))
    return texts


def summarize_session(events, label):
    sessions = sessions_of(events)
    print(f"===== {label}: {len(sessions)} examiner sessions =====")
    for i, s in enumerate(sessions):
        n_turns = sum(1 for e in s if e["event"].get("type") == "turn_start")
        ended = "agent_end" if any(
            e["event"].get("type") == "agent_end" for e in s
        ) else "UNTERMINATED"
        calls = tool_calls(s)
        offered, touched = set(), set()
        for e in s:
            ev = e["event"]
            if ev.get("type") == "message_start":
                m = ev.get("message", {})
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    offered.update(TASK_RE.findall(m["content"]))
        for _name, tid, _ok, _s in calls:
            if tid:
                touched.add(tid)
        counts = Counter(n for n, _t, _o, _s in calls)
        edits = [(t, o) for n, t, o, _s in calls if n in ("edit_verifier", "propose_verifier")]
        acc = sum(1 for _t, o in edits if o is True)
        ref = sum(1 for _t, o in edits if o is False)
        print(f"-- session {i}: turns={n_turns} end={ended} offered={len(offered)} "
              f"touched={len(touched)} tools={dict(counts)} edits(acc={acc},ref={ref})")
        # post-refusal behavior: what tool follows each refused edit
        for j, (n, tid, ok, _s) in enumerate(calls):
            if n in ("edit_verifier", "propose_verifier") and ok is False:
                nxt = calls[j + 1][0] + (f":{calls[j+1][1]}" if calls[j + 1][1] else "") \
                    if j + 1 < len(calls) else "<session end>"
                print(f"   refused edit on {tid} -> next: {nxt}")
        texts = turn_texts(s)
        last = [t for t in texts if t[1]][-2:] if any(t[1] for t in texts) else []
        for turn, text, _tc in last:
            print(f"   last text turn {turn}: {(text or '')[:300]}")
    return sessions


def aggregate(label, paths):
    tot_tools, tot_edits, acc, ref = Counter(), 0, 0, 0

    for path in paths:
        for e in load_bus(path):
            if e["agent"] != "examiner":
                continue
            ev = e["event"]
            if ev.get("type") == "tool_execution_start":
                tot_tools[ev.get("tool_name")] += 1
            if ev.get("type") == "tool_execution_end" and ev.get("tool_name") in (
                "edit_verifier", "propose_verifier"):
                tot_edits += 1
                s = json.dumps(ev.get("result", ev.get("content", "")))
                if "refused" in s[:2000].lower():
                    ref += 1
                else:
                    acc += 1
    print(f"##### {label}: tools={dict(tot_tools)} edits={tot_edits} acc={acc} ref={ref}")
    return tot_tools


if __name__ == "__main__":
    build_paths = [f"{BUILD}/.work-airline/bus.jsonl", f"{BUILD}/.work-retail/bus.jsonl"]
    build_sessions = {}
    for env, path in zip(("airline", "retail"), build_paths, strict=True):
        build_sessions[env] = summarize_session(load_bus(path), f"build/{env}")
    aggregate("build total", build_paths)
    print()
    ab_paths = [f"{SCRATCH}/{d}/bus.jsonl" for d in
                ("with1", "with2", "with3", "without1", "without2", "without3")]
    for d, path in zip(("with1", "with2", "with3", "without1", "without2", "without3"), ab_paths, strict=True):
        try:
            summarize_session(load_bus(path), f"examab2/{d}")
        except FileNotFoundError:
            print(f"examab2/{d}: no bus yet")
    aggregate("examab2 with", ab_paths[:3])
    aggregate("examab2 without", ab_paths[3:])
