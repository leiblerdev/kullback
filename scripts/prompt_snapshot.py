"""Snapshot proof for D245: stable system heads for the single message stages.

Builds the request messages for the intent stage and the one shot reference judge over ten
invented inputs, and checks that for every input the concatenation of the head's system text
and the user text holds every sentence of the old single message form, which is kept in tree
(`_intent_prompt`, `judge_prompt`) byte identical to origin/main. Writes the messages to a
file given as the first argument, or /tmp/prompt_snapshot.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
WORKTREE = HERE.parents[1]
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(WORKTREE / "tests"))

from builder.test_intent import cancel_trace  # noqa: E402
from examiner.test_reference import cancel_run, empty_run  # noqa: E402
from kullback.builder import intent as intent_mod  # noqa: E402
from kullback.examiner import reference as ref  # noqa: E402
from kullback.runner.records import Task  # noqa: E402


def sentences(text: str) -> Counter:
    return Counter(line.strip() for line in text.splitlines() if line.strip())


def intent_inputs():
    out = []
    words = ["cancel", "redirect", "hold", "refund", "extend",
             "renew", "reserve", "release", "transfer", "close"]
    for i, verb in enumerate(words):
        order = f"W{i + 1}"
        traces = [cancel_trace(f"r{i}a", order), cancel_trace(f"r{i}b", order)]
        task = Task(id=f"snapshot_{verb}", run_ids=[f"r{i}a", f"r{i}b"])
        out.append((f"intent/{verb}", task, traces))
    return out


def judge_inputs():
    verbs = ["cancel", "redirect", "hold", "refund", "extend",
             "renew", "reserve", "release", "transfer", "close"]
    out = []
    for i, verb in enumerate(verbs):
        groups = ref.group([cancel_run(f"{verb}a", delivery=f"#D10{i}"),
                            empty_run(f"{verb}b")])
        out.append((f"judge/{verb}", f"{verb} the pending delivery",
                    ["a reader may be told the standing of their own delivery"], groups))
    return out


def main() -> int:
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/prompt_snapshot.json")
    record: dict = {"inputs": []}
    failures = 0
    for name, task, traces in intent_inputs():
        members = [t for t in traces if t.trace_id in set(task.run_ids)]
        old = intent_mod._intent_prompt(members, {"cancel_order"})
        messages = intent_mod._intent_messages(members, {"cancel_order"})
        new = messages[0]["content"] + "\n" + messages[1]["content"]
        ok = sentences(old) == sentences(new)
        failures += not ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} "
              f"(system {len(messages[0]['content'].split())} words, "
              f"user {len(messages[1]['content'].split())} words)")
        record["inputs"].append({"name": name, "messages": messages})
    for name, intent_text, policy, groups in judge_inputs():
        old = ref.judge_prompt(intent_text, policy, groups)
        messages = ref._judge_messages(intent_text, policy, groups)
        new = messages[0]["content"] + "\n" + messages[1]["content"]
        ok = sentences(old) == sentences(new)
        failures += not ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} "
              f"(system {len(messages[0]['content'].split())} words, "
              f"user {len(messages[1]['content'].split())} words)")
        record["inputs"].append({"name": name, "messages": messages})
    for final in (False, True):
        _, intent_text, policy, groups = judge_inputs()[0]
        old = ref.judge_prompt(intent_text, policy, groups, final)
        messages = ref._judge_messages(intent_text, policy, groups, final)
        ok = sentences(old) == sentences(messages[0]["content"] + "\n" + messages[1]["content"])
        failures += not ok
        print(f"judge/final={final}: {'PASS' if ok else 'FAIL'}")
    intent_head = intent_mod._intent_head()
    judge_head = ref._judge_head()
    record["heads"] = {
        "intent_words": len(intent_head.split()),
        "judge_words": len(judge_head.split()),
        "intent_est_tokens": round(len(intent_head.split()) * 1.3),
        "judge_est_tokens": round(len(judge_head.split()) * 1.3),
        "tokenizer": "word count times 1.3; the repo has no prompt tokenizer",
    }
    print(f"intent head: {record['heads']['intent_words']} words, "
          f"about {record['heads']['intent_est_tokens']} tokens")
    print(f"judge head: {record['heads']['judge_words']} words, "
          f"about {record['heads']['judge_est_tokens']} tokens")
    dest.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {dest}")
    print("FAILURES: %d" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
