"""Step 2: agentic judge. One agent per failed Run, with read tools over the
Run, its Task, Reference and Verifier. Classifies the failure cause.

Cause classes are generic. No corpus, tool, Task or model names in code or
prompts. Every name below is read from the workdir at run time.
"""
import json
from pathlib import Path

from kullback.runner import budget as budget_mod
from kullback.runner import target as target_mod
from kullback.runner import verdict as verdict_mod
from kullback.runner.records import Verifier, read_json

JUDGE_ID = "bedrock/global.anthropic.claude-opus-5-5"
ENV_PATH = Path("/Users/krishuagarwal/Desktop/Programming/leibler/kullback/.env")
MAX_ROUNDS = 4

CAUSES = ("candidate_weakness", "wrong_reference", "environment_fault",
          "user_dead_end", "verifier_fault")

SYSTEM = """You judge why one agent Run failed its check. Read only what you need.
Cause classes, pick exactly one:
- candidate_weakness: the Run had what it needed (answers arrived, reads named
  rows) and still acted wrongly: nothing where an action belonged, a wrong
  action, a wrong row, or a skipped step. Name the missing skill in a few
  generic words (what had to be done, not which call or row).
- wrong_reference: the comparison Run itself looks wrong: it stopped short of
  the stated goal, acted against the stated goal, or its end matches a goal
  no stated goal asked for. The failure may say more about the comparison
  than about the judged Run.
- environment_fault: the world errored, crashed, or answered through a
  fallback path (an assisted or fallback-marked step, a missed overlay, a
  cannot-answer where the question was answerable). The agent did not cause it.
- user_dead_end: the conversation partner ran out of answers (unavailable,
  restated, exhausted markers) while the agent was still asking. The Run died
  of missing facts, not of a wrong action.
- verifier_fault: the Run did what the comparison Run did (same actions on
  the same rows) and still failed, or it failed on a conversation atom about
  wording rather than on what changed in the world.
Rules: cite row ids and counts only, never quote message text. Answer as one
JSON object: {"cause": one of the five, "skill": generic skill name or null,
"confidence": 0 to 1, "rationale": one or two short sentences}."""


def make_model():
    from kullback.ai import provider
    provider.load_dotenv(ENV_PATH)
    provider.enable_live_calls_from_env()
    return provider.live_model(JUDGE_ID)


def cost_of(reply) -> float:
    return budget_mod.call_cost(reply.usage, JUDGE_ID)


class Reader:
    """Read tools over one workdir. All returned names come from stored files."""

    def __init__(self, workdir: Path, evidence: dict):
        self.workdir = workdir
        self.by_id = {item["run_id"]: item for item in evidence}

    def tools(self) -> list[dict]:
        def tool(name: str, desc: str) -> dict:
            return {"name": name, "description": desc,
                    "input_schema": {"type": "object",
                                     "properties": {"id": {"type": "string"}},
                                     "required": ["id"]}}
        return [
            tool("read_run", "Full call sequence of a Run by run id, with errors."),
            tool("read_reference", "Call sequence of a Task comparison Run by task id."),
            tool("read_verifier", "Every check atom of a Task by task id."),
            tool("read_transcript", "First and last conversation turns of a Run by run id."),
        ]

    def call(self, name: str, arg: str):
        if name == "read_run":
            return self._run(arg, full=True)
        if name == "read_reference":
            return self._reference(arg)
        if name == "read_verifier":
            return self._verifier(arg)
        if name == "read_transcript":
            return self._transcript(arg)
        return {"error": "unknown tool"}

    def _path(self, run_id: str) -> Path:
        return self.workdir / "runs" / f"{run_id}.jsonl"

    def _run(self, run_id: str, full: bool):
        item = self.by_id.get(run_id, {})
        if item.get("unreadable") or not self._path(run_id).is_file():
            return {"run_id": run_id, "unreadable": True}
        run = verdict_mod.load_run(self._path(run_id))
        calls = target_mod.run_calls(run)
        seq = []
        for call in calls[:40]:
            args = {key: str(value)[:80] for key, value in
                    (call.get("args") or {}).items()}
            seq.append({"call": call["name"], "error": bool(call["error"]),
                        "args": args if full else {}})
        return {"run_id": run_id, "calls": seq, "user": item.get("user", {})}

    def _reference(self, task_id: str):
        for item in self.by_id.values():
            if item.get("task_id") == task_id and "reference" in item:
                return {"task_id": task_id, **item["reference"]}
        return {"task_id": task_id, "missing": True}

    def _verifier(self, task_id: str):
        for item in self.by_id.values():
            if item.get("task_id") == task_id and "verifier" in item:
                return {"task_id": task_id, **item["verifier"]}
        body = read_json(self.workdir / "verifiers" / f"{task_id}.json", None)
        if not body:
            return {"task_id": task_id, "missing": True}
        verifier = Verifier.model_validate(body)
        return {"task_id": task_id, "atoms": [
            {"id": str(atom.id), "kind": str(atom.kind)} for atom in verifier.atoms or ()]}

    def _transcript(self, run_id: str):
        if not self._path(run_id).is_file():
            return {"run_id": run_id, "unreadable": True}
        run = verdict_mod.load_run(self._path(run_id))
        texts = []
        for event in run.events or ():
            if event.type in ("user_turn", "model_call"):
                payload = event.payload or {}
                text = str(payload.get("text") or payload.get("content") or "")
                if len(text) > 200:
                    text = text[:200] + " ..."
                texts.append({"turn": event.type, "text": text})
        keep = texts[:3] + texts[-3:] if len(texts) > 6 else texts
        return {"run_id": run_id, "turns": keep}


def ask(model, messages, tools):
    """One model call with retries on throttle and server errors."""
    import time as _time
    last = None
    for wait in (5, 15, 45, 120):
        try:
            return model.query(messages, tools=tools)
        except Exception as error:
            last = error
            status = getattr(error, "status", None)
            if status is not None and status < 429:
                raise
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            _time.sleep(wait)
    raise last


def judge_one(model, reader: Reader, item: dict) -> dict:
    inner = dict(item)
    inner.pop("harness_note", None)
    messages = [{"role": "user", "content":
                 "Classify this failed Run. Evidence:\n" + json.dumps(inner)[:6000]}]
    spent = 0.0
    rounds = 0
    try:
        for _ in range(MAX_ROUNDS):
            reply = ask(model, [{"role": "system", "content": SYSTEM}] + messages,
                        reader.tools())
            spent += cost_of(reply)
            for call in reply.tool_calls or ():
                result = reader.call(call.name, (call.arguments or {}).get("id", ""))
                messages.append({"role": "assistant", "content": reply.content or "",
                                 "tool_calls": [{"id": call.id, "name": call.name,
                                                 "arguments": call.arguments or {}}]})
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": json.dumps(result, default=str)[:4000]})
                rounds += 1
            text = (reply.content or "").strip()
            if text and not (reply.tool_calls):
                return {**parse(text, item), "rounds": rounds, "cost_usd": spent}
            if rounds >= MAX_ROUNDS:
                messages.append({"role": "user", "content":
                                 "Answer now with the JSON object only."})
                reply = ask(model, [{"role": "system", "content": SYSTEM}] + messages,
                            reader.tools())
                spent += cost_of(reply)
                return {**parse(reply.content or "", item), "rounds": rounds,
                        "cost_usd": spent}
    except Exception as error:
        return {"run_id": item["run_id"], "task_id": item["task_id"],
                "cause": "undetermined", "skill": None, "confidence": 0.0,
                "rationale": f"judge call failed: {type(error).__name__} "
                             f"status {getattr(error, 'status', None)}",
                "rounds": rounds, "cost_usd": spent}


def parse(text: str, item: dict) -> dict:
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        body = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"run_id": item["run_id"], "task_id": item["task_id"],
                "cause": "undetermined", "skill": None, "confidence": 0.0,
                "rationale": "judge answer was not JSON"}
    cause = str(body.get("cause") or "undetermined")
    if cause not in CAUSES:
        cause = "undetermined"
    skill = body.get("skill")
    skill = str(skill)[:120] if skill else None
    if cause != "candidate_weakness":
        skill = None
    try:
        confidence = float(body.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {"run_id": item["run_id"], "task_id": item["task_id"], "cause": cause,
            "skill": skill, "confidence": min(max(confidence, 0.0), 1.0),
            "rationale": str(body.get("rationale") or "")[:400]}
