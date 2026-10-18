"""One Kullback Task as it runs inside a Harbor container: the tools, the call log, the Verifier (D363).

A Harbor agent is a process with a shell, not a model the Runner drives turn by turn, so the world
has to outlive every call the agent makes. Each call is one short process: it rebuilds the Task's
world through the same Episode a Run uses, plays the call log from the start, plays the new call,
appends it to the log and writes the world back to `db.json`. Playing the log again from the
Starting state under one seed is what keeps ids and times the same across processes, with no
state of the Router's own to persist.

The agent reaches the tools two ways from this one code path: `tools` on the PATH (`tools list`,
`tools call <name> --json '{...}'`) and an MCP server over stdio. One tool more than the Task's,
`reply_to_user`, is how the agent speaks to the user. There is no live user in a Harbor trial, so
every message is answered with the user's standing consent line: a confirm-before-write rule then
still asks the agent to ask before it writes, and the user can never say no.

The Verifier runs the same play over the log the agent left, then scores the Run with the
Verdict by code. The reward is gated then weighted as the Verdict scores it, and it fails closed:
a gated item that cannot be scored by code gives 0 and is named.

Judge items are scored by the model JUDGE_VAR names at test time, one call per item (D328), when it
is set and live calls are allowed (the Harness's own switch); its answers and spend go to judge.json.
Unset, refused, unreadable or out after its retry, the judge leaves the trial unscored (D405): no reward,
a reward.json that says why, and exit UNSCORED_EXIT, so Harbor records an exception rather than a 0 or a 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.ai.provider import (
    BEDROCK_BEARER_VARS,
    BEDROCK_KEY_VARS,
    BEDROCK_SESSION_VAR,
    LIVE_ENV_VAR,
    live_model,
)
from kullback.judge.items import judge_items
from kullback.runner.records import ItemResult, UserRules, Verifier, read_json, write_json
from kullback.runner.verdict import load_run, verdict
from kullback.runner.world.environment import BuiltEnvironment
from kullback.runner.world.episode import Episode

APP = Path("/app")
WORLD_DIR = "world"
CALLS_NAME = ".calls.jsonl"
DB_NAME = "db.json"
# What the export writes beside the world: the Task id and the instruction the user opens with.
TASK_NAME = "harbor-task.json"
REPLY_TOOL = "reply_to_user"
CONSENT_LINE = "Yes, please go ahead with what I asked for. I have nothing to add."
REPLY_SPEC = {"name": REPLY_TOOL,
              "description": "Send a message to the user. The last message you send is your final reply.",
              "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                             "required": ["text"]}}
# What a call after the end of the conversation answers: a reply that handed the user over ends it.
ENDED = "the conversation has ended"
MAX_TURNS = 10_000
SEED = 0
MCP_PROTOCOL = "2024-11-05"
# The judge model, a 'provider/model' id read at test time; no task file names a model.
JUDGE_VAR = "KULLBACK_JUDGE_MODEL"
# What a test run passes through to reach a judge: the model, the live switch, the keys and region.
JUDGE_ENV = (JUDGE_VAR, LIVE_ENV_VAR, *BEDROCK_BEARER_VARS, *BEDROCK_KEY_VARS, BEDROCK_SESSION_VAR,
             "AWS_REGION", "ANTHROPIC_API_KEY")
JUDGE_NAME = "judge.json"
# D405: a judge item that did not come back answered makes the trial unscored, never a pass or a fail. The
# verifier exits with this code and reward.json carries a word where Harbor takes a number, so Harbor records
# the trial as an exception whose reason names the judge status.
UNSCORED_EXIT = 3
ANSWERED = "answered"
NOT_CONFIGURED = "not_configured"
NOT_STARTED = "not_started"


class ConsentUser:
    """The user of a Harbor trial: opens with the instruction, then answers every message with consent."""

    def __init__(self, opening: str):
        self.opening = opening
        self.done = False
        self.events: list = []
        self._spoken = False

    def reply(self, messages: Any) -> str:
        if not self._spoken:
            self._spoken = True
            return self.opening
        return CONSENT_LINE


class HarborEnvironment(BuiltEnvironment):
    """The Task's world with the consent user: no recorded rules, never driven by the agent user."""

    def rules(self, task: Any) -> Any:
        return UserRules()

    def agent_driven(self, task_id: str) -> bool:
        return False


class Play:
    """One play of a call log: the Episode it ran in, each entry's answer, and the user."""

    def __init__(self, episode: Episode, answers: list[dict], user: ConsentUser, done: bool):
        self.episode, self.answers, self.user, self.done = episode, answers, user, done


def _task(app: Path) -> dict:
    body = read_json(app / WORLD_DIR / TASK_NAME, None)
    if not isinstance(body, dict) or not body.get("task_id"):
        raise SystemExit(f"{app / WORLD_DIR / TASK_NAME} names no Task")
    return body


def play(app: Path, entries: Iterable[dict], outdir: Path) -> Play:
    """Play every entry from the Starting state: a tool call is a turn with that call, a reply a turn of text."""
    task = _task(app)
    users: list[ConsentUser] = []

    def make_user(env: Any, task_record: Any, router: Any, rules: Any, reference: Any) -> ConsentUser:
        users.append(ConsentUser(str(task.get("instruction") or "")))
        return users[-1]

    episode = Episode(HarborEnvironment(app / WORLD_DIR), outdir=outdir, max_turns=MAX_TURNS,
                      user_factory=make_user)
    episode.reset(str(task["task_id"]), seed=SEED)
    answers: list[dict] = []
    done = False
    for index, entry in enumerate(entries):
        name, args = str(entry.get("name") or ""), dict(entry.get("arguments") or {})
        if done:
            answers.append({"result": None, "error": ENDED})
            continue
        if name == REPLY_TOOL:
            out = episode.step({"content": str(args.get("text") or ""), "tool_calls": []})
            answers.append({"result": out.user_message or "", "error": None})
            done = out.done
            continue
        out = episode.step({"content": "", "tool_calls": [{"id": f"call-{index}", "name": name,
                                                            "arguments": args}]})
        result = out.results[0] if out.results else {"result": None, "error": "the call was not answered"}
        answers.append({"result": result.get("result"), "error": result.get("error")})
        done = out.done
    return Play(episode, answers, users[0], done)


def finish(played: Play) -> None:
    """End the Run: the user is done, so the turn after the last entry closes it."""
    played.user.done = True
    if not played.done:
        played.episode.step({"content": "", "tool_calls": []})
        played.done = True


def read_log(app: Path) -> list[dict]:
    path = app / CALLS_NAME
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tool_specs(app: Path) -> list[dict]:
    """Every tool the agent may call: the Task's own, then reply_to_user."""
    return HarborEnvironment(app / WORLD_DIR).tool_specs() + [REPLY_SPEC]


def argument_error(spec: dict, arguments: dict) -> Optional[str]:
    """Why a call's arguments do not fit its tool's spec, naming the parameters it takes, or None (D409).

    A required parameter left out or a parameter the spec does not name is refused, for the Task's tools and
    reply_to_user alike, with no alias for any name: a reply sent under another name would otherwise reach the
    Run as an empty turn while the agent was answered as if it had been heard.
    """
    parameters = spec.get("parameters") or {}
    expected = sorted(parameters.get("properties") or {})
    required = [str(name) for name in parameters.get("required") or ()]
    missing = [name for name in required if name not in arguments]
    unknown = sorted(name for name in arguments if name not in expected)
    if not missing and not unknown:
        return None
    wrong = ([f"missing {', '.join(missing)}"] if missing else []) + ([f"unknown {', '.join(unknown)}"] if unknown else [])
    return (f"{spec['name']} takes {', '.join(expected) or 'no parameters'} (required: {', '.join(required) or 'none'}); "
            + "; ".join(wrong))


def call(app: Path, name: str, arguments: dict) -> dict:
    """Make one call: play the log and the call, append it with its answer, write the world back.

    The shell `tools` command and the MCP server both enter here, so both refuse a call whose arguments do not
    fit the spec; a refused call is not logged and nobody answers it (D409).
    """
    specs = {spec["name"]: spec for spec in tool_specs(app)}
    if name not in specs:
        return {"result": None, "error": f"no tool named {name}"}
    wrong = argument_error(specs[name], arguments)
    if wrong:
        return {"result": None, "error": wrong}
    entries = read_log(app)
    with tempfile.TemporaryDirectory() as scratch:
        played = play(app, [*entries, {"name": name, "arguments": arguments}], Path(scratch))
        answer = played.answers[-1]
        world = played.episode._router.world()  # noqa: SLF001 - the Episode exposes no world reader
    line = {"name": name, "arguments": arguments, "result": answer["result"], "error": answer["error"]}
    with (app / CALLS_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
    write_json(app / DB_NAME, world)
    return answer


# --- the reward ---------------------------------------------------------------------


def _row(item: ItemResult, unscored: str = "") -> dict:
    return {"id": item.id, "kind": item.kind, "gate": item.gate, "weight": item.weight,
            "held": item.holds, "why": item.why, **({"status": unscored} if unscored else {})}


def reward_of(items: list[ItemResult], *, failed_closed: bool = False) -> dict:
    """The gated then weighted reward of a Verdict's items, failing closed on an open gate.

    The same formula as the Verdict's own score (runner/verdict.py `_score`, D329): 0 when a gate does
    not hold; past the gates the gate weight, at least half of the whole, plus the held weighted
    items, over the gate weight plus every weighted item. Two differences, both closed: a gate that
    cannot be scored gives 0 where the Verdict gives no number, and a weighted item that cannot be
    scored leaves the denominator where the Verdict masks the whole score.
    """
    gates = [item for item in items if item.gate]
    weighted = [item for item in items if not item.gate]
    unscored_gates = [item.id for item in gates if item.holds is None]
    rows = [_row(item, _status(item)) for item in items]
    if failed_closed or unscored_gates or any(item.holds is False for item in gates):
        return {"reward": 0.0, "items": rows, "unscored_gates": unscored_gates}
    return {"reward": _weighted(gates, weighted), "items": rows, "unscored_gates": [],
            "unscored": [item.id for item in weighted if item.holds is None]}


def _status(item: ItemResult) -> str:
    if item.holds is not None:
        return ""
    return "unscored_gate" if item.gate else "unscored"


def _weighted(gates: list[ItemResult], weighted: list[ItemResult]) -> float:
    """Past the gates: the gate weight, at least half, plus the held weighted items, over the gate weight
    plus every weighted item that could be scored."""
    scored = [item for item in weighted if item.holds is not None]
    rest = sum(item.weight for item in scored)
    gate_weight = max(sum(item.weight for item in gates), rest)
    total = gate_weight + rest
    return 1.0 if not total else (gate_weight + sum(item.weight for item in scored if item.holds)) / total


def judge(verifier: Verifier, run: Any, logs: Path) -> Optional[dict]:
    """The judge's answers to the Verifier's judge items on `run`, one call each, or None when no judge
    is named; judge.json records the model, each score and why, and the spend, never into reward.json."""
    if not any(atom.judge for atom in verifier.atoms):
        return None
    model_id = os.environ.get(JUDGE_VAR, "").strip()
    record: dict = {"model": model_id or None, "items": [], "cost_usd": 0.0, "error": None}
    answers = None
    if not model_id:
        record["error"] = f"{JUDGE_VAR} is not set"
    else:
        try:
            answers = judge_items(live_model(model_id), verifier.atoms, run, model_id=model_id)
        except Exception as error:  # noqa: BLE001 - a judge that cannot start leaves the items unscored
            record["error"] = f"{type(error).__name__}: {str(error)[:200]}"
    for result in (answers or {}).values():
        record["items"].append({"id": result.id, "score": result.score, "why": result.why,
                                "error": result.error, "status": result.status,
                                "stop_reason": result.stop_reason, "retries": result.retries,
                                "cost_usd": result.cost_usd})
    record["cost_usd"] = round(sum(row["cost_usd"] for row in record["items"]), 6)
    write_json(logs / JUDGE_NAME, record)
    return answers


def score(app: Path, verifier_path: Path, logs: Path, *, use_judge: bool = True) -> dict:
    """Play the agent's log, score the Run, write reward.txt, reward.json and items.json under `logs`;
    `use_judge` False scores with no judge whatever the environment names (the export's own check)."""
    verifier = Verifier.model_validate(read_json(verifier_path, {}))
    with tempfile.TemporaryDirectory() as scratch:
        played = play(app, read_log(app), Path(scratch))
        finish(played)
        env = played.episode.env
        run = load_run(played.episode.run_file())
        flagged = {sig.name for sig in env.sigs if getattr(sig, "unclassified", False)}
        answers = judge(verifier, run, logs) if use_judge else None
        result = verdict(run, verifier, env.canon_rules, answers, schema=env.schema,
                         write_tools=env.write_tools() or None, flagged_tools=flagged)
    closed = result.class_ == "env_error"
    body = reward_of(list(result.items), failed_closed=closed)
    body.update({"class": result.class_, "failing_atom": result.failing_atom,
                 "verdict_score": result.score, "notes": list(result.notes)})
    unjudged = {item: status for item, status in judge_statuses(verifier, answers).items()
                if status != ANSWERED} if use_judge else {}
    logs.mkdir(parents=True, exist_ok=True)
    if unjudged:
        return _unscored(body, unjudged, logs)
    (logs / "reward.txt").write_text(f"{body['reward']}\n", encoding="utf-8")
    # Harbor reads reward.json before reward.txt and takes every value in it as a reward, so it
    # carries the number alone; the items go beside it, where the trial keeps them.
    write_json(logs / "reward.json", {"reward": body["reward"]})
    write_json(logs / "items.json", body)
    return body


def judge_statuses(verifier: Verifier, answers: Optional[dict]) -> dict[str, str]:
    """Each judge item's status after the judge's own retry (D400): answered, unreadable, outage, or not
    configured or not started where no judge answered anything."""
    fallback = NOT_CONFIGURED if not os.environ.get(JUDGE_VAR, "").strip() else NOT_STARTED
    return {atom.id: (getattr((answers or {}).get(atom.id), "status", "") or fallback)
            for atom in verifier.atoms if atom.judge}


def _unscored(body: dict, unjudged: dict[str, str], logs: Path) -> dict:
    """The trial with no reward (D405): items.json keeps every item and each judge status, reward.json says
    unscored with the reason, and no reward.txt is left for a reader to take as a number."""
    reason = "unscored: judge " + ", ".join(f"{item} {status}" for item, status in sorted(unjudged.items()))
    for row in body["items"]:
        if row["id"] in unjudged:
            row["status"] = unjudged[row["id"]]
    body.update({"reward": None, "unscored_reason": reason, "judge_status": unjudged,
                 "reward_had_it_been_scored": body["reward"]})
    (logs / "reward.txt").unlink(missing_ok=True)
    write_json(logs / "reward.json", {"reward": reason})
    write_json(logs / "items.json", body)
    return body


# --- the two front doors ------------------------------------------------------------


def tools_main(argv: Optional[list[str]] = None, app: Path = APP) -> int:
    """`tools list` and `tools call <name> --json '{...}'`, printing JSON."""
    parser = argparse.ArgumentParser(prog="tools", description="The Task's tools.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Name every tool with its parameters.")
    one = sub.add_parser("call", help="Call one tool.")
    one.add_argument("name")
    one.add_argument("--json", dest="arguments", default="{}", help="The arguments as one JSON object.")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps(tool_specs(app), indent=2, ensure_ascii=False))
        return 0
    try:
        arguments = json.loads(args.arguments)
    except ValueError as exc:
        print(json.dumps({"error": f"--json is not JSON: {exc}"}))
        return 2
    if not isinstance(arguments, dict):
        print(json.dumps({"error": "--json must be one JSON object"}))
        return 2
    answer = call(app, args.name, arguments)
    print(json.dumps(answer, ensure_ascii=False, default=str))
    return 1 if answer.get("error") else 0


def _mcp_answer(app: Path, request: dict) -> Optional[dict]:
    method, ident = request.get("method"), request.get("id")
    if ident is None:
        return None  # a notification takes no answer
    if method == "initialize":
        result: Any = {"protocolVersion": MCP_PROTOCOL, "capabilities": {"tools": {}},
                       "serverInfo": {"name": "kullback-task", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": spec["name"], "description": spec.get("description") or "",
                             "inputSchema": spec.get("parameters") or {"type": "object"}}
                            for spec in tool_specs(app)]}
    elif method == "tools/call":
        params = request.get("params") or {}
        answer = call(app, str(params.get("name") or ""), dict(params.get("arguments") or {}))
        text = json.dumps(answer.get("error") or answer.get("result"), ensure_ascii=False, default=str)
        result = {"content": [{"type": "text", "text": text}], "isError": bool(answer.get("error"))}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": f"no method {method}"}}
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def mcp_main(app: Path = APP, stdin: Any = None, stdout: Any = None) -> int:
    """An MCP server over stdio, one JSON-RPC message per line, serving the same calls as `tools`."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    for line in stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except ValueError:
            continue
        answer = _mcp_answer(app, request) if isinstance(request, dict) else None
        if answer is not None:
            stdout.write(json.dumps(answer, ensure_ascii=False) + "\n")
            stdout.flush()
    return 0


def verify_main(argv: Optional[list[str]] = None, app: Path = APP) -> int:
    parser = argparse.ArgumentParser(prog="kullback-harbor-verify")
    parser.add_argument("verifier", type=Path)
    parser.add_argument("--logs", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args(argv)
    body = score(app, args.verifier, args.logs)
    print(json.dumps({"reward": body["reward"], "class": body["class"],
                      "unscored_gates": body["unscored_gates"], "unscored_reason": body.get("unscored_reason")}))
    return UNSCORED_EXIT if body.get("unscored_reason") else 0


def main(argv: Optional[list[str]] = None) -> int:
    """`python -m kullback.hub.harbor_world tools|mcp|verify ...`, the three entry points in the image."""
    argv = list(sys.argv[1:] if argv is None else argv)
    door = argv.pop(0) if argv else ""
    if door == "tools":
        return tools_main(argv)
    if door == "mcp":
        return mcp_main()
    if door == "verify":
        return verify_main(argv)
    print("usage: python -m kullback.hub.harbor_world tools|mcp|verify ...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
