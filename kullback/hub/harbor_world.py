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
a gated item that cannot be scored (a judge item with no judge) gives 0 and is named.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

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


def call(app: Path, name: str, arguments: dict) -> dict:
    """Make one call: play the log and the call, append it with its answer, write the world back."""
    known = {spec["name"] for spec in tool_specs(app)}
    if name not in known:
        return {"result": None, "error": f"no tool named {name}"}
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


def score(app: Path, verifier_path: Path, logs: Path) -> dict:
    """Play the agent's log, score the Run, write reward.txt, reward.json and items.json under `logs`."""
    verifier = Verifier.model_validate(read_json(verifier_path, {}))
    with tempfile.TemporaryDirectory() as scratch:
        played = play(app, read_log(app), Path(scratch))
        finish(played)
        env = played.episode.env
        run = load_run(played.episode.run_file())
        flagged = {sig.name for sig in env.sigs if getattr(sig, "unclassified", False)}
        result = verdict(run, verifier, env.canon_rules, None, schema=env.schema,
                         write_tools=env.write_tools() or None, flagged_tools=flagged)
    closed = result.class_ == "env_error"
    body = reward_of(list(result.items), failed_closed=closed)
    body.update({"class": result.class_, "failing_atom": result.failing_atom,
                 "verdict_score": result.score, "notes": list(result.notes)})
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").write_text(f"{body['reward']}\n", encoding="utf-8")
    # Harbor reads reward.json before reward.txt and takes every value in it as a reward, so it
    # carries the number alone; the items go beside it, where the trial keeps them.
    write_json(logs / "reward.json", {"reward": body["reward"]})
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
                      "unscored_gates": body["unscored_gates"]}))
    return 0


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
