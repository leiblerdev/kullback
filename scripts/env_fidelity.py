"""How close is the Environment we built to the real one?

`xdomain_check.py` answers that for the parts a build produces without a model: the mined tool
kinds, the mined tables, the Starting state. This answers it for the part only a live build
produces, which is the tool bodies the model wrote.

The measure is behaviour, not source. Two implementations of `get_order_details` that share no
line can be the same tool, and two that read alike can differ on the one order that is cancelled.
So both sides are handed the same seed database and the same recorded call, and what is compared
is what came back and what changed underneath.

The seed database is the real one on both sides on purpose. Our own Starting state is mined from
the traces and is scored separately; running our tools on it here would fold two different errors
into one number and we would not know which we were looking at.

The reference runs in its own process under its own dependencies (scripts/tau2_reference.py),
because tau2 needs litellm and pandas and the harness needs three packages.

    uv run python scripts/env_fidelity.py .work-retail retail --per-tool 25
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from xdomain_check import VENDOR, norm  # noqa: E402

from kullback.builder import compile_env  # noqa: E402
from kullback.builder.build import CANON_RULES  # noqa: E402
from kullback.gates.confinement import source_confinement  # noqa: E402
from kullback.runner import canon  # noqa: E402
from kullback.runner.records import EntitySchema, ToolSig  # noqa: E402

REFERENCE = REPO / "scripts" / "tau2_reference.py"


def _read(path: Path, fallback: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def seed_blob(domain: str) -> dict:
    """The seed file as tau2's own tools want it, table shapes untouched."""
    import tomllib
    folder = VENDOR / "data" / "tau2" / "domains" / domain
    if (folder / "db.json").exists():
        return json.loads((folder / "db.json").read_text(encoding="utf-8"))
    return tomllib.loads((folder / "db.toml").read_text(encoding="utf-8"))


def recorded_calls(workdir: Path, per_tool: int) -> dict[str, list[dict]]:
    """Up to `per_tool` recorded calls of each tool, newest file first, deduplicated by arguments.

    Deduplicated because a corpus of 456 simulations asks for the same user forty times, and forty
    identical calls measure one thing forty times over.
    """
    out: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, set] = defaultdict(set)
    for path in sorted((workdir / "traces").glob("*.json")):
        trace = _read(path, {}) or {}
        for call in trace.get("tool_calls") or []:
            name = call.get("name")
            if not name or len(out[name]) >= per_tool:
                continue
            key = json.dumps(call.get("args") or {}, sort_keys=True, default=str)
            if key in seen[name]:
                continue
            seen[name].add(key)
            out[name].append(call)
    return dict(out)


class Reference:
    """The real toolkit, one call at a time, over a pipe."""

    def __init__(self, domain: str, venv_python: Path):
        env = {"PYTHONPATH": str(VENDOR / "src"), "PATH": "/usr/bin:/bin"}
        self.proc = subprocess.Popen(
            [str(venv_python), str(REFERENCE), domain],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, env=env)
        ready = json.loads(self.proc.stdout.readline())
        self.tables = ready.get("tables") or []

    def call(self, tool: str, args: dict) -> dict:
        self.proc.stdin.write(json.dumps({"tool": tool, "args": args}, default=str) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def ours(toolkit_source: str, blob: dict, tool: str, args: dict) -> dict:
    """One call against the body the model wrote, on a fresh copy of the same seed database."""
    db = json.loads(json.dumps(blob, default=str))
    toolkit = compile_env.load_toolkit(toolkit_source, db)
    # The state to watch is the toolkit's, not the dict handed in: load_toolkit validates that dict
    # into a pydantic model, so a write lands on the model and the dict never moves. Reading the
    # dict made every write tool look like it wrote nothing, which is a bug in this file and not a
    # finding about the harness.
    def state():
        return json.dumps(plain(toolkit.db.model_dump(mode="json")), sort_keys=True, default=str)

    before = state()
    out: dict[str, Any] = {"tool": tool}
    fn = getattr(toolkit, tool, None)
    if fn is None:
        out["error"] = "NotBuilt: the build produced no body for this tool"
        out["changed"] = False
        return out
    try:
        out["result"] = fn(**(args or {}))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["changed"] = state() != before
    return out


def plain(value: Any) -> Any:
    """Both sides return whatever their own data model returns, and one of them is pydantic.
    A model and the dict it dumps to are the same answer, so both are reduced to JSON first."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def _payload(value: Any) -> Any:
    """A tool that answers with a JSON string and one that answers with the object are the same
    tool. tau2's own tools return both shapes, so the string is opened before comparing."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "[{":
            try:
                return json.loads(text)
            except ValueError:
                return value
    return value


def verdict(mine: dict, real: dict, rules: Any) -> str:
    """One word for one call.

    `error` and `result` are kept apart on purpose: a tool that raises where the real one answers
    is a different failure from one that answers differently, and folding them together would hide
    the first behind the second.
    """
    if "error" in mine and "error" in real:
        # Both refusing is agreement about behaviour but not necessarily about what the customer
        # is told, and the message is part of the environment. Kept apart so the headline number
        # cannot be flattered by a message our body copied out of the transport rather than the
        # tool: the traces carry tau2's own "Error: " prefix and a body that repeats it is wrong.
        a, b = str(mine["error"]).split(": ", 1)[-1], str(real["error"]).split(": ", 1)[-1]
        return "both_error" if a.strip() == b.strip() else "both_error_other_message"
    if "error" in mine:
        return "only_ours_errored"
    if "error" in real:
        return "only_theirs_errored"
    a = canon.canonicalize(norm(plain(_payload(mine.get("result")))), rules)
    b = canon.canonicalize(norm(plain(_payload(real.get("result")))), rules)
    if a != b:
        return "result_differs"
    if bool(mine.get("changed")) != bool(real.get("changed")):
        return "effect_differs"
    return "same"



# One word for why a call missed, so the headline splits by cause and not only by tool. The
# verdict says what was observed; the cause says which part of the Builder owns it.
CAUSE_OWNER = {
    "confinement": "compile_tools: the body used getattr, __dict__ or a denied builtin",
    "missing_import": "compile_tools: the body used a module it never imported",
    "row_access": "compile_tools: the body treated a pydantic row as a dict",
    "result_shape": "compile_tools: same value, different wrapping (scalar boxed, key renamed)",
    "error_prefix": "compile_tools: the body copied the transport's error prefix into the message",
    "error_message": "compile_tools: both refused, the wording differs",
    "schema_shape": "mine: our table layout differs from the real one, so a real row is not found (retail: items nested under products.variants, mined since D106)",
    "body_error": "compile_tools: the body raised where the real tool answered",
    "value": "compile_tools: a different answer with no shape or error explanation",
    "effect": "compile_tools: same answer, different state change",
    "real_errored": "reference: the real tool refused where ours answered",
}


def _peel(text: str) -> str:
    """Drop every leading `Word: ` prefix an error message carries (exception name, transport)."""
    parts = str(text).split(": ")
    while len(parts) > 1 and parts[0].strip().replace("_", "").isalnum():
        parts = parts[1:]
    return ": ".join(parts).strip()


def cause(word: str, mine: dict, real: dict) -> str:
    if word == "not_confined":
        return "confinement"
    if word == "same" or word == "both_error":
        return "none"
    if word == "both_error_other_message":
        return "error_prefix" if _peel(mine.get("error")) == _peel(real.get("error")) else "error_message"
    if word == "only_ours_errored":
        error = str(mine.get("error") or "")
        if error.startswith("NameError"):
            return "missing_import"
        if error.startswith("AttributeError"):
            return "row_access"
        if "not found" in error.lower():
            return "schema_shape"
        return "body_error"
    if word == "only_theirs_errored":
        return "real_errored"
    if word == "result_differs":
        ours, theirs = plain(_payload(mine.get("result"))), plain(_payload(real.get("result")))
        if isinstance(ours, dict) and len(ours) == 1 and not isinstance(theirs, dict) and \
                str(next(iter(ours.values()))) == str(theirs):
            return "result_shape"
        if isinstance(ours, dict) and isinstance(theirs, dict) and \
                sorted(map(str, ours.values())) == sorted(map(str, theirs.values())):
            return "result_shape"
        if not isinstance(ours, dict) and not isinstance(theirs, dict) and str(ours) == str(theirs):
            return "result_shape"
        return "value"
    if word == "effect_differs":
        return "effect"
    return "value"


def check(workdir: Path, domain: str, venv_python: Path, per_tool: int) -> dict:
    schema = EntitySchema.model_validate(_read(workdir / "schema.json", {}) or {})
    sigs = [ToolSig.model_validate(s) for s in _read(workdir / "tool_sigs.json", []) or []]
    bodies = _read(workdir / "bodies.json", {}) or {}
    if not bodies:
        raise SystemExit(f"{workdir}/bodies.json is empty; run a build with a model first")
    rules = canon.load_rules(workdir / CANON_RULES)
    blob, calls = seed_blob(domain), recorded_calls(workdir, per_tool)
    assisted = set(_read(workdir / "environment.json", {}).get("assisted_tools") or [])

    # A body the confinement gate refuses cannot be loaded at all, and load_toolkit refuses the
    # whole module for one of them, which is the right answer for the Runner and the wrong answer
    # for a measurement. The refused tools are named and set aside; the rest are still scored,
    # because "we could not load it" and "it answered differently" are different findings and
    # folding them together would hide however many of each there are.
    full = compile_env.module_source(schema, sigs, bodies)
    unconfined = {line.split(" ", 1)[0]: line for line in source_confinement(full)}
    source = compile_env.module_source(
        schema, [g for g in sigs if g.name not in unconfined],
        {name: body for name, body in bodies.items() if name not in unconfined})

    reference = Reference(domain, venv_python)
    per_tool_rows, totals = {}, Counter()
    causes: dict[str, Counter] = defaultdict(Counter)
    examples: list[dict] = []
    try:
        for tool in sorted(set(bodies) | set(calls)):
            counts = Counter()
            if tool in unconfined:
                n = len(calls.get(tool, []))
                counts["not_confined"] = n
                totals["not_confined"] += n
                causes["confinement"][tool] += n
                per_tool_rows[tool] = {"calls": n, "assisted": tool in assisted,
                                       "why": unconfined[tool], **counts}
                continue
            for call in calls.get(tool, []):
                args = call.get("args") or {}
                mine, real = ours(source, blob, tool, args), reference.call(tool, args)
                word = verdict(mine, real, rules)
                why = cause(word, mine, real)
                counts[word] += 1
                totals[word] += 1
                if why != "none":
                    causes[why][tool] += 1
                if word != "same" and len(examples) < 40:
                    examples.append({"tool": tool, "args": args, "verdict": word, "cause": why,
                                     "ours": str(mine.get("error") or mine.get("result"))[:300],
                                     "real": str(real.get("error") or real.get("result"))[:300]})
            per_tool_rows[tool] = {"calls": sum(counts.values()), "assisted": tool in assisted,
                                   **counts}
    finally:
        reference.close()

    scored = sum(totals.values()) - totals["not_confined"]
    recorded = sum(totals.values())
    agree = totals["same"] + totals["both_error"]
    by_cause = {why: {"calls": sum(tools.values()), "tools": dict(sorted(tools.items()))}
                for why, tools in sorted(causes.items(), key=lambda kv: -sum(kv[1].values()))}
    return {"domain": domain, "workdir": str(workdir), "tools": len(bodies),
            "assisted": sorted(assisted), "calls_scored": scored, "calls_recorded": recorded,
            "agreement": round(agree / scored, 4) if scored else 0.0,
            "agreement_all": round(agree / recorded, 4) if recorded else 0.0,
            "totals": dict(totals), "by_cause": by_cause,
            "per_tool": per_tool_rows, "examples": examples}


def report(result: dict) -> str:
    lines = [f"# Environment fidelity: {result['domain']}", "",
             f"{result['tools']} tools built, {len(result['assisted'])} assisted. "
             f"{result['calls_scored']} recorded calls replayed against the real toolkit "
             f"on the real seed database.", "",
             f"**Agreement: {result['agreement']:.1%}** "
             f"({result['totals'].get('same', 0)} same, "
             f"{result['totals'].get('both_error', 0)} both refused; "
             f"{result['totals'].get('not_confined', 0)} calls belong to tools the confinement "
             f"gate refused to load and are not scored)", "",
             "| tool | calls | same | both refused | result differs | effect differs "
             "| only ours failed | only theirs failed | other message | not confined | assisted |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for tool, row in sorted(result["per_tool"].items(),
                            key=lambda kv: (-(kv[1].get("result_differs", 0)
                                              + kv[1].get("only_ours_errored", 0)), kv[0])):
        lines.append(f"| `{tool}` | {row['calls']} | {row.get('same', 0)} "
                     f"| {row.get('both_error', 0)} | {row.get('result_differs', 0)} "
                     f"| {row.get('effect_differs', 0)} | {row.get('only_ours_errored', 0)} "
                     f"| {row.get('only_theirs_errored', 0)} "
                     f"| {row.get('both_error_other_message', 0)} "
                     f"| {row.get('not_confined', 0)} | {'yes' if row['assisted'] else ''} |")
    by_cause = result.get("by_cause") or {}
    if by_cause:
        recorded = result.get("calls_recorded") or 1
        lines += ["", f"Over every recorded call, refused tools counted as misses: "
                      f"**{result.get('agreement_all', 0):.1%}**", "",
                  "## Where the misses come from", "",
                  "| cause | calls | share of recorded | tools | owner |", "|---|---:|---:|---|---|"]
        for why, row in by_cause.items():
            tools = ", ".join(f"`{t}` {n}" for t, n in row["tools"].items())
            lines.append(f"| {why} | {row['calls']} | {row['calls'] / recorded:.1%} | {tools} "
                         f"| {CAUSE_OWNER.get(why, '')} |")
    if result["examples"]:
        lines += ["", "## Where they part", ""]
        for item in result["examples"][:15]:
            lines += [f"**`{item['tool']}`** ({item['verdict']}, {item.get('cause', '')}) "
                      f"`{json.dumps(item['args'])[:160]}`",
                      f"- ours: `{item['ours']}`", f"- real: `{item['real']}`", ""]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("domain", choices=["retail", "airline", "telecom"])
    parser.add_argument("--per-tool", type=int, default=25)
    parser.add_argument("--venv", type=Path, required=True,
                        help="Python of a venv with tau2's dependencies installed.")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = check(args.workdir, args.domain, args.venv, args.per_tool)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
