"""Where a workdir stands: six rows a newcomer reads before anything else.

The entry screen and `kullback doctor` render these rows. Both read files and
variable names only, never secret values, and never call a model. Each row says
whether its step is done and the screen always ends on the one next step, so a
clean workdir answers "what now" without listing every command.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass
class Row:
    """One step of a workdir: done, and the detail a person acts on."""

    name: str
    done: bool
    detail: str
    next_key: str


def _dotenv_added() -> dict[str, str]:
    """The current directory's .env file as values, without touching the environment.

    Exactly the file provider.load_dotenv reads, parsed the same way, but into a
    fresh dict. A workdir's own .env does not count, because the builds never read
    it: only the current directory's .env reaches a model call.
    """
    from kullback.ai.provider import load_dotenv

    try:
        return load_dotenv(env={})
    except OSError:
        return {}


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _live_values(env: Mapping[str, str]) -> dict[str, str]:
    """What the live switch and the key rows read: the cwd .env under the passed environment.

    Exported values win, the way load_dotenv leaves them alone when it fills in
    the rest from the file. Presence only, values never reach a row.
    """
    merged = _dotenv_added()
    merged.update(env)
    return merged


def key_source(name: str, session: Mapping[str, Any],
               env: Optional[Mapping[str, str]] = None) -> str:
    """Where one key variable's value comes from: this session, .env, the remembered
    store, the environment, or nowhere.

    A session key wins because _apply_key overwrote the environment with it. Otherwise
    the first store that names it in provider order (exported environment, then .env,
    then the remembered store) is its source, unless the shell overrode that store
    with another value, in which case the environment is. Names only, never values.
    Reads env, the process environment when not given, so a caller passing its own
    merged mapping gets that mapping's source. The screen imports this for /login.
    """
    values = os.environ if env is None else env
    value = values.get(name)
    if name in session and value is not None:
        return "this session"
    if not value:
        return "missing"
    dotted = _dotenv_added()
    if name in dotted:
        return ".env" if dotted[name] == value else "environment"
    try:
        from kullback.ai import credentials

        stored = credentials.load_credentials({})
    except Exception:
        stored = {}
    if name in stored:
        return "auth.json" if stored[name] == value else "environment"
    return "environment"


def _key_groups(model: str) -> tuple[tuple[str, ...], ...]:
    """The key variable groups a model reads, or nothing when it names none.

    One place beside the screen's own _credential_source, reached lazily so
    this module never imports its parent at load time. An id no resolver
    reaches names no group, which reads as a model row that is not done.
    """
    from kullback.tui import _credential_source

    try:
        groups, _ = _credential_source(model, "")
    except ValueError:
        return ()
    return groups


def _describe_groups(groups: tuple[tuple[str, ...], ...]) -> str:
    """The groups in words: one group fully set is enough, so groups join with or."""
    return " or ".join(" and ".join(group) for group in groups)


def where_it_stands(workdir: Path, env: Mapping[str, str], model: str,
                    session: Optional[Mapping[str, Any]] = None) -> list[Row]:
    """Six rows for a workdir, in the order a newcomer works through them.

    Reads ingest_summary.json, rounds.json and runner_version.json off the
    workdir and variable names off env over the current directory's .env, the
    same two places a build reads them from. Names only: a set variable is
    named with where it came from (this session, .env, auth.json or the
    environment), its value never appears. Session maps a /login key to what
    the shell held before, so the screen's held keys read as this session.
    """
    workdir = Path(workdir)
    key_env = _live_values(env)
    held: Mapping[str, Any] = session if session is not None else {}
    rows: list[Row] = []

    from kullback.ai.provider import LIVE_ENV_VAR

    groups = _key_groups(model) if model else ()
    set_groups = [group for group in groups if group and all(key_env.get(var) for var in group)]
    if set_groups:
        shown = set_groups[0]
        source = key_source(shown[0], held, env=key_env)
        rows.append(Row("model", True, f"{' and '.join(shown)} set ({source})", "/login"))
    elif groups:
        rows.append(Row("model", False, f"needs {_describe_groups(groups)}", "/login"))
    else:
        rows.append(Row("model", False, f"{model or 'no model'} names no key variable", "/login"))

    if _truthy(key_env.get(LIVE_ENV_VAR, "")):
        source = key_source(LIVE_ENV_VAR, held, env=key_env)
        rows.append(Row("live calls", True, f"on ({LIVE_ENV_VAR}=1 in {source})", "/login"))
    else:
        rows.append(Row("live calls", False, f"off: put {LIVE_ENV_VAR}=1 in .env or export it", "/login"))

    try:
        summaries = json.loads((workdir / "ingest_summary.json").read_text(encoding="utf-8"))
        summaries = summaries if isinstance(summaries, list) else None
    except (OSError, ValueError):
        summaries = None
    if summaries:
        files = len(summaries)
        runs = sum(int(row.get("runs") or 0) for row in summaries if isinstance(row, dict))
        word = "file" if files == 1 else "files"
        trace_word = "trace" if runs == 1 else "traces"
        rows.append(Row("traces", True, f"{files} {word}, {runs} {trace_word}", "kullback ingest"))
    else:
        rows.append(Row("traces", False, "none yet", "kullback ingest"))

    try:
        rounds = json.loads((workdir / "rounds.json").read_text(encoding="utf-8"))
        rounds = [row for row in rounds if isinstance(row, dict)] if isinstance(rounds, list) else []
    except (OSError, ValueError):
        rounds = []
    if rounds:
        last = rounds[-1]
        counts = last.get("counts") if isinstance(last.get("counts"), dict) else {}
        trusted_ids = counts.get("trusted_ids")
        trusted = len(trusted_ids) if isinstance(trusted_ids, list) else int(counts.get("trusted") or 0)
        detail = f"round {last.get('round', len(rounds))}: trusted {trusted}"
        if counts.get("fidelity") is not None or counts.get("tasks") is not None:
            detail += f", fidelity {counts.get('fidelity', 0)}/{counts.get('tasks', 0)}"
        rows.append(Row("build", True, detail, "/build"))
    else:
        rows.append(Row("build", False, "not started", "/build"))

    try:
        version_body = json.loads((workdir / "runner_version.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        version_body = None
    if isinstance(version_body, dict):
        version = str(version_body.get("runner_version") or "")
        rows.append(Row("runner", True, f"frozen {version[:12]}" if version else "frozen",
                        "kullback freeze-runner"))
    elif (workdir / "runner_version.json").is_file():
        rows.append(Row("runner", True, "frozen", "kullback freeze-runner"))
    else:
        rows.append(Row("runner", False, "not frozen", "kullback freeze-runner"))

    rows.append(Row("publish", False, "needs fidelity 0.90 over Tasks", "kullback publish"))
    return rows


def next_step(rows: list[Row]) -> str:
    """The first row not done, as one sentence with the command to run."""
    from kullback.ai.provider import LIVE_ENV_VAR

    for row in rows:
        if row.done:
            continue
        if row.name == "model":
            return "run /login to choose a model and set its key."
        if row.name == "live calls":
            return f"put {LIVE_ENV_VAR}=1 in .env or export it to turn live calls on."
        if row.name == "traces":
            return "run kullback ingest with a customer export file to bring traces."
        if row.name == "build":
            return "run /build to start the Builder over the ingested traces."
        if row.name == "runner":
            return "run kullback freeze-runner to freeze the Runner."
        return "reach fidelity 0.90 over Tasks, then run kullback publish."
    return "every row is done."
