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
from typing import Mapping


@dataclass
class Row:
    """One step of a workdir: done, and the detail a person acts on."""

    name: str
    done: bool
    detail: str
    next_key: str


def _dotenv_values(path: Path) -> dict[str, str]:
    """The KEY=VALUE lines of one .env file, without touching the environment.

    Same line rules as provider.load_dotenv (blank lines and comments skipped,
    optional export prefix and matching quotes stripped), but read only: the
    checklist names where a variable came from without changing what is set.
    """
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in out:
            out[key] = value
    return out


def _dotenv_keys(workdir: Path) -> set[str]:
    """The variable names set in a .env file: the caller's cwd or the workdir.

    Live calls are turned on from the shell or from a .env file in either
    place, so both count and the detail can say which one held the variable.
    """
    return set(_dotenv_values(Path(".env"))) | set(_dotenv_values(Path(workdir) / ".env"))


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _live_values(env: Mapping[str, str], workdir: Path) -> dict[str, str]:
    """What the live switch reads: the passed environment over both .env files.

    Exported values win, the way load_dotenv leaves them alone when it fills
    in the rest from the file.
    """
    merged = dict(_dotenv_values(Path(".env")))
    merged.update(_dotenv_values(Path(workdir) / ".env"))
    merged.update(env)
    return merged


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


def _source_of(variable: str, dotenv_keys: set[str]) -> str:
    """Where a set variable came from: the shell's environment or a .env file."""
    if variable in os.environ:
        return "environment"
    if variable in dotenv_keys:
        return ".env"
    return "environment"


def _describe_groups(groups: tuple[tuple[str, ...], ...]) -> str:
    """The groups in words: one group fully set is enough, so groups join with or."""
    return " or ".join(" and ".join(group) for group in groups)


def where_it_stands(workdir: Path, env: Mapping[str, str], model: str) -> list[Row]:
    """Six rows for a workdir, in the order a newcomer works through them.

    Reads ingest_summary.json, rounds.json and runner_version.json off the
    workdir and variable names off env and the .env files. Names only: a set
    variable is named with where it came from, its value never appears.
    """
    workdir = Path(workdir)
    dotenv_keys = _dotenv_keys(workdir)
    live = _live_values(env, workdir)
    rows: list[Row] = []

    from kullback.ai.provider import LIVE_ENV_VAR

    groups = _key_groups(model) if model else ()
    # A key in a .env file counts as set: the builds find it through load_dotenv.
    # Presence only, the value never reaches a row.
    key_env = _live_values(env, workdir)
    set_groups = [group for group in groups if group and all(key_env.get(var) for var in group)]
    if set_groups:
        shown = set_groups[0]
        source = _source_of(shown[0], dotenv_keys)
        rows.append(Row("model", True, f"{' and '.join(shown)} set ({source})", "/login"))
    elif groups:
        rows.append(Row("model", False, f"needs {_describe_groups(groups)}", "/login"))
    else:
        rows.append(Row("model", False, f"{model or 'no model'} names no key variable", "/login"))

    if _truthy(live.get(LIVE_ENV_VAR, "")):
        source = ".env" if LIVE_ENV_VAR in dotenv_keys and LIVE_ENV_VAR not in os.environ else "environment"
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
