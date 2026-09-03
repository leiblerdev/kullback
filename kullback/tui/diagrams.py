"""Diagrammatic views of the kullback build: the pipeline as a flow, the loop as beats, the layering as a stack.

Why a separate module. `kullback/tui/__init__.py` owns the live Board and the commands; this module
owns pure rendering over plain data (stage order and states, round records, nothing else), so the
diagrams work on any tree: the single-pass Builder with only `pipeline/state.json` on disk, and the
phase-5 loop with `rounds.json` beside it. No import of `builder`, `rounds`, `examiner` or `gates`
lives here, which is also what keeps the TUI importable in tests without a workdir or a model.

The mark table duplicates `MARKS` in `__init__` on purpose: importing it would make this module
depend on the Screen, and the Screen depends on this module for rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from rich.text import Text

MARKS = {
    "pending": ("·", "dim"), "start": ("▸", "yellow"), "ran": ("✔", "green"),
    "cached": ("✔", "cyan"), "rolled_back": ("↺", "yellow"), "failed": ("✘", "red"),
    "stopped": ("■", "magenta"), "crashed": ("✘", "red"),
}

LAYERS = (
    ("builder", "examiner", "the two agents (extensions on the agent core)"),
    ("gates", None, "code no agent can write"),
    ("agent", "runner", "the loop and the frozen loop"),
    ("ai", None, "providers, stream, pricing"),
)


def newest_hashes(workdir: Any, order: list[str]) -> dict[str, str]:
    """The content hash each stage wrote, by stage name; a stage with no cache file wrote nothing to reuse."""
    try:
        files = sorted(Path(workdir, "cache").glob("*.json"))
    except OSError:
        return {}
    newest: dict[str, Path] = {}
    for path in files:
        stage = path.name.split(".")[0]
        try:
            if stage not in newest or path.stat().st_mtime > newest[stage].stat().st_mtime:
                newest[stage] = path
        except OSError:
            continue
    out = {}
    for name in order:
        path = newest.get(name)
        if path is not None:
            out[name] = path.name.split(".")[1] if "." in path.name else ""
    return out


def read_rounds_file(workdir: Any) -> list[dict]:
    """The round records a workdir recorded, oldest first; none when the loop has not run here.

    Reads `rounds.json` (phase 5 writes one RoundRecord per round) as plain dicts, so this works
    without importing the records. Anything unreadable is no rounds, not an error: `/loop` before
    the first round is a normal screen state.
    """
    try:
        body = json.loads(Path(workdir, "rounds.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = body if isinstance(body, list) else []
    return [dict(row) for row in rows if isinstance(row, dict)]


def dag_text(order: list[str], status: dict[str, str], attempts: Optional[dict[str, int]] = None,
             hashes: Optional[dict[str, str]] = None) -> Text:
    """The pipeline as a downward flow: one box per stage, a ▼ between stages, provenance on the right."""
    out = Text()
    if not order:
        return Text("no stages yet — /build to start, /status for the last build", style="dim")
    attempts, hashes = attempts or {}, hashes or {}
    inner = max(len(name) + len(_state_label(status.get(name, "pending"), attempts.get(name, 0))) + 12
                for name in order)
    inner = max(inner, 30)
    for i, name in enumerate(order):
        state = status.get(name, "pending")
        mark, colour = MARKS.get(state, ("·", "dim"))
        label = _state_label(state, attempts.get(name, 0))
        digest = (hashes.get(name) or "")[:8]
        out.append(f"┌─ {name} " + "─" * max(1, inner - len(name) - 3) + "┐\n", style="dim")
        row = f"│ {mark} {label}"
        out.append(row, style=colour)
        pad = inner - len(row) + (len(digest) + 1 if digest else 0)
        if digest:
            out.append(" " * max(1, pad - len(digest) - 1))
            out.append(digest, style="dim")
        else:
            out.append(" " * max(1, pad))
        out.append("│\n", style="dim")
        out.append("└" + "─" * (inner + 1) + "┘", style="dim")
        if i < len(order) - 1:
            out.append("\n" + " " * (inner // 2) + "▼\n", style="dim")
    return out


def _state_label(state: str, tries: int) -> str:
    return state if tries <= 1 else f"{state} ×{tries}"


def loop_text(rounds: list[dict], current_round: int = 0, agent: str = "") -> Text:
    """The loop as beats: Builder beat, gates, Examiner beat, round_end, with the counts the gates reported."""
    out = Text()
    if not rounds and not current_round:
        return Text("single-pass Builder — no rounds.json yet (the loop writes one record per round)",
                    style="dim")
    for row in rounds:
        n = row.get("round", "?")
        counts = dict(row.get("counts") or {})
        out.append(f"round {n} ── [builder beat] ── gates ──▶ [examiner beat] ──▶ round_end\n", style="bold")
        out.append(f"  {_counts_line(counts)}\n", style="dim")
        pending = row.get("pending_findings") or []
        if row.get("exit"):
            style = "bold red" if (row["exit"] != "done" or row.get("failed")) else "bold"
            out.append(f"  exit: {row['exit']}" + (" (failed)" if row.get("failed") else "") + "\n",
                         style=style)
        elif pending:
            out.append(f"  {len(pending)} finding(s) owed the Builder a beat — the round continued\n",
                         style="yellow")
        if row.get("exit_note"):
            out.append(f"  note: {row['exit_note']}\n", style="dim")
    if current_round and (not rounds or current_round != rounds[-1].get("round")):
        beat = f", {agent} beat ●" if agent else ""
        out.append(f"round {current_round}{beat}\n", style="yellow")
    return out


def _counts_line(counts: dict) -> str:
    spend = counts.get("spend") or {}
    try:
        dollars = float(spend.get("total") or 0.0)
    except (TypeError, ValueError):
        dollars = 0.0
    return (f"fidelity {counts.get('fidelity', 0)}/{counts.get('tasks', 0)} · "
            f"trusted {counts.get('trusted', 0)} · refused {counts.get('refused_count', 0)} · "
            f"probes passing {counts.get('probes_passing', 0)} · spend ${dollars:,.4f}")


def layers_text() -> Text:
    """The kullback layering from `pyproject.toml`'s import-linter contracts, top (agents) to bottom (providers)."""
    out = Text()
    width = 34
    for i, (left, right, note) in enumerate(LAYERS):
        edge = ("┌", "┐") if i == 0 else (("└", "┘") if i == len(LAYERS) - 1 else ("├", "┤"))
        title = left if right is None else f"{left} + {right}"
        line = f"{edge[0]} {title} " + "─" * max(1, width - len(title) - 3) + edge[1]
        out.append(line + f"  {note}\n", style="cyan" if i == 0 else ("yellow" if i == 1 else "dim"))
    out.append("build ──▶ gates ──▶ examine ──▶ round_end ──▶ follow-ups ──▶ build …\n", style="dim")
    out.append("(one agent holds the stream at a time; findings reach the Builder as follow-ups)", style="dim")
    return out
