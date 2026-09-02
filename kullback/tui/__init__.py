"""kullback's terminal view: one screen that shows a build while it is happening.

Why this exists, and why it is this small. Feynman is built on Pi, and the whole of the
difference is that Feynman does not add a better chat: it adds named workflows over a fixed
domain, and it keeps the provenance of what each one produced. That is the same shape as the
harness. The pipeline is already the named workflow, and the content-addressed cache is already
the provenance, so the screen has nothing to invent. It names the stages, shows which one is
running, and shows the hash each one wrote. Everything a general agent TUI carries and this one
does not (a chat pane, a tool picker, an approval prompt) is absent because the pipeline is the
conversation.

The two numbers a live build actually turns on are here and nowhere else in one place: what the
gates said, and what has been spent against the ceiling. Both are read from the files the build
writes, never from a variable this module keeps, so a screen that dies mid build loses nothing.
"""

from __future__ import annotations

import json
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

GLYPHS = {
    "k": ("#  #", "# # ", "##  ", "# # ", "#  #"),
    "u": ("#  #", "#  #", "#  #", "#  #", " ## "),
    "l": ("#   ", "#   ", "#   ", "#   ", "####"),
    "b": ("### ", "#  #", "### ", "#  #", "### "),
    "a": (" ## ", "#  #", "####", "#  #", "#  #"),
    "c": (" ###", "#   ", "#   ", "#   ", " ###"),
}

# The screen never invents a status. These are the words pipeline.py already writes into
# state.json, so a stage reads the same here as it does in the report.
MARKS = {
    "pending": ("·", "dim"), "start": ("▸", "yellow"), "ran": ("✔", "green"),
    "cached": ("✔", "cyan"), "rolled_back": ("↺", "yellow"), "failed": ("✘", "red"),
    "stopped": ("■", "magenta"), "crashed": ("✘", "red"),
}

HELP = """\
/build [--iterate] [--file PATH]   run the Builder over the ingested traces
/run TASK [--count N]              run the Candidate against the built Environment
/status                            the last build's stages, gates and spend
/keys                              which provider keys this shell can see
/help                              this
/quit                              leave\
"""


def banner(word: str = "kullback") -> Text:
    out = Text()
    for row in range(5):
        for letter in word:
            out.append(GLYPHS[letter][row].replace("#", "█"), style="bold")
            out.append(" ")
        out.append("\n")
    return out


def _as_dict_event(event: Any) -> Optional[dict]:
    """A typed stage event of the agent core as the dict the board reads; anything else is not for the board."""
    kind = getattr(event, "type", None)
    if kind == "stage_start":
        return {"kind": "stage", "stage": event.name, "state": "start", "attempt": 1}
    if kind == "stage_end":
        counts = dict(getattr(event, "counts", None) or {})
        return {"kind": "stage", "stage": event.name, "state": str(counts.get("status") or "ran"),
                "attempt": int(counts.get("attempts") or 1)}
    return None


@dataclass
class Board:
    """What the screen knows. Every field is filled from an event or from a file on disk."""

    workdir: Path
    title: str = ""
    order: list[str] = field(default_factory=list)
    status: dict[str, str] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    started: dict[str, float] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    gates: list[dict] = field(default_factory=list)
    outcome: str = ""
    ceiling: Optional[float] = None

    def event(self, event: Any) -> None:
        if not isinstance(event, dict):
            event = _as_dict_event(event)
            if event is None:
                return
        kind, stage = event.get("kind"), event.get("stage") or ""
        if kind == "gate":
            self.gates.append(event)
            return
        if kind == "pipeline":
            self.outcome = str(event.get("state") or "")
            return
        if stage not in self.order:
            self.order.append(stage)
        state = str(event.get("state") or "")
        self.status[stage] = state
        self.attempts[stage] = int(event.get("attempt") or 0)
        if state == "start":
            self.started[stage] = time.monotonic()
        elif stage in self.started:
            self.seconds[stage] = time.monotonic() - self.started[stage]

    def stages(self) -> Table:
        table = Table.grid(padding=(0, 2))
        table.add_column(width=1)
        table.add_column(min_width=16)
        table.add_column(justify="right", min_width=11)
        table.add_column(justify="right", min_width=6)
        for name in self.order:
            state = self.status.get(name, "pending")
            mark, colour = MARKS.get(state, ("·", "dim"))
            tries = self.attempts.get(name, 0)
            label = state if tries <= 1 else f"{state} ×{tries}"
            took = self.seconds.get(name)
            table.add_row(Text(mark, style=colour), Text(name, style=colour),
                          Text(label, style=colour), Text(f"{took:.1f}s" if took else "", style="dim"))
        return table

    def provenance(self) -> Table:
        """The cache file each stage wrote: the stage's name and the hash of what it produced.

        Read off the directory rather than carried in memory, because that directory is what a
        later build will actually reuse. A stage with no row here produced nothing to reuse.
        """
        table = Table.grid(padding=(0, 2))
        table.add_column(min_width=16)
        table.add_column(style="dim")
        newest: dict[str, Path] = {}
        for path in sorted((self.workdir / "cache").glob("*.json")):
            stage = path.name.split(".")[0]
            if stage not in newest or path.stat().st_mtime > newest[stage].stat().st_mtime:
                newest[stage] = path
        for name in self.order:
            path = newest.get(name)
            if path is not None:
                table.add_row(Text(name, style="cyan"), path.name.split(".")[1])
        return table

    def money(self) -> Text:
        """Spend, from the file budget.py writes on every priced call."""
        totals = _read(self.workdir / "budget.json", {}).get("total") or {}
        spent = float(totals.get("usd") or 0.0)
        out = Text(f"${spent:,.4f}", style="bold")
        if self.ceiling:
            out.append(f" of ${self.ceiling:,.2f} ceiling", style="dim")
        out.append(f"   {int(totals.get('calls') or 0)} calls", style="dim")
        out.append(f"   {int(totals.get('input') or 0):,} in / {int(totals.get('output') or 0):,} out",
                   style="dim")
        if totals.get("unpriced_calls"):
            out.append(f"   {int(totals['unpriced_calls'])} unpriced", style="yellow")
        return out

    def verdict(self) -> Text:
        failed = [g for g in self.gates if not g.get("passed")]
        out = Text()
        out.append(f"{len(self.gates) - len(failed)} gates passed", style="green")
        if failed:
            out.append(f", {len(failed)} failed", style="red")
            latest = failed[-1]
            reason = "; ".join(latest.get("failures") or []) or "no reason given"
            out.append(f"\n  {latest.get('stage')}: {reason[:160]}", style="red")
        if self.outcome:
            out.append(f"\n{self.outcome}", style="bold" if self.outcome == "complete" else "bold red")
        return out

    def render(self) -> Panel:
        body = Table.grid(padding=(0, 4))
        body.add_column()
        body.add_column()
        body.add_row(self.stages(), self.provenance())
        return Panel(Group(body, Text(""), self.money(), self.verdict()),
                     title=self.title, title_align="left", border_style="dim")


def _values(words: list[str], flag: str) -> list[str]:
    """Every value given to `flag`, in order. A flag with nothing after it, or with another flag
    after it, is a mistake the person typing can act on, so it is said in those words rather than
    surfacing as an IndexError from inside the build."""
    out = []
    for i, word in enumerate(words):
        if word != flag:
            continue
        if i + 1 >= len(words) or words[i + 1].startswith("--"):
            raise ValueError(f"{flag} needs a value after it")
        out.append(words[i + 1])
    return out


def _read(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def _keys(env: dict[str, str]) -> Text:
    """Which keys are visible, never what they are. A live run fails here first, so it is asked here."""
    out = Text()
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HARNESS_ALLOW_MODEL_REQUESTS"):
        value = env.get(name)
        shown = "set" if value else "missing"
        out.append(f"{name:<32}", style="dim")
        out.append(f"{shown}\n", style="green" if value else "red")
    return out


class Screen:
    """One console, one Board, and the small set of commands that drive the pipeline."""

    def __init__(self, workdir: Path, model: Optional[str] = None, base_url: Optional[str] = None,
                 console: Optional[Console] = None, runner: Any = None,
                 ceiling_usd: Optional[float] = None):
        self.workdir, self.model, self.base_url = Path(workdir), model, base_url
        self.ceiling_usd = ceiling_usd
        self.console = console or Console()
        self.runner = runner  # injected in tests; nothing here builds a live adapter

    def open(self) -> None:
        self.console.print(banner())
        # A long workdir path wrapping over three lines is the first thing you would see, so it
        # is cut rather than folded; the whole path is on the panel border of every build anyway.
        self.console.print(Text(f"  workdir {self.workdir}   model {self.model or 'none (no model calls)'}",
                                style="dim"), no_wrap=True, overflow="ellipsis")
        self.console.print(Text("  /help for commands\n", style="dim"))

    def command(self, line: str) -> bool:
        """One typed line. Returns False when the screen should close."""
        parts = shlex.split(line.strip())
        if not parts:
            return True
        verb, rest = parts[0].lstrip("/"), parts[1:]
        if verb in ("quit", "exit", "q"):
            return False
        if verb == "help":
            self.console.print(HELP)
        elif verb == "keys":
            self.console.print(_keys(dict(os.environ)))
        elif verb == "status":
            self._status()
        elif verb == "build":
            self._live("build", lambda emit: self._build(emit, rest))
        elif verb == "run":
            if not rest or rest[0].startswith("--"):
                self.console.print(Text("run needs a task id: /run TASK [--count N]", style="red"))
            else:
                self._live(f"run {rest[0]}", lambda emit: self._run(emit, rest))
        else:
            self.console.print(Text(f"no command {verb}; /help", style="red"))
        return True

    def _live(self, title: str, work: Any) -> None:
        board = Board(self.workdir, title=title, ceiling=self.ceiling_usd)
        with Live(board.render(), console=self.console, refresh_per_second=8) as live:
            def on_event(event: dict) -> None:
                board.event(event)
                live.update(board.render())
            try:
                work(on_event)
            except ValueError as exc:
                # No stage has started when the typed line itself was wrong, so it is a usage
                # message and not a build outcome (Greptile, PR 1).
                if board.order:
                    board.outcome = f"{type(exc).__name__}: {exc}"
                else:
                    board.outcome = f"{exc}   (/help for usage)"
            except Exception as exc:  # a failed build is a result to read, not a traceback to lose
                board.outcome = f"{type(exc).__name__}: {exc}"
            live.update(board.render())

    def _build(self, on_event: Any, rest: list[str]) -> None:
        from kullback.builder import agent as builder_agent
        files = [Path(value) for value in _values(rest, "--file")]
        (self.runner or builder_agent.run_builder)(workdir=self.workdir, iterate="--iterate" in rest,
                                                   model=self._adapter(), files=files, on_event=on_event,
                                                   ceiling_usd=self.ceiling_usd)

    def _run(self, on_event: Any, rest: list[str]) -> None:
        from kullback.builder import build as builder
        counts = _values(rest, "--count")
        try:
            count = int(counts[-1]) if counts else 1
        except ValueError:
            raise ValueError(f"--count takes a whole number, not {counts[-1]!r}") from None
        if count < 1:
            raise ValueError(f"--count takes a number of runs, not {count}")
        on_event({"kind": "stage", "stage": rest[0], "state": "start", "attempt": 1})
        (self.runner or builder.run_batch)(workdir=self.workdir, task_id=rest[0],
                                           model=self._adapter(), count=count,
                                           ceiling_usd=self.ceiling_usd)
        on_event({"kind": "stage", "stage": rest[0], "state": "ran", "attempt": 1})
        on_event({"kind": "pipeline", "state": "complete"})

    def _adapter(self) -> Any:
        """No model means no model. The screen refuses to guess one, the same as the CLI."""
        if self.model is None:
            return None
        from kullback.ai import provider
        return provider.live_model(self.model, self.base_url)

    def _status(self) -> None:
        """The last build, read back off disk. No stage runs to answer this."""
        state = _read(self.workdir / "pipeline" / "state.json", {})
        board = Board(self.workdir, title="last build")
        board.order = list(state.get("statuses") or {})
        board.status = dict(state.get("statuses") or {})
        board.attempts = dict(state.get("attempts") or {})
        # pipeline.py writes each GateResult with as_dict, and GateResult.passed carries the
        # alias "pass", so that is the key on disk; report.py reads the same file the same way.
        board.gates = [{"stage": g.get("stage"), "passed": g.get("pass", g.get("passed")),
                        "failures": g.get("failures") or []} for g in state.get("gates") or []]
        board.outcome = str(state.get("status") or "no build yet")
        self.console.print(board.render())


def loop(workdir: Path, model: Optional[str] = None, base_url: Optional[str] = None,
         ceiling_usd: Optional[float] = None) -> None:
    screen = Screen(workdir, model=model, base_url=base_url, ceiling_usd=ceiling_usd)
    screen.open()
    while True:
        try:
            line = screen.console.input("[bold]›[/bold] ")
        except (EOFError, KeyboardInterrupt):
            screen.console.print()
            return
        if not screen.command(line):
            return
