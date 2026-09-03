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

from kullback.tui import diagrams

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
/map                               the pipeline as a diagram: stages, states, hashes
/loop                              the loop as beats: Builder, gates, Examiner, round ends
/layers                            the kullback layering as a diagram
/keys                              which provider keys this shell can see
/login [provider/model] [--set KEY=VALUE ...] [--base-url URL]
                                  use this model, with keys held in memory only
/logout                            forget the keys set with /login
/help                              this
/quit                              leave\
"""


# The welcome gradient, Aura-style: dusty blue over teal, cream and pink into purple, one stop
# per letter left to right. Style only: banner().plain is unchanged, so no test reads a color.
GRADIENT = [(128, 159, 197), (149, 226, 227), (233, 213, 161), (239, 143, 172), (171, 112, 219)]


def _gradient_at(position: float) -> str:
    """A hex color on the gradient; 0.0 is the first letter, 1.0 the last."""
    position = min(1.0, max(0.0, position))
    scaled = position * (len(GRADIENT) - 1)
    low, high = int(scaled), min(len(GRADIENT) - 1, int(scaled) + 1)
    mix = scaled - low
    rgb = tuple(round(a + (b - a) * mix) for a, b in zip(GRADIENT[low], GRADIENT[high], strict=True))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def banner(word: str = "kullback") -> Text:
    out = Text()
    for row in range(5):
        for i, letter in enumerate(word):
            colour = _gradient_at(i / max(1, len(word) - 1))
            out.append(GLYPHS[letter][row].replace("#", "█"), style=f"bold {colour}")
            out.append(" ")
        out.append("\n")
    return out


def status_segments(workdir: Any, model: Optional[str]) -> Text:
    """The status line under the banner, Aura-style segments: model, live switch, spend, workdir.

    Everything is read, never asked: the live switch from the environment, the spend from the
    budget file the runner wrote (absent before the first build), the workdir cut, not folded."""
    try:
        from kullback.ai.provider import enable_live_calls_from_env
        live = enable_live_calls_from_env()
    except Exception:
        live = False
    try:
        from kullback.runner.budget import load_totals
        spent = float(load_totals(workdir)["total"].get("usd") or 0.0)
    except Exception:
        spent = 0.0
    out = Text()
    out.append(f"model {model or 'none (no model calls)'}", style="bold")
    out.append("  ·  live on" if live else "  ·  live off (no model call will be made)",
                 style="green" if live else "yellow")
    if spent > 0:
        out.append(f"  ·  spend ${spent:,.4f}", style="dim")
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


def _keys(env: dict[str, str], session: set[str] = frozenset()) -> Text:
    """Which keys are visible, never what they are. A live run fails here first, so it is asked here."""
    out = Text()
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HARNESS_ALLOW_MODEL_REQUESTS"):
        value = env.get(name)
        shown = "set" if value else "missing"
        if name in session and value:
            shown += " (this session)"
        out.append(f"{name:<32}", style="dim")
        out.append(f"{shown}\n", style="green" if value else "red")
    extra = sorted(session - {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HARNESS_ALLOW_MODEL_REQUESTS"})
    for name in extra:
        out.append(f"{name:<32}", style="dim")
        out.append("set (this session)\n", style="green")
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
        # Keys handed over with /login --set, mapped to what the shell held before (None
        # means it held nothing), so /logout restores the shell instead of just deleting.
        self.session_keys: dict[str, Optional[str]] = {}

    def open(self) -> None:
        self.console.print(banner())
        self.console.print(status_segments(self.workdir, self.model))
        # A long workdir path wrapping over three lines is the first thing you would see, so it
        # is cut rather than folded; the whole path is on the panel border of every build anyway.
        self.console.print(Text(f"  workdir {self.workdir}", style="dim"),
                                no_wrap=True, overflow="ellipsis")
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
            self.console.print(_keys(dict(os.environ), set(self.session_keys)))
        elif verb == "login":
            self._login(rest)
        elif verb == "logout":
            self._logout()
        elif verb == "status":
            self._status()
        elif verb == "map":
            self._map()
        elif verb == "loop":
            self._show_loop()
        elif verb == "layers":
            self.console.print(diagrams.layers_text())
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

    def _login(self, rest: list[str]) -> None:
        """Use this model from here on, with keys held in memory only.

        `/login` alone inspects: the current model, where its calls go, which variable
        holds its key and whether that variable is set. `/login provider/model` resolves
        the id the same way a build does (an adapter of its own, else the models.dev
        snapshot, else the --base-url given here) and refuses in words when nothing
        reaches it. `--set KEY=VALUE` puts keys into this process's environment so a
        pasted key works without touching .env or the shell; values are never printed
        and never written to the workdir, and /logout restores what the shell held.
        """
        sets = _values(rest, "--set")
        base_urls = _values(rest, "--base-url")
        model = next((word for word in rest if not word.startswith("--")), "")
        applied = []
        try:
            for item in sets:
                name, sep, value = item.partition("=")
                if not sep or not name:
                    raise ValueError(f"--set takes KEY=VALUE, not {item!r}")
                if name not in self.session_keys:
                    self.session_keys[name] = os.environ.get(name)
                os.environ[name] = value
                applied.append(name)
            if model:
                self._resolve(model, base_urls[-1] if base_urls else None)
                self.model = model
                if base_urls:
                    self.base_url = base_urls[-1]
        except ValueError as exc:
            self.console.print(Text(str(exc), style="red"))
            return
        if applied:
            self.console.print(Text(f"keys held for this session: {', '.join(applied)}", style="dim"))
        self.console.print(self._login_status())

    def _resolve(self, model: str, base_url: Optional[str]) -> None:
        """The id reaches a model, or the reason it does not. Assigns nothing; reports everything."""
        from kullback.ai import provider as pv

        provider_name, _ = pv.split_model_id(model)  # the 'provider/model' shape, or words saying so
        if provider_name in pv.ADAPTERS or base_url:
            return
        try:
            endpoint = pv.registry_endpoint(model)
        except Exception:
            endpoint = None
        if endpoint is None:
            raise ValueError(
                f"{model} has no adapter of its own and the models.dev snapshot names no host "
                f"for {provider_name!r}; pass --base-url")
        if not endpoint.openai_shaped:
            raise ValueError(
                f"models.dev serves {provider_name!r} through {endpoint.adapter}, which is not "
                f"the OpenAI request shape this Harness builds; pass --base-url for one that is")

    def _login_status(self) -> Text:
        """The current model, where its calls go, and whether its key is set. Names only, never values."""
        from kullback.ai import provider as pv

        out = Text()
        if not self.model:
            return Text("no model: /login provider/model to use one", style="dim")
        out.append(f"model {self.model}\n", style="bold")
        provider_name, _ = pv.split_model_id(self.model)
        key_var, host = "", self.base_url or ""
        adapter_cls = pv.ADAPTERS.get(provider_name)
        if adapter_cls is not None:
            key_var = adapter_cls.key_env_var
            host = host or "built-in adapter"
        else:
            try:
                endpoint = pv.registry_endpoint(self.model)
            except Exception:
                endpoint = None
            if endpoint is not None:
                key_var = endpoint.key_env_var
                host = host or endpoint.base_url
        if host:
            out.append(f"host {host}\n", style="dim")
        if key_var:
            out.append(f"{key_var:<32}", style="dim")
            out.append("set\n" if os.environ.get(key_var) else "missing\n",
                         style="green" if os.environ.get(key_var) else "red")
        else:
            out.append("no key variable: this endpoint takes none\n", style="dim")
        try:
            live = pv.enable_live_calls_from_env()
        except Exception:
            live = False
        out.append("live calls on" if live else "live calls off (no model call will be made)",
                   style="green" if live else "yellow")
        return out

    def _logout(self) -> None:
        """Forget the keys set with /login: the shell gets back exactly what it held."""
        for name, previous in self.session_keys.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        count = len(self.session_keys)
        self.session_keys.clear()
        self.console.print(Text(f"cleared {count} session key(s)" + (f"; model still {self.model}" if self.model else ""), style="dim"))

    def _adapter(self) -> Any:
        """No model means no model. The screen refuses to guess one, the same as the CLI."""
        if self.model is None:
            return None
        from kullback.ai import provider
        return provider.live_model(self.model, self.base_url)

    def _map(self) -> None:
        """The pipeline as a diagram, read back off disk like /status. Runs no stage."""
        state = _read(self.workdir / "pipeline" / "state.json", {})
        order = list(state.get("statuses") or {})
        self.console.print(diagrams.dag_text(order, dict(state.get("statuses") or {}),
                                             dict(state.get("attempts") or {}),
                                             diagrams.newest_hashes(self.workdir, order)))

    def _show_loop(self) -> None:
        """The loop as beats, read back off disk like /status. No rounds.json is the single-pass Builder."""
        self.console.print(diagrams.loop_text(diagrams.read_rounds_file(self.workdir)))

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
