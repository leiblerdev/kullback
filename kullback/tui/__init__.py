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
/build [--iterate] [--file PATH]   run the Builder and the Examiner in rounds over the ingested traces
/run TASK [--count N]              run the Candidate against the built Environment
/status                            the last build's stages, gates, rounds and spend
/map                               the pipeline as a diagram: stages, states, hashes
/loop                              the loop as beats: Builder, gates, Examiner, round ends
/layers                            the kullback layering as a diagram
/sessions                          builds running now and before, and which to watch
/watch N                           watch session N from the sessions list
/keys                              which provider keys this shell can see
/login [provider/model] [--set KEY=VALUE ...] [--base-url URL]
                                  use this model, with keys held in memory only
/logout                            forget the keys set with /login
/help                              this
/quit                              leave\
"""


# The welcome gradient, in the brand: leibler.dev is black with white text, and its only
# gradient is white fading into gray (site.css :root: --fg #fff, --fg-2 #a3a3a3, --fg-3
# #8a8a8a). White into gray left to right, one stop per letter. Style only: banner().plain
# is unchanged, so no test reads a color.
GRADIENT = [(255, 255, 255), (163, 163, 163), (138, 138, 138)]

# What kullback is, in one line on the entry screen. The leibler.dev wording, shortened:
# the harness that rebuilds your environment from traces and checks the rebuild by replay.
TAGLINE = "Rebuilds your environment from traces, checks the rebuild by replay, and grades any model inside it."

# Every command in one table: name, usage, what it does. The entry screen and the / menu
# are rendered from this, so a command added here appears in both; HELP stays a literal
# beside it, and a test fails when a table name is missing from HELP, so the two cannot drift.
COMMANDS = [
    ("build", "/build [--iterate] [--file PATH]", "run the Builder over the ingested traces"),
    ("run", "/run TASK [--count N]", "run the Candidate against the built Environment"),
    ("status", "/status", "the last build's stages, gates, rounds and spend"),
    ("map", "/map", "the pipeline as a diagram: stages, states, hashes"),
    ("loop", "/loop", "the loop as beats: Builder, gates, Examiner, round ends"),
    ("layers", "/layers", "the kullback layering as a diagram"),
    ("sessions", "/sessions", "builds running now and before, and which to watch"),
    ("watch", "/watch N", "watch session N from the sessions list"),
    ("keys", "/keys", "which provider keys this shell can see"),
    ("login", "/login [provider/model] [--set KEY=VALUE ...] [--base-url URL]",
     "use this model, with keys held in memory only"),
    ("logout", "/logout", "forget the keys set with /login"),
    ("help", "/help", "this"),
    ("quit", "/quit", "leave"),
]


def filter_commands(fragment: str) -> list[tuple[str, str, str]]:
    """The / menu's matches: commands whose name holds the fragment, in table order.

    Pure, so the menu is tested without a console: what you see when you type /frag."""
    needle = fragment.strip().lower().lstrip("/")
    return [row for row in COMMANDS if needle in row[0].lower()]


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
    """A typed stage, round or beat event of the agent core as the dict the board reads; anything else
    is not for the board. The round and beat shapes are the ones rounds.emit sends to on_event."""
    kind = getattr(event, "type", None)
    if kind == "stage_start":
        return {"kind": "stage", "stage": event.name, "state": "start", "attempt": 1}
    if kind == "stage_end":
        counts = dict(getattr(event, "counts", None) or {})
        return {"kind": "stage", "stage": event.name, "state": str(counts.get("status") or "ran"),
                "attempt": int(counts.get("attempts") or 1)}
    if kind in ("round_start", "round_end"):
        return {"kind": "round", "state": "start" if kind == "round_start" else "end", "round": event.round,
                "counts": dict(getattr(event, "counts", None) or {}), "exit": getattr(event, "exit", None)}
    if kind in ("beat_start", "beat_end"):
        return {"kind": "beat", "state": "start" if kind == "beat_start" else "end",
                "agent": event.agent, "round": event.round}
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
    round: int = 0
    agent: str = ""
    rounds: list[dict] = field(default_factory=list)

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
        if kind == "round":
            self.round = int(event.get("round") or 0)
            if event.get("state") == "end":
                self.rounds.append({"round": self.round, "counts": dict(event.get("counts") or {}),
                                    "exit": event.get("exit")})
                self.agent = ""
            return
        if kind == "beat":
            self.round = int(event.get("round") or self.round)
            self.agent = str(event.get("agent") or "") if event.get("state") == "start" else ""
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

    def beat(self) -> Text:
        """Which round it is and who holds the stream (D128); nothing before the first round."""
        if not self.round:
            return Text("")
        line = Text(f"round {self.round}", style="bold")
        if self.agent:
            line.append(f", {self.agent} beat", style="yellow")
        return line

    def rounds_table(self) -> Table:
        """One row per finished round: the counts the gates reported and the exit if the round ended on one."""
        table = Table.grid(padding=(0, 2))
        for name in ("round", "fidelity", "trusted", "refused", "probes passing", "spend", "exit"):
            table.add_column(justify="right" if name not in ("round", "exit") else "left")
        table.add_row(*[Text(name, style="dim") for name in
                        ("round", "fidelity", "trusted", "refused", "probes passing", "spend", "exit")])
        for row in self.rounds:
            counts = row.get("counts") or {}
            spend = float((counts.get("spend") or {}).get("total") or 0.0)
            table.add_row(Text(str(row.get("round"))),
                          Text(f"{counts.get('fidelity', 0)}/{counts.get('tasks', 0)}"),
                          Text(str(counts.get("trusted", 0))), Text(str(counts.get("refused_count", 0))),
                          Text(str(counts.get("probes_passing", 0))), Text(f"${spend:,.4f}"),
                          Text(str(row.get("exit") or ""), style="bold" if row.get("exit") else "dim"))
        return table

    def render(self) -> Panel:
        body = Table.grid(padding=(0, 4))
        body.add_column()
        body.add_column()
        body.add_row(self.stages(), self.provenance())
        parts: list[Any] = [self.beat(), body] if self.round else [body]
        if self.rounds:
            parts += [Text(""), self.rounds_table()]
        parts += [Text(""), self.money(), self.verdict()]
        return Panel(Group(*parts), title=self.title, title_align="left", border_style="dim")


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
        # The / menu's numbered list, waiting for a bare number: (kind, rows). Kind is
        # "commands" (rows are COMMANDS entries) or "sessions" (rows are heartbeat dicts).
        self._pending: Optional[tuple[str, list]] = None

    def open(self) -> None:
        """The entry screen: what this is, how it stands, what you can do, what is running.

        Brand order, leibler.dev style: the word, one line saying what it is, the live
        numbers, then numbered sections (commands, sessions) in dim labels and white values."""
        self.console.print(banner())
        self.console.print(Text(f"  {TAGLINE}", style="dim"))
        self.console.print(status_segments(self.workdir, self.model))
        # A long workdir path wrapping over three lines is the first thing you would see, so it
        # is cut rather than folded; the whole path is on the panel border of every build anyway.
        self.console.print(Text(f"  workdir {self.workdir}", style="dim"),
                                no_wrap=True, overflow="ellipsis")
        self.console.print(Text("\n  01 commands", style="bold"))
        for name, _, blurb in COMMANDS:
            line = Text(f"    /{name:<10}", style="white")
            line.append(blurb, style="dim")
            self.console.print(line)
        self.console.print(Text("    type / to filter", style="dim"))
        self._print_sessions(limit=5)
        self.console.print()

    def command(self, line: str) -> bool:
        """One typed line. Returns False when the screen should close.

        A line that is only "/" or "/fragment" opens the menu: one match runs it, several
        print numbered and wait for a bare number, none says so. A bare number answers the
        last numbered list (commands or sessions). Anything else clears the waiting list."""
        stripped = line.strip()
        if not stripped:
            return True
        if stripped.isdigit() and self._pending is not None:
            return self._pick(int(stripped))
        self._pending = None
        parts = shlex.split(stripped)
        if not parts:
            return True
        verb, rest = parts[0].lstrip("/"), parts[1:]
        if verb in ("quit", "exit", "q"):
            return False
        if parts[0].startswith("/") and (verb == "" or (verb not in self._verbs() and rest == [])):
            return self._menu(parts[0][1:])
        if verb == "help":
            self.console.print(HELP)
        elif verb == "keys":
            self.console.print(_keys(dict(os.environ), set(self.session_keys)))
        elif verb == "login":
            self._login(rest)
            if not rest:
                self._login_menu()
        elif verb == "sessions":
            self._print_sessions(limit=None)
        elif verb == "watch":
            self._watch(rest)
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

    @staticmethod
    def _verbs() -> set[str]:
        return {name for name, _, _ in COMMANDS}

    def _menu(self, fragment: str) -> bool:
        """Type / and pick: one match runs now, several wait for a bare number."""
        matches = filter_commands(fragment)
        if not matches:
            self.console.print(Text(f"no command matches /{fragment}; /help", style="red"))
            return True
        if len(matches) == 1:
            return self.command("/" + matches[0][0])
        self.console.print(Text("  pick a number:", style="dim"))
        for i, (name, _, blurb) in enumerate(matches, 1):
            line = Text(f"    {i}  /{name:<10}", style="white")
            line.append(blurb, style="dim")
            self.console.print(line)
        self._pending = ("commands", matches)
        return True

    def _pick(self, number: int) -> bool:
        """A bare number answers the last numbered list, then the list is gone."""
        kind, rows = self._pending or (None, [])
        self._pending = None
        if number < 1 or number > len(rows):
            self.console.print(Text(f"pick 1-{len(rows)} from the list above", style="red"))
            return True
        if kind == "commands":
            return self.command("/" + rows[number - 1][0])
        return self._watch([str(number)], rows=rows)

    def _print_sessions(self, limit: Optional[int]) -> None:
        """Builds running now and before, newest first. Alive means its pid still runs."""
        from kullback.runner import heartbeat

        records = heartbeat.read_all()
        self.console.print(Text("\n  02 sessions", style="bold"))
        if not records:
            self.console.print(Text("    none yet: /build starts one here", style="dim"))
            return
        shown = records if limit is None else records[:limit]
        for i, record in enumerate(shown, 1):
            mark = "●" if heartbeat.alive(record.get("pid")) else "○"
            colour = "green" if heartbeat.alive(record.get("pid")) else "dim"
            line = Text(f"    {i}  {mark} ", style=colour)
            line.append(str(record.get("workdir") or "?"), style="white")
            rest = " ".join(part for part in (
                str(record.get("model") or "no model"),
                str(record.get("exit") or record.get("status") or ""),
                f"${float(record.get('spend_usd') or 0):,.4f}",
            ) if part)
            line.append(f"  {rest}", style="dim")
            self.console.print(line, no_wrap=True, overflow="ellipsis")
        self.console.print(Text("    /watch N to watch one here", style="dim"))
        self._pending = ("sessions", shown)

    def _watch(self, rest: list[str], rows: Optional[list] = None) -> bool:
        """Watch session N: this screen now reads that build's workdir. The build itself
        keeps running where it is; watching never touches it."""
        from kullback.runner import heartbeat

        records = rows if rows is not None else heartbeat.read_all()
        if not rest or not rest[0].isdigit():
            self.console.print(Text("watch which? /sessions lists them with numbers", style="red"))
            if records:
                self._pending = ("sessions", records)
            return True
        number = int(rest[0])
        if number < 1 or number > len(records):
            self.console.print(Text(f"pick 1-{len(records)} from /sessions", style="red"))
            return True
        self.workdir = Path(records[number - 1]["workdir"])
        self.open()
        return True

    def _login_menu(self) -> None:
        """Bare /login walks to a key: provider, model, key variable, secret, done.

        The secret is read with getpass so it never echoes; names are printed, values never.
        Every answer also works inline (/login provider/model --set KEY=VALUE), this menu
        only asks the same questions one at a time."""
        import getpass

        providers = ["anthropic", "openai", "opencode-go"]
        self.console.print(Text("  log in where?", style="bold"))
        for i, name in enumerate(providers, 1):
            self.console.print(Text(f"    {i}  {name}", style="white"))
        choice = self._ask("    provider [1-3 or name]: ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(providers):
            provider_name = providers[int(choice) - 1]
        elif choice in providers:
            provider_name = choice
        else:
            self.console.print(Text("pick 1-3 or a provider name", style="red"))
            return
        default_model = {"anthropic": "anthropic/claude-opus-5",
                         "openai": "openai/gpt-4.1-mini",
                         "opencode-go": "opencode-go/glm-5.3-flash"}[provider_name]
        model = self._ask(f"    model [{default_model}]: ").strip() or default_model
        key_var = self._key_var_for(provider_name, model)
        self.console.print(Text(f"    {key_var} holds the key (names only, value stays hidden)",
                                style="dim"))
        try:
            secret = getpass.getpass(f"    {key_var}: ")
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return
        if not secret:
            self.console.print(Text("empty key: nothing held", style="red"))
            return
        self._apply_key(key_var, secret)
        self.console.print(Text(f"keys held for this session: {key_var}", style="dim"))
        try:
            self._resolve(model, None)
        except ValueError as exc:
            self.console.print(Text(str(exc), style="red"))
            return
        self.model = model
        self.console.print(self._login_status())

    def _ask(self, prompt: str) -> str:
        """One question to the person typing. A method so tests can answer without stdin."""
        try:
            return self.console.input(prompt)
        except (EOFError, KeyboardInterrupt, OSError):
            return ""

    def _apply_key(self, name: str, value: str) -> None:
        """Hold one key for this session: first sighting remembers what the shell held
        (None means it held nothing) so /logout restores instead of deleting."""
        if name not in self.session_keys:
            self.session_keys[name] = os.environ.get(name)
        os.environ[name] = value

    @staticmethod
    def _key_var_for(provider_name: str, model: str) -> str:
        """Which variable holds this model's key: the adapter's, else the registry's."""
        from kullback.ai import provider as pv

        adapter_cls = pv.ADAPTERS.get(provider_name)
        if adapter_cls is not None:
            return adapter_cls.key_env_var
        try:
            endpoint = pv.registry_endpoint(model)
        except Exception:
            endpoint = None
        if endpoint is not None and endpoint.key_env_var:
            return endpoint.key_env_var
        return f"{provider_name.upper().replace('-', '_')}_API_KEY"

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
        files = [Path(value) for value in _values(rest, "--file")]
        runner = self.runner
        if runner is None:
            # the same entry the CLI uses: whole rounds, Builder then Examiner (D126)
            from kullback import rounds
            runner = rounds.run_rounds
        runner(workdir=self.workdir, iterate="--iterate" in rest, model=self._adapter(), files=files,
               on_event=on_event, ceiling_usd=self.ceiling_usd)

    def _run(self, on_event: Any, rest: list[str]) -> None:
        runner = self.runner
        if runner is None:
            from kullback.builder import build as builder
            runner = builder.run_batch
        counts = _values(rest, "--count")
        try:
            count = int(counts[-1]) if counts else 1
        except ValueError:
            raise ValueError(f"--count takes a whole number, not {counts[-1]!r}") from None
        if count < 1:
            raise ValueError(f"--count takes a number of runs, not {count}")
        on_event({"kind": "stage", "stage": rest[0], "state": "start", "attempt": 1})
        runner(workdir=self.workdir, task_id=rest[0], model=self._adapter(), count=count,
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
                self._apply_key(name, value)
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
        """The last build, read back off disk. No stage runs to answer this.

        A loop build is rounds first (rounds.json), gates beside them (gates.json), spend
        under both (budget.json); a single-pass build is the pipeline state. 'No build yet'
        is said only when none of those files exist, never while a build is spending."""
        rounds = _read(self.workdir / "rounds.json", [])
        gates = _read(self.workdir / "gates.json", [])
        totals = _read(self.workdir / "budget.json", {}).get("total") or {}
        state = _read(self.workdir / "pipeline" / "state.json", {})
        if isinstance(rounds, list) and rounds and not state:
            self.console.print(self._rounds_status(rounds, gates, totals))
            return
        if state or gates or float(totals.get("usd") or 0.0) > 0:
            board = Board(self.workdir, title="last build")
            board.order = list(state.get("statuses") or {})
            board.status = dict(state.get("statuses") or {})
            board.attempts = dict(state.get("attempts") or {})
            # pipeline.py writes each GateResult with as_dict, and GateResult.passed carries the
            # alias "pass", so that is the key on disk; report.py reads the same file the same way.
            board.gates = [{"stage": g.get("stage"), "passed": g.get("pass", g.get("passed")),
                            "failures": g.get("failures") or []} for g in state.get("gates") or []]
            # rounds.json is the driver's record: one row per round, the exit on the last (D126).
            rows = _read(self.workdir / "rounds.json", [])
            board.rounds = [{"round": r.get("round"), "counts": r.get("counts") or {}, "exit": r.get("exit")}
                            for r in rows if isinstance(r, dict)]
            if board.rounds:
                board.round = int(board.rounds[-1].get("round") or 0)
            board.outcome = str(state.get("status") or "build started, no round closed yet")
            self.console.print(board.render())
            return
        self.console.print(Text("no build yet: /build starts one here", style="dim"))

    def _rounds_status(self, rounds: list, gates: list, totals: dict) -> Text:
        """One screenful for a loop build with no pipeline state: where the loop stands,
        what the gates said, what the Examiner found, what it cost.

        Dim labels, white values, leibler.dev style. Round counts come in two shapes -
        the driver's flat counts (fidelity, tasks, trusted, refused_count, spend) and the
        loop record's (trusted_ids, refused map, pending_findings) - and both read here."""
        out = Text()
        out.append(f"last build - {self.workdir.name}\n", style="bold")
        last = rounds[-1] if isinstance(rounds[-1], dict) else {}
        exit_word = str(last.get("exit") or "running")
        line = Text(f"  round {last.get('round', len(rounds))}", style="white")
        line.append(f" · {exit_word}", style="red" if last.get("failed") else "green")
        if last.get("exit_note"):
            line.append(f" - {str(last['exit_note'])[:120]}", style="dim")
        out.append_text(line)
        out.append("\n")
        gate_list = gates if isinstance(gates, list) else []
        if gate_list:
            marks = []
            for gate in gate_list:
                if not isinstance(gate, dict):
                    continue
                passed = gate.get("pass", gate.get("passed", True))
                marks.append(f"{gate.get('stage', '?')} {'✔' if passed else '✘'}")
            staged = Text("  stages ", style="dim")
            staged.append(" · ".join(marks), style="white")
            out.append_text(staged)
            out.append("\n")
            failed = [g for g in gate_list
                      if isinstance(g, dict) and not g.get("pass", g.get("passed", True))]
            verdict = Text(f"  gates {len(gate_list) - len(failed)} passed", style="green")
            if failed:
                verdict.append(f", {len(failed)} failed", style="red")
                reason = "; ".join(failed[-1].get("failures") or []) or "no reason given"
                verdict.append(f"\n    {failed[-1].get('stage')}: {reason[:160]}", style="red")
            out.append_text(verdict)
            out.append("\n")
        counts = last.get("counts") or {}
        trusted = counts.get("trusted_ids")
        trusted_n = len(trusted) if isinstance(trusted, list) else int(counts.get("trusted") or 0)
        refused = counts.get("refused")
        refused_n = len(refused) if isinstance(refused, dict) else int(
            refused if isinstance(refused, int) else counts.get("refused_count") or 0)
        fidelity = Text("  fidelity ", style="dim")
        if counts.get("fidelity") is not None or counts.get("tasks") is not None:
            fidelity.append(f"{counts.get('fidelity', 0)}/{counts.get('tasks', 0)} tasks", style="white")
        else:
            fidelity.append("no counts yet", style="white")
        fidelity.append(f" · trusted {trusted_n}", style="white")
        fidelity.append(f" · refused {refused_n}", style="white")
        out.append_text(fidelity)
        out.append("\n")
        pending = last.get("pending_findings") or []
        out.append_text(Text(f"  examiner {len(pending)} finding(s) open", style="dim"))
        out.append("\n")
        round_spend = float((counts.get("spend") or {}).get("total") or 0.0)
        spent = round_spend or float(totals.get("usd") or 0.0)
        spend = Text(f"  spend ${spent:,.4f}", style="bold")
        if self.ceiling_usd:
            spend.append(f" of ${self.ceiling_usd:,.2f}", style="dim")
        calls = int(totals.get("calls") or 0)
        if calls:
            spend.append(f" · {calls} calls", style="dim")
            spend.append(f" · {int(totals.get('input') or 0):,} in / "
                         f"{int(totals.get('output') or 0):,} out", style="dim")
        out.append_text(spend)
        return out


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
