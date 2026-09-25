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
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kullback.ai.provider import DEFAULT_MODEL
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
/build [--file PATH]              run the autonomous Builder session over the ingested traces,
                                  in the background: the screen keeps taking commands
/nudge TEXT                        steer the running session: delivered before its next model turn
/tell TEXT                         follow up: delivered when the running session would stop
/stop                              cancel the running session after its current step
/context                           the session's context: window, used, share, compactions
/compact                           compact the running session's context when its turn ends
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

# How many of a watched build's own events stay on screen under the board. Enough to see a stage's
# calls going by, few enough that the board itself is never pushed off the top.
FEED_LINES = 12
# Seconds /nudge, /tell and /stop wait for a live build in another process to answer on its bus.
STEER_TIMEOUT = 10.0

# The providers /login walks to by name, and the model each one starts at: a cheap, tool-capable
# model the provider actually serves, since the menu is where a person tries a provider for the
# first time. deepseek-flash is the id the DeepSeek endpoint serves (its docs' quick start, and a
# GET of its own model list, 2026-09-10); the OpenRouter default is the cheapest tool-capable model
# in the registry snapshot at a whole megatoken of context, and its nested id is deliberate, since
# an id with a second slash is the shape OpenRouter mostly speaks. Providers the local registry
# adds are appended to this in _login_defaults, never repeated here. Bedrock starts at the harness
# default model, so a login lands on the model the builds run on; OpenAI at gpt-6-luna, the
# default before it.
LOGIN_DEFAULT_MODELS = {
    "anthropic": "anthropic/claude-opus-5",
    "openai": "openai/gpt-6-luna",
    "opencode-go": "opencode-go/glm-5.3-flash",
    "deepseek": "deepseek/deepseek-flash",
    "openrouter": "openrouter/qwen/qwen3.7-flash",
    "bedrock": DEFAULT_MODEL,
}


def registry_refusal(catalog: Optional[dict], model: str) -> Optional[str]:
    """Why an id cannot be reached through the registry, in the words a person is shown, or None
    when it can be.

    The two questions model_for asks of a provider with no adapter of its own: a host to post to,
    and a request shape this Harness builds, read off the model row's own npm before the provider's,
    because the provider field cannot say that one model rides a gateway and speaks another vendor's
    shape and the row can. One function, so the menu offers exactly what picking it accepts, and
    refuses exactly what a Run would refuse.
    """
    from kullback.ai import pricing
    from kullback.ai import provider as pv

    provider_name, _ = pv.split_model_id(model)
    if provider_name in pv.ADAPTERS:
        return None
    if pricing.endpoint_from_catalog(catalog, model) is None:
        return (f"{model} has no adapter of its own and the models.dev snapshot names no host for "
                f"{provider_name!r}; refresh the snapshot with live calls on, or pass --base-url")
    shape = pricing.model_adapter_for(catalog, model)
    if shape not in pricing.OPENAI_SHAPED:
        return (f"models.dev serves {model} through {shape}, which is not the OpenAI request "
                f"shape this Harness builds; pass --base-url for one that is")
    return None

# Every command in one table: name, usage, what it does. The entry screen and the / menu
# are rendered from this, so a command added here appears in both; HELP stays a literal
# beside it, and a test fails when a table name is missing from HELP, so the two cannot drift.
COMMANDS = [
    ("build", "/build [--file PATH]", "run the Builder over the ingested traces, in the background"),
    ("nudge", "/nudge TEXT", "steer the running session before its next model turn"),
    ("tell", "/tell TEXT", "follow up when the running session would stop"),
    ("stop", "/stop", "cancel the running session after its current step"),
    ("context", "/context", "the session's context: window, used, share, compactions"),
    ("compact", "/compact", "compact the running session's context when its turn ends"),
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
    """The status line under the banner: the spend of the workdir this screen is open on.

    The model and the live switch used to sit here, and they lied: they are this shell's own
    settings, not the settings of the build you are looking at, so a screen watching a live
    build read 'model none, live off' while that build was spending. Each session names its
    own model in the session list, which is where that belongs. Spend is read, never asked,
    off the budget file the runner wrote (absent before the first build)."""
    try:
        from kullback.runner.budget import load_totals
        spent = float(load_totals(workdir)["total"].get("usd") or 0.0)
    except Exception:
        spent = 0.0
    out = Text()
    if spent > 0:
        out.append(f"  spend ${spent:,.4f}", style="dim")
    return out


def live_heartbeats(workdir: Any) -> list[dict]:
    """The heartbeats of builds running now on this workdir, newest first.

    Running means the pid is alive and the heartbeat still says running: a build this screen ran
    leaves a heartbeat whose pid (the screen's own) outlives the build, so the pid alone is not
    enough. Paths are compared absolute, the form heartbeat.beat writes."""
    from kullback.runner import heartbeat

    here = Path(workdir).expanduser().absolute()
    return [r for r in heartbeat.read_all()
            if Path(str(r.get("workdir"))).expanduser().absolute() == here
            and r.get("status") == "running" and heartbeat.alive(r.get("pid"))]


def in_flight(workdir: Any) -> Optional[str]:
    """What a build that has not stopped is doing now, or None when nothing is running here.

    rounds.json gets a row when a round closes, and pipeline/state.json says "complete" about the
    Builder's pipeline and nothing about the loop, so a build an hour into round 2 read "round 1,
    complete" while both agents were still spending. Everything here is derived from files that do
    move: the heartbeat's pid says the build is alive, rounds.json says which round last closed, and
    the ledger says what has been spent since. Stages the pipeline does not name are work outside
    it, which is the Examiner's beat; they are counted, not named as the Examiner, because the
    screen states what it read rather than what it inferred."""
    if not live_heartbeats(workdir):
        return None
    rounds = _read(Path(workdir) / "rounds.json", [])
    # A row lands here when the round ends, so the row is the close. `exit` is not the test: it
    # names the reason a round stopped the loop and is null for a round that simply finished.
    closed = [r for r in rounds if isinstance(r, dict) and r.get("round")]
    totals = _read(Path(workdir) / "budget.json", {})
    spent = float((totals.get("total") or {}).get("usd") or 0.0)
    calls = int((totals.get("total") or {}).get("calls") or 0)
    # Each row holds what its own round spent, not the running total (rounds.py resets beat_spend
    # every round), so what the round in flight has spent is the ledger less all the closed ones.
    before = sum(_round_spend(record) for record in closed)
    parts = [f"round {len(closed) + 1} running"]
    if closed:
        parts.append(f"${spent - before:,.4f} since round {len(closed)} closed")
    parts.append(f"${spent:,.4f} and {calls:,} calls in all")
    named = set(_read(Path(workdir) / "pipeline" / "state.json", {}).get("statuses") or {})
    outside = {stage: bucket for stage, bucket in (totals.get("stages") or {}).items()
               if stage not in named and int(bucket.get("calls") or 0)}
    if outside:
        listed = ", ".join(f"{stage} {int(bucket['calls']):,}" for stage, bucket in sorted(outside.items()))
        parts.append(f"off the pipeline: {listed} calls")
    return " · ".join(parts)


def _round_spend(record: dict) -> float:
    spend = (record.get("counts") or {}).get("spend")
    try:
        return float(spend.get("total") if isinstance(spend, dict) else (spend or 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_dict_event(event: Any) -> Optional[dict]:
    """A typed stage, round, beat or tool event of the agent core as the dict the board reads; anything
    else is not for the board."""
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
    if kind == "tool_execution_start":
        return {"kind": "stage", "stage": event.tool_name, "state": "start", "attempt": 1}
    if kind == "tool_execution_end":
        return {"kind": "stage", "stage": event.tool_name,
                "state": "error" if getattr(event, "is_error", False) else "ran", "attempt": 1}
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


# How many transcript lines the screen keeps for its panel. The terminal's own scrollback holds the
# whole session when the screen runs it; the panel is what /status and /watch show under the board.
TRANSCRIPT_LINES = 200

# One tool line's arguments are cut here, so a call that writes a whole file is one line, not fifty.
ARGS_WIDTH = 100


def _args_summary(arguments: Any, width: int = ARGS_WIDTH) -> str:
    """A tool call's arguments as one line: key=value, the first line of each value, cut to fit."""
    if not isinstance(arguments, dict):
        return ""
    parts = []
    for key, value in arguments.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        first = (text.splitlines() or [""])[0]
        if len(first) > 40 or first != text:
            first = first[:40] + "..."
        parts.append(f"{key}={first}")
    line = ", ".join(parts)
    return line if len(line) <= width else line[:width - 3] + "..."


class Transcript:
    """The Builder session as it happens, one line per thing it did, from its typed events.

    This is tau's event adapter in small: the harness stream in, display lines out, nothing kept
    that a replay of the same events would not rebuild, so the live session and a bus read by
    /watch come out the same. Assistant text streams a line at a time: each delta is added to the
    open line, and the line is printed when the model ends it, because a line-oriented screen
    that takes commands on the same terminal cannot redraw half a line under the prompt. A tool
    call is one line when it starts ("tool: args") and one when it ends (the first line of the
    result, then each gate ruling it carries, then the findings when it is examine), the shape
    cli._session_subscriber prints. A turn end prints the footer, which the caller supplies,
    because the context fill and the spend are the harness's and the ledger's, not the events'.
    """

    def __init__(self, on_line: Optional[Any] = None, footer: Optional[Any] = None,
                 keep: int = TRANSCRIPT_LINES):
        self.on_line, self.footer, self.keep = on_line, footer, keep
        self.lines: list[tuple[str, str]] = []
        self._open = ""
        self._streamed = False

    def say(self, text: str, style: str = "") -> None:
        self.lines.append((text, style))
        del self.lines[:-self.keep]
        if self.on_line is not None:
            self.on_line(text, style)

    def _stream_text(self, event: Any) -> None:
        """One text delta appended to the open line, printing each line it completes."""
        stream = getattr(event, "stream_event", None)
        if getattr(stream, "type", None) == "text_delta":
            self._streamed = True
            self._open += stream.delta
            while "\n" in self._open:
                line, self._open = self._open.split("\n", 1)
                self.say(f"  {line}", "white")

    def _compaction(self, event: Any) -> None:
        """One compaction event as the line the feed keeps."""
        replaced = len(event.replaces_entry_ids)
        self.say(f"compaction by {event.by}: {replaced} entries became one summary"
                 + (f" ({event.reason})" if event.reason else ""), "magenta")

    def _steer(self, event: Any) -> bool:
        """One steer request or ack from another process as the line the feed keeps."""
        kind = getattr(event, "type", None)
        if kind == "steer_request":
            said = f": {_first_line(event.text)}" if event.text else ""
            self.say(f"steer {event.kind} from {event.sender or 'another process'}{said}", "cyan")
            return True
        if kind == "steer_ack":
            self.say(f"steer {event.kind} {event.outcome.replace('_', ' ')}"
                     + (f": {event.reason}" if event.reason else ""),
                     "red" if event.outcome == "refused" else "cyan")
            return True
        return False

    def event(self, event: Any) -> None:
        kind = getattr(event, "type", None)
        if kind == "message_start":
            self._open, self._streamed = "", False
        elif kind == "message_update":
            self._stream_text(event)
        elif kind == "message_end":
            self._message_end(event.message)
        elif kind == "tool_execution_start":
            self.say(f"▸ {event.tool_name}: {_args_summary(event.arguments)}", "yellow")
        elif kind == "tool_execution_end":
            self._tool_end(event)
        elif kind == "turn_end" and self.footer is not None:
            self.say(f"  {self.footer(event.turn)}", "dim")
        elif kind == "compaction":
            self._compaction(event)
        elif kind == "custom_message":
            self.say(f"queued ({event.deliver_as}): {_first_line(event.content)}", "cyan")
        elif self._steer(event):
            pass
        elif kind == "error":
            self.say(f"error: {event.message}", "red")
        elif kind == "agent_end":
            self.say("session run ended", "dim")

    def _message_end(self, message: Any) -> None:
        role = getattr(message, "role", None)
        if role == "assistant":
            rest = self._open if self._streamed else str(getattr(message, "content", None) or "")
            for line in rest.splitlines():
                self.say(f"  {line}", "white")
            self._open, self._streamed = "", False
        elif role == "user":
            self.say(f"› {_first_line(str(getattr(message, 'content', '') or ''))}", "cyan")

    def _tool_end(self, event: Any) -> None:
        result = event.result
        mark, style = ("✘", "red") if event.is_error else ("✔", "green")
        self.say(f"{mark} {event.tool_name}: {_first_line(str(result.content or ''))}", style)
        details = result.details or {}
        for ruling in details.get("rulings") or ():
            if isinstance(ruling, dict):
                accepted = ruling.get("accepted")
                self.say(f"    ruling {ruling.get('name')}: {'accepted' if accepted else 'rejected'}"
                         f" ({ruling.get('line')})", "green" if accepted else "red")
        if event.tool_name == "examine":
            for finding in details.get("findings") or ():
                if isinstance(finding, dict):
                    self.say(f"    finding {finding.get('kind')} on {finding.get('task_id')}: "
                             f"{finding.get('path')}: {finding.get('change')}", "magenta")

    def render(self, limit: int = FEED_LINES * 2) -> Text:
        out = Text()
        for text, style in self.lines[-limit:]:
            out.append(f"{text}\n", style=style or None)
        return out


def _spent(workdir: Path) -> float:
    return float((_read(Path(workdir) / "budget.json", {}).get("total") or {}).get("usd") or 0.0)


def _first_line(text: str, width: int = 160) -> str:
    first = (text.strip().splitlines() or [""])[0]
    return first if len(first) <= width else first[:width - 3] + "..."


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


def _credential_source(model: str, host: str) -> tuple[tuple[tuple[str, ...], ...], str]:
    """The key variable groups a model reads, and the host its calls go to unless one was given."""
    from kullback.ai import provider as pv

    provider_name, _ = pv.split_model_id(model)
    adapter_cls = pv.ADAPTERS.get(provider_name)
    if adapter_cls is not None:
        # The adapter names its own variables; one that signs requests reads several.
        return adapter_cls.credential_vars(), host or "built-in adapter"
    try:
        endpoint = pv.registry_endpoint(model)
    except Exception:
        endpoint = None
    if endpoint is None:
        return (), host
    groups = ((endpoint.key_env_var,),) if endpoint.key_env_var else ()
    return groups, host or endpoint.base_url


def _append_key_lines(out: Text, groups: tuple[tuple[str, ...], ...]) -> None:
    """One line per key variable saying set or missing, the alternative groups joined by `or`."""
    for index, group in enumerate(groups):
        if index:
            out.append("or\n", style="dim")
        for key_var in group:
            out.append(f"{key_var:<32}", style="dim")
            out.append("set\n" if os.environ.get(key_var) else "missing\n",
                         style="green" if os.environ.get(key_var) else "red")
    if not groups:
        out.append("no key variable: this endpoint takes none\n", style="dim")


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
        # The Builder session /build started here: its thread, its harness once session.build
        # hands it over, the board and transcript its events feed, and the last harness, which
        # /context reads after the session ends. Nothing else about a session is kept here.
        self._session: Optional[threading.Thread] = None
        self.harness: Any = None
        self.last_harness: Any = None
        self.board: Optional[Board] = None
        self.transcript: Optional[Transcript] = None
        self._compact_asked = False
        # How long /nudge, /tell and /stop wait for a live build elsewhere to answer on its bus.
        self.steer_timeout = STEER_TIMEOUT

    def attach(self) -> None:
        """Follow the live build on this workdir at once (what /watch does for its heartbeat), or
        say there is none and show /status."""
        live = live_heartbeats(self.workdir)
        if not live:
            self.console.print(Text(f"  no live build in {self.workdir}", style="dim"),
                               no_wrap=True, overflow="ellipsis")
            self._status()
            return
        self._watch(["1"], rows=live[:1])

    def open(self) -> None:
        """The entry screen: what this is, how it stands, what you can do, what is running.

        Brand order, leibler.dev style: the word, one line saying what it is, the live
        numbers, then numbered sections (commands, sessions) in dim labels and white values."""
        self.console.print(banner())
        self.console.print(Text(f"  {TAGLINE}", style="dim"))
        segments = status_segments(self.workdir, self.model)
        if segments.plain:
            self.console.print(segments)
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
        # A nudge is free text, so it is taken whole: shlex would refuse "don't" and eat quotes.
        head, _, tail = stripped.partition(" ")
        if head.lstrip("/") in ("nudge", "tell"):
            self._queue(head.lstrip("/"), tail.strip())
            return True
        parts = shlex.split(stripped)
        if not parts:
            return True
        verb, rest = parts[0].lstrip("/"), parts[1:]
        if verb in ("quit", "exit", "q"):
            self.close()
            return False
        if parts[0].startswith("/") and (verb == "" or (verb not in self._verbs() and rest == [])):
            return self._menu(parts[0][1:])
        if verb == "help":
            # markup off: rich reads a bracketed word as a style tag, so /login's own
            # [provider/model] was swallowed and the help said less than the command takes.
            self.console.print(HELP, markup=False)
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
            self._start_build(rest)
        elif verb == "stop":
            self._stop()
        elif verb == "context":
            self._context()
        elif verb == "compact":
            self._compact()
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
                (f"cache saved ${float(record.get('cache_saved_usd') or 0):,.4f}"
                 if record.get("cache_saved_usd") is not None else ""),
            ) if part)
            line.append(f"  {rest}", style="dim")
            self.console.print(line, no_wrap=True, overflow="ellipsis")
        self.console.print(Text("    /watch N to watch one here", style="dim"))
        self._pending = ("sessions", shown)

    def _watch(self, rest: list[str], rows: Optional[list] = None) -> bool:
        """Watch session N: this screen reads that build's workdir and follows it while it runs.

        Following is reading, on a timer: the board is rebuilt from the files the build writes,
        so watching never touches the build and a watcher that dies loses nothing. It used to
        reopen the entry screen instead, which showed the commands again and not the build, so
        watching a running build looked like nothing had happened. A build whose pid is gone is
        shown once, because there is nothing left to follow."""
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
        record = records[number - 1]
        self.workdir = Path(record["workdir"])
        self.console.print(Text(f"  watching {self.workdir}", style="dim"), no_wrap=True,
                           overflow="ellipsis")
        if heartbeat.alive(record.get("pid")):
            self._follow(record.get("pid"))
        else:
            self.console.print(self._status_renderable())
        return True

    def _follow(self, pid: Any, every_seconds: float = 1.0) -> None:
        """Re-read this build's files until its pid goes or its heartbeat says it ended, or until
        the person stops watching.

        Two things are read: the board, off the records the build writes, and the build's own
        event stream. A Builder session writes every event of its harness to workdir/bus.jsonl,
        and that is read through the same Transcript the screen uses for a session it runs itself,
        so watching another process reads the same lines as running here. A build with no bus
        falls back to the feed (kullback.runner.feed). Ctrl-C stops the watching, never the build:
        they are different processes, and the build does not know anyone is here.

        A build that writes neither is still followed. For those the calls are read off the reply
        cache instead, which every build writes."""
        from kullback.agent.bus import Bus
        from kullback.runner import feed, heartbeat

        bus = Bus(self.workdir / "bus.jsonl")
        transcript = Transcript()
        seq = 0

        def since(offset: int, mtime: float, first: bool) -> tuple[list[str], int, float]:
            nonlocal seq
            if bus.path.is_file():
                # One pass of the follower: every record after the last one read, then stop, so
                # the pid check above decides when watching ends, not the log. The Bus remembers
                # the byte offset it reached, so each pass reads only what was appended since.
                for record in bus.tail(seq, stop=lambda: True):
                    seq = record.seq
                    transcript.event(record.event)
                return [], offset, mtime  # the transcript holds them
            if feed.path_for(self.workdir).is_file():
                rows, offset = feed.read_since(self.workdir, offset)
            else:
                rows, mtime = feed.derived_since(
                    self.workdir, mtime, limit=FEED_LINES if first else None)
            return [feed.describe(row) for row in rows], offset, mtime

        def shown(recent: list[str]) -> Any:
            return self._watching(recent, transcript if bus.path.is_file() else None)

        lines, offset, mtime = since(0, 0.0, True)
        recent = lines[-FEED_LINES:]
        with Live(shown(recent), console=self.console, refresh_per_second=4) as live:
            try:
                # The pid alone never ends this: a build another screen started carries that
                # screen's pid, which outlives the build. Its final beat says done or failed.
                while heartbeat.alive(pid) and heartbeat.terminal_status(self.workdir, pid) is None:
                    time.sleep(every_seconds)
                    lines, offset, mtime = since(offset, mtime, False)
                    recent = (recent + lines)[-FEED_LINES:]
                    live.update(shown(recent))
                lines, offset, mtime = since(offset, mtime, False)
                recent = (recent + lines)[-FEED_LINES:]
                live.update(shown(recent))
            except KeyboardInterrupt:
                pass
        end = heartbeat.terminal_status(self.workdir, pid)
        if end is not None:
            self.console.print(Text(f"  build {end}; the build is untouched", style="dim"))
        else:
            self.console.print(Text("  stopped watching; the build is untouched", style="dim"))

    def _watching(self, recent: list[str], transcript: Optional[Transcript] = None) -> Any:
        """The board with the build's last few events under it: state above, story below."""
        board = self._status_renderable()
        if transcript is not None and transcript.lines:
            return Group(board, transcript.render(FEED_LINES))
        if not recent:
            return Group(board, Text("  waiting for the build's next call", style="dim"))
        lines = Text()
        for line in recent:
            lines.append(f"  {line}\n", style="dim")
        return Group(board, lines)

    def _login_menu(self) -> None:
        """Bare /login walks to a key: provider, model, key variable, secret, done.

        The secret is read with getpass so it never echoes; names are printed, values never.
        Every answer also works inline (/login provider/model --set KEY=VALUE), this menu
        only asks the same questions one at a time."""
        import getpass

        defaults = self._login_defaults()
        providers = list(defaults)
        self.console.print(Text("  log in where?", style="bold"))
        for i, name in enumerate(providers, 1):
            self.console.print(Text(f"    {i}  {name}", style="white"))
        span = f"1-{len(providers)}"
        choice = self._ask(f"    provider [{span} or name]: ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(providers):
            provider_name = providers[int(choice) - 1]
        elif choice in providers:
            provider_name = choice
        else:
            self.console.print(Text(f"pick {span} or a provider name", style="red"))
            return
        default_model = defaults[provider_name]
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
        """One question to the person typing. A method so tests can answer without stdin.

        Escaped, because rich reads the bracketed hint ("[1-6 or name]", "[default model]") as a
        style tag and the person was asked "model :" with the default gone."""
        from rich.markup import escape

        try:
            return self.console.input(escape(prompt))
        except (EOFError, KeyboardInterrupt, OSError):
            return ""

    def _apply_key(self, name: str, value: str) -> None:
        """Hold one key for this session: first sighting remembers what the shell held
        (None means it held nothing) so /logout restores instead of deleting."""
        if name not in self.session_keys:
            self.session_keys[name] = os.environ.get(name)
        os.environ[name] = value

    @staticmethod
    def _login_defaults() -> dict[str, str]:
        """The providers /login offers and the model each starts at.

        The named ones first, in the order a person is most likely to want them, then every
        provider the local registry adds, at its first model, so a provider models.dev does not
        list is one entry in that file away from being offered here too. Nothing is hand-listed
        twice: the local rows come from the same lookup that resolves the model.

        Every one of them, hand-listed or from the file, is then put through the refusal /login
        itself would give it. A choice that fails on the line after picking it is worse than one
        that was never listed, and on a machine with no snapshot yet that is most of the hand
        listed ones: they are reached through the registry, and the registry is a file that is not
        there. Typing the id still says so, and says how to get the file.
        """
        from kullback.ai import pricing
        from kullback.ai import provider as pv

        try:
            # The snapshot the resolver reads, so the menu and the model it then resolves are
            # always looking at the same two files.
            catalog = pricing.refresh(path=pv.REGISTRY_SNAPSHOT_PATH)
            local = pricing.local_providers(pv.REGISTRY_SNAPSHOT_PATH)
        except Exception:
            catalog, local = None, {}
        candidates = dict(LOGIN_DEFAULT_MODELS)
        for name, entry in local.items():
            models = list((entry.get("models") or {})) if isinstance(entry, dict) else []
            if name not in candidates and models:
                candidates[name] = f"{name}/{models[0]}"
        return {name: model for name, model in candidates.items()
                if registry_refusal(catalog, model) is None}

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
        return pv.key_var_for_provider(provider_name)

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

    # --- the session /build runs here ---

    def running(self) -> bool:
        """A session started here is still going."""
        return self._session is not None and self._session.is_alive()

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until the session started here ends: the quit path and the tests use it."""
        if self._session is not None:
            self._session.join(timeout)

    def close(self) -> None:
        """Leaving with a session running cancels it and lets it write its last step, rather than
        killing the thread mid write of the session file."""
        if self.running():
            self.console.print(Text("stopping the running session first", style="dim"))
            if self.harness is not None:
                self.harness.cancel()
            self.wait(30)

    def _start_build(self, rest: list[str]) -> None:
        """Start the Builder session on its own thread; this screen keeps taking commands.

        session.build is the same entry the CLI uses. It hands back the live harness through
        on_harness, which is what /nudge, /tell, /stop, /compact and /context act on. Every event
        feeds the board and the transcript; the transcript prints each line as it lands, so the
        session scrolls past in the terminal above the prompt, and the board is printed when the
        session ends (and on /status while it runs)."""
        if self.running():
            self.console.print(Text("a session is already running here: /nudge, /tell or /stop it",
                                    style="red"))
            return
        try:
            files = [Path(value) for value in _values(rest, "--file")]
        except ValueError as exc:
            self.console.print(Text(f"{exc}   (/help for usage)", style="red"))
            return
        runner = self.runner
        if runner is None:
            from kullback.builder import session as session_mod
            runner = session_mod.build
        board = Board(self.workdir, title="build", ceiling=self.ceiling_usd)
        transcript = Transcript(on_line=self._print_line, footer=self._footer)
        self.board, self.transcript, self._compact_asked = board, transcript, False

        def on_event(event: Any) -> None:
            board.event(event)
            transcript.event(event)

        def work() -> None:
            # The same heartbeat and feed `kullback build` writes, so a build started on this
            # screen is listed under /sessions on any other screen, /watch and in_flight see it,
            # and `kullback steer` can find it.
            from kullback.runner import feed, heartbeat

            feed.start(self.workdir, model=self.model, ceiling_usd=self.ceiling_usd)
            pulse = heartbeat.pulse(self.workdir, self.model, "running")
            failed = True
            try:
                result = runner(workdir=self.workdir, model=self._adapter(), files=files,
                                ceiling_usd=self.ceiling_usd,
                                subscribers=[on_event, self._compact_when_asked],
                                on_harness=self._attach)
                failed = isinstance(result, dict) and result.get("stopped") == "error"
                if isinstance(result, dict) and result.get("stopped"):
                    board.outcome = (f"stopped: {result['stopped']}; trusted {result.get('trusted', 0)}, "
                                     f"refused {result.get('refused', 0)}, open {result.get('open', 0)}")
            except Exception as exc:  # a failed build is a result to read, not a traceback to lose
                board.outcome = f"{type(exc).__name__}: {exc}"
            finally:
                pulse.stop()
                heartbeat.beat(self.workdir, self.model, "failed" if failed else "done")
                self.harness = None
                self.console.print(board.render())

        self.console.print(Text("  build started; /nudge TEXT, /tell TEXT, /stop, /context while it runs",
                                style="dim"))
        self._session = threading.Thread(target=work, name="kullback-build", daemon=True)
        self._session.start()

    def _attach(self, harness: Any) -> None:
        self.harness = self.last_harness = harness

    def _print_line(self, text: str, style: str) -> None:
        self.console.print(Text(text, style=style or ""))

    def _footer(self, turn: int) -> str:
        """The line under each turn: the turn, the context fill off the harness, the spend off the
        ledger, and the model. tau keeps these in a status bar; a line per turn is the same
        numbers on a screen that scrolls."""
        parts = [f"turn {turn}"]
        if self.harness is not None:
            estimate = self.harness.context.estimate()
            parts.append(f"context {estimate.tokens:,} of {estimate.window:,} ({estimate.fill:.0%})")
        spent = _spent(self.workdir)
        parts.append(f"spend ${spent:,.4f}" + (f" of ${self.ceiling_usd:,.2f}" if self.ceiling_usd else ""))
        if self.model:
            parts.append(self.model)
        return " · ".join(parts)

    def _refuse_without_session(self, what: str) -> bool:
        if self.running() and self.harness is not None:
            return False
        self.console.print(Text(f"no session running: {what} reaches a session /build started here, "
                                "or a live build elsewhere through `kullback steer`", style="red"))
        return True

    def _steer_remote(self, kind: str, text: str = "") -> bool:
        """Steer a live build on this workdir that another process runs, through its bus.

        False when there is none, so the caller refuses as before. The screen waits for the
        build's ack, up to `steer_timeout` seconds, and prints what it said."""
        if self.running() or not live_heartbeats(self.workdir):
            return False
        from kullback.agent import steer

        request_id = steer.request(self.workdir, kind, text, sender="screen")
        ack = steer.wait_for_ack(self.workdir, request_id, self.steer_timeout)
        if ack is None:
            self.console.print(Text(f"sent {kind} to the live build; no answer within "
                                    f"{self.steer_timeout:g} seconds", style="yellow"))
        else:
            self.console.print(Text(f"{kind} {ack.outcome.replace('_', ' ')}"
                                    + (f": {ack.reason}" if ack.reason else ""),
                                    style="red" if ack.outcome == "refused" else "cyan"))
        return True

    def _queue(self, kind: str, text: str) -> None:
        """/nudge steers (delivered before the next model turn), /tell follows up (delivered when
        the run would otherwise stop). Both are the harness's own queues; the screen only says
        what it queued, and the transcript shows the turn again when it is delivered."""
        if not text:
            self.console.print(Text(f"/{kind} needs text: /{kind} TEXT", style="red"))
            return
        if self._steer_remote(kind, text) or self._refuse_without_session(f"/{kind}"):
            return
        if kind == "nudge":
            self.harness.steer(text)
            when = "before the next model turn"
        else:
            self.harness.follow_up(text)
            when = "when the current work ends"
        line = f"queued {kind} ({when}): {text}"
        if self.transcript is not None:
            self.transcript.say(line, "cyan")
        else:
            self._print_line(line, "cyan")

    def _stop(self) -> None:
        if self._steer_remote("stop") or self._refuse_without_session("/stop"):
            return
        self.harness.cancel()
        self.console.print(Text("cancel asked: the session stops before its next step", style="yellow"))

    def _compact(self) -> None:
        """Compaction mutates the transcript the loop is using, so it is never run from this
        thread: it is asked for here and runs inside the session at the next turn end (tau
        refuses a compaction during a run for the same reason)."""
        if self._refuse_without_session("/compact"):
            return
        self._compact_asked = True
        self.console.print(Text("compaction asked: it runs when the current turn ends", style="dim"))

    def _compact_when_asked(self, event: Any) -> Any:
        """A subscriber on the session's own loop: at the turn end after /compact, it returns the
        compaction for the harness to await there; otherwise nothing."""
        if (getattr(event, "type", None) != "turn_end" or not self._compact_asked
                or self.harness is None):
            return None
        self._compact_asked = False
        return self.harness.compact("/compact from the screen")

    def _context(self) -> None:
        """The context of the running session, else of the last one this screen ran."""
        harness = self.harness or self.last_harness
        if harness is None:
            self.console.print(Text("no session yet: /build starts one, and /context reads it",
                                    style="dim"))
            return
        estimate = harness.context.estimate()
        stats = harness.context_stats
        out = Text(("running session" if self.running() else "last session") + "\n", style="bold")
        out.append(f"  window {estimate.window:,} tokens, used {estimate.tokens:,}, "
                   f"share {estimate.fill:.0%} (compacts at {estimate.line:.0%}, "
                   f"estimated from {estimate.source})\n", style="white")
        out.append(f"  compactions {stats.compactions} ({stats.mechanical_summaries} by code), "
                   f"{stats.entries_replaced} entries replaced", style="dim")
        if stats.fill_at_turn_end:
            out.append(f"\n  fill at turn end: last {stats.fill_at_turn_end[-1]:.0%}, "
                       f"most {max(stats.fill_at_turn_end):.0%}", style="dim")
        self.console.print(out)

    def _run(self, on_event: Any, rest: list[str]) -> None:
        runner = self.runner
        if runner is None:
            from kullback.runner import tool as runner_tool
            runner = runner_tool.reroll
        counts = _values(rest, "--count")
        try:
            count = int(counts[-1]) if counts else 1
        except ValueError:
            raise ValueError(f"--count takes a whole number, not {counts[-1]!r}") from None
        if count < 1:
            raise ValueError(f"--count takes a number of runs, not {count}")
        on_event({"kind": "stage", "stage": rest[0], "state": "start", "attempt": 1})
        runner(environment_dir=self.workdir, task_id=rest[0], model=self._adapter(), count=count,
               workdir=self.workdir)
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
        from kullback.ai import pricing
        from kullback.ai import provider as pv

        provider_name, _ = pv.split_model_id(model)  # the 'provider/model' shape, or words saying so
        if provider_name in pv.ADAPTERS or base_url:
            return
        try:
            catalog = pricing.refresh(path=pv.REGISTRY_SNAPSHOT_PATH)
        except Exception:
            catalog = None
        refusal = registry_refusal(catalog, model)
        if refusal:
            raise ValueError(refusal)

    def _login_status(self) -> Text:
        """The current model, where its calls go, and whether its key is set. Names only, never values."""
        from kullback.ai import provider as pv

        out = Text()
        if not self.model:
            return Text("no model: /login provider/model to use one", style="dim")
        out.append(f"model {self.model}\n", style="bold")
        groups, host = _credential_source(self.model, self.base_url or "")
        if host:
            out.append(f"host {host}\n", style="dim")
        _append_key_lines(out, groups)
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
        """While a session runs here, its live board with the transcript under it; otherwise the
        last build read back off disk."""
        if self.running() and self.board is not None and self.transcript is not None:
            self.console.print(Group(self.board.render(), Panel(
                self.transcript.render(), title="transcript", title_align="left", border_style="dim")))
            return
        self.console.print(self._status_renderable())

    def _status_renderable(self) -> Any:
        """The last build, read back off disk. No stage runs to answer this.

        A loop build is rounds first (rounds.json), gates beside them (gates.json), spend
        under both (budget.json); a single-pass build is the pipeline state. 'No build yet'
        is said only when none of those files exist, never while a build is spending.

        Returned rather than printed because /watch renders the same thing on a timer: what
        you see watching a build is exactly what /status says about it."""
        rounds = _read(self.workdir / "rounds.json", [])
        gates = _read(self.workdir / "gates.json", [])
        totals = _read(self.workdir / "budget.json", {}).get("total") or {}
        state = _read(self.workdir / "pipeline" / "state.json", {})
        if isinstance(rounds, list) and rounds and not state:
            return self._rounds_status(rounds, gates, totals)
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
            # A build that is still running says what it is doing; the pipeline's own "complete"
            # is about the Builder's stages and says nothing about the round in flight.
            board.outcome = in_flight(self.workdir) or str(
                state.get("status") or "build started, no round closed yet")
            return board.render()
        return Text("no build yet: /build starts one here", style="dim")

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
         ceiling_usd: Optional[float] = None, attach: bool = False) -> None:
    """The screen's read-and-run loop; `attach` follows the workdir's live build before the prompt."""
    screen = Screen(workdir, model=model, base_url=base_url, ceiling_usd=ceiling_usd)
    screen.open()
    if attach:
        screen.attach()
    while True:
        try:
            line = screen.console.input("[bold]›[/bold] ")
        except (EOFError, KeyboardInterrupt):
            screen.console.print()
            screen.close()
            return
        if not screen.command(line):
            return
