"""The Textual app shell: header, tab routing, palette, sessions, keys, quit.

One full-screen app over a workdir. It opens on Home, or on Watch when a build
is live in this workdir. Keys 2, 5 and 6 and the palette entries traces, tasks,
runs, publish and synthesise mount those views from kullback.tui.views.
A command line pinned to the bottom runs every slash command of the COMMANDS
table inside the app: views mount, text answers show in a modal, and bare
`kullback` opens this app on the current directory.
Quitting never touches the build child, which runs detached.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Callable, Optional

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from kullback.tui import COMMANDS, live_heartbeats, status_segments
from kullback.tui.build_view import BuildView
from kullback.tui.command_line import CommandLine
from kullback.tui.home import HomeView
from kullback.tui.views import VIEWS
from kullback.tui.views.traces import TracesView
from kullback.tui.watch import WatchView

APP_CSS = """
#hdr { height: 1; }
#view { height: 1fr; }
#cmd-result { height: 1; }
#cmdline { height: auto; }
#cmd-menu { height: auto; max-height: 10; }
#home-checklist { height: auto; max-height: 8; }
#output-form { width: 80%; height: auto; max-height: 80%; }
#transcript { width: 1fr; }
#sidebar { width: 30%; min-width: 68; }
#watch-main { height: 1fr; }
#steer { height: 3; }
"""

# Line-screen commands the palette lists beside the views.
PALETTE_COMMANDS = ("status", "sessions", "watch", "build", "keys", "login", "logout", "help",
                    "quit")
KEYS_HELP = """keys: 1 home, 2 traces, 3 build, 4 watch, 5 tasks, 6 runs. ctrl+k commands \
(publish, synthesise), ctrl+r machine sessions, ? keys, ctrl+d quit. In watch: enter nudges, \
alt+enter tells, esc asks before stopping."""


# The views by their palette name, off the registry the views package owns.
VIEW_CLASSES = {title.lower(): cls for _, title, cls in VIEWS}


class PaletteModal(ModalScreen[Optional[str]]):
    """Every view and the line-screen commands that still make sense, one per row."""

    def __init__(self, choose: Callable[[str], None]) -> None:
        super().__init__()
        self.choose = choose

    def compose(self):  # type: ignore[override]
        yield Vertical(
            *[Button(name, id=f"view-{name}")
              for name in ("home", "traces", "build", "watch", "tasks", "runs", "publish",
                           "synthesise")],
            *[Button(name, id=f"command-{name}") for name in PALETTE_COMMANDS],
            id="palette-list")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = str(event.button.id or "")
        if button_id.startswith(("view-", "command-")):
            name = button_id.split("-", 1)[1]
        else:
            name = str(event.button.label)
        self.dismiss(None)
        self.choose(name)

class SessionsModal(ModalScreen[None]):
    """The machine's sessions off the heartbeats, newest first."""

    def __init__(self, choose: Callable[[dict], None]) -> None:
        super().__init__()
        self.choose = choose

    def compose(self):  # type: ignore[override]
        from kullback.runner import heartbeat

        records = heartbeat.read_all()[:20]
        if not records:
            yield Static("no sessions on this machine", id="sessions-empty")
            return
        for i, record in enumerate(records):
            alive = heartbeat.alive(record.get("pid"))
            mark = "live" if alive and record.get("status") == "running" else "done"
            yield Button(f"{i + 1} {mark} {record.get('workdir')} {record.get('model') or ''}",
                         id=f"session-{i}")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        from kullback.runner import heartbeat

        records = heartbeat.read_all()[:20]
        try:
            index = int(str(event.button.id).split("-")[1])
        except (IndexError, ValueError):
            self.dismiss(None)
            return
        self.dismiss(None)
        if 0 <= index < len(records):
            self.choose(records[index])


class KeysModal(ModalScreen[None]):
    """The keys for this view."""

    def compose(self):  # type: ignore[override]
        yield Static(KEYS_HELP, id="keys-body")

    BINDINGS = [Binding("escape", "dismiss", "Close"), Binding("question_mark", "dismiss", "Close")]


class OutputModal(ModalScreen[None]):
    """A slash command's result text, with esc or a close button to leave."""

    BINDINGS = [Binding("escape", "dismiss", "Close")]

    def __init__(self, title: str, text: str) -> None:
        super().__init__()
        self._title = title
        self._text = text

    def compose(self):  # type: ignore[override]
        yield Vertical(
            Static(self._title, id="output-title"),
            VerticalScroll(Static(Text(self._text), id="output-body")),
            Button("close", id="output-close"),
            id="output-form")

    @on(Button.Pressed, "#output-close")
    def _closed(self) -> None:
        self.dismiss(None)

    def text(self) -> str:
        """The result as plain text, for tests to read without rendering."""
        return self._text


class KullbackApp(App):
    """kullback's full-screen app over one workdir."""

    CSS = APP_CSS
    BINDINGS = [
        Binding("1", "show_home", "Home"),
        Binding("2", "show_traces", "Traces"),
        Binding("3", "show_build", "Build"),
        Binding("4", "show_watch", "Watch"),
        Binding("5", "show_tasks", "Tasks"),
        Binding("6", "show_runs", "Runs"),
        # Priority chords: an Input eats ctrl+k and ctrl+d for editing, so the app
        # takes them first. Escape stays unprioritised so a modal keeps it. Slash
        # stays unprioritised too, so it types into inputs instead of refocusing.
        Binding("ctrl+k", "palette", "Commands", priority=True),
        Binding("ctrl+r", "sessions", "Sessions", priority=True),
        Binding("question_mark", "keys", "Keys"),
        Binding("ctrl+d", "quit_app", "Quit", priority=True),
        Binding("slash", "focus_command", "Commands"),
        Binding("escape", "confirm_stop", "Stop"),
    ]

    def __init__(self, workdir: Any, model: Optional[str] = None, base_url: Optional[str] = None,
                 ceiling_usd: Optional[float] = None) -> None:
        super().__init__()
        self.workdir = Path(workdir)
        self.model = model
        self.base_url = base_url or ""
        self.ceiling_usd = ceiling_usd
        # Keys held with /login, mapped to what the shell held before (None means
        # it held nothing), so /logout restores the shell instead of just deleting.
        self.session_keys: dict[str, Optional[str]] = {}
        # The last one-line result, for tests to read without rendering.
        self.last_said = ""
        self.header = Static("", id="hdr")
        self.body = Vertical(id="view")
        self.result = Static("", id="cmd-result")
        self.cmdline = CommandLine(on_run=self.run_command)
        self.current: Any = None

    def compose(self) -> ComposeResult:
        yield self.header
        yield self.body
        yield self.result
        yield self.cmdline

    def on_mount(self) -> None:
        if live_heartbeats(self.workdir):
            self.show_watch()
        else:
            self.show_home()
        self.refresh_header()
        self.set_interval(5.0, self.refresh_header)

    def check_action(self, action: str, parameters: tuple) -> Optional[bool]:
        """Number keys, slash and ? type into inputs instead of switching views there."""
        from textual.widgets import Input

        if action in ("show_home", "show_traces", "show_build", "show_watch", "show_tasks",
                      "show_runs", "keys", "focus_command") and isinstance(self.focused, Input):
            return False
        return True

    def refresh_header(self) -> None:
        """The workdir, its total spend, and where the commands live."""
        out = Text(f"kullback  {self.workdir} ", style="bold")
        out.append_text(status_segments(self.workdir, self.model))
        out.append("   / commands  ? keys", style="dim")
        self.header.update(out)

    def say(self, line: str) -> None:
        """One result line under the view, kept for tests to read without rendering."""
        self.last_said = line
        try:
            self.query_one("#cmd-result", Static).update(line)
        except Exception:
            pass

    def show_text(self, title: str, text: str) -> None:
        """A command's longer result in a modal the person dismisses with esc."""
        self.push_screen(OutputModal(title, text))

    def prefill_command(self, text: str) -> None:
        """Put text in the command line and focus it, for checklist fixes to run."""
        self.query_one("#cmdline", CommandLine).prefill(text)

    def action_focus_command(self) -> None:
        """Slash anywhere focuses the command line with a / ready to filter."""
        if self.screen.id != "_default":
            return
        try:
            line = self.query_one("#cmdline", CommandLine)
        except Exception:
            return
        line.focus_line()
        if not line.cmd_input.value:
            line.cmd_input.value = "/"

    def run_command(self, line: str) -> None:
        """Run one slash command typed in the command line, showing its result here.

        View commands mount their view; text commands run the same code the line
        screen runs and show it in a modal. Nothing shells out to a subprocess for
        answers. Every exception becomes a result line, never a crash, since a
        message handler must not throw.
        """
        try:
            self._dispatch(line)
        except Exception as exc:
            self.say(f"{type(exc).__name__}: {exc}")

    def _dispatch(self, line: str) -> None:
        """One line to its view or its action; unknown verbs say so with /help."""
        text = line.strip()
        if not text:
            return
        # A nudge is free text, so it is taken whole: shlex would refuse "don't".
        head, _, _ = text.partition(" ")
        if head.lstrip("/") in ("nudge", "tell"):
            self._run_screen_text(text)
            return
        try:
            parts = shlex.split(text)
        except ValueError as exc:
            self.say(f"{exc}   (/help for usage)")
            return
        if not parts:
            return
        verb, rest = parts[0].lstrip("/"), parts[1:]
        if verb in ("quit", "exit", "q"):
            self.exit()
        elif verb in ("home", "doctor"):
            self.show_home()
        elif verb in ("traces", "tasks", "runs", "publish", "synthesise"):
            self.show_view(verb)
        elif verb == "build":
            self._build_command(rest)
        elif verb == "watch":
            self._watch_command(rest)
        elif verb == "sessions":
            self.action_sessions()
        elif verb == "help":
            self._help()
        elif verb == "login" and not rest:
            from kullback.tui.login_modal import LoginModal

            self.push_screen(LoginModal(), self._login_saved)
        elif verb == "ingest":
            self._ingest(rest)
        elif verb in {row[0] for row in COMMANDS}:
            self._run_screen_text(text)
        else:
            self.say(f"no command {verb}; /help lists every command")

    def _run_screen_text(self, line: str) -> None:
        """Run a line-screen command off a capturing Screen and show what it said.

        The Screen holds no session of its own here, so session commands answer
        off live builds on this workdir or say nothing is running. The first line
        lands under the view; longer answers open a modal too.
        """
        import io

        from rich.console import Console

        buf = io.StringIO()
        screen = self._screen_for(Console(file=buf, width=100, force_terminal=False,
                                          no_color=True))
        try:
            screen.command(line)
        except Exception as exc:
            self.say(f"{type(exc).__name__}: {exc}")
            return
        if screen.model != (self.model or screen.model):
            self.model = screen.model
        if (screen.base_url or "") != self.base_url:
            self.base_url = screen.base_url or ""
        self._refresh_home()
        out = buf.getvalue().strip()
        if not out:
            self.say("done")
            return
        self.say(out.splitlines()[0][:160])
        if "\n" in out:
            self.show_text(line.strip().split()[0], out)

    def _screen_for(self, console: Any) -> Any:
        """A line Screen answering off this workdir, sharing this app's held keys.

        Its ask is stubbed to "n", so a remember question keeps today's
        session-only behaviour instead of reading stdin under the app.
        """
        from kullback.tui import Screen

        screen = Screen(self.workdir, model=self.model, base_url=self.base_url or None,
                        console=console, ceiling_usd=self.ceiling_usd)
        screen.session_keys = self.session_keys
        screen._ask = lambda prompt: "n"  # noqa: E731
        return screen

    def _refresh_home(self) -> None:
        """Home reads again after a command changed the workdir or the keys."""
        if isinstance(self.current, HomeView):
            self.current.session = self.session_keys
            self.current.model = self.model
            self.current.refresh_view()
        self.refresh_header()

    def _help(self) -> None:
        """Every command with its one-line meaning, straight from the shared table."""
        self.show_text("/help", "\n".join(f"/{name:<10} {blurb}" for name, _, blurb in COMMANDS))

    def _login_saved(self, payload: Optional[dict]) -> None:
        """A modal login lands on its default model and Home's key rows update."""
        if not payload:
            return
        if payload["key_var"] not in self.session_keys:
            self.session_keys[payload["key_var"]] = payload["previous"]
        self.model = payload["model"]
        self._refresh_home()
        self.say(f"logged in {payload['provider']}: "
                 f"key {payload['key_var']} saved in ~/.kullback/auth.json")

    def _build_command(self, rest: list[str]) -> None:
        """Open the Build view, prefilling the model, judge model and ceiling it takes."""
        args = self._flag_values(rest)
        if isinstance(args, str):
            self.say(args)
            return
        self.show_build(prefill={"model": args.get("--model"), "judge": args.get("--judge-model"),
                                 "ceiling": args.get("--ceiling-usd")})
        if "--file" in rest:
            self.say("build opened; --file is not a Build field, bring files with /ingest PATH first")
        else:
            self.say("build opened")

    @staticmethod
    def _flag_values(rest: list[str]) -> Any:
        """Flag words to their values, or the words saying a flag needs one."""
        out: dict[str, str] = {}
        index = 0
        while index < len(rest):
            word = rest[index]
            if word.startswith("--"):
                if index + 1 >= len(rest) or rest[index + 1].startswith("--"):
                    return f"{word} needs a value"
                out[word] = rest[index + 1]
                index += 2
            else:
                index += 1
        return out

    def _watch_command(self, rest: list[str]) -> None:
        """Watch this workdir, or session N of the sessions list on its own workdir."""
        from kullback.runner import heartbeat

        records = heartbeat.read_all()
        if rest:
            first = rest[0]
            if not first.isdigit() or not 1 <= int(first) <= len(records):
                if records:
                    self.say(f"pick 1-{len(records)} from /sessions")
                    self.action_sessions()
                else:
                    self.say("no sessions yet: /build starts one here")
                return
            self.workdir = Path(records[int(first) - 1]["workdir"])
        self.show_watch()

    def _ingest(self, rest: list[str]) -> None:
        """Ingest trace files into the workdir with the builder's own entry, then refresh.

        The same ingest_file the CLI calls, one row per file with its runs, tool
        calls and gate result, and the same ingest_summary.json the checklist
        reads, so Home's traces row turns done here too.
        """
        from kullback.builder import ingest as ingest_mod

        floor: Optional[float] = None
        dry = False
        paths: list[str] = []
        index = 0
        while index < len(rest):
            word = rest[index]
            if word == "--dry-run":
                dry = True
                index += 1
            elif word == "--intake-floor":
                if index + 1 >= len(rest):
                    self.say("--intake-floor needs a value between 0 and 1")
                    return
                try:
                    floor = float(rest[index + 1])
                except ValueError:
                    self.say(f"--intake-floor takes a number, not {rest[index + 1]!r}")
                    return
                index += 2
            elif word.startswith("--"):
                self.say(f"unknown flag {word}: /ingest PATH [PATH ...] [--intake-floor N] [--dry-run]")
                return
            else:
                paths.append(word)
                index += 1
        if floor is not None and not 0.0 <= floor <= 1.0:
            self.say("--intake-floor must lie within [0, 1]")
            return
        if not paths:
            self.say("ingest needs a file: /ingest PATH [PATH ...]")
            return
        default_floor = ingest_mod.MIN_TASK_ELIGIBLE_SHARE
        rows: list[str] = []
        if dry:
            for path in paths:
                try:
                    result = ingest_mod.dry_run(path, self.workdir,
                                                intake_floor=default_floor if floor is None else floor)
                except Exception as exc:
                    rows.append(f"{path}: {type(exc).__name__}: {exc}")
                    continue
                rows.append(self._dry_row(path, result))
            self.show_text("/ingest --dry-run", "\n".join(rows))
            self.say(rows[0][:160] if rows else "done")
            return
        summaries: list[dict] = []
        for path in paths:
            try:
                if floor is None:
                    summary = ingest_mod.ingest_file(path, self.workdir)
                else:
                    summary = ingest_mod.ingest_file(path, self.workdir, intake_floor=floor)
            except Exception as exc:
                rows.append(f"{path}: {type(exc).__name__}: {exc}")
                self.show_text("/ingest", "\n".join(rows))
                self.say(rows[-1][:160])
                return
            summaries.append(summary)
            rows.append(self._ingest_row(path, summary))
        try:
            (Path(self.workdir) / "ingest_summary.json").write_text(
                json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            rows.append(f"ingest_summary.json was not written: {exc}")
        self._refresh_home()
        if isinstance(self.current, TracesView):
            self.show_view("traces")
        self.show_text("/ingest", "\n".join(rows))
        self.say(rows[0][:160] if rows else "done")

    def _ingest_row(self, path: str, summary: dict) -> str:
        """One ingested file: its runs, its tool calls and what the gate said."""
        gate = summary.get("gate") or {}
        metrics = gate.get("metrics") or {}
        passed = gate.get("pass", gate.get("passed", True))
        share = float(metrics.get("eligible_share") or 0.0)
        floor = float(metrics.get("floor") or 0.0)
        return (f"{path} format {summary.get('format')} "
                f"traces {summary.get('runs')} calls {summary.get('tool_calls')} "
                f"eligible {share:.2f} vs {floor:.2f} "
                f"tools {self._ingested_tools(str(summary.get('raw_hash') or ''))} "
                f"{'ready' if passed else 'gate fail'}")

    @staticmethod
    def _dry_row(path: str, result: dict) -> str:
        """One dry-run file: whether it would ingest and what it would give."""
        if result.get("passed"):
            status = "ready"
        else:
            failures = result.get("failures") or result.get("reasons") or ["no reason given"]
            status = f"not read: {failures[0]}"
        return (f"{path} format {result.get('format')} confidence {float(result.get('confidence') or 0):.2f} "
                f"traces {result.get('traces')} eligible {float(result.get('eligible_share') or 0):.2f} "
                f"vs {float(result.get('floor') or 0):.2f} tools {result.get('tools')} {status}")

    def _ingested_tools(self, raw_hash: str) -> int:
        """How many distinct tool names the ingested traces call, eligible and set aside."""
        names: set[str] = set()
        for folder in ("traces", "evidence_traces"):
            root = Path(self.workdir) / folder
            if not root.is_dir():
                continue
            for cand in root.glob("*.json"):
                try:
                    body = json.loads(cand.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(body, dict) or body.get("raw_hash") != raw_hash:
                    continue
                for call in body.get("tool_calls") or []:
                    if isinstance(call, dict) and call.get("name"):
                        names.add(call["name"])
        return len(names)

    def _show(self, view: Any) -> None:
        self.body.remove_children()
        self.body.mount(view)
        self.current = view
        if isinstance(view, HomeView):
            view.refresh_view()
        # Keep focus inside the view on something that does not eat keys, once
        # the mount lands: the command line and the views' inputs take keys
        # only once focused, so the view keys keep working after every switch.
        self.call_after_refresh(self._settle_focus, view)

    def _settle_focus(self, view: Any) -> None:
        """Focus the view's first focusable widget that is not an input."""
        from textual.widgets import Input

        if self.current is not view:
            return
        for widget in view.query("*"):
            if widget.focusable and not isinstance(widget, Input):
                widget.focus()
                return

    def show_home(self) -> None:
        self._show(HomeView(self.workdir, self.model, self.base_url, session=self.session_keys))

    def show_build(self, prefill: Optional[dict] = None) -> None:
        """Open the Build view, prefilling the model, judge model and ceiling it takes."""
        view = BuildView(self.workdir, self.model, self.base_url, on_started=self.show_watch)
        for key, field in (("model", "model_init"), ("judge", "judge_init"),
                           ("ceiling", "ceiling_init")):
            value = (prefill or {}).get(key)
            if value:
                setattr(view, field, str(value))
        self._show(view)

    def show_watch(self) -> None:
        self._show(WatchView(self.workdir, self.ceiling_usd))

    def show_view(self, name: str) -> None:
        """Mount a registry view by palette name, over this workdir."""
        self._show(VIEW_CLASSES[name](self.workdir))

    def action_show_home(self) -> None:
        self.show_home()

    def action_show_traces(self) -> None:
        self.show_view("traces")

    def action_show_build(self) -> None:
        self.show_build()

    def action_show_watch(self) -> None:
        self.show_watch()

    def action_show_tasks(self) -> None:
        self.show_view("tasks")

    def action_show_runs(self) -> None:
        self.show_view("runs")

    def action_palette(self) -> None:
        self.push_screen(PaletteModal(self._palette_chosen))

    def _palette_chosen(self, name: Optional[str]) -> None:
        if name in ("home", "build", "watch"):
            {"home": self.show_home, "build": self.show_build,
             "watch": self.show_watch}[name]()
        elif name in VIEW_CLASSES:
            self.show_view(str(name))
        elif name == "sessions":
            self.action_sessions()
        elif name == "quit":
            self.action_quit_app()

    def action_sessions(self) -> None:
        self.push_screen(SessionsModal(self._session_chosen))

    def _session_chosen(self, record: dict) -> None:
        from pathlib import Path as _Path

        here = _Path(self.workdir).expanduser().absolute()
        there = _Path(str(record.get("workdir") or "")).expanduser().absolute()
        if there == here:
            self.show_watch()
        else:
            self.notify(f"that build lives in {record.get('workdir')}")

    def action_keys(self) -> None:
        self.push_screen(KeysModal())

    def action_confirm_stop(self) -> None:
        """Esc anywhere asks before stopping, since the watch input may not hold focus."""
        if isinstance(self.current, WatchView):
            self.current.action_confirm_stop()

    def action_quit_app(self) -> None:
        self.exit()
