"""The Textual app shell: header, tab routing, palette, sessions, keys, quit.

One full-screen app over a workdir. It opens on Home, or on Watch when a build
is live in this workdir. Keys 2, 5 and 6 and the palette entries traces, tasks,
runs, publish and synthesise mount those views from kullback.tui.views.
Quitting never touches the build child, which runs detached.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from kullback.ai.provider import DEFAULT_MODEL
from kullback.tui import live_heartbeats, status_segments
from kullback.tui.build_view import BuildView
from kullback.tui.home import HomeView
from kullback.tui.views import VIEWS
from kullback.tui.watch import WatchView

APP_CSS = """
#hdr { height: 1; }
#view { height: 1fr; }
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

class MessageModal(ModalScreen[None]):
    """One palette command's output as plain text: status, keys, login, logout, help."""

    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self._title = title
        self._body = body

    def compose(self):  # type: ignore[override]
        yield Static(f"{self._title}", id="message-title")
        yield Static(self._body, id="message-body")

    BINDINGS = [Binding("escape", "dismiss", "Close")]



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
        # takes them first. Escape stays unprioritised so a modal keeps it.
        Binding("ctrl+k", "palette", "Commands", priority=True),
        Binding("ctrl+r", "sessions", "Sessions", priority=True),
        Binding("question_mark", "keys", "Keys"),
        Binding("ctrl+d", "quit_app", "Quit", priority=True),
        Binding("escape", "confirm_stop", "Stop"),
    ]

    def __init__(self, workdir: Any, model: Optional[str] = None, base_url: Optional[str] = None,
                 ceiling_usd: Optional[float] = None) -> None:
        super().__init__()
        self.workdir = Path(workdir)
        self.model = model
        self.base_url = base_url or ""
        self.ceiling_usd = ceiling_usd
        self.header = Static("", id="hdr")
        self.body = Vertical(id="view")
        self.current: Any = None

    def compose(self) -> ComposeResult:
        yield self.header
        yield self.body

    def on_mount(self) -> None:
        if live_heartbeats(self.workdir):
            self.show_watch()
        else:
            self.show_home()
        self.refresh_header()
        self.set_interval(5.0, self.refresh_header)

    def check_action(self, action: str, parameters: tuple) -> Optional[bool]:
        """Number keys and ? type into inputs instead of switching views there."""
        from textual.widgets import Input

        if action in ("show_home", "show_traces", "show_build", "show_watch", "show_tasks",
                      "show_runs", "keys") and isinstance(self.focused, Input):
            return False
        return True

    def refresh_header(self) -> None:
        """The workdir, its total spend, and where the commands live."""
        out = Text(f"kullback  {self.workdir} ", style="bold")
        out.append_text(status_segments(self.workdir, self.model))
        out.append("   ctrl+k commands  ? keys", style="dim")
        self.header.update(out)

    def _show(self, view: Any) -> None:
        self.body.remove_children()
        self.body.mount(view)
        self.current = view
        if isinstance(view, HomeView):
            view.refresh_view()

    def show_home(self) -> None:
        self._show(HomeView(self.workdir, self.model, self.base_url))

    def show_build(self) -> None:
        self._show(BuildView(self.workdir, self.model, self.base_url,
                             on_started=self.show_watch))

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
        action = self._command_table().get(str(name) if name else "")
        if action is not None:
            action()

    def _command_table(self) -> dict[str, Callable[[], None]]:
        """Every palette command in one table, so the palette cannot drift from the handlers.

        View names mount their registry view; status, keys, login, logout and help show
        what the same slash command prints on the line screen, through its own renderer."""
        table: dict[str, Callable[[], None]] = {
            "home": self.show_home,
            "build": self.show_build,
            "watch": self.show_watch,
            "status": self.show_status,
            "sessions": self.action_sessions,
            "keys": self.show_keys,
            "login": self.show_login,
            "logout": self.run_logout,
            "help": self.show_help,
            "quit": self.action_quit_app,
        }
        for view_name in VIEW_CLASSES:
            table.setdefault(view_name, lambda name=view_name: self.show_view(name))
        return table

    def show_status(self) -> None:
        """Palette status shows what slash status prints: the last build read off disk."""
        from kullback.tui import Screen

        renderable = Screen(self.workdir, self.model, self.base_url)._status_renderable()
        plain = renderable.plain if hasattr(renderable, "plain") else str(renderable)
        self.push_screen(MessageModal("status", plain))

    def show_keys(self) -> None:
        """Palette keys shows what slash keys prints: which provider keys are visible."""
        import os

        from kullback.tui import _keys

        model = self.model or DEFAULT_MODEL
        self.push_screen(MessageModal(
            "keys", _keys(dict(os.environ), model=model, host=self.base_url or "").plain))

    def show_login(self) -> None:
        """Palette login shows what slash login prints: the model and its key sources."""
        from kullback.tui import Screen

        text = Screen(self.workdir, self.model, self.base_url)._login_status()
        self.push_screen(MessageModal("login", text.plain))

    def run_logout(self) -> None:
        """Palette logout runs the slash logout handler, showing what it cleared."""
        from io import StringIO

        from rich.console import Console

        from kullback.tui import Screen

        console = Console(file=StringIO(), width=100)
        Screen(self.workdir, self.model, self.base_url, console=console)._logout([])
        self.push_screen(MessageModal("logout", str(console.file.getvalue())))

    def show_help(self) -> None:
        """Palette help shows the slash help text itself, so the two cannot drift."""
        from kullback.tui import HELP

        self.push_screen(MessageModal("help", HELP))

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
