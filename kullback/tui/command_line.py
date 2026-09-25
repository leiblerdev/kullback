"""The command line at the bottom of the app: slash commands inside KullbackApp.

Always on screen under every view, like the line screen's prompt. Typing /
opens the / menu filtered as you type, off the same COMMANDS table and
filter_commands the line screen reads, so the two never list different
commands. Tab completes to the first match, enter runs, esc closes the menu.
Enter on a menu row runs it; enter twice on a fragment with many matches runs
the highlighted row. Menu rebuilds await the ListView's own clear, since an
unawaited clear leaves the old rows beside the new ones under fast typing.
"""

from __future__ import annotations

from typing import Callable, Optional

from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, Label, ListItem, ListView

from kullback.tui import COMMANDS, filter_commands


class CommandLine(Vertical):
    """The app's prompt and its / menu, calling back with each line to run."""

    BINDINGS = [
        Binding("tab", "complete", "Complete", show=False),
        Binding("escape", "close_menu", "Close menu", show=False),
    ]

    def __init__(self, on_run: Callable[[str], None]) -> None:
        super().__init__(id="cmdline")
        self._on_run = on_run
        self.cmd_input = Input(placeholder="type / for commands", id="cmd-input")
        self.menu = ListView(id="cmd-menu")
        self._matches: list = []
        self._asked: Optional[str] = None
        self.menu.display = False

    def compose(self):  # type: ignore[override]
        yield self.cmd_input
        yield self.menu

    def prefill(self, text: str) -> None:
        """Put text in the line and focus it, so a checklist fix is one enter away."""
        self.cmd_input.value = text
        self.cmd_input.focus()

    def focus_line(self) -> None:
        self.cmd_input.focus()

    def menu_names(self) -> list[str]:
        """The menu's command names now, for tests to read without rendering."""
        return [row[0] for row in self._matches]

    def check_action(self, action: str, parameters: tuple) -> Optional[bool]:
        """Tab and esc only answer when they have something to do here."""
        if action == "complete":
            value = self.cmd_input.value
            if not value.startswith("/"):
                return False
            first, _, _ = value[1:].partition(" ")
            if not filter_commands(first):
                return False
        if action == "close_menu" and not self.menu.display:
            return False
        return True

    def action_complete(self) -> None:
        """Tab completes to the first match, keeping any arguments already typed."""
        value = self.cmd_input.value
        if not value.startswith("/"):
            return
        first, _, rest = value[1:].partition(" ")
        matches = filter_commands(first)
        if not matches:
            return
        completed = "/" + matches[0][0]
        completed += " " + rest.strip() + " " if rest.strip() else " "
        self.cmd_input.value = completed
        self.cmd_input.focus()

    async def action_close_menu(self) -> None:
        """Esc closes the menu and keeps the typed text for editing."""
        await self._hide_menu()
        self.cmd_input.focus()

    @on(Input.Changed, "#cmd-input")
    async def _filtered(self, event: Input.Changed) -> None:
        """A / opens the menu filtered by its first word; anything else closes it."""
        value = event.value
        if not value.startswith("/"):
            await self._hide_menu()
            return
        words = value[1:].split()
        self._matches = filter_commands(words[0] if words else "")
        self._asked = None
        await self._show_menu()

    @on(Input.Submitted, "#cmd-input")
    async def _submitted(self, event: Input.Submitted) -> None:
        """Enter runs: an exact name whole, one match completed, many picked by repeat."""
        line = event.value.strip()
        if not line:
            return
        self.cmd_input.value = ""
        await self.run_text(line)

    async def run_text(self, line: str) -> None:
        """Run one line, completing a unique fragment on the way."""
        text = line.strip()
        if not text:
            return
        if text.startswith("/"):
            first, _, rest = text[1:].partition(" ")
            names = {row[0] for row in COMMANDS}
            if first in names:
                await self._hide_menu()
                self._on_run(text)
                return
            matches = filter_commands(first)
            if len(matches) == 1:
                await self._hide_menu()
                extra = " " + rest.strip() if rest.strip() else ""
                self._on_run("/" + matches[0][0] + extra)
                return
            if matches:
                if self.menu.display and text == self._asked:
                    await self._run_highlighted(rest.strip())
                    return
                self._matches = matches
                self._asked = text
                await self._show_menu()
                return
        await self._hide_menu()
        self._on_run(text)

    @on(ListView.Selected, "#cmd-menu")
    async def _picked(self, event: ListView.Selected) -> None:
        """Enter or click on a menu row runs that command with no arguments."""
        try:
            index = list(self.menu.children).index(event.item)
        except ValueError:
            return
        if index >= len(self._matches):
            return
        name = self._matches[index][0]
        self.cmd_input.value = ""
        await self._hide_menu()
        self._on_run("/" + name)

    async def _run_highlighted(self, args: str) -> None:
        """The second enter on an open menu runs the highlighted row with the typed args."""
        try:
            index = int(self.menu.index)
        except (TypeError, ValueError):
            index = 0
        if not (0 <= index < len(self._matches)):
            index = 0
        name = self._matches[index][0]
        extra = " " + args if args else ""
        self.cmd_input.value = ""
        await self._hide_menu()
        self._on_run("/" + name + extra)

    async def _show_menu(self) -> None:
        """The matches as rows, or one row saying nothing matches."""
        await self.menu.clear()
        if self._matches:
            rows = [ListItem(Label(f"/{name:<10} {blurb}"), id=f"cmd-{name}")
                    for name, _, blurb in self._matches]
        else:
            rows = [ListItem(Label("no command matches; /help lists every command"),
                             id="cmd-none", disabled=True)]
        await self.menu.mount(*rows)
        self.menu.display = True
        self.menu.index = 0

    async def _hide_menu(self) -> None:
        await self.menu.clear()
        self.menu.display = False
        self._matches = []
        self._asked = None
