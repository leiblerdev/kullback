"""Start a capped build from the Textual app: the Build view.

The person types the model, the judge model and the ceiling, and enter starts
the build. Refusals use the line screen's own sentences, so the two screens
never disagree about why a build cannot start. Starting launches the same
`kullback build` the CLI runs as a detached child, so quitting the app leaves
the build running and the heartbeat, feed and bus land where Watch reads them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

from textual.containers import Vertical
from textual.widgets import Button, Input, Static

from kullback.ai.provider import DEFAULT_MODEL

# What the judge model field shows before the person types: the judges run on
# the build model itself unless named otherwise, the way `kullback build` does.
JUDGE_SAME = "same"


def last_ceiling_usd(workdir: Any) -> str:
    """The last ceiling used in this workdir, or nothing where none was recorded."""
    try:
        total = json.loads((Path(workdir) / "budget.json").read_text(encoding="utf-8")).get("total")
    except (OSError, ValueError):
        return ""
    if not isinstance(total, dict):
        return ""
    for key in ("ceiling_usd", "ceiling"):
        try:
            value = float(total.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return str(value)
    return ""


def default_launcher(argv: list[str], workdir: Any) -> subprocess.Popen:
    """Start the build detached: its own session, its output to workdir/build.log."""
    log = open(Path(workdir) / "build.log", "ab")
    try:
        return subprocess.Popen(argv, start_new_session=True, stdout=log, stderr=log,
                                stdin=subprocess.DEVNULL)
    finally:
        log.close()


class BuildView(Vertical):
    """Model, judge model, ceiling, and a start that refuses before spawning."""

    def __init__(self, workdir: Any, model: Optional[str] = None, base_url: str = "",
                 launcher: Optional[Callable[[list[str]], Any]] = None,
                 on_started: Optional[Callable[[], None]] = None) -> None:
        super().__init__(id="build")
        self.workdir = Path(workdir)
        self.base_url = base_url
        self.launcher = launcher or (lambda argv: default_launcher(argv, self.workdir))
        self.on_started = on_started
        # Initial values stay plain strings: an Input with a value needs the app
        # running, so the widgets stay empty until compose fills them in.
        self.model_init = model or DEFAULT_MODEL
        self.judge_init = JUDGE_SAME
        self.ceiling_init = last_ceiling_usd(workdir)
        self.model_input = Input(placeholder="model", id="model")
        self.judge_input = Input(placeholder="judge model", id="judge-model")
        self.ceiling_input = Input(placeholder="ceiling in USD", id="ceiling")
        self.message = Static("", id="build-message")
        self.start_button = Button("start build", id="start")

    def compose(self):  # type: ignore[override]
        self.model_input.value = self.model_init
        self.judge_input.value = self.judge_init
        self.ceiling_input.value = self.ceiling_init
        yield Static("model", classes="label")
        yield self.model_input
        yield Static("judge model", classes="label")
        yield self.judge_input
        yield Static("ceiling in USD", classes="label")
        yield self.ceiling_input
        yield self.start_button
        yield self.message

    def refusal(self) -> Optional[str]:
        """Why the build will not start, or None when it may.

        The first two sentences are the line screen's own: the check runs
        through Screen._build_refusal on this view's settings, so the wording
        lives in one place. The ceiling check is this view's own, since only
        this view asks for a ceiling.
        """
        from kullback.tui import Screen

        probe = SimpleNamespace(workdir=self.workdir, model=self.model_value(),
                                base_url=self.base_url)
        refusal = Screen._build_refusal(probe)
        if refusal is not None:
            return refusal
        try:
            ceiling = float((self.ceiling_input.value or "").strip())
        except ValueError:
            return "the ceiling needs a positive number in USD"
        if ceiling <= 0:
            return "the ceiling needs a positive number in USD"
        return None

    def model_value(self) -> str:
        return (self.model_input.value or "").strip() or DEFAULT_MODEL

    def judge_value(self) -> Optional[str]:
        value = (self.judge_input.value or "").strip()
        return None if value.lower() in ("", JUDGE_SAME) else value

    def ceiling_value(self) -> str:
        return (self.ceiling_input.value or "").strip()

    def build_argv(self) -> list[str]:
        """The `kullback build` this view starts, as the test records it."""
        argv = [sys.executable, "-m", "kullback.cli", "build",
                "--workdir", str(self.workdir),
                "--model", self.model_value(),
                "--ceiling-usd", self.ceiling_value()]
        judge = self.judge_value()
        if judge is not None:
            argv += ["--judge-model", judge]
        if self.base_url:
            argv += ["--base-url", self.base_url]
        return argv

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button is self.start_button:
            self.start_build()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        self.start_build()

    def start_build(self) -> bool:
        """Refuse or launch; True when a child was started."""
        refusal = self.refusal()
        if refusal is not None:
            self.message.update(refusal)
            return False
        self.message.update("")
        self.launcher(self.build_argv())
        if self.on_started is not None:
            self.on_started()
        return True
