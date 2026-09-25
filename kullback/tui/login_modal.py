"""Bare /login inside the app: pick a provider, paste its key, save it.

The modal offers the providers the code already knows and starts each at its
default model, the same list the line screen's login menu walks. Saving writes
the key to the remembered store exactly as `kullback login` does, through
credentials.remember, and holds it in this process too. Names are shown,
values never, and nothing about the key is printed.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from textual import on
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static


class LoginModal(ModalScreen[Optional[dict]]):
    """One provider, one masked key field, save or cancel."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self.providers: dict[str, str] = self._providers()
        self.message = Static("", id="login-message")
        self.provider_select = Select(
            [(name, name) for name in self.providers] or [("none", "none")],
            value=next(iter(self.providers), "none"),
            id="login-provider",
        )
        self.key_input = Input(password=True, placeholder="paste the key", id="login-key")

    @staticmethod
    def _providers() -> dict[str, str]:
        """The providers /login offers and the model each starts at, never empty."""
        from kullback.tui import LOGIN_DEFAULT_MODELS, Screen

        try:
            defaults = Screen._login_defaults()
        except Exception:
            defaults = {}
        if defaults:
            return dict(defaults)
        return dict(LOGIN_DEFAULT_MODELS)

    def compose(self):  # type: ignore[override]
        yield Vertical(
            Static("log in: pick a provider, paste its key", id="login-title"),
            self.provider_select,
            self.key_input,
            self.message,
            Button("save", id="login-save"),
            Button("cancel", id="login-cancel"),
            id="login-form",
        )

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#login-cancel")
    def _cancelled(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#login-save")
    def _save_pressed(self) -> None:
        self.save()

    def save(self) -> Optional[dict]:
        """Save the typed key under its provider's variable and dismiss with what changed.

        Returns the dismissal payload, or None when nothing was saved. The payload
        names the provider, the key variable, the model the app now builds on, and
        what the shell held before, so the app can restore it on /logout.
        """
        from kullback.ai import credentials
        from kullback.tui import Screen

        provider = str(self.provider_select.value or "")
        if provider not in self.providers:
            self.message.update("pick a provider from the list")
            return None
        secret = self.key_input.value
        if not secret:
            self.message.update("empty key: nothing saved")
            return None
        key_var = Screen._key_var_for(provider, self.providers[provider])
        previous = os.environ.get(key_var)
        try:
            credentials.remember(key_var, secret)
        except ValueError as exc:
            self.message.update(f"{exc}: kept for this session only")
            return None
        os.environ[key_var] = secret
        payload: dict[str, Any] = {
            "provider": provider,
            "key_var": key_var,
            "model": self.providers[provider],
            "previous": previous,
        }
        self.dismiss(payload)
        return payload
