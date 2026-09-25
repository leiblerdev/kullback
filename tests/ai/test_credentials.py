"""The remembered key store: written once at 0600, read on every later launch. No network."""

from __future__ import annotations

import os
import stat

import pytest

from kullback.ai import credentials


def _isolated(monkeypatch, tmp_path):
    """Point the store at a tmp file, with no keys exported: no test touches the real home."""
    path = tmp_path / "auth.json"
    monkeypatch.setenv(credentials.AUTH_ENV_VAR, str(path))
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENCODE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return path


def test_a_remembered_key_is_written_at_mode_0600_and_read_back(monkeypatch, tmp_path):
    path = _isolated(monkeypatch, tmp_path)
    where = credentials.remember("OPENAI_API_KEY", "sk-remembered")
    assert where == path
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700
    assert credentials.stored_names() == ["OPENAI_API_KEY"]
    assert credentials.load_credentials({}) == {"OPENAI_API_KEY": "sk-remembered"}


def test_a_remembered_key_never_overrides_an_exported_one(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    credentials.remember("OPENAI_API_KEY", "sk-remembered")
    env = {"OPENAI_API_KEY": "sk-exported"}
    assert credentials.load_credentials(env) == {}
    assert env == {"OPENAI_API_KEY": "sk-exported"}


def test_forget_removes_only_that_key(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    credentials.remember("OPENAI_API_KEY", "sk-one")
    credentials.remember("ANTHROPIC_API_KEY", "sk-two")
    assert credentials.forget("OPENAI_API_KEY") is True
    assert credentials.forget("OPENAI_API_KEY") is False
    assert credentials.stored_names() == ["ANTHROPIC_API_KEY"]
    assert credentials.load_credentials({}) == {"ANTHROPIC_API_KEY": "sk-two"}


def test_loading_a_wide_file_reads_it_and_fixes_it_to_0600(monkeypatch, tmp_path):
    path = _isolated(monkeypatch, tmp_path)
    credentials.remember("OPENAI_API_KEY", "sk-remembered")
    os.chmod(path, 0o644)
    assert credentials.load_credentials({}) == {"OPENAI_API_KEY": "sk-remembered"}
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_a_file_that_does_not_parse_adds_nothing(monkeypatch, tmp_path):
    path = _isolated(monkeypatch, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert credentials.load_credentials({}) == {}
    assert credentials.stored_names() == []


def test_a_name_outside_the_pattern_is_refused(monkeypatch, tmp_path):
    path = _isolated(monkeypatch, tmp_path)
    for bad in ("lowercase", "A", "HAS SPACE", "9LIVES", "x" * 65):
        with pytest.raises(ValueError):
            credentials.remember(bad, "sk-nope")
        with pytest.raises(ValueError):
            credentials.forget(bad)
    assert not path.exists()
    assert credentials.stored_names() == []
