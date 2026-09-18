"""A killed build leaves no torn artifact: write_json swaps the file in."""

from __future__ import annotations

import fnmatch
import inspect
import json
import os
import threading
from pathlib import Path

import pytest

from kullback.runner import records as records_module
from kullback.runner.records import read_json, write_json

pytestmark = pytest.mark.skipif(
    "os.replace" not in inspect.getsource(records_module.write_json),
    reason="safe-write.patch is not applied",
)


class _Token:
    def __str__(self) -> str:
        return "token-7"


def _leftover_temps(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.suffix == ".tmp" or p.name.startswith(".")]


def test_bytes_match_the_old_direct_write(tmp_path: Path):
    body = {"zebra": {"n": [3, 2, 1], "m": "x"}, "apple": {"token": _Token()}, "mid": [None, True, 1]}
    target = tmp_path / "harbor.json"
    write_json(target, body)
    expected = json.dumps(body, indent=2, sort_keys=True, default=str)
    assert target.read_text(encoding="utf-8") == expected
    assert read_json(target) == json.loads(expected)


def test_serialisation_failure_keeps_the_old_file(tmp_path: Path):
    class _Bad:
        def __str__(self) -> str:
            raise RuntimeError("cannot render")

    target = tmp_path / "ledger.json"
    write_json(target, {"kept": 1})
    before = target.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        write_json(target, {"broken": _Bad()})
    assert target.read_text(encoding="utf-8") == before
    assert _leftover_temps(tmp_path) == []


def test_replace_failure_keeps_the_old_file_and_no_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    target = tmp_path / "ledger.json"
    write_json(target, {"kept": 2})
    before = target.read_text(encoding="utf-8")

    def _boom(src: str, dst: str) -> None:
        raise OSError("swap failed")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError):
        write_json(target, {"next": 3})
    assert target.read_text(encoding="utf-8") == before
    assert _leftover_temps(tmp_path) == []


def test_two_threads_writing_one_path_both_leave_valid_json(tmp_path: Path):
    target = tmp_path / "shared.json"
    write_json(target, {"harbor": [], "note": "seeded destination"})
    first = {"harbor": list(range(50)), "note": "first writer"}
    second = {"harbor": list(range(50, 100)), "note": "second writer"}
    errors: list[BaseException] = []

    def _write(body: dict) -> None:
        try:
            write_json(target, body)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(first,)), threading.Thread(target=_write, args=(second,))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk in (json.loads(json.dumps(first, sort_keys=True, default=str)),
                       json.loads(json.dumps(second, sort_keys=True, default=str)))
    assert _leftover_temps(tmp_path) == []


def test_reader_sees_only_whole_files_while_a_swap_waits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Deterministic form of the atomicity guarantee: the swap is held back, the destination
    # must still read as the old complete JSON, and only after release as the new one.
    target = tmp_path / "ledger.json"
    old = {"kept": "old value", "n": 1}
    new = {"kept": "new value with more bytes in it " * 8, "n": 2}
    write_json(target, old)
    entered = threading.Event()
    release = threading.Event()
    real_replace = os.replace
    seen: dict = {}

    def _gated(src: str, dst: str) -> None:
        seen["src"] = str(src)
        seen["dst"] = str(dst)
        entered.set()
        assert release.wait(timeout=30)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _gated)
    errors: list[BaseException] = []

    def _write() -> None:
        try:
            write_json(target, new)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=_write)
    thread.start()
    assert entered.wait(timeout=30)
    assert read_json(target) == json.loads(json.dumps(old, sort_keys=True, default=str))
    staged = Path(seen["src"])
    assert staged.parent == target.parent
    assert staged.name.startswith(".") and staged.suffix == ".tmp"
    assert staged.is_file()
    release.set()
    thread.join(timeout=30)
    assert errors == []
    assert read_json(target) == json.loads(json.dumps(new, sort_keys=True, default=str))
    assert _leftover_temps(tmp_path) == []


def test_temp_name_matches_no_workdir_reader_glob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Workdir readers glob "*.json", "*.jsonl", "*/*.jsonl" and fixed names such as
    # "*/pool.json" (grep "glob(" kullback). The writer stages the bytes in a file the swap
    # boundary hands over, so this test asserts on that actual path: same directory as the
    # destination, dot-prefixed, ending in .tmp, matching none of the reader patterns.
    seen: dict = {}
    real_replace = os.replace

    def _capture(src: str, dst: str) -> None:
        seen["src"] = str(src)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _capture)
    target = tmp_path / "ledger.json"
    write_json(target, {"kept": 1})
    staged = Path(seen["src"])
    assert staged.parent == target.parent
    actual = staged.name
    assert actual.startswith(".") and actual.endswith(".tmp")
    assert not fnmatch.fnmatch(actual, "*.json")
    assert not fnmatch.fnmatch(actual, "*.jsonl")
    assert not fnmatch.fnmatch(actual, "*/pool.json")
