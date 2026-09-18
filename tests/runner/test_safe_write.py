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


def test_temp_name_matches_no_workdir_reader_glob():
    # Workdir readers glob "*.json", "*.jsonl", "*/*.jsonl" and fixed names such as
    # "*/pool.json" (grep "glob(" kullback). The temp file is ".<final name>.<unique>.tmp"
    # in the same directory, so it ends in .tmp and starts with a dot, and none of those
    # patterns match it.
    sample = ".ledger.json.8f3k.tmp"
    assert not fnmatch.fnmatch(sample, "*.json")
    assert not fnmatch.fnmatch(sample, "*.jsonl")
    assert not fnmatch.fnmatch(sample, "*/pool.json")
    assert sample.endswith(".tmp") and sample.startswith(".")
