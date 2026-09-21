from __future__ import annotations

import copy

import pytest

from kullback.laws import (
    ERROR_CHANGED_NOTHING,
    MINTED_IDS_UNIQUE,
    NOTHING_CRASHES_OR_HANGS,
    READ_CHANGES_NOTHING,
    SAME_WRITE_SAME_RESULT,
    SUCCESS_CHANGED_SOMETHING,
    WRITE_CAN_BE_READ_BACK,
    ArgSpec,
    Outcome,
    ToolInfo,
    check_laws,
)


def _scan_tool():
    return ToolInfo("scan", "read", [ArgSpec("slot", "int")])


def _store_tool():
    return ToolInfo("store", "write", [ArgSpec("item", "int")])


class HonestWorld:
    def __init__(self):
        self._cells = {}
        self._next = 1
        self._reads = 0

    def reset(self):
        self._cells = {}
        self._next = 1
        self._reads = 0

    def tools(self):
        return [_scan_tool(), _store_tool()]

    def snapshot(self):
        return {"cells": dict(self._cells), "next": self._next, "reads": self._reads}

    def _read(self, args):
        return Outcome(True, self._cells.get(args.get("slot")))

    def _write(self, args):
        item = args.get("item")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return Outcome(False, None, "bad item")
        new = self._next
        self._cells[new] = item
        self._next += 1
        return Outcome(True, new, None)

    def call(self, name, args):
        if name == "scan":
            return self._read(args)
        if name == "store":
            return self._write(args)
        return Outcome(False, None, "unknown tool")


class LoggingReadWorld(HonestWorld):
    def _read(self, args):
        self._reads += 1
        return Outcome(True, self._cells.get(args.get("slot")))


class CounterWriteWorld(HonestWorld):
    def __init__(self):
        super().__init__()
        self._hidden = 0

    def reset(self):
        super().reset()

    def _write(self, args):
        item = args.get("item")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return Outcome(False, None, "bad item")
        self._hidden += 1
        new = self._next
        self._cells[new] = (item, self._hidden)
        self._next += 1
        return Outcome(True, self._hidden, None)


class NoopWriteWorld(HonestWorld):
    def _write(self, args):
        return Outcome(True, None, None)


class HalfWriteWorld(HonestWorld):
    def __init__(self):
        super().__init__()
        self._marks = 0

    def reset(self):
        super().reset()
        self._marks = 0

    def snapshot(self):
        snap = super().snapshot()
        snap["marks"] = self._marks
        return snap

    def _write(self, args):
        self._marks += 1
        return Outcome(False, None, "bad item")


class GhostIdWorld(HonestWorld):
    def _write(self, args):
        item = args.get("item")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return Outcome(False, None, "bad item")
        new = self._next
        self._cells[new] = item
        self._next += 1
        return Outcome(True, new + 1000, None)


class RepeatIdWorld(HonestWorld):
    def __init__(self):
        super().__init__()
        self._last = None

    def reset(self):
        super().reset()
        self._last = None

    def snapshot(self):
        snap = super().snapshot()
        snap["last"] = self._last
        return snap

    def _write(self, args):
        item = args.get("item")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return Outcome(False, None, "bad item")
        new = self._next
        self._cells[new] = item
        self._next += 1
        self._last = "kid-1"
        return Outcome(True, "kid-1", None)


class RaisingWorld:
    def reset(self):
        pass

    def tools(self):
        return [ToolInfo("zap", "write", [])]

    def snapshot(self):
        return {"cells": {}}

    def call(self, name, args):
        raise RuntimeError("boom")


class EmptyWorld:
    def reset(self):
        pass

    def tools(self):
        return []

    def snapshot(self):
        return {}

    def call(self, name, args):
        raise AssertionError("must not be called")


def _run(world, **kwargs):
    params = {"seed": 11, "sequences": 12, "max_length": 4, "shrink_evaluations": 500}
    params.update(kwargs)
    return check_laws(world, **params)


def test_honest_world_holds_every_law():
    report = _run(HonestWorld())
    assert report.violations == ()
    assert report.consistency == 1.0
    assert sum(report.broken.values()) == 0
    assert sum(report.checked.values()) > 0


@pytest.mark.parametrize(
    "world_cls,law,tool,length",
    [
        (LoggingReadWorld, READ_CHANGES_NOTHING, "scan", 1),
        (CounterWriteWorld, SAME_WRITE_SAME_RESULT, "store", 1),
        (NoopWriteWorld, SUCCESS_CHANGED_SOMETHING, "store", 1),
        (HalfWriteWorld, ERROR_CHANGED_NOTHING, "store", 1),
        (GhostIdWorld, WRITE_CAN_BE_READ_BACK, "store", 1),
        (RepeatIdWorld, MINTED_IDS_UNIQUE, "store", 2),
        (RaisingWorld, NOTHING_CRASHES_OR_HANGS, "zap", 1),
    ],
    ids=[
        "read-mutates",
        "hidden-counter",
        "noop-success",
        "half-write",
        "ghost-id",
        "repeat-id",
        "raises",
    ],
)
def test_flawed_world_reports_its_law(world_cls, law, tool, length):
    report = _run(world_cls())
    assert [(v.law, v.tool) for v in report.violations] == [(law, tool)]
    (violation,) = report.violations
    assert len(violation.sequence) == length
    assert all(step.tool == tool for step in violation.sequence)
    assert report.consistency is not None
    assert report.consistency < 1.0


def test_same_seed_gives_same_report():
    first = _run(HonestWorld())
    second = _run(HonestWorld())
    assert first == second


def test_exemption_silences_exactly_one_law_for_one_tool():
    exempt = {"scan": [READ_CHANGES_NOTHING]}
    report = _run(LoggingReadWorld(), exempt=exempt)
    assert report.violations == ()
    assert report.consistency == 1.0
    other = _run(LoggingReadWorld(), exempt={"store": [READ_CHANGES_NOTHING]})
    assert [(v.law, v.tool) for v in other.violations] == [(READ_CHANGES_NOTHING, "scan")]


def test_empty_tool_list_gives_consistency_none():
    report = _run(EmptyWorld())
    assert report.consistency is None
    assert report.violations == ()
    assert report.checked == {}


class TypedWorld:
    def __init__(self):
        self._rows = []
        self._meta = {"flag": True, "rate": 1.5, "tag": "ok"}

    def reset(self):
        self._rows = []
        self._meta = {"flag": True, "rate": 1.5, "tag": "ok"}

    def tools(self):
        return [
            ToolInfo("fetch", "read", [ArgSpec("slot", "int")]),
            ToolInfo(
                "append",
                "write",
                [
                    ArgSpec("label", "str"),
                    ArgSpec("score", "float"),
                    ArgSpec("chosen", "bool"),
                    ArgSpec("note", "str", True),
                ],
            ),
        ]

    def snapshot(self):
        rows = copy.deepcopy(self._rows)
        meta = copy.deepcopy(self._meta)
        return {"rows": rows, "mirror": rows, "meta": meta, "copy": meta}

    def call(self, name, args):
        if name == "fetch":
            slot = args.get("slot")
            if isinstance(slot, bool) or not isinstance(slot, int):
                return Outcome(False, None, "bad slot")
            if slot < 0 or slot >= len(self._rows):
                return Outcome(False, None, "no row")
            return Outcome(True, self._rows[slot])
        if name == "append":
            row = {
                "id": len(self._rows),
                "label": args.get("label"),
                "score": args.get("score"),
                "chosen": args.get("chosen"),
            }
            self._rows.append(row)
            return Outcome(True, dict(row))
        return Outcome(False, None, "unknown tool")


class BadTypeWorld:
    def reset(self):
        pass

    def tools(self):
        return [ToolInfo("store", "write", [ArgSpec("blob", "bytes")])]

    def snapshot(self):
        return {}

    def call(self, name, args):
        return Outcome(True, None, None)


class FlakyWorld:
    def __init__(self):
        self._cells = {}
        self._calls = 0

    def reset(self):
        self._cells = {}

    def tools(self):
        return [_store_tool()]

    def snapshot(self):
        return dict(self._cells)

    def call(self, name, args):
        self._calls += 1
        if self._calls > 1:
            raise RuntimeError("flaky")
        item = args.get("item")
        if isinstance(item, bool) or not isinstance(item, int):
            return Outcome(False, None, "bad item")
        self._cells[0] = item
        return Outcome(True, 0, None)


def test_typed_world_holds_every_law():
    report = _run(TypedWorld(), sequences=30)
    assert report.violations == ()
    assert report.consistency == 1.0


def test_unsupported_argument_type_is_rejected():
    with pytest.raises(ValueError):
        _run(BadTypeWorld())


def test_flaky_world_reports_replay_failure_and_crash():
    report = _run(FlakyWorld())
    assert {(v.law, v.tool) for v in report.violations} == {
        (SAME_WRITE_SAME_RESULT, "store"),
        (NOTHING_CRASHES_OR_HANGS, "store"),
    }


def test_exemption_silences_error_law_for_half_write_tool():
    report = _run(HalfWriteWorld(), exempt={"store": [ERROR_CHANGED_NOTHING]})
    assert report.violations == ()
    assert report.consistency == 1.0


def test_full_exemption_checks_nothing():
    exempt = {
        "scan": [
            READ_CHANGES_NOTHING,
            SAME_WRITE_SAME_RESULT,
            SUCCESS_CHANGED_SOMETHING,
            ERROR_CHANGED_NOTHING,
            WRITE_CAN_BE_READ_BACK,
            MINTED_IDS_UNIQUE,
            NOTHING_CRASHES_OR_HANGS,
        ],
        "store": [
            READ_CHANGES_NOTHING,
            SAME_WRITE_SAME_RESULT,
            SUCCESS_CHANGED_SOMETHING,
            ERROR_CHANGED_NOTHING,
            WRITE_CAN_BE_READ_BACK,
            MINTED_IDS_UNIQUE,
            NOTHING_CRASHES_OR_HANGS,
        ],
    }
    report = _run(LoggingReadWorld(), exempt=exempt)
    assert report.violations == ()
    assert report.checked == {}
    assert report.consistency is None


@pytest.mark.parametrize(
    "kwargs",
    [{"max_length": 0}, {"sequences": -1}, {"call_budget_s": 0.0}],
    ids=["max-length", "sequences", "budget"],
)
def test_bad_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        _run(HonestWorld(), **kwargs)
