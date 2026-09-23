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


def _store_tool(mints=()):
    return ToolInfo("store", "write", [ArgSpec("item", "int")], mints=mints)


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
        return [_scan_tool(), _store_tool(mints=("id",))]

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
        return Outcome(True, {"id": new}, None)

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
        return Outcome(True, {"id": self._hidden}, None)


class NoopWriteWorld(HonestWorld):
    def tools(self):
        return [_scan_tool(), _store_tool()]

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
        return Outcome(True, {"id": new + 1000}, None)


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
        self._cells[new] = (item, "done")
        self._next += 1
        self._last = "kid-1"
        return Outcome(True, {"id": "kid-1", "status": "done"}, None)


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
    half = _run(HalfWriteWorld(), exempt={"store": [ERROR_CHANGED_NOTHING]})
    assert half.violations == ()
    assert half.consistency == 1.0


def test_a_world_with_nothing_to_check_gives_consistency_none():
    report = _run(EmptyWorld())
    assert report.consistency is None
    assert report.violations == ()
    assert report.checked == {}
    every_law = [
        READ_CHANGES_NOTHING,
        SAME_WRITE_SAME_RESULT,
        SUCCESS_CHANGED_SOMETHING,
        ERROR_CHANGED_NOTHING,
        WRITE_CAN_BE_READ_BACK,
        MINTED_IDS_UNIQUE,
        NOTHING_CRASHES_OR_HANGS,
    ]
    exempt_all = _run(LoggingReadWorld(), exempt={"scan": every_law, "store": every_law})
    assert exempt_all.violations == ()
    assert exempt_all.checked == {}
    assert exempt_all.consistency is None


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


@pytest.mark.parametrize(
    "world_cls,sequences", [(HonestWorld, 12), (TypedWorld, 30)], ids=["int-args", "typed-args"]
)
def test_an_honest_world_holds_every_law_whatever_its_argument_types(world_cls, sequences):
    report = _run(world_cls(), sequences=sequences)
    assert report.violations == ()
    assert report.consistency == 1.0
    assert sum(report.broken.values()) == 0
    assert sum(report.checked.values()) > 0


@pytest.mark.parametrize(
    "world_cls,kwargs",
    [
        (HonestWorld, {"max_length": 0}),
        (HonestWorld, {"sequences": -1}),
        (HonestWorld, {"call_budget_s": 0.0}),
        (BadTypeWorld, {}),
    ],
    ids=["max-length", "sequences", "budget", "argument-type"],
)
def test_bad_configuration_is_rejected(world_cls, kwargs):
    with pytest.raises(ValueError):
        _run(world_cls(), **kwargs)


class SecondLifeWorld:
    def __init__(self):
        self._cells = {}
        self._next = 1
        self._lives = 0

    def reset(self):
        self._cells = {}
        self._next = 1
        self._lives += 1

    def tools(self):
        return [_store_tool()]

    def snapshot(self):
        return {"cells": dict(self._cells), "next": self._next}

    def call(self, name, args):
        if self._lives > 1:
            raise RuntimeError("second life")
        item = args.get("item")
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return Outcome(False, None, "bad item")
        new = self._next
        self._cells[new] = item
        self._next += 1
        return Outcome(True, new, None)


class LiveReadWorld:
    def __init__(self):
        self._state = {"reads": 0}

    def reset(self):
        self._state = {"reads": 0}

    def tools(self):
        return [ToolInfo("scan", "read", [])]

    def snapshot(self):
        return self._state

    def call(self, name, args):
        self._state["reads"] += 1
        return Outcome(True, self._state["reads"], None)


@pytest.mark.parametrize(
    "world_cls,sequences", [(SecondLifeWorld, 1), (FlakyWorld, 12)], ids=["second-life", "flaky"]
)
def test_write_failing_only_on_replay_reports_both_laws(world_cls, sequences):
    report = _run(world_cls(), sequences=sequences)
    assert {(v.law, v.tool) for v in report.violations} == {
        (SAME_WRITE_SAME_RESULT, "store"),
        (NOTHING_CRASHES_OR_HANGS, "store"),
    }


def test_live_snapshot_mutating_read_is_still_caught():
    report = _run(LiveReadWorld())
    assert [(v.law, v.tool) for v in report.violations] == [(READ_CHANGES_NOTHING, "scan")]
    (violation,) = report.violations
    assert len(violation.sequence) == 1


class CrashProbeWorld:
    def __init__(self):
        self.cells = {}
        self.n = 0

    def reset(self):
        self.cells = {}
        self.n = 0

    def tools(self):
        return [
            ToolInfo("boom", "read"),
            ToolInfo("store", "write", [ArgSpec("text", "str")], mints=("id",)),
        ]

    def snapshot(self):
        return self.cells

    def call(self, name, args):
        if name == "boom":
            raise RuntimeError("always")
        self.n += 1
        key = f"c{self.n}"
        self.cells[key] = {"id": key, "text": args.get("text")}
        return Outcome(True, {"id": key, "text": args.get("text")})


class CallCountingWorld:
    def __init__(self, inner):
        self._inner = inner
        self.episodes = []

    def reset(self):
        self._inner.reset()
        self.episodes.append([])

    def tools(self):
        return self._inner.tools()

    def snapshot(self):
        return self._inner.snapshot()

    def call(self, name, args):
        self.episodes[-1].append(name)
        return self._inner.call(name, args)


def test_honest_write_not_blamed_for_other_tools_crash():
    report = check_laws(CrashProbeWorld(), seed=3, sequences=30)
    assert report.broken[("SAME_WRITE_SAME_RESULT", "store")] == 0
    assert [(v.law, v.tool) for v in report.violations] == [(NOTHING_CRASHES_OR_HANGS, "boom")]


class TwoSpacesWorld:
    def __init__(self):
        self._users = 0
        self._orders = 0

    def reset(self):
        self._users = 0
        self._orders = 0

    def tools(self):
        return [
            ToolInfo("create_user", "write", [], mints=("user.id",)),
            ToolInfo("create_order", "write", [], mints=("order.id",)),
        ]

    def snapshot(self):
        return {"users": self._users, "orders": self._orders}

    def call(self, name, args):
        if name == "create_user":
            self._users += 1
            return Outcome(True, {"user": {"id": self._users}})
        if name == "create_order":
            self._orders += 1
            return Outcome(True, {"order": {"id": self._orders}})
        return Outcome(False, None, "unknown tool")


class TwoPathsWorld:
    def __init__(self):
        self._made = 0

    def reset(self):
        self._made = 0

    def tools(self):
        return [ToolInfo("make", "write", [], mints=("a.id", "b.id"))]

    def snapshot(self):
        return {"made": self._made}

    def call(self, name, args):
        self._made += 1
        return Outcome(True, {"a": {"id": self._made}, "b": {"id": self._made}})


def test_crash_counts_match_planned_calls_only():
    inner = CrashProbeWorld()
    recording = CallCountingWorld(inner)
    report = check_laws(recording, seed=3, sequences=1)
    planned = recording.episodes[0]
    assert planned.count("boom") > 0
    assert planned.count("store") > 0
    for tool in ("boom", "store"):
        key = (NOTHING_CRASHES_OR_HANGS, tool)
        assert report.checked.get(key, 0) == planned.count(tool)


class StatusEchoWorld:
    def __init__(self):
        self._rows = []
        self._next = 1

    def reset(self):
        self._rows = []
        self._next = 1

    def tools(self):
        return [ToolInfo("append", "write", [ArgSpec("label", "str")], mints=("order.id",))]

    def snapshot(self):
        return {"rows": list(self._rows), "next": self._next}

    def call(self, name, args):
        label = args.get("label")
        if not isinstance(label, str):
            return Outcome(False, None, "bad label")
        new = self._next
        self._rows.append({"id": new, "status": "done"})
        self._next += 1
        return Outcome(True, {"order": {"id": new}, "status": "done"})


class MissingMintWorld:
    def __init__(self):
        self._made = 0

    def reset(self):
        self._made = 0

    def tools(self):
        return [ToolInfo("make", "write", [], mints=("id", "order.id", "blank"))]

    def snapshot(self):
        return {"made": self._made, "five": 5}

    def call(self, name, args):
        self._made += 1
        return Outcome(True, {"order": 5, "blank": None}, None)


@pytest.mark.parametrize(
    "world_cls",
    [TwoSpacesWorld, TwoPathsWorld, StatusEchoWorld],
    ids=["separate-spaces", "coinciding-paths", "repeated-status"],
)
def test_minted_ids_unique_per_path_hold_minted_law(world_cls):
    report = _run(world_cls())
    assert report.violations == ()
    assert report.consistency == 1.0


def test_missing_mint_path_breaks_minted_law():
    report = _run(MissingMintWorld())
    assert [(v.law, v.tool) for v in report.violations] == [(MINTED_IDS_UNIQUE, "make")]
    (violation,) = report.violations
    assert len(violation.sequence) == 1


def test_an_id_argument_draws_existing_ids_where_its_name_points_and_some_fresh_ones():
    import random

    from kullback.laws import _gen_args

    snapshot = {"widgets": {"w1": {"widget_id": "w1", "label": "plain"}}}
    schema = [ArgSpec("widget_id", "string", False, True)]
    pool = {"int": [], "str": ["zz", "b", "a", ""], "float": [], "bool": []}
    rng = random.Random(7)
    drawn = [_gen_args(schema, rng, pool, snapshot)["widget_id"] for _ in range(20)]
    assert "w1" in drawn
    assert any(value != "w1" for value in drawn)

    snapshot = {
        "groups": {"g1": {"group_id": "g1"}},
        "members": {"m1": {"member_id": "m1"}},
    }
    schema = [ArgSpec("group_id", "string", False, True, "groups")]
    pool = {"int": [], "str": ["zz"], "float": [], "bool": []}
    rng = random.Random(3)
    drawn = [_gen_args(schema, rng, pool, snapshot)["group_id"] for _ in range(30)]
    assert "g1" in drawn
    assert "m1" not in drawn, "a named table keeps the draw to its own rows"

    snapshot = {
        "groups": {"g1": {"tag": "t1"}},
        "members": {"m9": {"tag": "t9"}},
    }
    schema = [ArgSpec("tag", "string", False, True)]
    pool = {"int": [], "str": ["zz"], "float": [], "bool": []}
    rng = random.Random(3)
    drawn = [_gen_args(schema, rng, pool, snapshot)["tag"] for _ in range(30)]
    assert "t1" in drawn or "t9" in drawn
    assert "g1" not in drawn and "m9" not in drawn, "an unresolved id draws only same-named values"


