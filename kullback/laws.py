"""Environment consistency: laws that hold for any tool, checked off the recorded path.

A property harness generates call sequences by code, with no model, runs them
through a small world protocol and checks one function per law. A broken law is
shrunk to the shortest failing sequence with the shrinker in
kullback.consistency. The number in LawReport is named Environment consistency.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Protocol, Sequence

from kullback.consistency import shrink_sequence

READ_CHANGES_NOTHING = "READ_CHANGES_NOTHING"
SAME_WRITE_SAME_RESULT = "SAME_WRITE_SAME_RESULT"
SUCCESS_CHANGED_SOMETHING = "SUCCESS_CHANGED_SOMETHING"
ERROR_CHANGED_NOTHING = "ERROR_CHANGED_NOTHING"
WRITE_CAN_BE_READ_BACK = "WRITE_CAN_BE_READ_BACK"
MINTED_IDS_UNIQUE = "MINTED_IDS_UNIQUE"
NOTHING_CRASHES_OR_HANGS = "NOTHING_CRASHES_OR_HANGS"

LAW_NAMES = (
    READ_CHANGES_NOTHING,
    SAME_WRITE_SAME_RESULT,
    SUCCESS_CHANGED_SOMETHING,
    ERROR_CHANGED_NOTHING,
    WRITE_CAN_BE_READ_BACK,
    MINTED_IDS_UNIQUE,
    NOTHING_CRASHES_OR_HANGS,
)

_READ = "read"
_WRITE = "write"

_INT_TYPES = {"int", "integer", "long"}
_STR_TYPES = {"str", "string", "text"}
_FLOAT_TYPES = {"float", "double", "number"}
_BOOL_TYPES = {"bool", "boolean"}

_INVENTED_INTS = [-1, 0, 1, 2, 7, 100]
_INVENTED_STRS = ["", "a", "b", "zz"]
_INVENTED_FLOATS = [0.0, 1.5, -2.5]
_INVENTED_BOOLS = [True, False]


@dataclass
class Outcome:
    ok: bool
    value: Any = None
    error: str | None = None


@dataclass
class ArgSpec:
    name: str
    type: str = "int"
    optional: bool = False


@dataclass
class ToolInfo:
    name: str
    kind: str
    args: Sequence[ArgSpec] = ()


class LawWorld(Protocol):
    def reset(self) -> None: ...
    def tools(self) -> Sequence[ToolInfo]: ...
    def call(self, name: str, args: Mapping[str, Any]) -> Outcome: ...
    def snapshot(self) -> Any: ...


@dataclass
class Step:
    tool: str
    args: dict


@dataclass
class Entry:
    step: Step
    before: Any
    after: Any
    outcome: Outcome | None
    duration_s: float
    crashed: bool


@dataclass(frozen=True)
class LawViolation:
    law: str
    tool: str
    sequence: tuple
    status: str


@dataclass
class LawReport:
    checked: dict
    broken: dict
    violations: tuple
    consistency: float | None


def _scalars(value: Any, include_keys: bool = True) -> list:
    found: list = []
    seen: set = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, (str, int, float, bool)):
            found.append(current)
        elif isinstance(current, Mapping):
            if id(current) in seen:
                continue
            seen.add(id(current))
            if include_keys:
                stack.extend(current.keys())
            stack.extend(current.values())
        elif isinstance(current, (list, tuple, set, frozenset)):
            if id(current) in seen:
                continue
            seen.add(id(current))
            stack.extend(current)
    return found


def _contains(snapshot: Any, scalar: Any) -> bool:
    for candidate in _scalars(snapshot):
        if candidate == scalar and type(candidate) is type(scalar):
            return True
    return False


def _type_bucket(type_name: str) -> str:
    name = type_name.strip().lower()
    if name in _INT_TYPES:
        return "int"
    if name in _STR_TYPES:
        return "str"
    if name in _FLOAT_TYPES:
        return "float"
    if name in _BOOL_TYPES:
        return "bool"
    raise ValueError(f"unsupported argument type: {type_name!r}")


def _pool_from_snapshot(snapshot: Any) -> dict:
    pool: dict = {"int": [], "str": [], "float": [], "bool": []}
    for scalar in _scalars(snapshot):
        if isinstance(scalar, bool):
            pool["bool"].append(scalar)
        elif isinstance(scalar, int):
            pool["int"].append(scalar)
        elif isinstance(scalar, float):
            pool["float"].append(scalar)
        else:
            pool["str"].append(scalar)
    return pool


def _invented(bucket: str, rng: random.Random) -> Any:
    if bucket == "int":
        if rng.random() < 0.5:
            return rng.randint(-5, 200)
        return rng.choice(_INVENTED_INTS)
    if bucket == "str":
        if rng.random() < 0.5:
            return "".join(rng.choice("abcxyz") for _ in range(rng.randint(0, 4)))
        return rng.choice(_INVENTED_STRS)
    if bucket == "float":
        if rng.random() < 0.5:
            return rng.uniform(-10.0, 10.0)
        return rng.choice(_INVENTED_FLOATS)
    return rng.choice(_INVENTED_BOOLS)


def _gen_value(bucket: str, rng: random.Random, pool: dict) -> Any:
    known = pool.get(bucket, [])
    if known and rng.random() < 0.7:
        return rng.choice(known)
    return _invented(bucket, rng)


def _gen_args(schema: Sequence[ArgSpec], rng: random.Random, pool: dict) -> dict:
    args: dict = {}
    for spec in schema:
        if spec.optional and rng.random() < 0.5:
            continue
        args[spec.name] = _gen_value(_type_bucket(spec.type), rng, pool)
    return args


def _grow_pool(pool: dict, outcome: Outcome) -> None:
    for scalar in _scalars(outcome.value, False):
        if isinstance(scalar, bool):
            pool["bool"].append(scalar)
        elif isinstance(scalar, int):
            pool["int"].append(scalar)
        elif isinstance(scalar, float):
            pool["float"].append(scalar)
        else:
            pool["str"].append(scalar)


def _run_step(world: LawWorld, step: Step) -> Entry:
    before = world.snapshot()
    start = time.perf_counter()
    try:
        outcome = world.call(step.tool, dict(step.args))
    except Exception:
        duration = time.perf_counter() - start
        return Entry(step, before, world.snapshot(), None, duration, True)
    duration = time.perf_counter() - start
    return Entry(step, before, world.snapshot(), outcome, duration, False)


def _run_steps(world: LawWorld, steps: Sequence[Step]) -> list:
    return [_run_step(world, step) for step in steps]


def _outcome_equal(first: Outcome, second: Outcome) -> bool:
    return (
        first.ok == second.ok
        and first.value == second.value
        and first.error == second.error
    )


def _note(tally: dict, law: str, tool: str, held: bool, steps: list) -> None:
    record = tally.get((law, tool))
    if record is None:
        record = [0, 0, None]
        tally[(law, tool)] = record
    record[0] += 1
    if held:
        return
    record[1] += 1
    if record[2] is None:
        record[2] = list(steps)


def _exempted(exempt: Mapping[str, Collection[str]], law: str, tool: str) -> bool:
    names = exempt.get(tool)
    return names is not None and law in names


def _newly_absent(entry: Entry) -> list:
    return [s for s in _scalars(entry.outcome.value, False) if not _contains(entry.before, s)]


def _minted_now(entry: Entry) -> list:
    given = _scalars(entry.step.args, False)
    minted: list = []
    for scalar in _scalars(entry.outcome.value, False):
        if not _contains(entry.after, scalar):
            continue
        if any(type(known) is type(scalar) and known == scalar for known in given):
            continue
        minted.append(scalar)
    return minted


def _seen_before(seen: list, scalar: Any) -> bool:
    for old in seen:
        if type(old) is type(scalar) and old == scalar:
            return True
    return False


def _is_write_call(entry: Entry, kinds: dict) -> bool:
    return (
        not entry.crashed
        and entry.outcome is not None
        and kinds.get(entry.step.tool) == _WRITE
    )


def _check_read(
    entries: list, kinds: dict, tally: dict, steps: list, exempt: Mapping
) -> None:
    for entry in entries:
        if entry.crashed or kinds.get(entry.step.tool) != _READ:
            continue
        if _exempted(exempt, READ_CHANGES_NOTHING, entry.step.tool):
            continue
        _note(tally, READ_CHANGES_NOTHING, entry.step.tool, entry.before == entry.after, steps)


def _check_success(
    entries: list, kinds: dict, tally: dict, steps: list, exempt: Mapping
) -> None:
    for entry in entries:
        if not _is_write_call(entry, kinds) or not entry.outcome.ok:
            continue
        if _exempted(exempt, SUCCESS_CHANGED_SOMETHING, entry.step.tool):
            continue
        _note(
            tally,
            SUCCESS_CHANGED_SOMETHING,
            entry.step.tool,
            entry.before != entry.after,
            steps,
        )


def _check_error(entries: list, tally: dict, steps: list, exempt: Mapping) -> None:
    for entry in entries:
        if entry.crashed or entry.outcome is None or entry.outcome.ok:
            continue
        if _exempted(exempt, ERROR_CHANGED_NOTHING, entry.step.tool):
            continue
        _note(tally, ERROR_CHANGED_NOTHING, entry.step.tool, entry.before == entry.after, steps)


def _check_read_back(
    entries: list, kinds: dict, tally: dict, steps: list, exempt: Mapping
) -> None:
    for entry in entries:
        if not _is_write_call(entry, kinds) or not entry.outcome.ok:
            continue
        if _exempted(exempt, WRITE_CAN_BE_READ_BACK, entry.step.tool):
            continue
        minted = _newly_absent(entry)
        held = all(_contains(entry.after, scalar) for scalar in minted)
        _note(tally, WRITE_CAN_BE_READ_BACK, entry.step.tool, held, steps)


def _check_minted(
    entries: list, kinds: dict, tally: dict, steps: list, exempt: Mapping
) -> None:
    seen: list = []
    for entry in entries:
        if not _is_write_call(entry, kinds) or not entry.outcome.ok:
            continue
        current = _minted_now(entry)
        if not current:
            continue
        if _exempted(exempt, MINTED_IDS_UNIQUE, entry.step.tool):
            continue
        repeated = any(_seen_before(seen, scalar) for scalar in current)
        _note(tally, MINTED_IDS_UNIQUE, entry.step.tool, not repeated, steps)
        seen.extend(current)


def _check_crash(
    entries: list, tally: dict, steps: list, exempt: Mapping, call_budget_s: float
) -> None:
    for entry in entries:
        if _exempted(exempt, NOTHING_CRASHES_OR_HANGS, entry.step.tool):
            continue
        held = not entry.crashed and entry.duration_s <= call_budget_s
        _note(tally, NOTHING_CRASHES_OR_HANGS, entry.step.tool, held, steps)


def _replay_mismatch(world: LawWorld, steps: list, index: int, entry: Entry) -> bool:
    world.reset()
    try:
        for step in steps[:index]:
            world.call(step.tool, dict(step.args))
        before = world.snapshot()
        outcome = world.call(entry.step.tool, dict(entry.step.args))
        after = world.snapshot()
    except Exception:
        return True
    return (
        not _outcome_equal(entry.outcome, outcome)
        or before != entry.before
        or after != entry.after
    )


def _check_same_write(
    world: LawWorld, steps: list, entries: list, kinds: dict, tally: dict, exempt: Mapping
) -> None:
    for index, entry in enumerate(entries):
        if not _is_write_call(entry, kinds):
            continue
        if _exempted(exempt, SAME_WRITE_SAME_RESULT, entry.step.tool):
            continue
        held = not _replay_mismatch(world, steps, index, entry)
        _note(tally, SAME_WRITE_SAME_RESULT, entry.step.tool, held, steps)


def _check_all(
    world: LawWorld,
    steps: list,
    entries: list,
    kinds: dict,
    tally: dict,
    exempt: Mapping,
    call_budget_s: float,
) -> None:
    _check_read(entries, kinds, tally, steps, exempt)
    _check_same_write(world, steps, entries, kinds, tally, exempt)
    _check_success(entries, kinds, tally, steps, exempt)
    _check_error(entries, tally, steps, exempt)
    _check_read_back(entries, kinds, tally, steps, exempt)
    _check_minted(entries, kinds, tally, steps, exempt)
    _check_crash(entries, tally, steps, exempt, call_budget_s)


def _candidate_entries(world: LawWorld, candidate: Sequence[Step]) -> list:
    world.reset()
    return _run_steps(world, list(candidate))


def _make_fails(
    law: str, world: LawWorld, kinds: dict, tool: str, call_budget_s: float
):
    def fails(candidate: list) -> bool:
        steps = list(candidate)
        entries = _candidate_entries(world, steps)
        tally: dict = {}
        _check_all(world, steps, entries, kinds, tally, {}, call_budget_s)
        record = tally.get((law, tool))
        return record is not None and record[1] > 0

    return fails


def _run_planned(
    world: LawWorld, tools: list, rng: random.Random, max_length: int
) -> tuple:
    length = rng.randint(1, max_length)
    world.reset()
    pool = _pool_from_snapshot(world.snapshot())
    steps: list = []
    entries: list = []
    for _ in range(length):
        info = rng.choice(tools)
        step = Step(info.name, _gen_args(info.args, rng, pool))
        entry = _run_step(world, step)
        steps.append(step)
        entries.append(entry)
        if not entry.crashed and entry.outcome is not None and entry.outcome.ok:
            _grow_pool(pool, entry.outcome)
    return steps, entries


def _assemble(
    tally: dict, world: LawWorld, kinds: dict, call_budget_s: float, shrink_evaluations: int
) -> LawReport:
    checked: dict = {}
    broken: dict = {}
    violations: list = []
    for law in LAW_NAMES:
        names = sorted({key[1] for key in tally if key[0] == law})
        for tool in names:
            count, bad, first = tally[(law, tool)]
            checked[(law, tool)] = count
            broken[(law, tool)] = bad
            if bad == 0 or first is None:
                continue
            result = shrink_sequence(
                list(first),
                _make_fails(law, world, kinds, tool, call_budget_s),
                max_evaluations=shrink_evaluations,
            )
            violations.append(LawViolation(law, tool, tuple(result.subsequence), result.status))
    total = sum(checked.values())
    if total == 0:
        consistency = None
    else:
        consistency = (total - sum(broken.values())) / total
    return LawReport(checked, broken, tuple(violations), consistency)


def check_laws(
    world: LawWorld,
    *,
    seed: int = 0,
    sequences: int = 20,
    max_length: int = 5,
    exempt: Mapping[str, Collection[str]] | None = None,
    call_budget_s: float = 1.0,
    shrink_evaluations: int = 200,
) -> LawReport:
    """Generate call sequences by code and check the laws in LAW_NAMES.

    A call slower than call_budget_s is a NOTHING_CRASHES_OR_HANGS violation,
    measured after the fact; killing a hung call is the sandbox's job.
    Returns per law and per tool counts of checked and broken, the shrunk
    violations, and Environment consistency, None when nothing was checked.
    """
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if sequences < 0:
        raise ValueError("sequences must not be negative")
    if call_budget_s <= 0:
        raise ValueError("call_budget_s must be positive")
    tools = list(world.tools())
    if not tools:
        return LawReport({}, {}, (), None)
    gaps: Mapping[str, Collection[str]] = exempt or {}
    rng = random.Random(seed)
    kinds = {info.name: info.kind for info in tools}
    tally: dict = {}
    for _ in range(sequences):
        steps, entries = _run_planned(world, tools, rng, max_length)
        _check_all(world, steps, entries, kinds, tally, gaps, call_budget_s)
    return _assemble(tally, world, kinds, call_budget_s, shrink_evaluations)
