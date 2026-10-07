"""The forbidden writes and conduct rules a Spec of the older demand grammar asks for, by code.

No Reference: a Spec's Verifier is written from the Intent, the policy and the world (spec/items.py);
this module only keeps the demands of the older grammar compiling (D324's refusals and no-writes as
forbidden writes, confirmations and hand-offs as conduct). It never imports runner/expected.py.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple

from kullback.runner.records import Conduct, EndState, Forbidden, ValueSource
from kullback.spec.schema import Check, Spec

# The demands that become conduct rules, not atoms: a confirmation, a refusal, a hand-off.
CONDUCT_DEMANDS = ("refuse", "handoff")


class Gates(NamedTuple):
    """What the Verifier gates on beside its atoms."""
    expected: list[EndState]
    forbidden: list[Forbidden]
    conduct: list[Conduct]


def _fact_source(check: Check, spec: Spec) -> ValueSource:
    """A conduct's source: the user turn of the first fact the check answers, else the policy its because names."""
    facts = {fact.id: fact for fact in spec.intent.facts}
    fact = next((facts[f] for f in check.fact_ids if f in facts), None)
    if fact is not None:
        return ValueSource(kind="user_turn", ptr={"recording": fact.source.recording, "turn": fact.source.turn})
    return ValueSource(kind="policy", ptr={"clause": check.because})


def conduct_of(spec: Spec) -> list[Conduct]:
    """The conduct rules the grounded demands ask for: a confirmation before a write, a hand-off call."""
    out: list[Conduct] = []
    for check in spec.checks:
        demand = check.demand.get("demand")
        if demand == "ask" and check.demand.get("confirm_tool"):
            out.append(Conduct(kind="confirm_before_write", tool=check.demand["confirm_tool"],
                               source=_fact_source(check, spec)))
        elif demand == "handoff" and check.demand.get("tool"):
            out.append(Conduct(kind="handoff", tool=check.demand["tool"], source=_fact_source(check, spec)))
    return out


def forbidden_of(spec: Spec, write_tools: Iterable[str]) -> list[Forbidden]:
    """A refuse or no_write demand as forbidden writes: the demand's tool, else each write tool (D324)."""
    out: list[Forbidden] = []
    for check in spec.checks:
        if check.demand.get("demand") not in ("refuse", "no_write"):
            continue
        tools = [check.demand["tool"]] if check.demand.get("tool") else sorted(write_tools)
        entity = check.demand.get("entity")
        row = str(entity) if isinstance(entity, (str, int)) and entity != "" else None
        out += [Forbidden(kind="write", tool=tool, row_id=row, source=_fact_source(check, spec)) for tool in tools]
    return out


def gates_of(spec: Spec, write_tools: Iterable[str] = ()) -> Gates:
    """The gates of a demand-grammar Spec: no expected end state, its forbidden writes and conduct."""
    return Gates([], forbidden_of(spec, write_tools), conduct_of(spec))


__all__ = ["CONDUCT_DEMANDS", "Gates", "conduct_of", "forbidden_of", "gates_of"]
