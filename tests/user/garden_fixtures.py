"""The invented garden centre the user tests share: its vocabulary and one recorded delivery move.

The domain is invented: a garden centre that delivers plants, with a delivery slot, a plot number
and a membership tier. Nothing here names any customer's world.
"""

from __future__ import annotations

from kullback.runner.records import RawPtr, ToolCall, Trace, Turn
from kullback.user.vocabulary import GENERIC_FIELDS, FieldSpec, Vocabulary

# The shared conftest defines this too, but two test folders both name a module `conftest` and
# which one answers `from conftest import ...` depends on the order pytest put them on the path.
PTR = RawPtr(file_hash="testfile", sim_index=0)

VOCAB = Vocabulary(domain="garden", fields=[
    *[f.model_copy(deep=True) for f in GENERIC_FIELDS],
    FieldSpec(field="plot_id", kind="reference", pattern=r"\bPLOT-\d{4}\b",
              cues=[r"\bplot (?:id|number)\b", r"\byour plot\b"], aliases=["plot id"]),
    FieldSpec(field="delivery_slot", kind="value", pattern=r"\b(?:0[1-9]|1[0-9]|2[0-3]):00\b",
              cues=[r"\bdelivery slot\b", r"\bwhat time\b"], aliases=["delivery slot"]),
])


def turn(idx: int, role: str, content: str, calls: list[str] = ()) -> Turn:
    return Turn(idx=idx, role=role, content=content, tool_call_ids=list(calls), raw_ptr=PTR)


def trace_of(turns: list[Turn], calls: list[ToolCall] = ()) -> Trace:
    return Trace(trace_id="t1", raw_hash="h", ingest_version="v", source="test",
                 turns=turns, tool_calls=list(calls), raw_ptr=PTR)


def recorded_trace() -> Trace:
    """Three user turns: the goal, a plot id when asked, a slot when asked, then the close."""
    return trace_of([
        turn(0, "user", "Hello, please could you move my plant delivery to a later slot."),
        turn(1, "assistant", "Of course. Could you provide your plot number?"),
        turn(2, "user", "Thanks, my plot number is PLOT-4471."),
        turn(3, "assistant", "Thank you. What delivery slot would you like?"),
        turn(4, "user", "Please make it 16:00."),
        turn(5, "assistant", "Done, your delivery is moved. Anything else?"),
        turn(6, "user", "No, that is all. Thank you."),
    ], [ToolCall(id="c1", name="move_delivery", args={"plot_id": "PLOT-4471", "slot": "16:00"},
                 result={"plot_id": "PLOT-4471", "slot": "16:00", "courier_ref": "CR-90881"},
                 raw_ptr=PTR)])
