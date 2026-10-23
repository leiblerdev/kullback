"""A small workdir the scenario tests share: three Tasks with recordings, a policy, two tools, a world.

Every name is abstract: widgets, slots and a holder. No corpus, domain or customer word appears.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from kullback.ai.provider import ModelReply, ToolCallRequest
from kullback.builder.search import Chain, MemoSearch, TestSearch
from kullback.domain import destination_refusal
from kullback.runner.records import RawPtr, Task, Trace, Turn, as_dict, write_json

PTR = RawPtr(file_hash="fixture", sim_index=0)
POLICY = ("A widget may be moved to another slot only before it is sealed.\n"
          "A sealed widget cannot be moved; offer a replacement instead.\n"
          "Every holder must be verified before any change.")
TOOLS = [{"name": "move_widget", "kind": "write", "description": "Move one widget to another slot.",
          "args_fields": [{"name": "widget_id"}, {"name": "slot"}]},
         {"name": "get_widget", "kind": "read", "description": "Read one widget.",
          "args_fields": [{"name": "widget_id"}]}]
STATE = {"widget_rows": {"W1001": {"widget_id": "W1001", "slot_code": "two", "is_sealed": False,
                                  "holder_name": "Ada Byron", "holder_id": "holder_7"},
                        "W1002": {"widget_id": "W1002", "slot_code": "four", "is_sealed": True,
                                  "holder_name": "Ada Byron", "holder_id": "holder_7"}},
         "holder_rows": {"holder_7": {"holder_name": "Ada Byron", "contact_mail": "ada@example.org"}}}
OPENINGS = {"task-a1": "Hi, I need my widget moved to slot seven please.",
            "task-b2": "Can you move my widget to another slot? It is already sealed.",
            "task-c3": "My widget arrived broken and I want a new one."}
DESCRIPTION = "A service that keeps widgets in slots for their holders."


def _trace(trace_id: str, text: str) -> Trace:
    turns = [Turn(idx=0, role="assistant", content="Hello, how can I help?", raw_ptr=PTR),
             Turn(idx=1, role="user", content=text, raw_ptr=PTR),
             Turn(idx=2, role="assistant", content="Let me look.", raw_ptr=PTR)]
    return Trace(trace_id=trace_id, raw_hash="h", ingest_version="1", source="fixture", turns=turns, raw_ptr=PTR)


def make_workdir(root: Path) -> Path:
    workdir = Path(root) / "work"
    for task_id, text in OPENINGS.items():
        trace_id = f"rec-{task_id}"
        write_json(workdir / "tasks" / f"{task_id}.json", as_dict(Task(id=task_id, run_ids=[trace_id])))
        write_json(workdir / "traces" / f"{trace_id}.json", as_dict(_trace(trace_id, text)))
    write_json(workdir / "tool_sigs.json", TOOLS)
    write_json(workdir / "db.json", STATE)
    write_json(workdir / "environment.json", {"description": DESCRIPTION})
    (workdir / "env").mkdir(parents=True, exist_ok=True)
    (workdir / "env" / "policy.md").write_text(POLICY, encoding="utf-8")
    return workdir


def call(name: str, arguments: dict, call_id: str = "c1") -> ModelReply:
    return ModelReply(tool_calls=[ToolCallRequest(id=call_id, name=name, arguments=arguments)])


PAGE = ("<html><body><h1>Help</h1><p>Customers often ask to move a widget after it has been sealed. "
        "Sealed widgets stay where they are, so we offer a replacement.</p>"
        "<p>Some holders forget which slot they picked and ask us to tell them.</p></body></html>")


URL = "https://help.example/a"
# The domain module's destination check over a scripted resolver: every host answers with one public
# address, so no test resolves a name over the network.
REFUSE = partial(destination_refusal, resolve=lambda host: ["93.184.215.14"])


def web(workdir: Path, hits: dict | None = None, pages: dict | None = None) -> tuple[MemoSearch, TestSearch]:
    """The harness's reader as the live path builds it, memo over a chain, with a scripted provider inside."""
    inner = TestSearch(hits={"": [{"url": URL}]} if hits is None else hits,
                       pages={URL: PAGE} if pages is None else pages)
    return MemoSearch(Chain([inner]), workdir), inner


PLANNED = "a holder wants something done with a widget"
# Rows of the fixture world a bound scenario names, table to key.
BOUND = {"widget_rows": "W1001", "holder_rows": "holder_7"}


def planned(*situations: tuple[str, list[int]]) -> list[ModelReply]:
    """One plan session writing each (statement, openings) as a situation, by default one holding every
    opening, then stopping."""
    rows = situations or ((PLANNED, [0, 1, 2]),)
    calls = [call("write_situation", {"statement": text, "kind": "change", "openings": list(at)}, f"p{n}")
             for n, (text, at) in enumerate(rows)]
    return [*calls, ModelReply(content="Planned them.")]
