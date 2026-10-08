"""Tools the world records as actions, not writes (D308).

D282 records each call of a tool that ends the Run and writes nothing of its own as a row of an
actions table, and marks the tool a write so an atom can demand it. That row is the harness's record
of the call, never a change the customer's world made, so the Spec path takes such a tool out of the
write set it scores under (the write effects, the no-write atom, the cap, the extra-write check) and
demands it as an action instead: the call is in the Run's events, or it is not.

The evidence is the schema's own: the columns of the actions table name the tools whose calls they
record (`evidence.actions_of`). No tool name is read for meaning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kullback.runner.records import read_json


def _get(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def action_tools_of(schema: Any) -> set[str]:
    """The tools whose calls the schema records as rows of an actions table."""
    out: set[str] = set()
    for column in _get(schema, "columns") or ():
        out.update(str(tool) for tool in (_get(column, "evidence") or {}).get("actions_of") or ())
    return out


def action_tables_of(schema: Any) -> set[str]:
    """The tables whose columns record tool calls: a call's record, not a change of the world."""
    return {str(_get(column, "table")) for column in _get(schema, "columns") or ()
            if (_get(column, "evidence") or {}).get("actions_of") and _get(column, "table")}


def workdir_action_tools(workdir: Any) -> set[str]:
    """The action tools of a workdir's schema.json; none when the file is missing."""
    return action_tools_of(read_json(Path(workdir) / "schema.json", None) or {})


__all__ = ["action_tables_of", "action_tools_of", "workdir_action_tools"]
