"""Mined tool signatures the top-level tests build by hand."""

from __future__ import annotations

from kullback.runner.records import FieldStat, ToolSig


def tool_sig(name: str, kind: str, *args: str) -> ToolSig:
    """A mined tool signature whose arguments are all required strings."""
    return ToolSig(name=name, kind=kind,
                   args_fields=[FieldStat(name=arg, types=["str"], optional=False) for arg in args],
                   args_schema={"type": "object",
                                "properties": {arg: {"type": ["str"]} for arg in args},
                                "required": list(args)})
