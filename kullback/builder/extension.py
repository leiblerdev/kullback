"""The Builder as an extension on the agent core (D120, D123, ADR-0007).

`builder_extension(plan)` is a `setup(api)` the harness loads: it registers the six tools of
builder/tools.py, adds three short sections to the system prompt (what the Builder is, what it may
and may not do, the target vocabulary of this plan's graph), and installs two hooks. The
`tool_result` hook runs every registered gate bound to an artifact a tool produced (`gates_over`)
over the plan's store and appends the rulings to the result, in the text the model reads and in
`details`; the stages already recorded the same rulings in gates.json on their way, so the hook
writes nothing. The `tool_call` hook raises on any call whose arguments name a path under
kullback/gates or kullback/runner, which the loop turns into an is_error result: the gates and
the Runner are packages no agent writes (D122), and the refusal is in code, not in the prompt.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from kullback.agent.extensions import ExtensionAPI
from kullback.agent.messages import ToolCall
from kullback.agent.tools import ToolResult
from kullback.builder import build as build_module
from kullback.builder.build import TARGET_ALL, BuildPlan
from kullback.builder.tools import builder_tools, ruling_line
from kullback.gates import gates_over
from kullback.runner.records import as_dict

PROTECTED = ("kullback/gates", "kullback/runner")
# A path segment `gates/` or `runner/`, on its own or under kullback/, at the start of a value or
# after a separator; `gates.json`, `runs/` and `runner_version.json` do not match. The second
# alternative is the package directory named with nothing after it (`rm -r kullback/gates`), which
# the first would miss for want of a trailing separator; it asks for the `kullback/` prefix, so a
# bare word `gates` or `runner` in an argument is still an ordinary string.
_PROTECTED_PATH = re.compile(
    r"(?:^|[\s\"'=:,(\[{/\\])"
    r"(?:(?:kullback[/\\])?(?:gates|runner)[/\\]|kullback[/\\](?:gates|runner)(?=$|[\s\"',)\]}]))")

WHAT = ("You are the Builder. From a customer's recorded traces you build an Environment: the tools "
        "with their bodies, the Starting state, the Tasks with a Reference Run each, the Simulated "
        "user's rules, the policy as Constraints, and one Verifier per Task. Every step is a stage of "
        "a fixed graph with a gate over what it made; the scheduler decides the order, you decide "
        "which target to ask for.")
RULES = ("You may run any target through the build tools and read the rulings that come back. You "
         "may not edit code: there is no repair verb here, and any call naming a path under "
         "kullback/gates or kullback/runner is refused. The gates are the standard, not something to "
         "argue with; a failed ruling is reported as it is.")


def target_vocabulary(plan: BuildPlan) -> str:
    """The names `build(target)` accepts for this plan: the whole build, each stage, each artifact."""
    stages = build_module.stages(plan)
    names = ", ".join(s.name for s in stages)
    artifacts = ", ".join(a for s in stages for a in s.outputs)
    return (f"Targets: `{TARGET_ALL}` is the whole build. Stages: {names}. Artifacts: {artifacts}. "
            "A target's stale inputs are rebuilt first; current ones come from the cache.")


def names_protected_path(value: Any) -> Optional[str]:
    """The first string in a value (walked through dicts and lists) that names a protected path."""
    if isinstance(value, str):
        return value if _PROTECTED_PATH.search(value) else None
    if isinstance(value, dict):
        for item in value.values():
            found = names_protected_path(item)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = names_protected_path(item)
            if found is not None:
                return found
    return None


def no_agent_writes_gates_or_runner(call: ToolCall) -> None:
    """Raise on a call whose arguments reach into the gates or the Runner (D122); the loop makes it an error result."""
    found = names_protected_path(call.arguments)
    if found is not None:
        raise PermissionError(f"the arguments name {found!r}, a path under {' or '.join(PROTECTED)}, "
                              "which no agent may write (D122)")


no_agent_writes_gates_or_runner.hook_name = "no_agent_writes_gates_or_runner"  # type: ignore[attr-defined]


def rulings_over(plan: BuildPlan, produced: list) -> list:
    """Every registered gate bound to one of these artifacts, run over the plan's store, in registry order."""
    store = plan.store
    out, seen = [], set()
    for artifact in produced:
        for spec in gates_over(artifact):
            if spec.name in seen or any(name not in store for name in spec.artifacts):
                continue
            seen.add(spec.name)
            out.append(spec.fn(*[store[name] for name in spec.artifacts]))
    return out


def gate_rulings_hook(plan: BuildPlan) -> Callable[[ToolCall, ToolResult], Optional[ToolResult]]:
    def gate_rulings(call: ToolCall, result: ToolResult) -> Optional[ToolResult]:
        if result.is_error or not result.details:
            return None
        produced = result.details.get("produced") or []
        rulings = rulings_over(plan, list(produced))
        if not rulings:
            return None
        line = ruling_line("gate rulings", rulings)
        details = dict(result.details)
        details["gate_rulings"] = [as_dict(r) for r in rulings]
        return ToolResult(content=f"{result.content}\n{line}", details=details, is_error=False)

    gate_rulings.hook_name = "gate_rulings"  # type: ignore[attr-defined]
    return gate_rulings


def builder_extension(plan: BuildPlan) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: tools, prompt sections, the two hooks."""

    def setup(api: ExtensionAPI) -> None:
        for tool in builder_tools(plan, sink=api.harness.emit):
            api.register_tool(tool)
        api.add_prompt_section("builder", WHAT)
        api.add_prompt_section("builder_rules", RULES)
        api.add_prompt_section("builder_targets", target_vocabulary(plan))
        api.tool_call(no_agent_writes_gates_or_runner)
        api.tool_result(gate_rulings_hook(plan))

    return setup
