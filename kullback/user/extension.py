"""The agent user as an extension on the shared agent core (D214, ADR-0007).

`user_extension(ctx, box)` is a `setup(api)` the harness loads, the way `builder_extension(plan)`
and `examiner_extension(plan)` are. It registers the user's own tools, adds the prompt sections in
the order skills.py states them, catalogues the user skill so the ContextManager can cut and recall
it, adds the curated per-Task sections under their stable tags (context.py), and installs the two
refusals.

The refusals are in code and never in the prompt, for the reason D122 gives: a rule a model reads is
a rule a model can be talked out of. The first is the harness's own, the path block over the gates
and the Runner. The second is this agent's: a tool call whose arguments name the Environment's
tables or the Starting state is refused, because a user that can read the world is a user that can
hand the Candidate the answer it was supposed to look up, and no tool here reads them anyway, so the
refusal is what stops one being reached through a rewritten argument.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

from kullback.agent.extensions import ExtensionAPI, refuse_paths
from kullback.agent.harness import prompt_block
from kullback.gates import PROTECTED, names_protected_path
from kullback.user import skills
from kullback.user.context import TaskContext, sections
from kullback.user.tools import Toolbox, user_tools

# The words a call would have to name to be reaching for the world's own tables. They are the names
# the harness itself uses for those artifacts, not any customer's table names, so the refusal is
# general and a corpus that calls its tables anything at all is covered.
WORLD_NAMES = ("db.json", "starting_state", "overlays", "environment.json", "schema.json",
               "tool_sigs.json", "bodies.json")


def names_world(arguments) -> Optional[str]:
    """The first argument string that names the world's own records, or None."""
    for value in _strings(arguments):
        for name in WORLD_NAMES:
            if name in value:
                return name
    return None


def _strings(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for inner in value.values():
            yield from _strings(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            yield from _strings(inner)


no_agent_writes_gates_or_runner = refuse_paths(
    names_protected_path, f"a path under {' or '.join(PROTECTED)}, which no agent may write (D122)",
    "no_agent_writes_gates_or_runner")

no_user_reads_the_world = refuse_paths(
    names_world, "one of the world's own records; a user of this conversation sees what a person on "
                 "the phone sees and never the tables behind it (D214)",
    "no_user_reads_the_world")


def user_extension(ctx: TaskContext, box: Toolbox,
                   prefix: Sequence = (), asked: Iterable[str] = ()) -> Callable[[ExtensionAPI], None]:
    """The setup the harness loads: the tools, the prompt in skills.py's order, the two refusals."""

    def setup(api: ExtensionAPI) -> None:
        for tool in user_tools(box):
            api.register_tool(tool)
        api.add_prompt_section("user", prompt_block("task", skills.WHAT))
        api.add_prompt_section("user_tools", prompt_block("tools", skills.TOOLS))
        api.add_prompt_section("skills", api.context.skills_section())
        api.catalog_skill(skills.USER_SKILL_NAME, skills.USER_SKILL, loaded=True)
        api.add_prompt_section("user_examples", prompt_block("examples", skills.EXAMPLES))
        api.add_prompt_section("user_rules", prompt_block("rules", skills.RULES))
        for section in sections(ctx, prefix, asked):
            api.add_prompt_section(section.name, prompt_block(section.name, section.text))
        api.add_prompt_section("user_feedback", prompt_block("feedback", skills.FEEDBACK))
        api.add_prompt_section("user_stop", prompt_block("stop", skills.STOP))
        api.tool_call(no_agent_writes_gates_or_runner)
        api.tool_call(no_user_reads_the_world)

    return setup
