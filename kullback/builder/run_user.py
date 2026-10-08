"""The agent user a fresh Run meets, over its rule floor, built from the workdir (D214).

Moved from the Examiner's runners when the Examiner stopped running anything (D320): the
Builder's run tool is the one caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.runner.records import UserRules, read_json
from kullback.runner.world.environment import BuiltEnvironment
from kullback.spec.schema import goal_of
from kullback.user import account as account_mod
from kullback.user import context as context_mod
from kullback.user import factory as user_factory
from kullback.user.fidelity import vocabulary_of
from kullback.user.simulated import SimulatedUser
from kullback.user.value_strip import value_strip


def _seed_ids(anchor: Any, task_id: str, run_ids: Iterable[str]) -> list[str]:
    """The Task's Runs minus its anchor: held-out runs never seed a Reference (D81)."""
    run_ids = list(run_ids)
    if anchor is None:
        return run_ids
    seed_runs = getattr(anchor, "seed_runs", None)
    if callable(seed_runs):
        return list(seed_runs(task_id, run_ids))
    held = set((anchor.get("held_out", {}) or {}).get(task_id, []))
    return [run_id for run_id in run_ids if run_id not in held]


def _replay_rows(workdir: Any, task_id: str) -> dict:
    """The Task's replay rows by run id, tolerant of a missing or torn file."""
    body = read_json(Path(workdir) / "replays.json", {}) or {}
    rows = (body.get(task_id) or {}) if isinstance(body, dict) else {}
    if isinstance(rows, list):
        rows = {str(row.get("run_id") or row.get("trace_id")): row
                for row in rows if isinstance(row, dict)}
    return rows if isinstance(rows, dict) else {}


def _reference_id(workdir: Any, task_id: str, run_ids: Iterable[str], anchor: Any) -> Optional[str]:
    """The confirmed seed replay with user rules on disk, earliest first (D81, D112).

    The seed set is the Task's Runs minus the anchor's held-out Runs; among the confirmed rows
    the Reference is the first one whose rules file exists, the way the Builder's re-roll runner
    picks the rules its Simulated user answers from.
    """
    seeds = set(_seed_ids(anchor, task_id, run_ids))
    rows = _replay_rows(workdir, task_id)
    confirmed = [key for key, row in sorted(rows.items())
                 if key in seeds and isinstance(row, dict)
                 and row.get("confirmed") and row.get("path")]
    if not confirmed:
        return None
    rules_dir = Path(workdir) / "user_rules"
    for key in confirmed:
        trace_id = str(rows[key].get("trace_id") or key)
        if (rules_dir / f"{trace_id}.json").is_file():
            return trace_id
    first = rows[confirmed[0]]
    return str(first.get("trace_id") or confirmed[0])


def _rules_for(workdir: Any, reference_id: Optional[str]) -> Optional[UserRules]:
    """The Reference's user rules, or nothing where the Task has none to answer from."""
    if reference_id is None:
        return None
    path = Path(workdir) / "user_rules" / f"{reference_id}.json"
    if not path.is_file():
        return None
    try:
        return UserRules.model_validate(read_json(path))
    except ValueError:
        return None


def _make_user(workdir: Any, task_id: str, anchor: Any, router: Any, user_model: Any = None,
               corpus: Optional["_Corpus"] = None) -> Any:
    """The Simulated user of the Task's confirmed Reference, or nothing without rules.

    Built from the derivation's own inputs, the replay rows, the rules and the recordings, over
    the state reader of the Router the runner built: the Examiner never opens the compiled side
    itself (D123). The vocabulary is the build's (`vocabulary_of`, D326), the one a fresh Run reads.

    With a `user_model` the rule-driven user built here is the floor under the agent user (D214),
    which answers the questions the rules' cues miss instead of restating the goal. Without one
    this is the rule-driven user alone, exactly as before the switch existed.
    """
    env = BuiltEnvironment(workdir)
    task = env.task(task_id)
    reference_id = _reference_id(workdir, task_id, task.run_ids, anchor)
    rules = _rules_for(workdir, reference_id)
    if rules is None:
        return None
    writes = env.write_tools()
    traces = env.traces()
    reference = traces.get(reference_id) if reference_id is not None else None
    members = [traces[run_id] for run_id in task.run_ids if run_id in traces]
    goal_writes, goal_counts = goal_of(workdir, task_id, writes)
    answer_strip = value_strip(members) if members else None
    vocab = vocabulary_of(workdir)
    floor = SimulatedUser(
        rules, starting_state_reader=router.state, vocab=vocab, write_tools=writes,
        goal_writes=goal_writes, answer_strip=answer_strip, goal_counts=goal_counts)
    if user_model is None:
        return floor
    corpus = corpus if corpus is not None else _Corpus(env)
    record = context_mod.mine_record_values(reference)
    ctx = context_mod.curate(task_id, rules, reference, vocab=vocab, write_tools=writes,
                             record_fields=sorted(record))
    identity_fields = vocab.by_kind("identity")
    choices = account_mod.choice_book(
        reference, writes, customer=account_mod.customer_keys(rules, identity_fields),
        traces=corpus.traces, rules=corpus.rules, identity_fields=identity_fields)
    state = getattr(router, "state", None)
    account = account_mod.account_view(getattr(state, "shared", {}) or {}, floor.identity,
                                       corpus.columns,
                                       keys=account_mod.id_keys(rules, identity_fields))
    return user_factory.build_user(
        workdir, task_id, user_model, user_factory.PURPOSE_RUN, ctx=ctx, fallback=floor,
        record_values=record, vocab=vocab, write_tools=writes, goal_writes=goal_writes,
        answer_strip=answer_strip, trace=reference, goal_counts=goal_counts, choices=choices, account=account)


def run_user(workdir: Any, task_id: str, router: Any, user_model: Any) -> Any:
    """The Task's agent user over its rule floor for a fresh Run outside the derivation.

    The Builder's run tool meets the same user the Examiner's re-rolls meet: no anchor
    here, so held-out Runs are not filtered from the Reference pick the way the
    derivation filters them, the way the Builder's own rule user picks its rules.
    """
    return _make_user(workdir, task_id, None, router, user_model)


class _Corpus:
    """The workdir's recordings, their rules and the columns their users stated, mined once per user."""

    def __init__(self, env: BuiltEnvironment):
        self.traces = dict(env.traces())
        self.rules = {trace_id: rules for trace_id in self.traces
                      if (rules := _rules_for(env.root, trace_id)) is not None}
        self.columns = account_mod.stated_columns(self.traces)
