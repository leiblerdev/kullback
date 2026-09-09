"""Groups Runs into Categories by their confirmed write-tool set, then into Tasks by intent similarity (D83)."""

from __future__ import annotations

import itertools
import math
import re
from typing import Any, Iterable, Optional, Sequence

from kullback.ai.provider import Model
from kullback.builder.mine import is_assistant_call
from kullback.runner.records import Category, Task, ToolCall, Trace, content_hash

# Similarity is Jaccard over the user's words weighted by inverse document frequency, so the
# politeness and authentication chatter every Run repeats cannot decide membership. On the 456-Run
# tau2 retail corpus the shipped raw-token rule scored pair F1 0.276 at 0.3 and 0.720 only at 0.6,
# a cliff: the constant was doing the work. Weighted, the same corpus scores 0.719 at 0.3, 0.717 at
# 0.4 and 0.710 at 0.5 (tests/test_cluster.py runs both numbers), so 0.4 sits in the middle of a
# flat range rather than being a fitted constant (D83, D97, and the slice note in todo.md).
DEFAULT_THRESHOLD = 0.4
MIN_RUNS_GUARDED = 3  # D81: a Task with fewer Runs cannot hold one out, so it is unguarded
USER_TURNS_USED = 2  # the user's opening turns carry the intent; later turns are mostly repair
MAX_NAME_CHARS = 80

TOKEN_RE = re.compile(r"[a-z0-9_]+")
APOSTROPHE_RE = re.compile(r"[\u0027\u2019\u02bc]")  # "don't" is one word, not "don" and "t"
STOPWORDS = frozenset(
    """
    a an the and or but so then of to for in on at by from with without about into as is was were be been being
    are am it its this that these those there here i me my we our you your he she they them their his her
    do does did done have has had can could would should will shall may might must not no yes if when while
    because please thanks thank hi hello ok okay just also very really any some all
    dont doesnt didnt cant cannot couldnt wouldnt shouldnt wasnt werent isnt arent wont havent hasnt hadnt
    im ive ill id youre youve youll youd hes shes theyre theyve thats whats lets weve wed
    """.split()
)


def tokens(text: Optional[str]) -> list[str]:
    """Lowercase word tokens, apostrophes closed up, single letters and stopwords dropped."""
    if not text:
        return []
    lowered = APOSTROPHE_RE.sub("", text.lower())
    return [t for t in TOKEN_RE.findall(lowered) if len(t) > 1 and t not in STOPWORDS]


def write_tool_names(tool_sigs: Any) -> set[str]:
    """The write tools, from mined ToolSigs, a name-to-kind mapping, or a plain list of names."""
    if not tool_sigs:
        return set()
    if isinstance(tool_sigs, dict):
        return {name for name, kind in tool_sigs.items() if kind == "write"}
    names: set[str] = set()
    for sig in tool_sigs:
        if isinstance(sig, str):
            names.add(sig)
        elif getattr(sig, "kind", None) == "write":
            names.add(sig.name)
    return names


def confirmed_write_calls(trace: Trace, writes: set[str]) -> list[ToolCall]:
    """The Run's write calls that went through: a call with an error changed nothing, and only the
    assistant's own calls count toward what the Run wrote (docs/cross-domain-check.md, Judgement)."""
    return [c for c in trace.tool_calls if c.name in writes and c.error is None and is_assistant_call(c)]


def category_signature(trace: Trace, writes: set[str]) -> tuple[str, ...]:
    """The set of write tools this Run actually wrote through, sorted, as the Category key."""
    return tuple(sorted({c.name for c in confirmed_write_calls(trace, writes)}))


def category_id(signature: Sequence[str]) -> str:
    """Same write set, same id, in this corpus and the next."""
    return "cat_" + content_hash(list(signature))[:12]


def task_id(run_ids: Sequence[str]) -> str:
    """The id of the Task grouping these Runs, over the sorted Run ids and nothing else (D200).

    A Run belongs to exactly one Task, so the Run set names the Task on its own, and leaving the
    Category out of the hash is what makes the id survive a rebuild: re-mining a tool from read to
    write moves every Category id, and an id that moved with it took the Task's cache and its
    trusted ruling with it even though the same Runs were grouped the same way.
    """
    return "task_" + content_hash(sorted(run_ids))[:12]


def run_tokens(trace: Trace) -> set[str]:
    """What a Run's intent looks like to code: the words of its first user turns.

    The keys of the write call are deliberately not in here. Every Run of a Category wrote through
    the same tools, so those keys are the same for all of them and only add a constant to every
    similarity: two Runs with no word in common scored 0.308 that way and merged.
    """
    user_turns = [t for t in trace.turns if t.role == "user"][:USER_TURNS_USED]
    return {tok for turn in user_turns for tok in tokens(turn.content)}


def idf_weights(bags: Iterable[set[str]]) -> dict[str, float]:
    """How much one token is worth: log((1 + runs) / (1 + runs holding it)) + 1, smoothed.

    A token every Run says weighs 1, a token one Run says weighs log(n/2) more. The smoothing is
    what keeps a three-Run corpus sane: there the weights are nearly flat and this is plain Jaccard.
    """
    bags = list(bags)
    total = len(bags)
    document_frequency: dict[str, int] = {}
    for bag in bags:
        for token in bag:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    return {tok: math.log((1 + total) / (1 + df)) + 1.0 for tok, df in document_frequency.items()}


def similarity(a: set[str], b: set[str], weights: Optional[dict[str, float]] = None) -> float:
    """Token Jaccard, weighted when weights are given. An empty side scores zero, so a Run with no
    evidence is never forced into a cluster."""
    if not a or not b:
        return 0.0
    if weights is None:
        return len(a & b) / len(a | b)
    union = sum(weights.get(t, 1.0) for t in a | b)
    if union <= 0:
        return 0.0
    return sum(weights.get(t, 1.0) for t in a & b) / union


def first_line(text: Optional[str], max_chars: int) -> Optional[str]:
    """The first non-empty, whitespace-collapsed line of a model reply, truncated; None when there is none."""
    for line in (text or "").splitlines():
        line = " ".join(line.split())
        if line:
            return line[:max_chars]
    return None


def name_task(model: Optional[Model], traces: Sequence[Trace]) -> Optional[str]:
    """The one LLM step in clustering: it names a cluster, it never decides membership (D83)."""
    if model is None or not traces:
        return None
    reply = model.query([{"role": "user", "content": _name_prompt(traces)}])
    return first_line(reply.content, MAX_NAME_CHARS)


def _name_prompt(traces: Sequence[Trace]) -> str:
    lines = ["Name what these runs have in common, in at most six words.", ""]
    for trace in traces[:5]:
        said = [t.content for t in trace.turns if t.role == "user" and t.content][:USER_TURNS_USED]
        called = sorted({c.name for c in trace.tool_calls})
        lines.append(f"run {trace.trace_id}: user said {said}; tools called {called}")
    lines += ["", "Reply with the name only, on one line."]
    return "\n".join(lines)


def _cluster_by_intent(
    members: Sequence[Trace],
    bags: dict[str, set[str]],
    weights: dict[str, float],
    threshold: float,
) -> list[list[Trace]]:
    """Complete-linkage agglomeration: merge the two clusters whose every pair is similar enough.

    Leader clustering compared each Run only to the leaders before it, so the same three Runs came
    out as {A,B},{C} or {A,B,C} or {A},{B,C} depending on how their trace ids happened to sort.
    This is order free: only similarities decide, and a cluster holds only Runs that are all
    similar to each other (D83, section 8 determinism). Two equally good merges are decided by the
    tokens of the clusters, never by the trace ids, so renaming the Runs cannot move a Run.
    """
    by_id = {t.trace_id: t for t in members}
    order = sorted(by_id)
    groups: dict[int, list[str]] = {i: [tid] for i, tid in enumerate(order)}
    link: dict[tuple[int, int], float] = {
        (i, j): similarity(bags[order[i]], bags[order[j]], weights)
        for i, j in itertools.combinations(range(len(order)), 2)
    }

    def content(pair: tuple[int, int]) -> tuple[str, ...]:
        merged: set[str] = set()
        for index in pair:
            for tid in groups[index]:
                merged |= bags[tid]
        return tuple(sorted(merged))

    while len(groups) > 1:
        score = max(link.values())
        if score < threshold:
            break
        tied = [pair for pair, value in link.items() if value == score]
        keep, gone = tied[0] if len(tied) == 1 else min(tied, key=lambda pair: (content(pair), pair))
        groups[keep] = groups[keep] + groups[gone]
        del groups[gone]
        for other in groups:
            if other == keep:
                continue
            a, b = sorted((keep, other))
            c, d = sorted((gone, other))
            link[(a, b)] = min(link[(a, b)], link[(c, d)])
        for key in [k for k in link if gone in k]:
            del link[key]
    return [[by_id[tid] for tid in sorted(members_ids)] for _, members_ids in sorted(groups.items())]


def split_by_world(group: Sequence[Trace], worlds: dict[str, dict]) -> list[list[Trace]]:
    """Runs that saw one row in two versions before writing started in different worlds: different Tasks.

    `worlds` is compile_env.trace_worlds: per trace, a key to the version hash of its pre-write
    sightings under that key. The key names the tool that answered and the row it answered about,
    and the version is the canonical form of the recording alone (D216), so this split is a function
    of the recordings and the split of one recording set never moves between rounds. Nothing of the
    schema or of the readers is passed here or is inside a version.

    Greedy and order free in effect: traces are taken by id and each joins the first subgroup whose
    rows it does not contradict, so the same traces always land the same way.
    """
    subgroups: list[tuple[list[Trace], dict]] = []
    for trace in sorted(group, key=lambda t: t.trace_id):
        seen = worlds.get(trace.trace_id) or {}
        for members, world in subgroups:
            if all(world.get(key, version) == version for key, version in seen.items()):
                members.append(trace)
                world.update(seen)
                break
        else:
            subgroups.append(([trace], dict(seen)))
    return [members for members, _ in subgroups]


def cluster_runs(
    traces: Iterable[Trace],
    tool_sigs: Any = None,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    model: Optional[Model] = None,
    min_runs: int = MIN_RUNS_GUARDED,
    worlds: Optional[dict[str, dict]] = None,
) -> tuple[list[Category], list[Task]]:
    """Categories by write-tool signature, Tasks by intent similarity and starting world inside them (D83, D81, D74)."""
    writes = write_tool_names(tool_sigs)
    by_signature: dict[tuple[str, ...], list[Trace]] = {}
    bags: dict[str, set[str]] = {}
    for trace in sorted(traces, key=lambda t: t.trace_id):
        by_signature.setdefault(category_signature(trace, writes), []).append(trace)
        bags[trace.trace_id] = run_tokens(trace)
    # The weights come from the whole corpus, not from one Category, because a word is boilerplate
    # by how often the customer's users say it, not by which tool the Run happened to write with.
    weights = idf_weights(bags.values())

    categories: list[Category] = []
    tasks: list[Task] = []
    for signature in sorted(by_signature):
        cat_id = category_id(signature)
        task_ids: list[str] = []
        for cluster in _cluster_by_intent(by_signature[signature], bags, weights, threshold):
          for group in split_by_world(cluster, worlds or {}):
            run_ids = sorted(t.trace_id for t in group)
            tid = task_id(run_ids)
            tasks.append(
                Task(
                    id=tid,
                    category_id=cat_id,
                    run_ids=run_ids,
                    unguarded=len(run_ids) < min_runs,
                    name=name_task(model, group),
                )
            )
            task_ids.append(tid)
        categories.append(Category(id=cat_id, write_tools=list(signature), task_ids=task_ids))
    return categories, tasks


# D200: what a rebuild says about a frozen Task it did not reproduce. It is kept, so every number
# measured over the frozen list stays comparable across rounds, and it is named, so the drift is a
# finding rather than a silently smaller denominator.
FROZEN_ONLY_REASON = "the rebuild grouped this Task's Runs differently; the frozen Task is kept"
# D216: the same Task, re-examined against the recordings alone. The world split is not what
# stranded it, so whatever moved sits in the intent clustering and the flag is wrong to carry.
CLEARED_REASON = ("the recordings alone still group this Task's Runs together, so the starting "
                  "world did not strand it; the rebuild's intent clustering did")
GROUPING_FILE = "grouping.json"
# 1: the fingerprint of the frozen grouping and the two inputs it was taken over (D216).
GROUPING_FORMAT = 1


class UnexplainedRegrouping(RuntimeError):
    """The grouping moved while both of its inputs stood still, which the rule says cannot happen.

    The split of Runs into Tasks is a function of the recordings and of the row identity mined from
    them (D216). If the fingerprint of a round's grouping differs from the frozen one while the
    recording set and the homing are byte identical, some third thing reached the split, and that is
    a bug in the harness rather than a finding about the corpus. It raises here rather than being
    counted, because a count would let the very drift this rule exists to remove pass as weather.
    """


def _run_ids(task: Any) -> tuple[str, ...]:
    """A Task's Run ids, sorted, whether it arrives as a record or as a Task."""
    ids = task.get("run_ids") if isinstance(task, dict) else getattr(task, "run_ids", None)
    return tuple(sorted(str(run_id) for run_id in (ids or ())))


def grouping_fingerprint(tasks: Iterable[Any]) -> str:
    """The identity of a grouping: its sorted Run sets and nothing else.

    Task ids are hashed over the Run set already, so they would say the same thing, but only while
    the id scheme stands still. The Run sets are what two rounds are actually comparable on: the
    same recordings grouped the same way have one fingerprint however the ids were made.
    """
    return content_hash(sorted(_run_ids(task) for task in tasks))


def recordings_hash(traces: Iterable[Trace]) -> str:
    """The recording set as one hash: every Run's id with the hash of the raw recording behind it.

    This is one of the two inputs the split is allowed to depend on. A corpus that grew, shrank or
    was re-ingested moves it; a rebuild over the same recordings cannot.
    """
    return content_hash(sorted([str(trace.trace_id), str(trace.raw_hash)] for trace in traces))


def moved_input(fingerprint: str, inputs: dict, frozen: Optional[dict]) -> str:
    """Which input moved under a grouping that differs from the frozen one: the name, or "".

    The empty string is the ordinary answer: the fingerprint matches, so nothing moved and there is
    nothing to say on the round line. A fingerprint that differs is explained by the first input
    that differs, and a fingerprint that differs while both inputs stand still raises, because the
    rule says the grouping is a function of those two and nothing else.
    """
    if not isinstance(frozen, dict) or not frozen.get("fingerprint"):
        return ""
    if fingerprint == frozen.get("fingerprint"):
        return ""
    for name in ("recordings", "homing"):
        if inputs.get(name) != frozen.get(name):
            return name
    raise UnexplainedRegrouping(
        f"the grouping fingerprint moved from {frozen['fingerprint'][:12]} to {fingerprint[:12]} "
        f"while the recordings and the homing both stood still")


def world_split_reason(run_ids: Sequence[str], worlds: dict[str, dict]) -> Optional[str]:
    """Why the recordings alone split these Runs, or None when they would still group together.

    The same greedy rule `split_by_world` applies, run over Run ids rather than over records, so a
    frozen Task can be asked the question without the traces being on hand. The reason names the
    tool and the row, because that pair is what the Examiner has to go and look at: one tool showed
    one row in two versions before either Run wrote, and no overlay can hold both.
    """
    world: dict = {}
    for run_id in sorted(str(r) for r in run_ids):
        seen = worlds.get(run_id) or {}
        against = [key for key, version in sorted(seen.items()) if world.get(key, version) != version]
        if against:
            return f"Run {run_id} disagrees on {_where(against[0])} before either Run wrote"
        world.update(seen)
    return None


def _where(key: Any) -> str:
    """A world key as a sentence part: the tool that answered and the row it answered about.

    A key with no table is another requestor's own state, which no table of the customer's holds,
    so it is named by the requestor rather than by a row that does not exist.
    """
    if isinstance(key, (tuple, list)) and len(key) == 3:
        tool, table, row_id = (str(part) for part in key)
        named = f"{table} row {row_id}" if table else f"the state of the {row_id} requestor"
        return f"{named} as {tool or 'an unnamed tool'} showed it"
    return str(key)


def _as_task(record: Any) -> Task:
    """A frozen record as a Task, whether it was stored as a record or as a Task already."""
    return record if isinstance(record, Task) else Task.model_validate(dict(record))


def resume_frozen(tasks: Sequence[Task], frozen: Optional[Sequence[Any]],
                  *, min_runs: int = MIN_RUNS_GUARDED,
                  worlds: Optional[dict[str, dict]] = None) -> tuple[list[Task], dict]:
    """The Task list a rebuild is allowed to publish once a list has been frozen (D200).

    Every number a build is judged on is counted over the frozen list, so a re-split that drops a
    Task or gives it a new id makes two rounds incomparable and takes that Task's ruling with it.
    Here a frozen Task is never dropped and never re-split: it keeps its id, its Runs and its
    Category. A Run the frozen list does not hold is free to form a new Task, which is what lets a
    corpus grow; a Run the frozen list does hold stays where it was even when the rebuild would
    have put it somewhere else, and the freshly grouped Task it was going to join keeps only the
    Runs nobody had frozen (with the id those Runs address, since the id is over the Run set).

    Returns the Task list and the split status: how many Tasks came off the frozen list, how many
    the rebuild grouped, which are new, and which are `frozen_only`, meaning the rebuild would have
    removed them. A `frozen_only` count above zero is a finding, not a failure: the build goes on.

    Given `worlds`, every `frozen_only` Task is re-examined against the recordings alone (D216).
    D200 could only say the rebuild grouped a Task's Runs differently, which named no cause and left
    the flag standing for as long as the corpus lived. Now a Task whose Runs the recordings would
    still put in one world loses the flag with the reason, because the starting world did not strand
    it and the difference is the intent clustering's; a Task whose Runs the recordings really do
    split keeps the flag and carries the split's own reason, naming the tool and the row, which is a
    finding the Examiner can file rather than a count nobody can act on.
    """
    live = list(tasks)
    if frozen is None:
        return live, {"frozen": 0, "live": len(live), "added": [t.id for t in live],
                      "frozen_only": [], "reasons": {}, "cleared": {}}

    kept = [_as_task(record) for record in frozen]
    home = {run_id: task.id for task in kept for run_id in task.run_ids}
    by_id = {task.id: task for task in kept}
    # The same Runs are the same Task whatever the id was hashed over, so a list frozen by an older
    # harness is matched on its Run set as well as on its id.
    live_sets = {frozenset(task.run_ids) for task in live}
    live_ids = {task.id for task in live}
    reproduced = {task.id for task in kept
                  if task.id in live_ids or frozenset(task.run_ids) in live_sets}

    # A frozen Task with no name yet takes the name the rebuild found for the same Runs: a name is a
    # label, never identity, so it may improve between rounds where the id may not.
    named = {task.id: task.name for task in live if task.name}
    added: list[Task] = []
    for task in live:
        if task.id in by_id:
            continue
        free = [run_id for run_id in task.run_ids if run_id not in home]
        if not free:
            continue
        if len(free) == len(task.run_ids):
            added.append(task)
        else:
            added.append(task.model_copy(update={"id": task_id(free), "run_ids": sorted(free),
                                                 "unguarded": len(free) < min_runs}))
    for task in kept:
        if task.name is None and task.id in named:
            task.name = named[task.id]

    unreproduced = [task for task in kept if task.id not in reproduced]
    frozen_only: list[str] = []
    reasons: dict[str, str] = {}
    cleared: dict[str, str] = {}
    for task in unreproduced:
        if worlds is None:
            frozen_only.append(task.id)
            reasons[task.id] = FROZEN_ONLY_REASON
            continue
        reason = world_split_reason(task.run_ids, worlds)
        if reason is None:
            cleared[task.id] = CLEARED_REASON
        else:
            frozen_only.append(task.id)
            reasons[task.id] = reason
    return kept + added, {
        "frozen": len(kept), "live": len(live), "added": [t.id for t in added],
        "frozen_only": frozen_only, "reasons": reasons, "cleared": cleared,
    }
