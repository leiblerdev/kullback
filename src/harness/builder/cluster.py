"""Groups Runs into Categories by their confirmed write-tool set, then into Tasks by intent similarity (D83)."""

from __future__ import annotations

import itertools
import math
import re
from typing import Any, Iterable, Optional, Sequence

from harness.builder.mine import is_assistant_call
from harness.shared.provider import Model
from harness.shared.records import Category, Task, ToolCall, Trace, content_hash

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


def task_id(cat_id: str, run_ids: Sequence[str]) -> str:
    return "task_" + content_hash({"category_id": cat_id, "run_ids": list(run_ids)})[:12]


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

    `worlds` is compile_env.trace_worlds: per trace, row key to version hash of its pre-write
    sightings. Greedy and order free in effect: traces are taken by id and each joins the first
    subgroup whose rows it does not contradict, so the same traces always land the same way.
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
            tid = task_id(cat_id, run_ids)
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
