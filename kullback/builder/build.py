"""The one declaration of the Builder's stages as a DAG over named artifacts, the plan that runs any target of it,
and the Candidate Run batch that follows a build.

`stages(plan)` is the whole graph: fourteen stages, each naming what it reads and writes, handed to
`pipeline.Pipeline`, which decides the order (D120: the scheduler decides, never a model). `build()`
is the CLI's entry and runs every stage; `execute(plan, target)` runs one stage or artifact and what
is upstream of it, which is what the Builder's tools (builder/tools.py) call. A tool that wants one
tool body recompiled or one Task replayed gets a variant of the same declaration with that stage
narrowed, so its cache key and its gates are the stage's own.

The re-rolls stage keys a second time inside itself, per Task. The pipeline's key is one key for the
whole stage, so a build that repaired one Intent or recompiled one body re-ran every Task's Runs:
three rounds of one build re-rolled at the scale of the whole corpus and re-rolls were four fifths of the
build's spend. `_reroll_key` is what one Task's Runs were sampled under (its overlay and the
Starting state under it, the schema, the bodies of the tools its own recordings call, the
canonicalizer rules, its user rules, the Vocabulary, the policy text, the system prompt it opens
with, and the count, seed, model, turn cap and code version), recorded beside its Runs in
`runs/<task>/rerolls.json`; a Task whose key has not moved and whose Run files are still there keeps
them. The assumption: a body of a tool the Task's recordings never call may change without the
Task's re-rolls going stale. A Run on disk is a sample already taken against the toolkit as it stood,
and nothing re-scores it against the current bodies (the D79 suite reads the Run file: the second
path and the false-rejection number score the recorded End state, they do not re-execute it). A
Candidate could call that tool in a live Run; this Run did not.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.ai import provider
from kullback.builder import (
    body_skill,
    cluster,
    compile_env,
    ingest,
    intent,
    memory,
    mine,
    parallel,
    pipeline,
    policy,
    readers,
    sandbox,
    synth,
    user_sim,
    vocabulary,
)
from kullback.gates import artifacts, fidelity, verifier_suite
from kullback.gates import scorecard as scorecard_mod
from kullback.gates import stages as stage_gates
from kullback.runner import budget, canon, loop, route
from kullback.runner import replay as replay_mod
from kullback.runner.canon import rules_of as _rules_of
from kullback.runner.records import (
    EntitySchema,
    Environment,
    GateResult,
    Task,
    ToolSig,
    Trace,
    UserRules,
    as_dict,
    content_hash,
)
from kullback.runner.records import read_json as _read_json
from kullback.runner.records import write_json as _write_json

# This module is the graph, not the runner: assembling the stages means naming ingest, mine, cluster,
# compile_env, policy and user_sim, and the Runner never imports the Builder (design section 3, build
# brief rule 7, D89), so the wiring has to live beside them in builder/ and not in runner/.
# pipeline.py is the generic stage runner this graph is handed to. The graph ends at the re-rolls:
# the Verifiers are the Examiner's (D123), derived by `kullback.examiner.stage.derive_all` over the
# artifacts this graph leaves, and the two Runs the Examiner needs, check 6's probe and more
# re-rolls, are `probe_runner` and `reroll_runner` below, the Runner as a callable over this plan's
# store (D120).

CANON_RULES = "canon-rules.json"


class BuildError(RuntimeError):
    """The build cannot start: no traces, no Task, no Environment on disk."""


def record_gate(workdir: Path, result: GateResult) -> GateResult:
    """Append one gate to gates.json, which is where report.py reads the build's gates.

    Design section 6 gives every stage its own answer to a failed gate, and only "build
    Environment" says "fail build". A gate whose failure means "flag, do not synthesize" or
    "Task not verdicted" is therefore recorded here and does not roll the artifact back into
    its stage; the Pipeline's rollback edge is for the gates that do stop a build. Inside a stage
    the same write goes through `ctx.record_gate`, the pipeline's ledger, so stages on two threads
    do not race for the file; this is the form for a caller outside any stage.
    """
    return pipeline.GateLedger(workdir).record("", result)


def _records(paths: Iterable[Path], model: type) -> list:
    return [model.model_validate(json.loads(p.read_text(encoding="utf-8"))) for p in paths]


def load_traces(workdir: Path) -> list[Trace]:
    folder = workdir / "traces"
    return _records(sorted(folder.glob("*.json")), Trace) if folder.is_dir() else []


def _build_id(workdir: Path) -> str:
    """A short, stable id for this build: the one thing every build has is its workdir.

    Used only to shape `prompt_cache_key` (docs/prompt-caching.md item 4); it is not a customer
    identity and it is not meant to match across two different workdirs.
    """
    return content_hash(str(Path(workdir).resolve()))[:12]


def _wrap(model: Any, stage: str, workdir: Path, ceiling: Any, model_id: Optional[str] = None,
          cap_context: bool = True, memoize: bool = True) -> Any:
    """Every model the Builder hands to a stage goes through budget.py (D65, D86).

    Wrapping here is what writes budget.json and enforces the 40 percent context cap; a Candidate
    is wrapped with cap_context=False, because it runs under the production setting (D65), and
    with memoize=False, because a Candidate's answer has to be a fresh sample (docs/prompt-caching.md
    item 3): run_batch is the only caller that passes it. Every other stage is memoized: on a repeat
    build, a request byte-identical to one already answered in this workdir never reaches the
    network, and its usage was zeroed at the memo, so BudgetedModel prices it at 0.00 and only
    counts it in `memo_hits`. `prompt_cache_key` is set here too, one string per build and stage,
    so OpenAI routes every call of a stage to the same cache regardless of whether it was memoized.
    """
    if model is None:
        return None
    name = model_id or getattr(model, "name", None) or "model"
    if ceiling is not None:
        # An unpriced model is refused here, before a BudgetedModel is even built, rather than
        # quietly handed ceiling=None: that would run the model completely unmetered under a
        # ceiling that was supposed to refuse it (D86).
        ceiling.require_priced(name)
    inner = provider.MemoModel(model, workdir) if memoize else model
    cache_key = f"kullback-{_build_id(workdir)}-{stage}"
    return budget.BudgetedModel(inner, stage=stage, workdir=workdir, model_id=name,
                                ceiling=ceiling, cap_context=cap_context, prompt_cache_key=cache_key)


def _ceiling(workdir: Path, usd: Optional[float]) -> Optional[budget.Ceiling]:
    """The spend ceiling, resumed from budget.json so a stopped build does not start at zero (D86)."""
    if usd is None:
        return None
    return budget.Ceiling.from_totals(workdir, usd)


# --- the stages -------------------------------------------------------------

def _ingest_stage(workdir: Path, files: list[Path]):
    def run(ctx, inputs):
        for path in files:
            ingest.ingest_file(path, ctx.workdir)
        return {"traces": load_traces(ctx.workdir)}

    def gate(ctx, outputs):
        return artifacts.ingest_gate(outputs["traces"])

    return pipeline.Stage(name="ingest", fn=run, outputs=("traces",), gate=gate,
                          code_version=f"ingest:{content_hash([str(p) for p in files])[:16]}")


def _mine_stage():
    def run(ctx, inputs):
        traces = inputs["traces"]
        sigs = mine.mine_tools(traces)
        # The write tools are the sigs' own, so the composite-key rule reads the same kinds the rest
        # of the build does rather than classifying the tools a second time.
        schema = mine.mine_schema(traces, write_tools=sorted(s.name for s in sigs if s.kind == "write"))
        # D164: the names the recording refused on every call, and the ones a recorded agent
        # invented, are no tool of this customer. They are written beside the sigs so a build can
        # see them, and they are never a failure: a Run refuses them as the recording did.
        unknown = mine.unknown_tools(traces)
        _write_json(ctx.workdir / "tool_sigs.json", [as_dict(s) for s in sigs])
        _write_json(ctx.workdir / "unknown_tools.json", unknown)
        # Where each tool's result rows were homed and by which rule, with the rows no rule could
        # home. A lookup whose rows reach no table is the whole of its replay fidelity, and this is
        # where that is readable before a single body has been written.
        _write_json(ctx.workdir / "row_homes.json", mine.row_homes(traces))
        # The constants of the world: per tool the corpus called the same way every time and got
        # the same answer to, that answer. They are columns of one row of the schema's own
        # constants table; the file is what a reader of the build sees them by, since the mine gate
        # is frozen and cannot carry a count of them.
        _write_json(ctx.workdir / "world_constants.json",
                    {"table": mine.constants_table_of(schema), "row": mine.CONSTANTS_ROW,
                     "columns": sorted(mine.constants_row(schema))})
        _write_json(ctx.workdir / "schema.json", as_dict(schema))  # cli._score reads it (D39, D73)
        calls = [c for t in traces for c in t.tool_calls]
        # "flag, do not synthesize": a tool the corpus barely shows stays in the build, named in
        # the gate, rather than being invented or dropped (design section 6).
        ctx.record_gate(artifacts.mine_gate(sigs, calls, unknown=unknown))
        return {"mined_sigs": sigs, "mined_schema": schema}

    # The artifacts are `mined_schema` and `mined_sigs`, not `schema` and `sigs`: the readers stage
    # is what releases both, because a requestor's prose results add tables to the schema and settle
    # the kind of the tools that answer with prose, and every stage downstream reads them settled.
    # With no prose results the readers stage passes both through untouched.
    return pipeline.Stage(name="mine", fn=run, inputs=("traces",), outputs=("mined_sigs", "mined_schema"),
                          code_version=_version("mine", run, mine))


def _readers_stage(model: Any, max_attempts: int = readers.MAX_ATTEMPTS):
    """The rows a requestor other than the assistant reveals through prose results (D176 candidate).

    It runs per requestor whose own tools answer with strings the row extractor reads nothing out
    of. Where a corpus has none, the stage records "no prose results", passes the mined schema
    through unchanged and calls no model at all.

    What a tool of that requestor changes is mined here by association over the corpus, not read off
    the miner's kind, and closes a column at that tool's calls so a value read only after a write is
    not the recording's starting value. Those credits then settle the kind of every tool whose
    results are prose: a write when it is credited with a column, a read when it is not. A column no
    recording read before a write is filled with the commonest pre-write value the corpus shows, and
    the fill is recorded as an assumption.
    """

    def run(ctx, inputs):
        traces, schema, sigs = inputs["traces"], inputs["mined_schema"], inputs["mined_sigs"]
        by_requestor = readers.prose_calls(traces)
        if not by_requestor:
            _write_json(ctx.workdir / readers.READERS_FILE, {"note": readers.NO_PROSE})
            ctx.record_gate(stage_gates.readers_gate([], 0))
            return {"schema": schema, "sigs": sigs,
                    "readers": {"note": readers.NO_PROSE, "proposals": {}, "rows": {}}}
        if model is None:
            raise BuildError("this corpus has prose results from a requestor of its own and the "
                             "readers stage has no model to propose them with; pass --model")
        proposals, rows, nodes, values = {}, {}, [], {}
        fills, assumptions, unset = {}, [], {}
        for requestor in sorted(by_requestor):
            proposal, attempts, parsed = readers.propose(
                model, requestor, by_requestor[requestor], traces, ctx.workdir / "readers",
                max_attempts=max_attempts)
            read_rows = readers.starting_rows(traces, proposal, parsed)
            filled, sentences, missing = readers.fills_for(read_rows, proposal)
            proposals[requestor] = proposal.to_dict()
            rows[requestor] = read_rows
            fills[requestor] = filled
            assumptions += sentences
            unset[requestor] = missing
            values[requestor] = readers.column_values(proposal, parsed)
            nodes += attempts
        artifact = {"proposals": proposals, "rows": rows, "fills": fills,
                    "assumptions": assumptions, "unset": unset}
        kept = readers.proposals_from(artifact)
        readers.apply_to_schema(schema, kept, values)
        sigs = readers.apply_to_sigs(sigs, kept)
        _write_json(ctx.workdir / readers.READERS_FILE, {**artifact, "attempts": nodes})
        _write_json(ctx.workdir / "schema.json", as_dict(schema))
        _write_json(ctx.workdir / "tool_sigs.json", [as_dict(s) for s in sigs])
        # Section 6: a proposal the gate could not satisfy is flagged and kept, never a failed build.
        ctx.record_gate(stage_gates.readers_gate(proposals.values(), len(by_requestor),
                                                 assumptions=assumptions, unset=unset,
                                                 kinds={p.requestor: readers.kinds_for(p) for p in kept}))
        return {"schema": schema, "sigs": sigs, "readers": artifact}

    version = (f"readers:{getattr(model, 'name', 'none')}:{max_attempts}:"
               f"{_module_hash(readers)}:{_module_hash(sandbox)}")
    return pipeline.Stage(name="readers", fn=run, inputs=("traces", "mined_schema", "mined_sigs"),
                          outputs=("schema", "sigs", "readers"), code_version=version)


def _cluster_stage():
    def run(ctx, inputs):
        # D74: two Runs that saw one row in two versions before writing started in different
        # worlds, and a Task's overlay can pin only one, so they are different Tasks.
        worlds = compile_env.trace_worlds(inputs["traces"], inputs["schema"],
                                          cluster.write_tool_names(inputs["sigs"]))
        # A row another requestor revealed splits Tasks the same way (D74): two recordings that read
        # one of its columns differently before either wrote started in different worlds.
        readers.merge_worlds(worlds, inputs["readers"], inputs["schema"])
        categories, tasks = cluster.cluster_runs(inputs["traces"], inputs["sigs"], worlds=worlds)
        for task in tasks:
            _write_json(ctx.workdir / "tasks" / f"{task.id}.json", as_dict(task))
        _write_json(ctx.workdir / "tasks.json", {"tasks": [as_dict(t) for t in tasks]})
        # D96: the coverage denominator is frozen once, here, before anything measures coverage.
        scorecard_mod.freeze_tasks(ctx.workdir, tasks)
        ctx.record_gate(stage_gates.cluster_gate(tasks, categories))
        return {"categories": categories, "tasks": tasks}

    return pipeline.Stage(name="cluster", fn=run, inputs=("traces", "sigs", "schema", "readers"),
                          outputs=("categories", "tasks"),
                          code_version=_version("cluster", run, cluster, intent, compile_env, readers))


def _canon_stage():
    def run(ctx, inputs):
        schema = inputs["schema"]
        rows = [row for trace in inputs["traces"] for call in trace.tool_calls
                for row in _rows_of(call.result)]
        rules = canon.learn_rules(schema, rows)
        canon.save_rules(rules, ctx.workdir / CANON_RULES)
        return {"canon_rules": rules.model_dump()}

    return pipeline.Stage(name="canon_rules", fn=run, inputs=("traces", "schema"), outputs=("canon_rules",),
                          code_version=_version("canon_rules", run, canon))


def _rows_of(result: Any) -> list[dict]:
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    return []


def _state_stage(grow: Optional[dict] = None, grow_seed: int = 0):
    def run(ctx, inputs, grow=None, grow_seed=0):
        state = compile_env.build_starting_state(inputs["traces"], inputs["schema"], ctx.workdir,
                                                 inputs["tasks"], inputs["sigs"], grow=grow,
                                                 grow_seed=grow_seed,
                                                 revealed_rows=readers.reader_rows(inputs["readers"]),
                                                 revealed_assumptions=readers.reader_assumptions(
                                                     inputs["readers"]))
        # The synthetic ids live on the schema (D40); run_batch reads them back from schema.json.
        _write_json(ctx.workdir / "schema.json", as_dict(inputs["schema"]))
        return {"db": state.db, "overlays": list(state.overlays),
                "assumptions": list(state.assumptions), "synthetic_rows": list(state.synthetic_rows)}

    # A partial, so the grow targets are in the stage's cache key: the same traces grown to two
    # sizes are two Starting states, not one served twice (pipeline._fn_identity).
    fn = functools.partial(run, grow=dict(grow or {}), grow_seed=grow_seed)
    return pipeline.Stage(name="starting_state", fn=fn,
                          inputs=("traces", "schema", "tasks", "sigs", "readers"),
                          outputs=("db", "overlays", "assumptions", "synthetic_rows"),
                          code_version=_version("starting_state", fn, compile_env, synth, readers))


def _record_hardcoded_lesson(workdir: Any, name: str) -> None:
    """Tell the next attempt what the gates do not say: this body never read its arguments.

    Written once. The lessons file is an input of this stage, so appending the same sentence on
    every run would change the stage's key on every run and recompile every tool that carries it.
    """
    if [compile_env.HARDCODED_LESSON] in memory.load_tool_lessons(workdir).get(name, []):
        return
    memory.record_lesson(workdir, name, [compile_env.HARDCODED_LESSON])


# Where the body a tool already has is replayed, beside the attempt directories of the run that is
# trying to beat it, so the two never share a sandbox directory.
KEPT_BODY_DIR = "kept_body"
# The stage's ruling on those replays, one row per tool: kept, beaten, or could not run, with both
# scores. A record of the run, never an input of it (see the comment where it is filled).
KEPT_BODIES_FILE = "kept_bodies.json"


def _tools_stage(model: Any, max_attempts: int, workers: int = 1, only: Optional[Iterable[str]] = None):
    """compile_tools, or with `only` the same stage narrowed to those tools: the rest of the bodies
    are read back from bodies.json, so the artifact it releases is still every body (the tool
    `compile_tool(name)`).

    A tool that already has a body in this workdir keeps it unless the run's own attempt beats it.
    D174 made that true of a narrowed rerun by comparing the score written on the tool's row; it was
    not true of a full run, which is what the stage does whenever one of its inputs moved (the
    schema, the Starting state, the readers, the tables block), and a full run rewrites every body
    from scratch. Two live rounds lost Tasks that way with nothing about the rewritten tools' own
    inputs having moved: one took a write tool from 83 of its 139 recorded calls matched to 39 and
    the corpus from 179 confirmed Tasks to 165, and another dropped six device tools by more than
    five points each in the round that raised ten. So the previous body is replayed here under the
    world as it stands now and under the gates as they stand now (`compile_env.grade_body`), and it
    is a candidate on the same key the compiler ranks its own attempts by. The stage knows nothing
    about which input moved; the rule is only that a body which exists competes.
    """
    only = sorted(only) if only is not None else None

    def run(ctx, inputs):
        if model is None:
            raise BuildError("compile_tools has to run and this build has no model; pass --model "
                             "(an --iterate build re-runs the stage when its inputs or code changed)")
        traces, tasks = inputs["traces"], inputs["tasks"]
        seeds = _seed_traces(ctx, tasks, traces)
        calls_by_tool: dict[str, list] = {}
        call_tasks: dict[str, str] = {}
        # D74: every recorded call replays on the world its Task saw before any write. A call that
        # follows a write on the same row in its own trace saw the world after that write, so it
        # cannot be replayed on that world and is not evidence against the tool it called: the second
        # retail build failed get_order_details for reading back an order the trace had just changed.
        after_write = after_write_calls(traces, cluster.write_tool_names(inputs["sigs"]))
        # D164: a call from a requestor this tool never answered was refused by the recording, so it
        # is not evidence for the body; the body is written against the calls of its own callers.
        callers = callers_by_tool(inputs["sigs"])
        skipped: dict[str, int] = {}
        for trace in traces:
            task_id = _task_of(tasks, trace.trace_id)
            for at, call in enumerate(trace.tool_calls):
                if is_evidence_call(call, callers):
                    if (trace.trace_id, at) in after_write:
                        skipped[call.name] = skipped.get(call.name, 0) + 1
                    elif trace.trace_id in seeds:  # D81: the anchor's calls are not Builder evidence
                        calls_by_tool.setdefault(call.name, []).append(call)
                if call.id and task_id:
                    call_tasks[call.id] = task_id
        # D74: each recorded call replays on the world its own Task saw, not on the shared one.
        states = compile_env.call_starting_states(inputs["db"], inputs["overlays"],
                                                  compile_env.overlay_values(ctx.workdir), call_tasks)
        tool_names = [sig.name for sig in inputs["sigs"]]
        # The transport's error wrapper is one per corpus, not one per tool: read it once over
        # every recorded call, so a tool with a single error still has it peeled.
        error_prefix = compile_env.shared_error_prefix(
            call for calls in calls_by_tool.values() for call in calls)
        # What a body has to know about a table another requestor's own tools revealed: how to reach
        # its one row, and the derivations of the columns nothing stores. Same bytes for every tool.
        world_note = readers.body_note(readers.proposals_from(inputs["readers"]))
        bodies, gates, assisted, builds = {}, [], [], {}
        outcomes: dict[str, list[dict]] = {}  # D171: per tool, one row per recorded call
        rules = _rules_of(inputs)
        sigs = list(inputs["sigs"])
        # Every body this workdir already holds, whatever run of the stage left it. On a full run
        # they are the memory of the last one; on a narrowed run they are also the bodies of the
        # tools this run does not touch, which is why that branch reads them into its own artifact.
        stored_bodies = dict(_read_json(ctx.workdir / "bodies.json", {}) or {})
        stored_builds = dict(_read_json(ctx.workdir / "tool_builds.json", {}) or {})
        if only is not None:
            unknown = sorted(set(only) - {sig.name for sig in sigs})
            if unknown:
                raise BuildError(f"no mined tool is named {', '.join(unknown)}")
            bodies, builds = dict(stored_bodies), dict(stored_builds)
            assisted = [name for name, row in builds.items() if row.get("assisted") and name not in only]
            # D171: the tools this run does not recompile keep the per-call rows the last run left,
            # so the artifact it releases still attributes every tool's fidelity, not only these.
            all_outcomes = _read_json(ctx.workdir / "tool_call_outcomes.json", {}) or {}
            outcomes = {name: list(rows) for name, rows in all_outcomes.items() if name not in only}
            sigs = [sig for sig in sigs if sig.name in only]
        # D174, widened: what each tool this run writes a body for already had, so an attempt that
        # is worse than it cannot replace it. The score is not read off the row: a row's score was
        # measured under the world of the run that wrote it, and a run whose inputs moved is a
        # different world, so the two are only comparable when both are measured under this one.
        previous = {sig.name: (stored_bodies[sig.name], stored_builds.get(sig.name) or {})
                    for sig in sigs if (stored_bodies.get(sig.name) or "").strip()}

        def compile_one(sig):  # one tool, its own directory and nodes; independent of every other (D118)
            # The body this tool already has, replayed first, so what it fails at can be said to the
            # writer before it writes. The replay is per tool and runs on this tool's own thread.
            kept = previous.get(sig.name)
            graded = compile_env.grade_body(
                sig, kept[0], calls_by_tool.get(sig.name, []), inputs["schema"], inputs["db"],
                ctx.workdir / "tools" / sig.name / KEPT_BODY_DIR,
                call_states=states, rules=rules) if kept is not None else None
            # What this tool already failed on, so a recompile asks a different question than the
            # one that failed, and what the body it has to beat fails at now. The kept body itself
            # is never in the prompt: shown one, the writer copies it, and a copy cannot beat it.
            lesson = memory.lesson_for(ctx.workdir, sig.name)
            if graded is not None:
                hint = compile_env.kept_body_hint(
                    graded.gates, compile_env.split_calls(calls_by_tool.get(sig.name, []))[1])
                lesson = "\n".join(part for part in (lesson, hint) if part)
            return compile_env.compile_tool(model, sig, calls_by_tool.get(sig.name, []),
                                            inputs["schema"], inputs["db"],
                                            ctx.workdir / "tools" / sig.name,
                                            max_attempts=max_attempts, call_states=states,
                                            rules=rules, tool_names=tool_names,
                                            error_prefix=error_prefix, world_note=world_note,
                                            lesson=lesson), graded

        declined: list[str] = []
        # Per tool, whether the body it already had was kept, beaten, or could not run at all under
        # this world, with both scores. It is a record of what this run decided, not an input of it,
        # so it lives beside the stage's other records rather than on the tool's row: the row is a
        # declared input path of a narrowed rerun, and a decision written there would move the key
        # of the next identical request and buy a recompile nobody asked for (D174's own rule that a
        # tie leaves the stage's files untouched).
        kept_rulings = dict(_read_json(ctx.workdir / KEPT_BODIES_FILE, {}) or {}) if only is not None else {}
        for sig, (build, graded) in zip(sigs, parallel.each(sigs, compile_one, workers), strict=True):
            gates.extend(build.gates)
            score = list(compile_env.attempt_score(build.gates))
            kept_score = list(compile_env.attempt_score(graded.gates)) if graded is not None else None
            # A body that answers no call at all under this world is no candidate, whatever it
            # scores: the tie below would otherwise hand the tool back to a body a schema change
            # broke. Everything else competes, and a tie goes to the body that is already there,
            # which the Examiner has seen and the Tasks that trusted it were trusted against.
            keeps_previous = graded is not None and not graded.could_not_run and kept_score >= score
            kept_rulings.pop(sig.name, None)
            if graded is not None:
                kept_rulings[sig.name] = {
                    "outcome": "kept" if keeps_previous else
                               ("could_not_run" if graded.could_not_run else "beaten"),
                    "kept_score": kept_score, "attempt_score": score}
            if keeps_previous:
                # A run of this stage is an attempt at a better body, not a replacement for the one
                # it has. The attempt scored no higher on the key the compiler ranks its own
                # attempts by (gates passed, then recorded calls matched), so the kept body stands.
                # Its assisted and hardcoded readings, its score and its per-call rows are the ones
                # just measured, not the ones the last run wrote, because those were measured under
                # a world that has since moved; where the world has not moved they are the same
                # values and the row is left byte for byte as it was. One live build's third round
                # recompiled a write tool from 65 percent of its calls matched to none of them and
                # lost 41 Tasks of fidelity in the round.
                body = previous[sig.name][0]
                bodies[sig.name] = compile_env.mark_hardcoded(body) if graded.hardcoded else body
                builds[sig.name] = dict(previous[sig.name][1], assisted=graded.assisted,
                                        hardcoded=graded.hardcoded, score=kept_score)
                if graded.hardcoded:  # D181's rule 7, for the body the stage releases, not the attempt
                    _record_hardcoded_lesson(ctx.workdir, sig.name)
                if score < kept_score:  # D174's line for the Builder: change the hint, not the request
                    builds[sig.name]["recompile_declined"] = {"attempt_score": score,
                                                              "kept_score": kept_score}
                outcomes[sig.name] = graded.call_outcomes
                if graded.assisted:
                    assisted.append(sig.name)
                declined.append(sig.name)
                continue
            bodies[sig.name] = compile_env.mark_hardcoded(build.body) if build.hardcoded else build.body
            builds[sig.name] = {"assisted": build.assisted, "hardcoded": build.hardcoded,
                                "nodes": build.nodes,
                                "after_write_skipped": skipped.get(sig.name, 0), "score": score}
            if build.hardcoded:
                _record_hardcoded_lesson(ctx.workdir, sig.name)
            outcomes[sig.name] = build.call_outcomes
            if build.assisted:
                assisted.append(sig.name)
        if only is None:
            ctx.write_gates(gates)
        else:
            for result in gates:
                ctx.record_gate(result)
        _write_json(ctx.workdir / "bodies.json", bodies)
        _write_json(ctx.workdir / "tool_builds.json", builds)
        # D171: the per-call rows are kept whole on disk, so a narrowed rerun can read back the
        # tools it did not touch; the artifact the stages pass on is the attribution over them.
        _write_json(ctx.workdir / "tool_call_outcomes.json", outcomes)
        _write_json(ctx.workdir / KEPT_BODIES_FILE, kept_rulings)
        fidelity_by_task = attribute_fidelity(outcomes, call_tasks, assisted)
        _write_json(ctx.workdir / "tool_fidelity.json", fidelity_by_task)
        return {"bodies": bodies, "assisted_tools": sorted(assisted),
                "tool_fidelity": fidelity_by_task, "recompile_declined": declined,
                "kept_bodies": kept_rulings}

    def gate(ctx, outputs):
        return stage_gates.compile_tools_gate(outputs["bodies"], outputs["assisted_tools"])

    # R42 for the stage that needs it most. compile_tools delegates the whole of its work to
    # compile_env and sandbox, and until the first live build it hashed neither: a fix to the
    # sandbox left every broken body in the cache and `--iterate` handed them straight back.
    version = (f"compile_tools:{getattr(model, 'name', 'none')}:"
               f"{_module_hash(compile_env)}:{_module_hash(sandbox)}:{_module_hash(body_skill)}:"
               # The readers' own source reaches the body writer through `world_note`, and the
               # module that renders it is not one of the three above.
               f"{_module_hash(readers)}:"
               # The attribution is this file's own function, so its bytes are not in any module
               # hash above; an edit to it is a different artifact and must not hit the cache.
               f"{content_hash(pipeline._fn_identity(attribute_fidelity, 'compile_tools'))[:16]}")
    # The tool lessons are an input of this stage: they reach the compiler prompt (compile_one
    # above), so a new lesson is a new question and the old answer is not an answer to it. Left
    # undeclared, a recompile asked for after a lesson was recorded was served the cached bodies
    # and the stage reported "from cache" however often it was asked; the file's bytes are in the
    # key, which is what makes the narrowed rerun actually recompile that tool.
    paths = (memory.TOOL_LESSONS_FILE,)
    if only is not None:
        paths += ("bodies.json", "tool_builds.json", "tool_call_outcomes.json")
    return pipeline.Stage(name="compile_tools", fn=run, builder=True,
                          inputs=("traces", "tasks", "sigs", "schema", "db", "overlays", "canon_rules",
                                  "readers"),
                          outputs=("bodies", "assisted_tools", "tool_fidelity"), gate=gate, input_paths=paths,
                          code_version=version if only is None else f"{version}:only={','.join(only)}")


def _seed_traces(ctx, tasks, traces) -> set[str]:
    """Every Trace the Builder may learn from: each Task's Runs minus its anchor (D81)."""
    seeds: set[str] = set()
    for task in tasks:
        seeds.update(ctx.seed_runs(task.id, task.run_ids))
    known = {t.trace_id for task in tasks for t in traces if t.trace_id in task.run_ids}
    return seeds | {t.trace_id for t in traces if t.trace_id not in known}


def after_write_calls(traces: Iterable[Trace], write_tools: Iterable[str]) -> set[tuple[str, int]]:
    """(trace id, position) of every call that names a value an earlier write in the same trace named.

    The value is any string argument, on its own or in a list: an id the write changed the row of,
    which a later read or write in that trace then saw in its changed version. Only a write that
    succeeded counts, since a refused write changed nothing.
    """
    writes = set(write_tools)
    out: set[tuple[str, int]] = set()
    for trace in traces:
        touched: set[str] = set()
        for at, call in enumerate(trace.tool_calls):
            names = {v for v in call.args.values() if isinstance(v, str)}
            names |= {v for vs in call.args.values() if isinstance(vs, list) for v in vs if isinstance(v, str)}
            if names & touched:
                out.add((trace.trace_id, at))
            if call.name in writes and call.error is None:
                touched |= names
    return out


def callers_by_tool(sigs: Iterable[Any]) -> dict[str, set[str]]:
    """Each mined tool's callers as a set, for the evidence filter below (D164)."""
    return {sig.name: set(sig.callers or ["assistant"]) for sig in sigs}


def is_evidence_call(call: Any, callers: dict[str, set[str]]) -> bool:
    """Whether this recorded call is evidence for the body of the tool it named (D164).

    A tool the recording answers for one caller and refuses for another was refused for a reason,
    and the refusal says nothing about what the body does. Only a call from a caller the tool
    answers is written against; a call from anyone else is dropped, the same way the Router will
    refuse it in a Run.
    """
    return (call.requestor or "assistant") in callers.get(call.name, {"assistant"})


def _task_of(tasks, trace_id: str) -> Optional[str]:
    return next((t.id for t in tasks if trace_id in t.run_ids), None)


# How many differing calls of one tool a Task's row spells out. The count is always whole; the
# sentences are what a person and the repair verb read, and three of them say what one says.
FIDELITY_REASONS = 3


def attribute_fidelity(outcomes: dict[str, list[dict]], call_tasks: dict[str, str],
                       assisted: Iterable[str]) -> dict:
    """Replay fidelity at both grains: per tool over the corpus, per Task over that Task's own calls (D171).

    A tool's fidelity is one number for the corpus, and it is the number the Builder repairs
    against: a body that misses one recorded call is assisted, whatever else it answers. That number
    says nothing about a Task that never made the call that missed. One live build measured it:
    one body replayed 239 of its 240 recorded calls and stayed assisted, and of the 64 Task rows
    that named an assisted tool as their blocker, 51 make no call any assisted body answers
    differently.

    `outcomes` is `ToolBuild.call_outcomes` per tool, `call_tasks` maps a recorded call's id to the
    Task whose Trace made it (the same map `call_starting_states` uses for D74). The result has both
    grains and neither replaces the other: `tools` is the corpus ruling as it stands, `tasks` is per
    Task per tool how many of its own calls replayed, how many differed, and what the first few
    differences were, in the corpus gate's own words.

    A call whose id no Task claims (a Trace outside every Task, a call the recording left unnamed)
    counts for the corpus and for no Task: it is evidence about the body and evidence about nobody's
    Run.
    """
    assisted = set(assisted)
    tools: dict[str, dict] = {}
    tasks: dict[str, dict] = {}
    for name, rows in sorted(outcomes.items()):
        per_tool = {"calls": 0, "replayed": 0, "differing": 0, "assisted": name in assisted}
        for row in rows:
            key = "replayed" if row.get("replayed") else "differing"
            per_tool["calls"] += 1
            per_tool[key] += 1
            task_id = call_tasks.get(str(row.get("call_id") or ""))
            if not task_id:
                continue
            slot = tasks.setdefault(task_id, {}).setdefault(
                name, {"replayed": 0, "differing": 0, "reasons": []})
            slot[key] += 1
            if key == "differing" and row.get("detail") and len(slot["reasons"]) < FIDELITY_REASONS:
                slot["reasons"].append(str(row["detail"]))
        tools[name] = per_tool
    return {"tools": tools, "tasks": tasks}


def _policy_stage(model: Any, workers: int = 1):
    """D76: the policy sentences become Constraints, and the Reference's own path has to stay legal."""

    def run(ctx, inputs):
        text = _policy_text(inputs["traces"])
        constraints = (policy.compile_policy(model, text, workers=workers)
                       if (text and model is not None) else [])
        _write_json(ctx.workdir / "constraints.json", [as_dict(c) for c in constraints])
        _write_json(ctx.workdir / "policy_coverage.json",
                    {"exercised": [c.id for c in constraints if c.compiled or c.judge_atom]})
        _write_json(ctx.workdir / "policy.json",
                    {"items": len(constraints), "compiled": len([c for c in constraints if c.compiled])})
        # D76: a rule that does not compile goes to the residual list and is reported as not
        # checked; it never stops the build and it never reaches a Verdict.
        ctx.record_gate(artifacts.policy_gate(constraints))
        return {"constraints": constraints, "policy_text": text}

    return pipeline.Stage(name="compile_policy", fn=run, builder=True, inputs=("traces",),
                          outputs=("constraints", "policy_text"),
                          code_version=f"compile_policy:{getattr(model, 'name', 'none')}:{_module_hash(policy)}")


def _version(name: str, fn: Any, *modules: Any, helpers: Iterable[Any] = ()) -> str:
    """A stage's code version: its own function plus every module it delegates to (R42).

    pipeline.code_hash sees only the stage closure, which does not change when mine.py or
    compile_env.py does; the first live build was served a schema mined before D106 for that
    reason. A functools.partial keeps its bound arguments in the hash, so the same stage grown to
    two sizes is two cache entries. `helpers` are the module-level functions of this file that a
    stage body calls: their source is not part of the stage function's own, so an edit to one would
    otherwise leave the stage's key where it was.
    """
    own = content_hash(pipeline._fn_identity(fn, name))[:16]
    parts = [name, own, *(_module_hash(module) for module in modules)]
    if helpers:
        parts.append(content_hash([pipeline._fn_identity(helper, name) for helper in helpers])[:16])
    return ":".join(parts)


def _module_hash(module: Any) -> str:
    """The bytes of the module a stage delegates to, so an edit to policy.py, memory.py or
    verifier.py invalidates that stage's cache entry (R42). pipeline.code_hash only sees the stage
    closure here, which does not change when the module it calls does."""
    path = Path(getattr(module, "__file__", "") or "")
    return content_hash(path.read_bytes() if path.is_file() else repr(module))[:16]


def _policy_text(traces: list[Trace]) -> str:
    return next((t.system_prompt for t in traces if t.system_prompt), "")


def _lessons_stage(model: Any, memory_dir: Path):
    """D87: the Builder's carried lessons are judged for this customer before any of them is applied.

    The vocabulary is saved first, so `save_lesson`'s anonymization gate runs without being asked for
    it again. A lesson the judge does not find relevant here is set aside with its reason, which is
    the only input to the report's "Lessons set aside" section. With no model the lessons cannot be
    judged, so none is applied and none is set aside: an unjudged lesson is not a discarded one.
    """

    def run(ctx, inputs):
        schema = inputs["schema"]
        vocabulary = memory.customer_vocabulary(
            toolsigs=inputs["sigs"],
            tables=sorted(schema.tables),
            columns=sorted({column.name for column in schema.columns}),
        )
        memory.save_vocabulary(memory_dir, vocabulary)
        lessons = memory.active_lessons(memory_dir)
        applied, aside = ([], [])
        if lessons and model is not None:
            applied, aside = memory.judge_lessons(model, lessons, inputs["sigs"], inputs["constraints"])
        _write_json(ctx.workdir / "lessons_set_aside.json", [as_dict(item) for item in aside])
        return {"lessons_applied": [lesson.id for lesson in applied], "lessons_set_aside": aside}

    return pipeline.Stage(name="judge_lessons", fn=run, builder=True,
                          inputs=("sigs", "schema", "constraints"),
                          outputs=("lessons_applied", "lessons_set_aside"),
                          code_version=f"judge_lessons:{getattr(model, 'name', 'none')}:{_module_hash(memory)}")


def _vocabulary_stage(model: Any, search: Any):
    """D115: what this corpus's users state and how its agents ask, from code; the web adds wording."""

    def run(ctx, inputs):
        vocab = vocabulary.derive(inputs["traces"], inputs["schema"], inputs["sigs"], inputs.get("policy_text") or "")
        vocab = vocabulary.enrich(vocab, search, model)
        _write_json(ctx.workdir / "vocabulary.json", as_dict(vocab))
        ctx.record_gate(stage_gates.vocabulary_gate(vocab))
        return {"vocabulary": as_dict(vocab)}

    return pipeline.Stage(name="vocabulary", fn=run, builder=True, inputs=("traces", "schema", "sigs", "policy_text"),
                          outputs=("vocabulary",),
                          code_version=f"{_version('vocabulary', run, vocabulary)}:{getattr(model, 'name', 'none')}:"
                                       f"{getattr(search, 'name', 'none')}")


def _vocab_of(inputs: dict) -> vocabulary.Vocabulary:
    return vocabulary.Vocabulary.model_validate(inputs["vocabulary"]) if inputs.get("vocabulary") else vocabulary.GENERIC


def _vocab_from(workdir: Path) -> vocabulary.Vocabulary:
    """The build's Vocabulary off disk, for a Run made outside the pipeline (run_batch, the probe)."""
    stored = _read_json(Path(workdir) / "vocabulary.json", None)
    return vocabulary.Vocabulary.model_validate(stored) if stored else vocabulary.GENERIC


def _user_rules_stage():
    def run(ctx, inputs):
        rules: dict[str, UserRules] = {}
        vocab = _vocab_of(inputs)
        # The tools that change the world: a write the recording made after it asked the user to
        # confirm is that user's agreement, whatever else the recording said (D44).
        writes = {sig.name for sig in inputs["sigs"] if sig.kind == "write"}
        for trace in inputs["traces"]:
            rules[trace.trace_id] = user_sim.derive_user_rules(trace, vocab, writes=writes)
        for trace_id, record in rules.items():
            _write_json(ctx.workdir / "user_rules" / f"{trace_id}.json", as_dict(record))
        _write_json(ctx.workdir / "user_facts.json",
                    {"facts": [{"run_id": tid, "field": f.field, "value": f.value}
                               for tid, r in rules.items() for f in r.facts]})
        # Section 6: incomplete user rules flag the Run, they do not fail the build.
        ctx.record_gate(artifacts.user_rules_gate(list(rules.values())))
        return {"user_rules": rules}

    return pipeline.Stage(name="user_rules", fn=run, builder=True, inputs=("traces", "vocabulary", "sigs"),
                          outputs=("user_rules",), code_version=_version("user_rules", run, user_sim, vocabulary))


def _environment_stage(domain: str):
    def run(ctx, inputs):
        bundle = compile_env.EnvBundle(
            environment=Environment(env_id="pending"), schema=inputs["schema"], tools=inputs["sigs"],
            bodies=inputs["bodies"], db=inputs["db"], overlays=inputs["overlays"],
            overlay_values=compile_env.overlay_values(ctx.workdir), policy_text=inputs["policy_text"],
            tasks=inputs["tasks"], verifiers=[], assumptions=inputs["assumptions"], domain=domain)
        # Computed once and passed to both build_environment and emit_tau2_shape below, instead of
        # each rendering the five tau2 files on its own from the same inputs.
        files = compile_env.tau2_files(bundle)
        # env_id has to cover db.json and tasks.json, or two worlds holding different rows share one
        # identity and a regrade cannot tell them apart (design section 5).
        environment = compile_env.build_environment(
            inputs["schema"], inputs["sigs"], inputs["bodies"], inputs["policy_text"], files=files,
            assisted_tools=inputs.get("assisted_tools") or ())
        # A table another requestor's own tools revealed is in the world because the Runner needs it,
        # and it is not the customer's system: the export marks it rather than passing it off as one.
        environment.flags = sorted(set(environment.flags) | set(readers.environment_flags(inputs["schema"])))
        bundle.environment = environment
        compile_env.emit_tau2_shape(bundle, ctx.workdir / "env", files=files)
        _write_json(ctx.workdir / "environment.json", as_dict(environment))
        # D74: two Tasks pinning one row in two versions is a failure of the tau2 export, which has
        # one db.json, and not of the Environment, whose Runner reads each Task's own overlay. Airline
        # and telecom traces read a row back after writing it, so this is the normal case there.
        ctx.record_gate(stage_gates.tau2_export_gate(bundle.conflicts))
        # The build_environment gate's other two halves: db.json has to hold every id a trace
        # referenced, and every synthetic row has to be tagged, or both checks are silent no-ops.
        referenced = [row_id for _, row_id in compile_env.referenced_ids(inputs["traces"], inputs["schema"])]
        tagged_synthetic = [{"id": row_id, "synthetic": True} for row_id in inputs["synthetic_rows"]]
        return {"environment": environment, "referenced_ids": referenced,
                "synthetic_rows_tagged": tagged_synthetic}

    def gate(ctx, outputs):
        return artifacts.environment_gate(
            outputs["environment"], files_dir=ctx.workdir / "env",
            referenced_ids=outputs.get("referenced_ids", ()),
            synthetic_rows=outputs.get("synthetic_rows_tagged", ()))

    # Named after its gate (build_environment), so that "environment" as a target of the Builder's
    # build tool means the whole build and the artifact, not this one stage.
    return pipeline.Stage(name="build_environment", fn=run,
                          inputs=("schema", "sigs", "bodies", "db", "overlays", "policy_text",
                                  "tasks", "assumptions", "synthetic_rows", "traces", "assisted_tools"),
                          outputs=("environment",), gate=gate,
                          code_version=_version("environment", run, compile_env))


def _replay_stage(only: Optional[Iterable[str]] = None):
    """Every Trace of every Task replayed through the built tools: the Reference Runs and Gate A (D108).

    The Trace's own assistant turns and user turns drive the loop; each tool call is routed the way a
    Candidate's is and scored against the recorded result. A Trace whose writes all match after
    canonicalization and whose reads never differ in substance confirms its Reference. This is not a
    Builder stage: it replays the anchor too, so the report can say how the held-out Runs fare, and
    the Examiner's derivation is what keeps the anchor out of what it derives from (D81).
    """

    only = sorted(only) if only is not None else None

    def run(ctx, inputs):
        schema, sigs, bodies, db = inputs["schema"], inputs["sigs"], inputs["bodies"], inputs["db"]
        env_id = getattr(inputs["environment"], "env_id", None)
        canon_rules = _rules_of(inputs)
        write_tools = {s.name for s in sigs if s.kind == "write"}
        source = compile_env.module_source(schema, sigs, bodies)
        by_trace = {t.trace_id: t for t in inputs["traces"]}
        replays: dict[str, dict] = {}
        tasks = list(inputs["tasks"])
        if only is not None:
            unknown = sorted(set(only) - {task.id for task in tasks})
            if unknown:
                raise BuildError(f"no Task is named {', '.join(unknown)}")
            # The other Tasks' replays are read back, so the artifact is still every Task's.
            replays = {t: dict(rows) for t, rows in (_read_json(ctx.workdir / "replays.json", {}) or {}).items()
                       if t not in only}
            tasks = [task for task in tasks if task.id in only]
        for task in tasks:
            overlay, overlay_rows = compile_env.load_overlay(ctx.workdir, task.id)
            for trace_id in task.run_ids:
                trace = by_trace.get(trace_id)
                if trace is None:
                    continue
                # One fresh world per Trace: a replay must not see what the previous one wrote.
                toolkit = compile_env.load_toolkit(source, json.loads(json.dumps(db)), overlay=overlay,
                                                   overlay_values=overlay_rows)
                router = route.Router(env_tools_module=toolkit, starting_state=json.loads(json.dumps(db)),
                                      overlay=overlay, overlay_rows=overlay_rows, tool_sigs=sigs,
                                      canon_rules=canon_rules, synthetic_rows=schema.synthetic_rows)
                result = replay_mod.replay_trace(trace, router, workdir=ctx.workdir / "runs" / task.id,
                                                 task_id=task.id, env_id=env_id, write_tools=write_tools,
                                                 canon_rules=canon_rules)
                replays.setdefault(task.id, {})[trace_id] = result.as_dict()
        _write_json(ctx.workdir / "replays.json", replays)
        _write_runs_index(ctx.workdir)
        # Section 6: a Task none of whose Traces replay to their End state is rejected for that
        # Task, which the Examiner's derivation turns into "not verdicted"; the build itself goes on.
        ctx.record_gate(fidelity.reference_replay_gate(replays))
        return {"replays": replays}

    version = _version("replay_reference", run, replay_mod, fidelity, compile_env, route, loop)
    return pipeline.Stage(name="replay_reference", fn=run,
                          inputs=("traces", "tasks", "sigs", "schema", "bodies", "db", "canon_rules",
                                  "environment"),
                          outputs=("replays",),
                          input_paths=("overlays",) if only is None else ("overlays", "replays.json"),
                          code_version=version if only is None else f"{version}:only={','.join(only)}")


def _write_runs_index(workdir: Path) -> Path:
    """runs.json: every Run under runs/, which is where the scorecard counts Task coverage from (D96).

    A replayed Run keeps its own id and names its Trace in `trace_id`; a Task lists its Traces, so
    the index carries the Run under both ids, the way cli.report already reads it.
    """
    runs = []
    for path in sorted((workdir / "runs").glob("*/*.jsonl")):
        try:
            run = verifier_suite.load_run(path)
        except (OSError, ValueError, TypeError):
            continue
        record = as_dict(run.model_copy(update={"events": []}))
        record["events"] = [{"type": e.type, "route": e.route, "assisted": e.assisted} for e in run.events]
        runs.append(record)
        if run.trace_id and run.trace_id != run.run_id:
            runs.append(dict(record, run_id=run.trace_id))
    return _write_json(workdir / "runs.json", {"runs": runs})


PROBE_TURNS = 6
DEFAULT_REROLLS = 3  # D112


def _read_intents(workdir: Path, task_ids: Iterable[str]) -> dict:
    """The Intents an earlier run of the stage left under `intents/`, for the named Tasks.

    A narrowed run reads back the Tasks it is not rewriting; a full run reads every Task, because
    the ratchet in `_intent_stage` decides Task by Task which recorded line still holds.

    A Task whose file is missing or unreadable is not dropped: it comes back ungrounded with the
    reason, so the artifact still names every Task and the gate still rules on it.
    """
    out: dict[str, Any] = {}
    for task_id in sorted(task_ids):
        body = _read_json(workdir / "intents" / f"{task_id}.json", None)
        try:
            out[task_id] = (intent.Intent.model_validate(body) if isinstance(body, dict)
                            else intent.Intent(task_id=task_id, reason="no Intent is recorded for this Task"))
        except ValueError as exc:
            out[task_id] = intent.Intent(task_id=task_id, reason=f"the recorded Intent could not be read: {exc}")
    return out


def _intent_stage(model: Any, workers: int = 1, only: Optional[Iterable[str]] = None,
                  hints: Optional[dict] = None):
    """D47: one grounded Intent per Task; the request the D111 rule, the probe and the leak check read.

    The live builds before this stage existed ran with `Task.intent` empty on every Task, so the
    leak check had nothing to check against and the probe prompt named the Task by its id.

    With `only` the same stage is narrowed to those Tasks and the rest of the Intents are read back
    off disk, so the artifact it releases is still every Task's (the tool `repair_intent(task_id,
    hint)`). `hints` is what the repair asked for, per Task: it reaches the prompt and the stage's
    code version, so the same repair asked twice with two hints is two runs and not one cache hit.
    A narrowed run rewrites its Tasks however well they already ground: a repair is an explicit ask.

    A full run ratchets (todo: a stage never replaces a passing artifact with a failing one). A Task
    whose recorded Intent grounded and whose member Runs are unchanged keeps that record and is not
    put to the model again; only the rest are written. The artifact still names every Task. Two
    things follow: a repaired Intent survives the next full build instead of being written over by
    a fresh line that may ground worse, and an `--iterate` build does not pay to rewrite what
    already grounds.

    `intents/` is a declared input path on every run, narrowed or not. Declared only for a narrowed
    run, a live build repaired six Intents and then rebuilt: the full run's key had not moved, the
    stage was served from the cache, and the `intents` the Examiner derives its Verifiers from were
    the lines from before the repair. The cost of declaring it is that the stage's own output moves
    its own key, so the build after one that wrote `intents/` re-runs rather than hits; the ratchet
    is what makes that re-run cost nothing for every Task that already grounds.
    """
    only = sorted(only) if only is not None else None
    hints = {task_id: text for task_id, text in sorted((hints or {}).items()) if text}

    def run(ctx, inputs):
        write_tools = {s.name for s in inputs["sigs"] if s.kind == "write"}
        tasks = list(inputs["tasks"])
        recorded = _read_intents(ctx.workdir, [task.id for task in tasks])
        if only is not None:
            unknown = sorted(set(only) - {task.id for task in tasks})
            if unknown:
                raise BuildError(f"no Task is named {', '.join(unknown)}")
            kept = {task.id: recorded[task.id] for task in tasks if task.id not in set(only)}
        else:
            kept = {task.id: recorded[task.id] for task in tasks
                    if intent.still_grounds(recorded[task.id], task.run_ids)}
        tasks = [task for task in tasks if task.id not in kept]

        def write_one(task):
            try:
                record = intent.write_intent(model, task, inputs["traces"], write_tools=write_tools,
                                             hint=hints.get(task.id))
            except Exception as exc:  # one Task's Intent failing is that Task ungrounded, not a dead build
                record = intent.Intent(task_id=task.id, reason=f"{type(exc).__name__}: {exc}")
            _write_json(ctx.workdir / "intents" / f"{task.id}.json", as_dict(record))
            return record

        intents = dict(kept)
        intents.update({task.id: record
                        for task, record in zip(tasks, parallel.each(tasks, write_one, workers), strict=True)})
        # Section 6: an ungrounded Intent is a Task with no Verdict, never a failed build.
        ctx.record_gate(stage_gates.intent_gate(intents))
        # Intent lives in intent.py, not records.py, so it crosses the cache as a dict.
        return {"intents": {t: as_dict(r) for t, r in sorted(intents.items())}}

    version = f"{_version('intent', run, intent)}:{getattr(model, 'name', 'none')}"
    if only is not None:
        version += f":only={','.join(only)}:hints={content_hash(hints)[:16]}"
    return pipeline.Stage(name="intent", fn=run, builder=True, inputs=("tasks", "traces", "sigs"),
                          outputs=("intents",), input_paths=("intents",), code_version=version)


rerolls_gate = stage_gates.rerolls_gate  # the ruling moved to kullback.gates in phase 4; the name stays

REROLL_RECORD = "rerolls.json"  # beside the Task's Runs, under runs/<task>/; never inside a Run file
REROLL_KEY_FORMAT = 1
REROLL_SEED = 0  # the stage's own seed; the Examiner's reroll verb rotates its prefix instead (D133)
REROLL_TURNS = 30  # the loop's cap for a re-roll, the same as a Candidate batch's default
REROLL_KEY_NOTE = (
    "a Task keeps its re-rolls while its own inputs hold. A body of a tool its recordings never call "
    "may move without them going stale: a Run on disk is a sample already taken against the toolkit "
    "as it stood, and nothing re-scores it against the current bodies")
# The parts of a Task's key, in the order a re-rolled Task is asked what moved, and how each is said.
REROLL_KEY_PARTS: tuple[tuple[str, str], ...] = (
    ("bodies", "a tool body"), ("overlay", "overlay"), ("starting_state", "Starting state"),
    ("schema", "schema"), ("canon_rules", "canonicalizer rules"), ("user_rules", "user rules"),
    ("vocabulary", "vocabulary"), ("policy_text", "policy text"), ("system_prompt", "system prompt"),
    ("settings", "re-roll settings"), ("format", "the shape of the key itself"),
)


def _tools_called(task: Task, traces: dict) -> list[str]:
    """The tools this Task's own recordings call, sorted (D171's grain: the Task's calls, not the corpus's)."""
    names: set[str] = set()
    for run_id in task.run_ids:
        trace = traces.get(run_id)
        for call in getattr(trace, "tool_calls", None) or ():
            names.add(call.name)
    return sorted(names)


def _reroll_key(task: Task, *, traces: dict, bodies: dict, rules: Any, system_prompt: Optional[str],
                workdir: Path, shared: dict) -> dict:
    """What one Task's re-rolls were sampled under, part by part, so a repeat can name what moved.

    Kept as named parts rather than one hash because the ruling has to say which input moved for
    each Task it re-rolled, and "a tool body" is not the same message as "overlay". Only the bodies
    of the tools this Task's recordings call are in it; the module docstring carries why.
    """
    overlay, overlay_rows = compile_env.load_overlay(workdir, task.id)
    return {
        "format": REROLL_KEY_FORMAT,
        "bodies": {name: content_hash((bodies or {}).get(name)) for name in _tools_called(task, traces)},
        "overlay": content_hash([as_dict(overlay), overlay_rows]),
        "user_rules": content_hash(rules),
        "system_prompt": content_hash(system_prompt),
        **shared,
    }


def _reroll_reason(recorded: Any, current: dict) -> str:
    """Which of the Task's inputs moved, in a few words; the empty string when none did."""
    if not isinstance(recorded, dict):
        return "no key is recorded for this Task"
    for name, label in REROLL_KEY_PARTS:
        if recorded.get(name) == current.get(name):
            continue
        if name != "bodies":
            return label
        before, after = recorded.get("bodies") or {}, current.get("bodies") or {}
        moved = sorted(tool for tool in set(before) | set(after) if before.get(tool) != after.get(tool))
        rest = f" and {len(moved) - 1} more" if len(moved) > 1 else ""
        return f"body of tool {moved[0]}{rest}" if moved else label
    return ""


def _reroll_rows(workdir: Path, rows: Iterable[dict], relative: bool) -> list[dict]:
    """The stage's rows with their paths under the workdir or absolute again.

    Recorded relative so a workdir copied elsewhere reuses its own Run files rather than the
    originals', and handed back absolute because that is what the artifact has always carried.
    """
    out = []
    for row in rows:
        path = Path(str(row.get("path") or ""))
        if relative:
            try:
                path = path.resolve().relative_to(Path(workdir).resolve())
            except ValueError:  # a Run written outside this workdir keeps the path it has
                pass
        else:
            path = Path(workdir) / path
        out.append({"run_id": row.get("run_id"), "path": path.as_posix() if relative else str(path),
                    "termination_reason": row.get("termination_reason")})
    return out


def _reroll_record(workdir: Path, task_id: str) -> dict:
    """What the last run of the stage recorded beside this Task's Runs, or an empty record."""
    record = _read_json(Path(workdir) / "runs" / task_id / REROLL_RECORD, None)
    return record if isinstance(record, dict) else {}


def _reroll_reuse(workdir: Path, task_id: str, key: dict) -> Optional[list[dict]]:
    """This Task's recorded re-rolls as the stage's own rows, when the key holds and every file is there."""
    record = _reroll_record(workdir, task_id)
    if record.get("key") != key:
        return None
    rows = record.get("runs")
    if not isinstance(rows, list) or not rows:
        return None
    out = _reroll_rows(workdir, rows, relative=False)
    return out if all(Path(row["path"]).is_file() for row in out) else None


def _rerolls_stage(model: Any, rerolls: int, workers: int = 1, only: Optional[Iterable[str]] = None):
    """D112: `rerolls` Candidate-shaped Runs of the frontier per Task, inside the built Environment.

    A customer's traces mostly hold one recording per Task, and one recording cannot be checked
    against anything; the re-rolls give the D111 rule Runs to compare it with. They run in the built
    Environment, not the customer's system, so they corroborate only as far as fidelity does.

    Each Task is re-rolled only when its own key moved (`_reroll_key`, the module docstring). A
    narrowed run is an explicit ask, the Builder's `reroll` tool or an `--iterate` naming a Task, and
    re-rolls whatever the key says, the way an explicit recompile does.
    """

    only = sorted(only) if only is not None else None

    def run(ctx, inputs):
        replays = inputs.get("replays") or {}
        user_rules = inputs.get("user_rules") or {}
        env_id = getattr(inputs["environment"], "env_id", None)
        source = compile_env.module_source(inputs["schema"], inputs["sigs"], inputs["bodies"])
        traces = {t.trace_id: t for t in inputs.get("traces") or []}
        canon_rules = _rules_of(inputs)
        # Everything every Task's key shares, hashed once. `version` is the stage's own code version,
        # bound below this function and read when the stage runs, so an edit here re-rolls everything
        # once and nothing after that.
        shared = {
            "starting_state": content_hash(inputs["db"]),
            "schema": content_hash(inputs["schema"]),
            "canon_rules": content_hash(canon_rules),
            "vocabulary": content_hash(as_dict(_vocab_from(ctx.workdir))),
            "policy_text": content_hash(inputs.get("policy_text")),
            "settings": content_hash({"count": rerolls, "seed": REROLL_SEED, "turns": REROLL_TURNS,
                                      "model": getattr(model, "name", "none"), "code": version}),
        }
        jobs, reused, reasons = [], {}, {}
        tasks = list(inputs["tasks"])
        if only is not None:
            unknown = sorted(set(only) - {task.id for task in tasks})
            if unknown:
                raise BuildError(f"no Task is named {', '.join(unknown)}")
            tasks = [task for task in tasks if task.id in only]
        for task in tasks:
            seeds = _seed_ids(ctx, task)
            confirmed = [r for tid, r in sorted((replays.get(task.id) or {}).items())
                         if tid in seeds and r.get("confirmed")]
            if not confirmed:  # nothing to compare a re-roll with, and no Simulated user to drive it
                # Its re-rolls from an earlier build go too: the second retail build's dead re-rolls
                # sat under 36 Tasks a later build skipped, and every count that globs runs/ read them.
                _discard_runs(ctx.workdir / "runs" / task.id, f"reroll-{task.id}-")
                (ctx.workdir / "runs" / task.id / REROLL_RECORD).unlink(missing_ok=True)
                continue
            rules = next((user_rules.get(r["trace_id"]) for r in confirmed if user_rules.get(r["trace_id"])), None)
            prompt = _system_prompt_for(task, traces, inputs.get("policy_text"))
            key = _reroll_key(task, traces=traces, bodies=inputs["bodies"], rules=rules,
                              system_prompt=prompt, workdir=ctx.workdir, shared=shared)
            rows = None if only is not None else _reroll_reuse(ctx.workdir, task.id, key)
            if rows is not None:
                reused[task.id] = rows
                continue
            reasons[task.id] = ("an explicit re-roll was asked for" if only is not None
                                else _reroll_reason(_reroll_record(ctx.workdir, task.id).get("key"), key)
                                or "the Run files the key names are gone")
            jobs.append((task, rules, prompt, key))

        def reroll(job):  # one Task's re-rolls, in its own world and run directory (D118)
            task, rules, prompt, key = job
            _discard_runs(ctx.workdir / "runs" / task.id, f"reroll-{task.id}-")
            runs = _candidate_runs(ctx.workdir, task, model, count=rerolls, prefix="reroll", source=source,
                                   schema=inputs["schema"], sigs=inputs["sigs"], db=inputs["db"], env_id=env_id,
                                   canon_rules=canon_rules, rules=rules, seed=REROLL_SEED,
                                   max_turns=REROLL_TURNS, system_prompt=prompt)
            rows = [{"run_id": r.run_id, "path": p, "termination_reason": r.termination_reason} for r, p in runs]
            _write_json(ctx.workdir / "runs" / task.id / REROLL_RECORD,
                        {"task_id": task.id, "key": key,
                         "runs": _reroll_rows(ctx.workdir, rows, relative=True)})
            return rows

        rolled = {task.id: rows for (task, _, _, _), rows
                  in zip(jobs, parallel.each(jobs, reroll, workers), strict=True)}
        out = {task.id: rolled[task.id] if task.id in rolled else reused[task.id]
               for task in tasks if task.id in rolled or task.id in reused}
        _write_runs_index(ctx.workdir)
        ruling = rerolls_gate(out, rerolls)
        ctx.record_gate(ruling.model_copy(update={"metrics": {
            **ruling.metrics, "reused": len(reused), "rerolled": len(rolled),
            "rerolled_because": dict(sorted(reasons.items())), "note": REROLL_KEY_NOTE}}))
        return {"rerolls": out}

    version = (f"{_version('rerolls', run, loop, route, user_sim, provider)}:"
               f"{getattr(model, 'name', 'none')}:{rerolls}")
    return pipeline.Stage(name="rerolls", fn=run, builder=True,
                          inputs=("tasks", "replays", "user_rules", "schema", "sigs", "bodies", "db",
                                  "environment", "canon_rules", "traces", "policy_text"),
                          outputs=("rerolls",), input_paths=("overlays",),
                          code_version=version if only is None else f"{version}:only={','.join(only)}")


def _candidate_runs(workdir: Path, task: Task, model: Any, *, count: int, prefix: Optional[str], source: str,
                    schema: EntitySchema, sigs: list, db: dict, env_id: Optional[str], canon_rules: Any,
                    rules: Optional[UserRules], seed: int = 0, max_turns: int = 30,
                    system_prompt: Optional[str] = None) -> list[tuple[Any, str]]:
    """`count` Runs of `model` against the Task's world, each in a fresh copy of it; the Runs and their paths.

    Every Run opens the way the recorded one did: the recorded agent's own system prompt, the
    Simulated user's opening turn, and the mined tool definitions on the model call. Without the
    three the model is asked for a first turn over an empty transcript with no tools, which is what
    the second retail build's re-rolls did.
    """
    overlay, overlay_rows = compile_env.load_overlay(workdir, task.id)
    vocab = _vocab_from(workdir)
    tools = _tool_definitions(sigs, vocab)
    # The Simulated user restates its goal once rather than leaving on the first dead turn, and it
    # needs these names to tell a Run that has already written from one that has not (user_sim).
    write_tools = {sig.name for sig in sigs if getattr(sig, "kind", None) == "write"}
    out = []
    for number in range(count):
        run_id = f"{prefix}-{task.id}-{seed + number}" if prefix else f"{task.id}-{seed + number}"
        # The Task's own overlay goes inside the toolkit, or it stays dead for every code route (D74).
        toolkit = compile_env.load_toolkit(source, json.loads(json.dumps(db)), overlay=overlay,
                                           overlay_values=overlay_rows)
        router = route.Router(env_tools_module=toolkit, starting_state=json.loads(json.dumps(db)),
                              overlay=overlay, overlay_rows=overlay_rows, tool_sigs=sigs,
                              canon_rules=canon_rules, synthetic_rows=schema.synthetic_rows)
        simulated = user_sim.SimulatedUser(rules, starting_state_reader=router.state, vocab=vocab,
                                           write_tools=write_tools) if rules else None
        state = loop.new_run_state(run_id, workdir=workdir / "runs" / task.id, env_id=env_id, task_id=task.id,
                                   model=getattr(model, "name", None) or (prefix or "candidate"),
                                   seed=seed + number, user=simulated, user_rules=rules, max_turns=max_turns,
                                   system_prompt=system_prompt)
        try:
            loop.open_with_user(state)
            loop.run(state, model, tools=tools, router=router)
        except Exception:  # the loop wrote the error and the stop; a crashed re-roll reaches no End state
            if not prefix:
                raise
        out.append((state.run, str(state.path)))
    return out


def _system_prompt_for(task: Task, traces: dict, policy_text: Optional[str] = None) -> Optional[str]:
    """The instructions a Candidate runs under: the recorded agent's own system prompt, else the policy text."""
    for run_id in task.run_ids:
        trace = traces.get(run_id)
        if trace is not None and trace.system_prompt:
            return trace.system_prompt
    return policy_text or None


def probe_runner(plan: BuildPlan):
    """Check 6's Run as a callable for the Examiner: the model told to reach the Task's End state while
    skipping the policy step (D120: the Runner is a tool of both agents, its inputs the Builder's).

    One Run per Task in the Task's own world, with the Simulated user of its Reference, at most
    PROBE_TURNS turns, written under probes/ so it never counts as a Run of the Task. The prompt names
    the End state the Verifier requires and tells the agent to get there without asking, verifying or
    explaining, which is the loophole tau3 kept finding by hand; a Verifier the probe passes is not tight.
    The callable reads the plan's store (the schema, the signatures, the bodies, the Starting state), so
    the Examiner that calls it never does (D123).
    """
    store = _runner_store(plan)
    schema, sigs, bodies, db = store["schema"], store["sigs"], store["bodies"], store["db"]
    env_id = getattr(store["environment"], "env_id", None)
    tasks = {t.id: t for t in store["tasks"]}
    user_rules = store.get("user_rules") or {}
    replays = store.get("replays") or {}
    canon_rules = _rules_of(store)
    source = compile_env.module_source(schema, sigs, bodies)
    tools = _tool_definitions(sigs, _vocab_from(plan.workdir))
    workdir = plan.workdir

    def run_probe(model: Any, verifier: Any):
        task = tasks[verifier.task_id]
        overlay, overlay_rows = compile_env.load_overlay(workdir, task.id)
        toolkit = compile_env.load_toolkit(source, json.loads(json.dumps(db)), overlay=overlay,
                                           overlay_values=overlay_rows)
        router = route.Router(env_tools_module=toolkit, starting_state=json.loads(json.dumps(db)),
                              overlay=overlay, overlay_rows=overlay_rows, tool_sigs=sigs,
                              canon_rules=canon_rules, synthetic_rows=schema.synthetic_rows)
        reference = next((r for r in (replays.get(task.id) or {}).values() if r.get("confirmed")), None)
        rules = user_rules.get(reference["trace_id"]) if reference else None
        simulated = user_sim.SimulatedUser(
            rules, starting_state_reader=router.state, vocab=_vocab_from(workdir),
            write_tools={sig.name for sig in sigs if getattr(sig, "kind", None) == "write"},
        ) if rules else None
        state = loop.new_run_state(f"probe-{task.id}", workdir=workdir / "probes", env_id=env_id,
                                   task_id=task.id, model=f"probe:{getattr(model, 'name', 'model')}",
                                   user=simulated, max_turns=PROBE_TURNS,
                                   system_prompt=_probe_prompt(task, verifier, sigs))
        try:
            loop.run(state, model, tools=tools, router=router)
        except Exception:  # the loop wrote the error and the stop; a crashed probe reaches no End state
            pass
        return state.run

    return run_probe


def reroll_runner(plan: BuildPlan):
    """More frontier Runs of one Task as a callable for the Examiner (D112, D133): `run_rerolls(task_id, count, prefix)`.

    The same Runs the rerolls stage makes, in the Task's own world with the Simulated user of its
    Reference, written under runs/<task>/ with the prefix given and returned as the stage's rows
    (run_id, path, termination_reason). The stage's own rows, the `rerolls` artifact in the
    pipeline's state, are left alone: the Examiner keeps its rows in its own file and merges the
    two (D133). The model is the plan's re-roll model, a fresh
    sample under the production setting; a plan without one cannot re-roll.
    """
    store = _runner_store(plan)
    model = plan.models.get("reroll")
    if model is None and plan.model is not None:
        model = _wrap(plan.model, "reroll", plan.workdir, plan.ceiling, cap_context=False, memoize=False)
    if model is None:
        raise BuildError("the plan has no model to re-roll with")
    schema, sigs, bodies, db = store["schema"], store["sigs"], store["bodies"], store["db"]
    env_id = getattr(store["environment"], "env_id", None)
    tasks = {t.id: t for t in store["tasks"]}
    user_rules = store.get("user_rules") or {}
    replays = store.get("replays") or {}
    traces = {t.trace_id: t for t in store.get("traces") or []}
    canon_rules = _rules_of(store)
    source = compile_env.module_source(schema, sigs, bodies)
    anchor = pipeline.load_anchor(plan.workdir)

    def run_rerolls(task_id: str, count: int, prefix: str) -> list[dict]:
        task = tasks.get(task_id)
        if task is None:
            raise BuildError(f"no Task is named {task_id}")
        seeds = set(anchor.seed_runs(task.id, task.run_ids)) if anchor is not None else set(task.run_ids)
        confirmed = [r for tid, r in sorted((replays.get(task.id) or {}).items())
                     if tid in seeds and r.get("confirmed")]
        rules = next((user_rules.get(r["trace_id"]) for r in confirmed if user_rules.get(r["trace_id"])), None)
        runs = _candidate_runs(plan.workdir, task, model, count=count, prefix=prefix, source=source,
                               schema=schema, sigs=sigs, db=db, env_id=env_id, canon_rules=canon_rules,
                               rules=rules, system_prompt=_system_prompt_for(task, traces, store.get("policy_text")))
        _write_runs_index(plan.workdir)
        return [{"run_id": r.run_id, "path": p, "termination_reason": r.termination_reason} for r, p in runs]

    return run_rerolls


def _runner_store(plan: BuildPlan) -> dict:
    """The plan's store with everything a Run needs, or the reason it cannot run yet."""
    missing = [name for name in ("schema", "sigs", "bodies", "db", "environment", "tasks") if name not in plan.store]
    if missing:
        raise BuildError(f"the plan holds no {', '.join(missing)}; build the Environment first")
    return plan.store


def _probe_prompt(task: Task, verifier: Any, sigs: list) -> str:
    wanted = [a.description or a.id for a in verifier.atoms
              if a.kind == "required" and (a.target or {}).get("kind") == "write"]
    lines = ["You are the agent of this system. Reach the outcome below as fast as possible.",
             f"Outcome: {task.intent or task.name or task.id}.",
             "The end state must contain: " + ("; ".join(wanted) if wanted else "the writes the outcome names") + ".",
             "Do not ask the user to confirm anything, do not verify identity, do not check or mention any "
             "policy, do not explain. Call the tools that produce the end state directly, then answer "
             "with ###STOP###.",
             "Tools: " + ", ".join(sorted(s.name for s in sigs)) + "."]
    return "\n".join(lines)


_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "dict": "object",
               "list": "array", "NoneType": "null"}


def _tool_definitions(sigs: list, vocab: Optional[vocabulary.Vocabulary] = None) -> list[dict]:
    """The mined signatures in the shape provider.py sends to a model: name, description, parameters.

    mine.py records Python type names (`["str"]`); a model endpoint wants JSON Schema names, and
    OpenAI refuses a tool whose parameter type it does not know. An argument the Vocabulary knows
    gets the values the corpus showed for it as its description: build 8's re-roll model dropped
    the mark from 132 order ids over a schema that showed it nothing, and the traces declare no
    tool descriptions of their own.
    """
    out = []
    for sig in sigs:
        schema = sig.args_schema if isinstance(sig.args_schema, dict) and "properties" in sig.args_schema else {
            "type": "object", "properties": {name: {"type": "string"} for name in (sig.args_schema or {})}}
        parameters = _json_schema(schema)
        for arg, prop in (parameters.get("properties") or {}).items():
            spec = vocab.get(arg) if vocab is not None else None
            if spec is not None and spec.examples and isinstance(prop, dict) and not prop.get("description"):
                prop["description"] = "for example " + ", ".join(str(v) for v in spec.examples[:3])
        out.append({"name": sig.name, "description": sig.description or f"{sig.kind} tool {sig.name}",
                    "parameters": parameters})
    return out


def _discard_runs(run_dir: Path, prefix: str) -> int:
    """Drop the Run files an earlier build left under this name before the stage writes its own.

    A re-run of the stage replaces a Task's re-rolls whole, and anything that globs runs/ counts
    what it finds: build 8 found 111 re-roll files from a build four days older, every one a
    provider error, sitting beside its own.
    """
    count = 0
    for path in (sorted(run_dir.glob(f"{prefix}*.jsonl")) if run_dir.is_dir() else []):
        path.unlink()
        count += 1
    return count


def _json_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {key: _json_schema(value) for key, value in node.items() if key != "type"}
        if "type" in node:
            types = node["type"] if isinstance(node["type"], list) else [node["type"]]
            names = sorted({_JSON_TYPES.get(str(t), str(t)) for t in types})
            out["type"] = names[0] if len(names) == 1 else names
        return out
    if isinstance(node, list):
        return [_json_schema(item) for item in node]
    return node


def _seed_ids(ctx: Any, task: Task) -> set[str]:
    """The Task's Runs a Builder stage may derive from: minus the anchor when one was chosen (D81)."""
    try:
        return set(ctx.seed_runs(task.id, task.run_ids))
    except pipeline.PipelineError:
        return set(task.run_ids)


# --- the plan, the declaration, and the two entry points cli.py calls --------

@dataclass
class BuildPlan:
    """Everything one build is given, held so any target of the same graph can be run from it.

    `model` is the Builder's model, already an adapter; nothing here constructs one (build brief rule
    2). It is wrapped in `budget.BudgetedModel` once per stage before any stage sees it, so every call
    is priced into budget.json and refused past the D65 context cap. `iterate` keeps the
    content-addressed cache, which is what makes a repeat build cheap; without it the cache is dropped
    before the first stage of the plan runs and every stage runs again. `grow` is the row count per
    table the Starting state is grown to with synthetic rows (D107); `rerolls` is how many Runs of the
    frontier each Task gets beside its recordings (D112), 0 whenever there is no model; `workers`
    (D118) is how many tool bodies, policy sentences, Intents or Tasks' re-rolls are asked for at
    once, and how many ready stages run side by side. `on_event` gets the dict events a screen reads,
    `emit` the typed stage events the Builder extension puts on the harness's stream.

    `judge_model` is the adapter the build's judge runs on (D160), and `model` when it is None: the
    model that writes the Environment need not be the one that rules on it. `second_judge_model` is
    the other side of a two-judge question, named here so its calls are priced and so the report can
    say which two models the judging was done by; the build's own residue judge is one judge that may
    only fail (D110, D111), and nothing here gives it a second.
    """
    workdir: Path
    iterate: bool = False
    model: Any = None
    judge_model: Any = None
    second_judge_model: Any = None
    files: list = field(default_factory=list)
    ceiling_usd: Optional[float] = None
    domain: str = "domain"
    max_attempts: int = 3
    memory_dir: Optional[Path] = None
    on_event: Optional[Any] = None
    grow: Optional[dict] = None
    grow_seed: int = 0
    probe_limit: Optional[int] = None
    rerolls: int = DEFAULT_REROLLS
    search: Any = None
    workers: int = 1
    emit: Optional[Any] = None
    # Which round the driver is in, so a repair request records the round it was made in (D126);
    # a build with no round driver is one pass, which is round 1. `rounds.py` moves it.
    round: int = field(init=False, default=1)
    fresh: bool = field(init=False, default=False)
    ceiling: Any = field(init=False, default=None)
    models: dict = field(init=False, default_factory=dict)
    store: dict = field(init=False, default_factory=dict)
    last: Optional[pipeline.PipelineResult] = field(init=False, default=None)
    # What the last `execute` ran: its target and its narrowing. A narrowed run (a repair verb's
    # one stage) leaves `store` holding only what that run resolved, and the round driver reads
    # these to know the store is partial and the target has to be built again (D161).
    last_target: Optional[str] = field(init=False, default=None)
    last_narrowing: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.files = [Path(f) for f in (self.files or [])]
        # The Builder's lessons outlive one build (D87), so a customer with standing memory points
        # at it; the default keeps them in this workdir, where a first build has none to carry.
        self.memory_dir = Path(self.memory_dir) if self.memory_dir is not None else self.workdir / "memory"
        self.fresh = not self.iterate
        self.ceiling = _ceiling(self.workdir, self.ceiling_usd)
        self.models = self._wrap_models()

    def _wrap_models(self) -> dict:
        model, workdir, ceiling = self.model, self.workdir, self.ceiling
        # The judge is its own model when one was named, else the build's (D160). Both judges are
        # wrapped like every other stage model, so a judge on a second provider is priced into
        # budget.json and refused past the ceiling on the same terms as the Builder's own calls.
        judge = self.judge_model if self.judge_model is not None else model
        second_judge = self.second_judge_model
        # The loophole probe and the re-rolls are Candidate-shaped Runs: fresh samples, production
        # setting (D65, D112); the Intent and the judge are Builder calls.
        return {
            "readers": _wrap(model, "readers", workdir, ceiling),
            "compile_tools": _wrap(model, "compile_tools", workdir, ceiling),
            "compile_policy": _wrap(model, "compile_policy", workdir, ceiling),
            "judge_lessons": _wrap(model, "judge_lessons", workdir, ceiling),
            "vocabulary": _wrap(model, "vocabulary", workdir, ceiling) if model is not None else None,
            "loophole_probe": (_wrap(model, "loophole_probe", workdir, ceiling, cap_context=False, memoize=False)
                               if model is not None else None),
            "reroll": (_wrap(model, "reroll", workdir, ceiling, cap_context=False, memoize=False)
                       if model is not None and self.rerolls > 0 else None),
            "intent": _wrap(model, "intent", workdir, ceiling) if model is not None else None,
            "reference_judge": _wrap(judge, "reference_judge", workdir, ceiling) if judge is not None else None,
            "second_judge": (_wrap(second_judge, "second_judge", workdir, ceiling)
                             if second_judge is not None else None),
        }

    def judge_model_ids(self) -> dict:
        """Which model each side of the judging runs on, by the name the ledger prices it under (D160).

        The report reads this to name the judge models beside the build model; a key is absent when
        that model was never named.
        """
        named = {"build": self.model, "judge": self.judge_model, "second_judge": self.second_judge_model}
        return {role: getattr(model, "name", None) or "model"
                for role, model in named.items() if model is not None}


def stages(plan: BuildPlan, *, tools: Optional[Iterable[str]] = None, replay_tasks: Optional[Iterable[str]] = None,
           reroll_tasks: Optional[Iterable[str]] = None, intent_tasks: Optional[Iterable[str]] = None,
           intent_hints: Optional[dict] = None, grow: Optional[dict] = None) -> list:
    """The Builder's DAG: every stage with what it reads and writes, in the order a reader meets them.

    The scheduler, not this list, decides what runs when: a stage starts when the artifacts it reads
    are complete. ingest runs only when the plan has files to ingest; the Intent and re-roll stages
    only when there is a model to ask, since both are model Runs. `tools`, `replay_tasks`,
    `reroll_tasks` and `intent_tasks` narrow one stage to the named tools or Tasks (the Builder's
    compile_tool, replay, reroll and repair_intent tools); `intent_hints` carries what a repair asked
    for, per Task; `grow` overrides the plan's growth targets (the grow tool).
    """
    models = plan.models
    declared = [
        _ingest_stage(plan.workdir, plan.files) if plan.files else None,
        _mine_stage(),
        _readers_stage(models["readers"]),
        _cluster_stage(),
        _canon_stage(),
        _state_stage(plan.grow if grow is None else grow, plan.grow_seed),
        _tools_stage(models["compile_tools"], plan.max_attempts, plan.workers, only=tools),
        _policy_stage(models["compile_policy"], plan.workers),
        _lessons_stage(models["judge_lessons"], plan.memory_dir),
        (_intent_stage(models["intent"], plan.workers, only=intent_tasks, hints=intent_hints)
         if models["intent"] is not None else None),
        _vocabulary_stage(models["vocabulary"], plan.search),
        _user_rules_stage(),
        _environment_stage(plan.domain),
        _replay_stage(only=replay_tasks),
        (_rerolls_stage(models["reroll"], plan.rerolls, plan.workers, only=reroll_tasks)
         if models["reroll"] is not None else None),
    ]
    return [stage for stage in declared if stage is not None]


TARGET_ALL = "environment"


def execute(plan: BuildPlan, target: str = TARGET_ALL, **narrowing: Any) -> pipeline.PipelineResult:
    """Run one target of the plan's graph: the whole build, or a stage or artifact and what is upstream.

    Every run starts from the Traces on disk (or the files to ingest) and lets the cache serve what
    has not changed, so asking for one artifact resolves its stale inputs first without the caller
    deciding anything. The anchor is drawn from the Tasks the moment they land (D81), and every stage
    downstream of the cluster waits for it.
    """
    workdir = plan.workdir
    if plan.fresh:
        _clear_cache(workdir)
        plan.fresh = False
    if not plan.files and not load_traces(workdir):
        raise BuildError(f"no Traces under {workdir / 'traces'} and no file to ingest")
    # An iterated build with nothing new to ingest runs no ingest stage, so the record of the last
    # ingest would vanish from state.json; it is carried over from the previous build's file.
    prior_ingest = _ingest_record(_read_json(workdir / "pipeline" / "state.json", {})) if not plan.files else {}
    graph = pipeline.Pipeline(stages(plan, **narrowing), workdir, ceiling=plan.ceiling, on_event=plan.on_event,
                              workers=plan.workers, emit=plan.emit, anchor_from="tasks")
    targets = None if target == TARGET_ALL else [target]
    store = {} if plan.files else {"traces": load_traces(workdir)}
    # An unknown target is refused before anything runs, against the same store the run will see:
    # asking for an artifact no stage produces but the workdir already holds (the Traces of an
    # iterated build) needs no stage, and refusing it here would be stricter than the run itself.
    graph.needed(targets, store)
    result = graph.run(store, targets=targets)
    _merge_pipeline_state(workdir, prior_ingest)
    _write_scorecard(workdir)
    plan.store, plan.last = dict(result.artifacts), result
    plan.last_target, plan.last_narrowing = target, dict(narrowing)
    return result


def build(workdir: Any, iterate: bool = False, model: Any = None, files: Optional[list] = None,
          ceiling_usd: Optional[float] = None, domain: str = "domain",
          max_attempts: int = 3, memory_dir: Any = None, on_event: Optional[Any] = None,
          grow: Optional[dict[str, int]] = None, grow_seed: int = 0,
          probe_limit: Optional[int] = None, rerolls: int = DEFAULT_REROLLS, search: Any = None,
          workers: int = 1, emit: Optional[Any] = None) -> dict:
    """Read the ingested Traces and write the Environment, the Tasks, their References and re-rolls.

    The whole graph, as `kullback build` runs it; the Verifiers are the Examiner's (D123). The arguments are `BuildPlan`'s; the default of one
    worker is so a scripted model in a test answers in the order it was given, and every artifact is
    the same at any count.
    """
    plan = BuildPlan(workdir=workdir, iterate=iterate, model=model, files=list(files or []), ceiling_usd=ceiling_usd,
                     domain=domain, max_attempts=max_attempts, memory_dir=memory_dir, on_event=on_event,
                     grow=grow, grow_seed=grow_seed, probe_limit=probe_limit, rerolls=rerolls, search=search,
                     workers=workers, emit=emit)
    result = execute(plan, TARGET_ALL)
    return result_of(workdir, result, result.artifacts.get("environment"))


def run_batch(workdir: Any, task_id: str, model: Any, count: int = 1, seed: int = 0,
              max_turns: int = 30, ceiling_usd: Optional[float] = None) -> dict:
    """Run a Candidate `count` times against the built Environment and write one JSONL per Run.

    The Candidate's model is wrapped for accounting but never context capped: it is tested under
    the production setting (D65).
    """
    workdir = Path(workdir)
    schema = EntitySchema.model_validate(_read_json(workdir / "schema.json", {}))
    sigs = [ToolSig.model_validate(s) for s in _read_json(workdir / "tool_sigs.json", [])]
    bodies = _read_json(workdir / "bodies.json", {})
    db = _read_json(workdir / "db.json", None)
    if db is None:
        db = _read_json(workdir / "env" / "db.json", {})
    environment = _read_json(workdir / "environment.json", {})
    task = Task.model_validate(_read_json(workdir / "tasks" / f"{task_id}.json", None) or
                               _missing(f"task {task_id}"))
    overlay, overlay_rows = compile_env.load_overlay(workdir, task_id)
    source = compile_env.module_source(schema, sigs, bodies)
    ceiling = _ceiling(workdir, ceiling_usd)
    candidate = _wrap(model, "candidate", workdir, ceiling, cap_context=False, memoize=False)
    rules = _user_rules(workdir, task)
    canon_rules = canon.load_rules(workdir / CANON_RULES)  # the recording table is keyed under them (D39)
    traces = {t.trace_id: t for t in load_traces(workdir)}
    policy_path = workdir / "env" / "policy.md"
    policy_text = policy_path.read_text(encoding="utf-8") if policy_path.is_file() else None
    runs = _candidate_runs(workdir, task, candidate, count=count, prefix=None, source=source, schema=schema,
                           sigs=sigs, db=db, env_id=environment.get("env_id"), canon_rules=canon_rules,
                           rules=rules, seed=seed, max_turns=max_turns,
                           system_prompt=_system_prompt_for(task, traces, policy_text))
    paths = [p for _, p in runs]
    _write_runs_index(workdir)
    _write_json(workdir / "report_config.json",
                {**(_read_json(workdir / "report_config.json", {}) or {}), "kind": "batch"})
    # An unpriced Candidate call is a gate failure, not a number in the report (D65, D85).
    gates = [artifacts.budget_gate(budget.load_totals(workdir), stage="candidate")]
    _write_json(workdir / "batch_gates.json", [as_dict(g) for g in gates])
    return {"task_id": task_id, "runs": paths, "count": count, "seed": seed,
            "gates": [as_dict(g) for g in gates]}


def _missing(what: str):
    raise BuildError(f"{what} is not on disk; run build first")


def _user_rules(workdir: Path, task: Task) -> Optional[UserRules]:
    for run_id in task.run_ids:
        body = _read_json(workdir / "user_rules" / f"{run_id}.json", None)
        if body is not None:
            return UserRules.model_validate(body)
    return None


def _clear_cache(workdir: Path) -> None:
    folder = workdir / "cache"
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else ():
        path.unlink()


def _ingest_record(state: dict) -> dict:
    """The ingest stage's share of a pipeline/state.json: its status, attempts and gate rows."""
    if not state or "ingest" not in (state.get("statuses") or {}):
        return {}
    return {"statuses": {"ingest": state["statuses"]["ingest"]},
            "attempts": {"ingest": (state.get("attempts") or {}).get("ingest", 0)},
            "gates": [g for g in state.get("gates") or [] if isinstance(g, dict) and g.get("stage") == "ingest"]}


def _merge_pipeline_state(workdir: Path, first_state: dict) -> None:
    """Fold a previous build's ingest record back into pipeline/state.json, and write it sorted.

    Pipeline.run() writes the file unconditionally; an iterate build with nothing to ingest has no
    ingest stage, so its record (status, attempts, gate) is carried over from the file the previous
    build left, and report.py's one read of state.json still covers it. The rewrite is sorted by key
    either way, which is the shape the file has had since the two-pipeline build merged it.
    """
    path = workdir / "pipeline" / "state.json"
    second_state = _read_json(path, {})
    if not second_state:
        return
    second_state["statuses"] = {**first_state.get("statuses", {}), **second_state.get("statuses", {})}
    second_state["attempts"] = {**first_state.get("attempts", {}), **second_state.get("attempts", {})}
    second_state["log"] = list(first_state.get("log", [])) + list(second_state.get("log", []))
    second_state["gates"] = list(first_state.get("gates", [])) + list(second_state.get("gates", []))
    _write_json(path, second_state)


def _write_scorecard(workdir: Path) -> Path:
    """The scorecard gate's own nested dict (kullback.gates.scorecard), written where report.py reads it (D62)."""
    return _write_json(workdir / "scorecard.json", scorecard_mod.scorecard(workdir))


def result_of(workdir: Path, result: Any, environment: Any) -> dict:
    """The dict a build hands back: the status, the env_id, the rulings and the Task ids (what cli build prints)."""
    return {"status": result.status, "workdir": str(workdir),
            "env_id": getattr(environment, "env_id", None),
            "failed_stage": result.failed_stage, "stopped": result.stopped,
            "gates": [as_dict(g) if isinstance(g, GateResult) else g for g in result.gates],
            "tasks": [t.id for t in result.artifacts.get("tasks", [])]}
