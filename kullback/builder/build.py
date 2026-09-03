"""The one declaration of the Builder's stages as a DAG over named artifacts, the plan that runs any target of it,
and the Candidate Run batch that follows a build.

`stages(plan)` is the whole graph: fourteen stages, each naming what it reads and writes, handed to
`pipeline.Pipeline`, which decides the order (D120: the scheduler decides, never a model). `build()`
is the CLI's entry and runs every stage; `execute(plan, target)` runs one stage or artifact and what
is upstream of it, which is what the Builder's tools (builder/tools.py) call. A tool that wants one
tool body recompiled or one Task replayed gets a variant of the same declaration with that stage
narrowed, so its cache key and its gates are the stage's own.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.ai import provider
from kullback.builder import (
    cluster,
    compile_env,
    ingest,
    intent,
    memory,
    mine,
    parallel,
    pipeline,
    policy,
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
        schema = mine.mine_schema(traces)
        _write_json(ctx.workdir / "tool_sigs.json", [as_dict(s) for s in sigs])
        _write_json(ctx.workdir / "schema.json", as_dict(schema))  # cli._score reads it (D39, D73)
        calls = [c for t in traces for c in t.tool_calls]
        # "flag, do not synthesize": a tool the corpus barely shows stays in the build, named in
        # the gate, rather than being invented or dropped (design section 6).
        ctx.record_gate(artifacts.mine_gate(sigs, calls))
        return {"sigs": sigs, "schema": schema}

    return pipeline.Stage(name="mine", fn=run, inputs=("traces",), outputs=("sigs", "schema"),
                          code_version=_version("mine", run, mine))


def _cluster_stage():
    def run(ctx, inputs):
        # D74: two Runs that saw one row in two versions before writing started in different
        # worlds, and a Task's overlay can pin only one, so they are different Tasks.
        worlds = compile_env.trace_worlds(inputs["traces"], inputs["schema"],
                                          cluster.write_tool_names(inputs["sigs"]))
        categories, tasks = cluster.cluster_runs(inputs["traces"], inputs["sigs"], worlds=worlds)
        for task in tasks:
            _write_json(ctx.workdir / "tasks" / f"{task.id}.json", as_dict(task))
        _write_json(ctx.workdir / "tasks.json", {"tasks": [as_dict(t) for t in tasks]})
        # D96: the coverage denominator is frozen once, here, before anything measures coverage.
        scorecard_mod.freeze_tasks(ctx.workdir, tasks)
        ctx.record_gate(stage_gates.cluster_gate(tasks, categories))
        return {"categories": categories, "tasks": tasks}

    return pipeline.Stage(name="cluster", fn=run, inputs=("traces", "sigs", "schema"),
                          outputs=("categories", "tasks"),
                          code_version=_version("cluster", run, cluster, intent, compile_env))


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
                                                 grow_seed=grow_seed)
        # The synthetic ids live on the schema (D40); run_batch reads them back from schema.json.
        _write_json(ctx.workdir / "schema.json", as_dict(inputs["schema"]))
        return {"db": state.db, "overlays": list(state.overlays),
                "assumptions": list(state.assumptions), "synthetic_rows": list(state.synthetic_rows)}

    # A partial, so the grow targets are in the stage's cache key: the same traces grown to two
    # sizes are two Starting states, not one served twice (pipeline._fn_identity).
    fn = functools.partial(run, grow=dict(grow or {}), grow_seed=grow_seed)
    return pipeline.Stage(name="starting_state", fn=fn, inputs=("traces", "schema", "tasks", "sigs"),
                          outputs=("db", "overlays", "assumptions", "synthetic_rows"),
                          code_version=_version("starting_state", fn, compile_env, synth))


def _tools_stage(model: Any, max_attempts: int, workers: int = 1, only: Optional[Iterable[str]] = None):
    """compile_tools, or with `only` the same stage narrowed to those tools: the rest of the bodies
    are read back from bodies.json, so the artifact it releases is still every body (the tool
    `compile_tool(name)`)."""
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
        skipped: dict[str, int] = {}
        for trace in traces:
            task_id = _task_of(tasks, trace.trace_id)
            for at, call in enumerate(trace.tool_calls):
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
        bodies, gates, assisted, builds = {}, [], [], {}
        rules = _rules_of(inputs)
        sigs = list(inputs["sigs"])
        if only is not None:
            unknown = sorted(set(only) - {sig.name for sig in sigs})
            if unknown:
                raise BuildError(f"no mined tool is named {', '.join(unknown)}")
            bodies = dict(_read_json(ctx.workdir / "bodies.json", {}) or {})
            builds = dict(_read_json(ctx.workdir / "tool_builds.json", {}) or {})
            assisted = [name for name, row in builds.items() if row.get("assisted") and name not in only]
            sigs = [sig for sig in sigs if sig.name in only]

        def compile_one(sig):  # one tool, its own directory and nodes; independent of every other (D118)
            return compile_env.compile_tool(model, sig, calls_by_tool.get(sig.name, []),
                                            inputs["schema"], inputs["db"],
                                            ctx.workdir / "tools" / sig.name,
                                            max_attempts=max_attempts, call_states=states,
                                            rules=rules, tool_names=tool_names,
                                            error_prefix=error_prefix)

        for sig, build in zip(sigs, parallel.each(sigs, compile_one, workers), strict=True):
            bodies[sig.name] = build.body
            gates.extend(build.gates)
            builds[sig.name] = {"assisted": build.assisted, "nodes": build.nodes,
                                "after_write_skipped": skipped.get(sig.name, 0)}
            if build.assisted:
                assisted.append(sig.name)
        if only is None:
            ctx.write_gates(gates)
        else:
            for result in gates:
                ctx.record_gate(result)
        _write_json(ctx.workdir / "bodies.json", bodies)
        _write_json(ctx.workdir / "tool_builds.json", builds)
        return {"bodies": bodies, "assisted_tools": sorted(assisted)}

    def gate(ctx, outputs):
        return stage_gates.compile_tools_gate(outputs["bodies"], outputs["assisted_tools"])

    # R42 for the stage that needs it most. compile_tools delegates the whole of its work to
    # compile_env and sandbox, and until the first live build it hashed neither: a fix to the
    # sandbox left every broken body in the cache and `--iterate` handed them straight back.
    version = (f"compile_tools:{getattr(model, 'name', 'none')}:"
               f"{_module_hash(compile_env)}:{_module_hash(sandbox)}")
    return pipeline.Stage(name="compile_tools", fn=run, builder=True,
                          inputs=("traces", "tasks", "sigs", "schema", "db", "overlays", "canon_rules"),
                          outputs=("bodies", "assisted_tools"), gate=gate,
                          input_paths=("bodies.json", "tool_builds.json") if only is not None else (),
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


def _task_of(tasks, trace_id: str) -> Optional[str]:
    return next((t.id for t in tasks if trace_id in t.run_ids), None)


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
        for trace in inputs["traces"]:
            rules[trace.trace_id] = user_sim.derive_user_rules(trace, vocab)
        for trace_id, record in rules.items():
            _write_json(ctx.workdir / "user_rules" / f"{trace_id}.json", as_dict(record))
        _write_json(ctx.workdir / "user_facts.json",
                    {"facts": [{"run_id": tid, "field": f.field, "value": f.value}
                               for tid, r in rules.items() for f in r.facts]})
        # Section 6: incomplete user rules flag the Run, they do not fail the build.
        ctx.record_gate(artifacts.user_rules_gate(list(rules.values())))
        return {"user_rules": rules}

    return pipeline.Stage(name="user_rules", fn=run, builder=True, inputs=("traces", "vocabulary"),
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


def _intent_stage(model: Any, workers: int = 1):
    """D47: one grounded Intent per Task; the request the D111 rule, the probe and the leak check read.

    The live builds before this stage existed ran with `Task.intent` empty on every Task, so the
    leak check had nothing to check against and the probe prompt named the Task by its id.
    """

    def run(ctx, inputs):
        write_tools = {s.name for s in inputs["sigs"] if s.kind == "write"}

        def write_one(task):
            try:
                record = intent.write_intent(model, task, inputs["traces"], write_tools=write_tools)
            except Exception as exc:  # one Task's Intent failing is that Task ungrounded, not a dead build
                record = intent.Intent(task_id=task.id, reason=f"{type(exc).__name__}: {exc}")
            _write_json(ctx.workdir / "intents" / f"{task.id}.json", as_dict(record))
            return record

        intents = {task.id: record
                   for task, record in zip(inputs["tasks"], parallel.each(inputs["tasks"], write_one, workers), strict=True)}
        # Section 6: an ungrounded Intent is a Task with no Verdict, never a failed build.
        ctx.record_gate(stage_gates.intent_gate(intents))
        # Intent lives in intent.py, not records.py, so it crosses the cache as a dict.
        return {"intents": {t: as_dict(r) for t, r in intents.items()}}

    return pipeline.Stage(name="intent", fn=run, builder=True, inputs=("tasks", "traces", "sigs"),
                          outputs=("intents",),
                          code_version=f"{_version('intent', run, intent)}:{getattr(model, 'name', 'none')}")


rerolls_gate = stage_gates.rerolls_gate  # the ruling moved to kullback.gates in phase 4; the name stays


def _rerolls_stage(model: Any, rerolls: int, workers: int = 1, only: Optional[Iterable[str]] = None):
    """D112: `rerolls` Candidate-shaped Runs of the frontier per Task, inside the built Environment.

    A customer's traces mostly hold one recording per Task, and one recording cannot be checked
    against anything; the re-rolls give the D111 rule Runs to compare it with. They run in the built
    Environment, not the customer's system, so they corroborate only as far as fidelity does.
    """

    only = sorted(only) if only is not None else None

    def run(ctx, inputs):
        replays = inputs.get("replays") or {}
        user_rules = inputs.get("user_rules") or {}
        env_id = getattr(inputs["environment"], "env_id", None)
        source = compile_env.module_source(inputs["schema"], inputs["sigs"], inputs["bodies"])
        traces = {t.trace_id: t for t in inputs.get("traces") or []}
        canon_rules = _rules_of(inputs)
        jobs = []
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
            if not confirmed:
                continue  # nothing to compare a re-roll with, and no Simulated user to drive it
            rules = next((user_rules.get(r["trace_id"]) for r in confirmed if user_rules.get(r["trace_id"])), None)
            jobs.append((task, rules))

        def reroll(job):  # one Task's re-rolls, in its own world and run directory (D118)
            task, rules = job
            runs = _candidate_runs(ctx.workdir, task, model, count=rerolls, prefix="reroll", source=source,
                                   schema=inputs["schema"], sigs=inputs["sigs"], db=inputs["db"], env_id=env_id,
                                   canon_rules=canon_rules, rules=rules,
                                   system_prompt=_system_prompt_for(task, traces, inputs.get("policy_text")))
            return [{"run_id": r.run_id, "path": p, "termination_reason": r.termination_reason} for r, p in runs]

        out = {task.id: rows for (task, _), rows in zip(jobs, parallel.each(jobs, reroll, workers), strict=True)}
        _write_runs_index(ctx.workdir)
        ctx.record_gate(rerolls_gate(out, rerolls))
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
    tools = _tool_definitions(sigs)
    out = []
    vocab = _vocab_from(workdir)
    for number in range(count):
        run_id = f"{prefix}-{task.id}-{seed + number}" if prefix else f"{task.id}-{seed + number}"
        # The Task's own overlay goes inside the toolkit, or it stays dead for every code route (D74).
        toolkit = compile_env.load_toolkit(source, json.loads(json.dumps(db)), overlay=overlay,
                                           overlay_values=overlay_rows)
        router = route.Router(env_tools_module=toolkit, starting_state=json.loads(json.dumps(db)),
                              overlay=overlay, overlay_rows=overlay_rows, tool_sigs=sigs,
                              canon_rules=canon_rules, synthetic_rows=schema.synthetic_rows)
        simulated = user_sim.SimulatedUser(rules, starting_state_reader=router.state, vocab=vocab) if rules else None
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
    tools = _tool_definitions(sigs)
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
        simulated = user_sim.SimulatedUser(rules, starting_state_reader=router.state,
                                           vocab=_vocab_from(workdir)) if rules else None
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


def _tool_definitions(sigs: list) -> list[dict]:
    """The mined signatures in the shape provider.py sends to a model: name, description, parameters.

    mine.py records Python type names (`["str"]`); a model endpoint wants JSON Schema names, and
    OpenAI refuses a tool whose parameter type it does not know.
    """
    out = []
    for sig in sigs:
        schema = sig.args_schema if isinstance(sig.args_schema, dict) and "properties" in sig.args_schema else {
            "type": "object", "properties": {name: {"type": "string"} for name in (sig.args_schema or {})}}
        out.append({"name": sig.name, "description": sig.description or f"{sig.kind} tool {sig.name}",
                    "parameters": _json_schema(schema)})
    return out


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
    """
    workdir: Path
    iterate: bool = False
    model: Any = None
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
    fresh: bool = field(init=False, default=False)
    ceiling: Any = field(init=False, default=None)
    models: dict = field(init=False, default_factory=dict)
    store: dict = field(init=False, default_factory=dict)
    last: Optional[pipeline.PipelineResult] = field(init=False, default=None)

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
        # The loophole probe and the re-rolls are Candidate-shaped Runs: fresh samples, production
        # setting (D65, D112); the Intent and the judge are Builder calls.
        return {
            "compile_tools": _wrap(model, "compile_tools", workdir, ceiling),
            "compile_policy": _wrap(model, "compile_policy", workdir, ceiling),
            "judge_lessons": _wrap(model, "judge_lessons", workdir, ceiling),
            "vocabulary": _wrap(model, "vocabulary", workdir, ceiling) if model is not None else None,
            "loophole_probe": (_wrap(model, "loophole_probe", workdir, ceiling, cap_context=False, memoize=False)
                               if model is not None else None),
            "reroll": (_wrap(model, "reroll", workdir, ceiling, cap_context=False, memoize=False)
                       if model is not None and self.rerolls > 0 else None),
            "intent": _wrap(model, "intent", workdir, ceiling) if model is not None else None,
            "reference_judge": _wrap(model, "reference_judge", workdir, ceiling) if model is not None else None,
        }


def stages(plan: BuildPlan, *, tools: Optional[Iterable[str]] = None, replay_tasks: Optional[Iterable[str]] = None,
           reroll_tasks: Optional[Iterable[str]] = None, grow: Optional[dict] = None) -> list:
    """The Builder's DAG: every stage with what it reads and writes, in the order a reader meets them.

    The scheduler, not this list, decides what runs when: a stage starts when the artifacts it reads
    are complete. ingest runs only when the plan has files to ingest; the Intent and re-roll stages
    only when there is a model to ask, since both are model Runs. `tools`, `replay_tasks` and
    `reroll_tasks` narrow one stage to the named tools or Tasks (the Builder's compile_tool, replay
    and reroll tools); `grow` overrides the plan's growth targets (the grow tool).
    """
    models = plan.models
    declared = [
        _ingest_stage(plan.workdir, plan.files) if plan.files else None,
        _mine_stage(),
        _cluster_stage(),
        _canon_stage(),
        _state_stage(plan.grow if grow is None else grow, plan.grow_seed),
        _tools_stage(models["compile_tools"], plan.max_attempts, plan.workers, only=tools),
        _policy_stage(models["compile_policy"], plan.workers),
        _lessons_stage(models["judge_lessons"], plan.memory_dir),
        _intent_stage(models["intent"], plan.workers) if models["intent"] is not None else None,
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
