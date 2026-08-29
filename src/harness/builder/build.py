"""The one wiring of the Builder stages over the Runner's Pipeline, and the Candidate Run batch that follows it."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from harness.builder import (
    cluster,
    compile_env,
    ingest,
    intent,
    memory,
    mine,
    policy,
    sandbox,
    synth,
    user_sim,
    vocabulary,
)
from harness.builder import reference as reference_mod
from harness.builder import verifier as verifier_mod
from harness.runner import loop, pipeline, route, validate
from harness.runner import replay as replay_mod
from harness.shared import budget, canon, parallel, provider
from harness.shared.records import (
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

# This module lives under builder/ and not under runner/pipeline.py on purpose: assembling the
# stages means naming ingest, mine, cluster, compile_env, policy, user_sim and verifier, and the
# Runner never imports the Builder (design section 3, build brief rule 7, D89). pipeline.py stays
# the stage runner; this is the graph it runs.

CANON_RULES = "canon-rules.json"
DEFAULT_CEILING_USD = 25.0


class BuildError(RuntimeError):
    """The build cannot start: no traces, no Task, no Environment on disk."""


# --- small file helpers -----------------------------------------------------

def _write_json(path: Path, body: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def record_gate(workdir: Path, result: GateResult) -> GateResult:
    """Append one gate to gates.json, which is where report.py reads the build's gates.

    Design section 6 gives every stage its own answer to a failed gate, and only "build
    Environment" says "fail build". A gate whose failure means "flag, do not synthesize" or
    "Task not verdicted" is therefore recorded here and does not roll the artifact back into
    its stage; the Pipeline's rollback edge is for the gates that do stop a build.
    """
    body = _read_json(workdir / "gates.json", []) or []
    body = [g for g in body if g.get("stage") != result.stage] + [as_dict(result)]
    _write_json(workdir / "gates.json", body)
    return result


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
        return validate.ingest_gate(outputs["traces"])

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
        record_gate(ctx.workdir, validate.mine_gate(sigs, calls))
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
        validate.freeze_tasks(ctx.workdir, tasks)
        empty = [t.id for t in tasks if not t.run_ids]
        record_gate(ctx.workdir, validate.gate(
            "cluster", [f"task {i} holds no Run" for i in empty],
            tasks=len(tasks), categories=len(categories)))
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


def _tools_stage(model: Any, max_attempts: int, workers: int = 1):
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

        def compile_one(sig):  # one tool, its own directory and nodes; independent of every other (D118)
            return compile_env.compile_tool(model, sig, calls_by_tool.get(sig.name, []),
                                            inputs["schema"], inputs["db"],
                                            ctx.workdir / "tools" / sig.name,
                                            max_attempts=max_attempts, call_states=states,
                                            rules=rules, tool_names=tool_names,
                                            error_prefix=error_prefix)

        for sig, build in zip(inputs["sigs"], parallel.each(inputs["sigs"], compile_one, workers), strict=True):
            bodies[sig.name] = build.body
            gates.extend(build.gates)
            builds[sig.name] = {"assisted": build.assisted, "nodes": build.nodes,
                                "after_write_skipped": skipped.get(sig.name, 0)}
            if build.assisted:
                assisted.append(sig.name)
        _write_json(ctx.workdir / "gates.json", [as_dict(g) for g in gates])
        _write_json(ctx.workdir / "bodies.json", bodies)
        _write_json(ctx.workdir / "tool_builds.json", builds)
        return {"bodies": bodies, "assisted_tools": sorted(assisted)}

    def gate(ctx, outputs):
        missing = sorted(name for name, body in outputs["bodies"].items() if not (body or "").strip())
        return validate.gate("compile_tools", [f"{name} has no body" for name in missing],
                             tools=len(outputs["bodies"]), assisted=len(outputs["assisted_tools"]))

    return pipeline.Stage(name="compile_tools", fn=run, builder=True,
                          inputs=("traces", "tasks", "sigs", "schema", "db", "overlays", "canon_rules"),
                          outputs=("bodies", "assisted_tools"), gate=gate,
                          # R42 for the stage that needs it most. compile_tools delegates the whole
                          # of its work to compile_env and sandbox, and until the first live build
                          # it hashed neither: a fix to the sandbox left every broken body in the
                          # cache and `--iterate` handed them straight back.
                          code_version=f"compile_tools:{getattr(model, 'name', 'none')}:"
                                       f"{_module_hash(compile_env)}:{_module_hash(sandbox)}")


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
        record_gate(ctx.workdir, validate.policy_gate(constraints))
        return {"constraints": constraints, "policy_text": text}

    return pipeline.Stage(name="compile_policy", fn=run, builder=True, inputs=("traces",),
                          outputs=("constraints", "policy_text"),
                          code_version=f"compile_policy:{getattr(model, 'name', 'none')}:{_module_hash(policy)}")


def _version(name: str, fn: Any, *modules: Any) -> str:
    """A stage's code version: its own function plus every module it delegates to (R42).

    pipeline.code_hash sees only the stage closure, which does not change when mine.py or
    compile_env.py does; the first live build was served a schema mined before D106 for that
    reason. A functools.partial keeps its bound arguments in the hash, so the same stage grown to
    two sizes is two cache entries.
    """
    own = content_hash(pipeline._fn_identity(fn, name))[:16]
    return ":".join([name, own, *(_module_hash(module) for module in modules)])


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
        derived = [f for f in vocab.fields if "generic" not in f.sources]
        record_gate(ctx.workdir, validate.gate(
            "vocabulary", [], fields=len(vocab.fields), derived=len(derived), searched=len(vocab.searched),
            web_aliases=sum(len(r.get("aliases") or []) for r in vocab.searched), notes=len(vocab.notes)))
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
        record_gate(ctx.workdir, validate.user_rules_gate(list(rules.values())))
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
        record_gate(ctx.workdir, validate.gate("tau2_export", list(bundle.conflicts),
                                               overlay_conflicts=len(bundle.conflicts)))
        # The build_environment gate's other two halves: db.json has to hold every id a trace
        # referenced, and every synthetic row has to be tagged, or both checks are silent no-ops.
        referenced = [row_id for _, row_id in compile_env.referenced_ids(inputs["traces"], inputs["schema"])]
        tagged_synthetic = [{"id": row_id, "synthetic": True} for row_id in inputs["synthetic_rows"]]
        return {"environment": environment, "referenced_ids": referenced,
                "synthetic_rows_tagged": tagged_synthetic}

    def gate(ctx, outputs):
        return validate.environment_gate(
            outputs["environment"], files_dir=ctx.workdir / "env",
            referenced_ids=outputs.get("referenced_ids", ()),
            synthetic_rows=outputs.get("synthetic_rows_tagged", ()))

    return pipeline.Stage(name="environment", fn=run,
                          inputs=("schema", "sigs", "bodies", "db", "overlays", "policy_text",
                                  "tasks", "assumptions", "synthetic_rows", "traces", "assisted_tools"),
                          outputs=("environment",), gate=gate,
                          code_version=_version("environment", run, compile_env))


def _rules_of(inputs: dict) -> canon.CanonRules:
    """The CanonRules the canon stage learned from this customer's own corpus (D39).

    Read off the pipeline rather than off disk: the rules are learned inside this same run, so a
    value read before the pipeline started would be the module defaults on every first build.
    """
    return canon.CanonRules.model_validate(inputs.get("canon_rules") or {})


def _replay_stage():
    """Every Trace of every Task replayed through the built tools: the Reference Runs and Gate A (D108).

    The Trace's own assistant turns and user turns drive the loop; each tool call is routed the way a
    Candidate's is and scored against the recorded result. A Trace whose writes all match after
    canonicalization and whose reads never differ in substance confirms its Reference. This is not a
    Builder stage: it replays the anchor too, so the report can say how the held-out Runs fare, and
    the Verifier stage is what keeps the anchor out of what it derives from (D81).
    """

    def run(ctx, inputs):
        schema, sigs, bodies, db = inputs["schema"], inputs["sigs"], inputs["bodies"], inputs["db"]
        env_id = getattr(inputs["environment"], "env_id", None)
        canon_rules = _rules_of(inputs)
        write_tools = {s.name for s in sigs if s.kind == "write"}
        source = compile_env.module_source(schema, sigs, bodies)
        by_trace = {t.trace_id: t for t in inputs["traces"]}
        replays: dict[str, dict] = {}
        for task in inputs["tasks"]:
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
        summary = replay_mod.summarize(replays)
        # Section 6: a Task none of whose Traces replay to their End state is rejected for that
        # Task, which the Verifier stage turns into "not verdicted"; the build itself goes on.
        failures = [f"task {task_id}: {replay_mod.unconfirmed_reason(per_task)}"
                    for task_id, per_task in sorted(replays.items())
                    if not any(r["confirmed"] for r in per_task.values())]
        record_gate(ctx.workdir, validate.gate("replay_reference", failures, **summary))
        return {"replays": replays}

    return pipeline.Stage(name="replay_reference", fn=run,
                          inputs=("traces", "tasks", "sigs", "schema", "bodies", "db", "canon_rules",
                                  "environment"),
                          outputs=("replays",), input_paths=("overlays",),
                          code_version=_version("replay_reference", run, replay_mod, compile_env, route, loop))


def _write_runs_index(workdir: Path) -> Path:
    """runs.json: every Run under runs/, which is where the scorecard counts Task coverage from (D96).

    A replayed Run keeps its own id and names its Trace in `trace_id`; a Task lists its Traces, so
    the index carries the Run under both ids, the way cli.report already reads it.
    """
    runs = []
    for path in sorted((workdir / "runs").glob("*/*.jsonl")):
        try:
            run = verifier_mod.load_run(path)
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
        record_gate(ctx.workdir, validate.gate(
            "intent", [f"task {t}: {r.reason}" for t, r in sorted(intents.items()) if not r.grounded],
            tasks=len(intents), grounded=sum(1 for r in intents.values() if r.grounded)))
        # Intent lives in intent.py, not records.py, so it crosses the cache as a dict.
        return {"intents": {t: as_dict(r) for t, r in intents.items()}}

    return pipeline.Stage(name="intent", fn=run, builder=True, inputs=("tasks", "traces", "sigs"),
                          outputs=("intents",),
                          code_version=f"{_version('intent', run, intent)}:{getattr(model, 'name', 'none')}")


def rerolls_gate(rerolls: dict[str, list[dict]], per_task: int) -> GateResult:
    """Green only when some re-roll finished: the third retail build's 597 re-rolls all stopped on a
    provider error and the gate passed over them, the same green-over-nothing the scorecard gate had."""
    total = sum(len(rows) for rows in rerolls.values())
    finished = sum(1 for rows in rerolls.values() for r in rows
                   if (r.get("termination_reason") or "") in verifier_mod.SUCCESS_TERMINATIONS)
    failures = ([f"no re-roll finished: {total} Runs and every one stopped on an error; a Run the frontier "
                 f"cannot complete says nothing about the Task"] if total and not finished else [])
    return validate.gate("rerolls", failures, tasks=len(rerolls), per_task=per_task, runs=total, finished=finished)


def _rerolls_stage(model: Any, rerolls: int, workers: int = 1):
    """D112: `rerolls` Candidate-shaped Runs of the frontier per Task, inside the built Environment.

    A customer's traces mostly hold one recording per Task, and one recording cannot be checked
    against anything; the re-rolls give the D111 rule Runs to compare it with. They run in the built
    Environment, not the customer's system, so they corroborate only as far as fidelity does.
    """

    def run(ctx, inputs):
        replays = inputs.get("replays") or {}
        user_rules = inputs.get("user_rules") or {}
        env_id = getattr(inputs["environment"], "env_id", None)
        source = compile_env.module_source(inputs["schema"], inputs["sigs"], inputs["bodies"])
        traces = {t.trace_id: t for t in inputs.get("traces") or []}
        canon_rules = _rules_of(inputs)
        jobs = []
        for task in inputs["tasks"]:
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
        record_gate(ctx.workdir, rerolls_gate(out, rerolls))
        return {"rerolls": out}

    return pipeline.Stage(name="rerolls", fn=run, builder=True,
                          inputs=("tasks", "replays", "user_rules", "schema", "sigs", "bodies", "db",
                                  "environment", "canon_rules", "traces", "policy_text"),
                          outputs=("rerolls",), input_paths=("overlays",),
                          code_version=f"{_version('rerolls', run, loop, route, user_sim, provider)}:"
                                       f"{getattr(model, 'name', 'none')}:{rerolls}")


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


def _request_text(task: Task, intents: dict, traces: dict) -> str:
    """What the user asked, for the judge: the grounded Intent, else the Task's name, else the first user turn."""
    record = intents.get(task.id)
    if record is not None and record.grounded and record.text:
        return record.text
    if task.intent or task.name:
        return task.intent or task.name or ""
    for run_id in task.run_ids:
        trace = traces.get(run_id)
        for turn in (trace.turns if trace else []):
            if turn.role == "user" and turn.content:
                return turn.content
    return ""


def _verifier_stage(probe_model: Any = None, probe_limit: Optional[int] = None, judge_model: Any = None):

    """One Verifier per Task from its References by the D111 rule, through the whole D79 suite.

    The References are the confirmed seed replays plus the finished re-rolls that agree on one End
    state after the recordings that broke a Hard constraint are out; the judge is the residue when
    two End states remain and fails at most one side (D110, D111). The Reference proper is the first
    recording of that group, the rest are the re-runs whose agreement sets required against allowed
    (D43) and the second path of check 5, and the anchor is never among them (D81). Before any of
    that, every compiled constraint is checked against the confirmed recordings corpus-wide and the
    ones they mostly break are demoted (D76). Check 4's wrong Run is built from the Reference by
    code; check 6's loophole probe is the one Run per Task the model executes, and `probe_limit`
    caps how many Tasks get one. A Task with no Reference is not verdicted.
    """

    def run(ctx, inputs):
        canon_rules = _rules_of(inputs)
        fn = verifier_mod._canon_fn(canon_rules)
        write_tools = {s.name for s in inputs["sigs"] if s.kind == "write"}
        read_tools = {s.name for s in inputs["sigs"] if s.kind != "write"}
        replays = inputs.get("replays") or {}
        rerolls = inputs.get("rerolls") or {}
        intents = {t: intent.Intent.model_validate(d) for t, d in (inputs.get("intents") or {}).items()}
        user_rules = inputs.get("user_rules") or {}
        traces = {t.trace_id: t for t in inputs.get("traces") or []}
        policy_lines = [c.text for c in inputs["constraints"]]
        seed_replays = {task.id: [r for tid, r in sorted((replays.get(task.id) or {}).items())
                                  if tid in _seed_ids(ctx, task) and r.get("confirmed") and r.get("path")]
                        for task in inputs["tasks"]}
        # D76, D111: the recordings are the frontier under the customer's real policy, so a compiled
        # rule they mostly break is a miscompiled rule; it is reported and gates nothing.
        compiled = [c for c in inputs["constraints"] if c.compiled or c.judge_atom]
        rates = reference_mod.constraint_rates(compiled, [r["path"] for rows in seed_replays.values() for r in rows],
                                               write_tools, fn, read_tools)
        constraints, demoted = reference_mod.demote(compiled, rates)
        # A demoted rule is a residual (D76): reported in the setup review, checked in no Verdict.
        # The compile_policy gate is recorded again over the final list, after the recordings had
        # their say, which is the order the reference check has to run in: on the second retail
        # build 15 of 39 compiled rules fired on confirmed recordings and poisoned every Verifier.
        by_id = {c.id: c for c in compiled}
        residual = [by_id[row["id"]].model_copy(update={"compiled": False, "judge_atom": False,
                                                          "residual_reason": row["reason"]})
                    for row in demoted]
        untouched = [c for c in inputs["constraints"] if not (c.compiled or c.judge_atom)]
        final = constraints + residual + untouched
        record_gate(ctx.workdir, validate.policy_gate(final))
        _write_json(ctx.workdir / "constraints_check.json",
                    {"rates": rates, "demoted": demoted, "constraints": [as_dict(c) for c in final]})
        assisted_tools = set(inputs.get("assisted_tools") or ())
        atoms = reference_mod.hard_atoms(constraints, write_tools, read_tools)
        probe = _probe_runner(ctx, inputs, write_tools, canon_rules) if probe_model is not None else None
        probed = 0
        verifiers, status, references = [], {}, {}
        for task in inputs["tasks"]:
            recordings = [reference_mod.load(r["path"], reference_mod.RECORDING, run_id=r["run_id"],
                                             trace_id=r["trace_id"], write_tools=write_tools, fn=fn, atoms=atoms)
                          for r in seed_replays[task.id]]
            recordings += [reference_mod.load(r["path"], reference_mod.REROLL, run_id=r["run_id"],
                                              write_tools=write_tools, fn=fn, atoms=atoms)
                           for r in rerolls.get(task.id, [])
                           if (r.get("termination_reason") or "") in verifier_mod.SUCCESS_TERMINATIONS]
            confirmation = reference_mod.confirm(recordings, request=_request_text(task, intents, traces),
                                                 policy_lines=policy_lines, judge=judge_model)
            references[task.id] = confirmation.as_dict()
            if not confirmation.references:
                reason = confirmation.reason or "no Run to confirm"
                if not seed_replays[task.id]:
                    reason = ("no seed Trace was replayed" if not (replays.get(task.id) or {}) else
                              replay_mod.unconfirmed_reason({t: r for t, r in replays[task.id].items()
                                                             if t in _seed_ids(ctx, task)}))
                # D49: the status names the blocking tool. A seed Trace that calls an assisted tool
                # replays through a body that failed the fidelity gates, so its divergence is the
                # tool's, and the setup review needs the tool's name, not the diff.
                assisted_used = sorted({c.name for tid in _seed_ids(ctx, task) if tid in traces
                                        for c in traces[tid].tool_calls if c.name in assisted_tools})
                if assisted_used:
                    reason = (f"the seed Trace calls {', '.join(assisted_used)}, an assisted tool whose body "
                              f"failed the fidelity gates (D49); {reason}")
                status[task.id] = {"reference_confirmed": False, "verifier_passed": False, "reason": reason,
                                   "recordings": len(seed_replays[task.id]),
                                   "rerolls": len(rerolls.get(task.id, [])), "judged": confirmation.judged,
                                   "assisted_tools": assisted_used}
                continue
            paths = [r.path for r in confirmation.references]
            first = confirmation.references[0]
            task_for = intent.apply_intent(task, intents[task.id]) if task.id in intents else task
            record = verifier_mod.derive_verifier(task_for, paths[0], paths[1:], canon_rules,
                                                  write_tools=write_tools, constraints=constraints,
                                                  successful_run_ids=[r.run_id for r in confirmation.references])
            may_probe = probe is not None and (probe_limit is None or probed < probe_limit)
            probed += int(may_probe)
            rules_trace = first.trace_id or next((r.trace_id for r in confirmation.references if r.trace_id), None)
            gates = verifier_mod.validate_verifier(
                record, paths[0], canon=canon_rules, write_tools=write_tools, seed_runs=paths[1:],
                wrong_run=verifier_mod.wrong_run(record, paths[0], canon_rules),
                alt_path_run=paths[1] if len(paths) > 1 else None,
                intent_text=task_for.intent, user_rules=user_rules.get(rules_trace),
                model=probe_model if may_probe else None, run_probe=probe)
            results = verifier_mod.d79_results(gates)
            passed = validate.verifier_gate(results).passed
            _write_json(ctx.workdir / "verifiers" / f"{task.id}.json", as_dict(record))
            status[task.id] = {"reference_confirmed": True, "verifier_passed": bool(passed),
                               "references": len(confirmation.references), "reference_kind": first.kind,
                               "recordings": len(seed_replays[task.id]), "rerolls": len(rerolls.get(task.id, [])),
                               "failed_recordings": dict(confirmation.failed), "judged": confirmation.judged,
                               "checks": results,
                               "not_run": [g.stage for g in gates if g.metrics.get("skipped")]}
            verifiers.append(record)
        _write_json(ctx.workdir / "task_status.json", status)
        _write_json(ctx.workdir / "references.json", references)
        # Section 6: a Task whose Verifier does not clear D79 is "not verdicted, Verifier
        # immature", which is a Task the report leaves uncounted, not a failed build.
        broken = [t for t, row in status.items()
                  if row["reference_confirmed"] and not row["verifier_passed"]]
        record_gate(ctx.workdir, validate.gate(
            "derive_verifier", [f"task {t}: the D79 suite did not pass" for t in broken],
            verifiers=len(verifiers), references=sum(1 for r in status.values() if r["reference_confirmed"]),
            passed=sum(1 for r in status.values() if r["verifier_passed"]), tasks=len(status),
            probed=probed, constraints_demoted=len(demoted),
            failed_recordings=sum(len(r.get("failed") or {}) for r in references.values()),
            judged=sum(1 for r in references.values() if r.get("judged")),
            disagreeing=sum(1 for r in references.values()
                            if not r["references"] and (r.get("reason") or "").startswith("recordings disagree"))))
        return {"verifiers": verifiers, "task_status": status}

    return pipeline.Stage(name="derive_verifier", fn=run, builder=True,
                          inputs=("tasks", "sigs", "constraints", "canon_rules", "replays", "rerolls", "intents",
                                  "user_rules", "schema", "bodies", "db", "environment", "traces", "assisted_tools"),
                          outputs=("verifiers", "task_status"), input_paths=("overlays",),
                          code_version=f"{_version('derive_verifier', run, verifier_mod, replay_mod, reference_mod, intent)}:"
                                       f"{getattr(probe_model, 'name', 'none')}:{probe_limit}:"
                                       f"{getattr(judge_model, 'name', 'none')}")


def _probe_runner(ctx: Any, inputs: dict, write_tools: set, canon_rules: Any):
    """Check 6's Run: the model told to reach the Task's End state while skipping the policy step.

    One Run per Task in the Task's own world, with the Simulated user of its Reference, at most
    PROBE_TURNS turns, written under probes/ so it never counts as a Run of the Task. The prompt names
    the End state the Verifier requires and tells the agent to get there without asking, verifying or
    explaining, which is the loophole tau3 kept finding by hand; a Verifier the probe passes is not tight.
    """
    schema, sigs, bodies, db = inputs["schema"], inputs["sigs"], inputs["bodies"], inputs["db"]
    env_id = getattr(inputs["environment"], "env_id", None)
    tasks = {t.id: t for t in inputs["tasks"]}
    user_rules = inputs.get("user_rules") or {}
    replays = inputs.get("replays") or {}
    source = compile_env.module_source(schema, sigs, bodies)
    tools = _tool_definitions(sigs)

    def run_probe(model: Any, verifier: Any):
        task = tasks[verifier.task_id]
        overlay, overlay_rows = compile_env.load_overlay(ctx.workdir, task.id)
        toolkit = compile_env.load_toolkit(source, json.loads(json.dumps(db)), overlay=overlay,
                                           overlay_values=overlay_rows)
        router = route.Router(env_tools_module=toolkit, starting_state=json.loads(json.dumps(db)),
                              overlay=overlay, overlay_rows=overlay_rows, tool_sigs=sigs,
                              canon_rules=canon_rules, synthetic_rows=schema.synthetic_rows)
        reference = next((r for r in (replays.get(task.id) or {}).values() if r.get("confirmed")), None)
        rules = user_rules.get(reference["trace_id"]) if reference else None
        simulated = user_sim.SimulatedUser(rules, starting_state_reader=router.state,
                                           vocab=_vocab_from(ctx.workdir)) if rules else None
        state = loop.new_run_state(f"probe-{task.id}", workdir=ctx.workdir / "probes", env_id=env_id,
                                   task_id=task.id, model=f"probe:{getattr(model, 'name', 'model')}",
                                   user=simulated, max_turns=PROBE_TURNS,
                                   system_prompt=_probe_prompt(task, verifier, sigs))
        try:
            loop.run(state, model, tools=tools, router=router)
        except Exception:  # the loop wrote the error and the stop; a crashed probe reaches no End state
            pass
        return state.run

    return run_probe


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


# --- the two entry points cli.py calls --------------------------------------

def build(workdir: Any, iterate: bool = False, model: Any = None, files: Optional[list] = None,
          ceiling_usd: Optional[float] = None, domain: str = "domain",
          max_attempts: int = 3, memory_dir: Any = None, on_event: Optional[Any] = None,
          grow: Optional[dict[str, int]] = None, grow_seed: int = 0,
          probe_limit: Optional[int] = None, rerolls: int = DEFAULT_REROLLS, search: Any = None,
          workers: int = 1) -> dict:
    """Read the ingested Traces and write the Environment, the Tasks and one Verifier each.

    `model` is the Builder's model, already an adapter; nothing here constructs one (build brief
    rule 2). It is wrapped in `budget.BudgetedModel` before any stage sees it, so every call is
    priced into budget.json and refused past the D65 context cap. `iterate` keeps the
    content-addressed cache, which is what makes a repeat build cheap; without it the cache is
    dropped and every stage runs again. `on_event` is handed to both Pipelines so a live view can
    watch one build across the two of them; nothing here reads it. `grow` is the row count per
    table the Starting state is grown to with synthetic rows (D107); by default nothing is grown.
    `rerolls` is how many Runs of the frontier each Task gets beside its recordings (D112); it is
    0 whenever there is no model, since a re-roll is a model Run. `workers` (D118) is how many
    tool bodies, policy sentences, Intents or Tasks' re-rolls are asked for at once; the CLI
    passes `parallel.DEFAULT_WORKERS`, the default here is one so a scripted model in a test
    answers in the order it was given, and every artifact is the same at any count.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # The Builder's lessons outlive one build (D87), so a customer with standing memory points at it;
    # the default keeps them in this workdir, where a first build has none to carry.
    memory_dir = Path(memory_dir) if memory_dir is not None else workdir / "memory"
    if not iterate:
        _clear_cache(workdir)
    ceiling = _ceiling(workdir, ceiling_usd)
    to_ingest = [Path(f) for f in (files or [])]
    if not to_ingest and not load_traces(workdir):
        raise BuildError(f"no Traces under {workdir / 'traces'} and no file to ingest")

    # Phase one reads the customer's traces; it produces the Tasks the anchor is drawn from, so the
    # anchor cannot exist before it and no stage here is a Builder stage (D81).
    # An iterated build with nothing new to ingest runs no ingest stage, so the record of the last
    # ingest would vanish from state.json; it is carried over from the previous build's file.
    prior_ingest = _ingest_record(_read_json(workdir / "pipeline" / "state.json", {})) if not to_ingest else {}
    read = pipeline.Pipeline(
        [s for s in [_ingest_stage(workdir, to_ingest) if to_ingest else None,
                     _mine_stage(), _cluster_stage()] if s is not None],
        workdir, ceiling=ceiling, on_event=on_event)
    first = read.run({} if to_ingest else {"traces": load_traces(workdir)})
    # Pipeline.run() writes its own statuses, log and gates to pipeline/state.json unconditionally;
    # the second Pipeline below writes the same fixed path and would otherwise overwrite this one's
    # record of ingest, mine and cluster. Captured here so it can be folded back in once the second
    # pipeline has run and written its own version of the file.
    first_state = _read_json(workdir / "pipeline" / "state.json", {})
    for key, value in prior_ingest.items():
        first_state[key] = {**value, **first_state.get(key, {})} if isinstance(value, dict) \
            else list(value) + list(first_state.get(key, []))
    if first.status != "complete":
        return _result(workdir, first, None)

    anchor = pipeline.choose_anchor(first.artifacts["tasks"], workdir)

    builder_model = _wrap(model, "compile_tools", workdir, ceiling)
    policy_model = _wrap(model, "compile_policy", workdir, ceiling)
    lessons_model = _wrap(model, "judge_lessons", workdir, ceiling)
    vocab_model = _wrap(model, "vocabulary", workdir, ceiling) if model is not None else None
    # The loophole probe is a Candidate-shaped Run: fresh samples, production setting (D65).
    probe_model = _wrap(model, "loophole_probe", workdir, ceiling, cap_context=False, memoize=False) \
        if model is not None else None
    # The re-rolls are Candidate-shaped too (D112); the Intent and the judge are Builder calls (D65).
    reroll_model = _wrap(model, "reroll", workdir, ceiling, cap_context=False, memoize=False) \
        if model is not None and rerolls > 0 else None
    intent_model = _wrap(model, "intent", workdir, ceiling) if model is not None else None
    judge_model = _wrap(model, "reference_judge", workdir, ceiling) if model is not None else None
    second = pipeline.Pipeline(
        [s for s in [_canon_stage(), _state_stage(grow, grow_seed), _tools_stage(builder_model, max_attempts, workers),
                     _policy_stage(policy_model, workers), _lessons_stage(lessons_model, memory_dir),
                     _intent_stage(intent_model, workers) if intent_model is not None else None,
                     _vocabulary_stage(vocab_model, search), _user_rules_stage(), _environment_stage(domain),
                     _replay_stage(),
                     _rerolls_stage(reroll_model, rerolls, workers) if reroll_model is not None else None,
                     _verifier_stage(probe_model, probe_limit, judge_model)] if s is not None],
        workdir, anchor=anchor, ceiling=ceiling, on_event=on_event)
    result = second.run(dict(first.artifacts))
    result.gates = list(first.gates) + list(result.gates)
    _merge_pipeline_state(workdir, first_state)
    _write_scorecard(workdir)
    return _result(workdir, result, result.artifacts.get("environment"))


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
    gates = [validate.budget_gate(budget.load_totals(workdir), stage="candidate")]
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
    """Fold phase one's statuses, attempts, log and gates back into pipeline/state.json.

    The second Pipeline's own run() already wrote its version of the file by the time this is
    called (runner/pipeline.py's write is unconditional and this module does not touch it); this
    merges phase one's record back in immediately after, so report.py's one read of state.json
    covers ingest, mine and cluster as well as the second pipeline's stages, instead of only
    whichever pipeline ran last.
    """
    if not first_state:
        return
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
    """validate.scorecard's own nested dict, written where report.py reads it (D62)."""
    return _write_json(workdir / "scorecard.json", validate.scorecard(workdir))


def _result(workdir: Path, result: Any, environment: Any) -> dict:
    return {"status": result.status, "workdir": str(workdir),
            "env_id": getattr(environment, "env_id", None),
            "failed_stage": result.failed_stage, "stopped": result.stopped,
            "gates": [as_dict(g) if isinstance(g, GateResult) else g for g in result.gates],
            "tasks": [t.id for t in result.artifacts.get("tasks", [])]}
