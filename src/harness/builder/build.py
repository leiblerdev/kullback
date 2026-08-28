"""The one wiring of the Builder stages over the Runner's Pipeline, and the Candidate Run batch that follows it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from harness.builder import cluster, compile_env, ingest, memory, mine, policy, user_sim
from harness.builder import verifier as verifier_mod
from harness.runner import loop, pipeline, route, validate
from harness.shared import budget, canon, provider
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

    return pipeline.Stage(name="mine", fn=run, inputs=("traces",), outputs=("sigs", "schema"))


def _cluster_stage():
    def run(ctx, inputs):
        categories, tasks = cluster.cluster_runs(inputs["traces"], inputs["sigs"])
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

    return pipeline.Stage(name="cluster", fn=run, inputs=("traces", "sigs"),
                          outputs=("categories", "tasks"))


def _canon_stage():
    def run(ctx, inputs):
        schema = inputs["schema"]
        rows = [row for trace in inputs["traces"] for call in trace.tool_calls
                for row in _rows_of(call.result)]
        rules = canon.learn_rules(schema, rows)
        canon.save_rules(rules, ctx.workdir / CANON_RULES)
        return {"canon_rules": rules.model_dump()}

    return pipeline.Stage(name="canon_rules", fn=run, inputs=("traces", "schema"), outputs=("canon_rules",))


def _rows_of(result: Any) -> list[dict]:
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    return []


def _state_stage():
    def run(ctx, inputs):
        state = compile_env.build_starting_state(inputs["traces"], inputs["schema"], ctx.workdir,
                                                 inputs["tasks"], inputs["sigs"])
        return {"db": state.db, "overlays": list(state.overlays),
                "assumptions": list(state.assumptions), "synthetic_rows": list(state.synthetic_rows)}

    return pipeline.Stage(name="starting_state", fn=run, inputs=("traces", "schema", "tasks", "sigs"),
                          outputs=("db", "overlays", "assumptions", "synthetic_rows"))


def _tools_stage(model: Any, max_attempts: int):
    def run(ctx, inputs):
        traces, tasks = inputs["traces"], inputs["tasks"]
        seeds = _seed_traces(ctx, tasks, traces)
        calls_by_tool: dict[str, list] = {}
        call_tasks: dict[str, str] = {}
        for trace in traces:
            task_id = _task_of(tasks, trace.trace_id)
            for call in trace.tool_calls:
                if trace.trace_id in seeds:  # D81: the anchor's calls are not Builder evidence
                    calls_by_tool.setdefault(call.name, []).append(call)
                if call.id and task_id:
                    call_tasks[call.id] = task_id
        # D74: each recorded call replays on the world its own Task saw, not on the shared one.
        states = compile_env.call_starting_states(inputs["db"], inputs["overlays"],
                                                  compile_env.overlay_values(ctx.workdir), call_tasks)
        tool_names = [sig.name for sig in inputs["sigs"]]
        bodies, gates, assisted, builds = {}, [], [], {}
        for sig in inputs["sigs"]:
            build = compile_env.compile_tool(model, sig, calls_by_tool.get(sig.name, []),
                                             inputs["schema"], inputs["db"],
                                             ctx.workdir / "tools" / sig.name,
                                             max_attempts=max_attempts, call_states=states,
                                             rules=_rules_of(inputs), tool_names=tool_names)
            bodies[sig.name] = build.body
            gates.extend(build.gates)
            builds[sig.name] = {"assisted": build.assisted, "nodes": build.nodes}
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
                          outputs=("bodies", "assisted_tools"), gate=gate)


def _seed_traces(ctx, tasks, traces) -> set[str]:
    """Every Trace the Builder may learn from: each Task's Runs minus its anchor (D81)."""
    seeds: set[str] = set()
    for task in tasks:
        seeds.update(ctx.seed_runs(task.id, task.run_ids))
    known = {t.trace_id for task in tasks for t in traces if t.trace_id in task.run_ids}
    return seeds | {t.trace_id for t in traces if t.trace_id not in known}


def _task_of(tasks, trace_id: str) -> Optional[str]:
    return next((t.id for t in tasks if trace_id in t.run_ids), None)


def _policy_stage(model: Any):
    """D76: the policy sentences become Constraints, and the Reference's own path has to stay legal."""

    def run(ctx, inputs):
        text = _policy_text(inputs["traces"])
        constraints = policy.compile_policy(model, text) if (text and model is not None) else []
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


def _user_rules_stage():
    def run(ctx, inputs):
        rules: dict[str, UserRules] = {}
        for trace in inputs["traces"]:
            rules[trace.trace_id] = user_sim.derive_user_rules(trace)
        for trace_id, record in rules.items():
            _write_json(ctx.workdir / "user_rules" / f"{trace_id}.json", as_dict(record))
        _write_json(ctx.workdir / "user_facts.json",
                    {"facts": [{"run_id": tid, "field": f.field, "value": f.value}
                               for tid, r in rules.items() for f in r.facts]})
        # Section 6: incomplete user rules flag the Run, they do not fail the build.
        record_gate(ctx.workdir, validate.user_rules_gate(list(rules.values())))
        return {"user_rules": rules}

    return pipeline.Stage(name="user_rules", fn=run, builder=True, inputs=("traces",),
                          outputs=("user_rules",))


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
            inputs["schema"], inputs["sigs"], inputs["bodies"], inputs["policy_text"], files=files)
        bundle.environment = environment
        compile_env.emit_tau2_shape(bundle, ctx.workdir / "env", files=files)
        _write_json(ctx.workdir / "environment.json", as_dict(environment))
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
                                  "tasks", "assumptions", "synthetic_rows", "traces"),
                          outputs=("environment",), gate=gate)


def _rules_of(inputs: dict) -> canon.CanonRules:
    """The CanonRules the canon stage learned from this customer's own corpus (D39).

    Read off the pipeline rather than off disk: the rules are learned inside this same run, so a
    value read before the pipeline started would be the module defaults on every first build.
    """
    return canon.CanonRules.model_validate(inputs.get("canon_rules") or {})


def _verifier_stage():
    """One Verifier per Task from the Task's own Reference replay (D91: no Run is executed here)."""

    def run(ctx, inputs):
        canon_rules = _rules_of(inputs)
        write_tools = {s.name for s in inputs["sigs"] if s.kind == "write"}
        constraints = [c for c in inputs["constraints"] if c.compiled or c.judge_atom]
        verifiers, status = [], {}
        for task in inputs["tasks"]:
            paths = sorted((ctx.workdir / "runs" / task.id).glob("*.jsonl"))
            if not paths:
                status[task.id] = {"reference_confirmed": False, "verifier_passed": False}
                continue
            record = verifier_mod.derive_verifier(task, paths[0], [str(p) for p in paths[1:]],
                                                  canon_rules, write_tools=write_tools,
                                                  constraints=constraints)
            gates = verifier_mod.validate_verifier(record, paths[0], canon=canon_rules,
                                                   write_tools=write_tools)
            passed = validate.verifier_gate(verifier_mod.d79_results(gates)).passed
            _write_json(ctx.workdir / "verifiers" / f"{task.id}.json", as_dict(record))
            status[task.id] = {"reference_confirmed": True, "verifier_passed": bool(passed)}
            verifiers.append(record)
        _write_json(ctx.workdir / "task_status.json", status)
        # Section 6: a Task whose Verifier does not clear D79 is "not verdicted, Verifier
        # immature", which is a Task the report leaves uncounted, not a failed build.
        broken = [t for t, row in status.items()
                  if row["reference_confirmed"] and not row["verifier_passed"]]
        record_gate(ctx.workdir, validate.gate(
            "derive_verifier", [f"task {t}: the D79 suite did not pass" for t in broken],
            verifiers=len(verifiers)))
        return {"verifiers": verifiers, "task_status": status}

    return pipeline.Stage(name="derive_verifier", fn=run, builder=True,
                          inputs=("tasks", "sigs", "constraints", "canon_rules"),
                          outputs=("verifiers", "task_status"),
                          code_version=f"derive_verifier:{_module_hash(verifier_mod)}")


# --- the two entry points cli.py calls --------------------------------------

def build(workdir: Any, iterate: bool = False, model: Any = None, files: Optional[list] = None,
          ceiling_usd: Optional[float] = None, domain: str = "domain",
          max_attempts: int = 3, memory_dir: Any = None) -> dict:
    """Read the ingested Traces and write the Environment, the Tasks and one Verifier each.

    `model` is the Builder's model, already an adapter; nothing here constructs one (build brief
    rule 2). It is wrapped in `budget.BudgetedModel` before any stage sees it, so every call is
    priced into budget.json and refused past the D65 context cap. `iterate` keeps the
    content-addressed cache, which is what makes a repeat build cheap; without it the cache is
    dropped and every stage runs again.
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
        workdir, ceiling=ceiling)
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
    second = pipeline.Pipeline(
        [_canon_stage(), _state_stage(), _tools_stage(builder_model, max_attempts),
         _policy_stage(policy_model), _lessons_stage(lessons_model, memory_dir),
         _user_rules_stage(), _environment_stage(domain), _verifier_stage()],
        workdir, anchor=anchor, ceiling=ceiling)
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
    paths = []
    for number in range(count):
        run_id = f"{task_id}-{seed + number}"
        # The Task's own overlay goes inside the toolkit, or it stays dead for every code route (D74).
        toolkit = compile_env.load_toolkit(source, json.loads(json.dumps(db)), overlay=overlay,
                                           overlay_values=overlay_rows)
        router = route.Router(env_tools_module=toolkit, starting_state=json.loads(json.dumps(db)),
                              overlay=overlay, overlay_rows=overlay_rows, tool_sigs=sigs,
                              canon_rules=canon_rules)
        simulated = user_sim.SimulatedUser(rules, starting_state_reader=router.state) if rules else None
        state = loop.new_run_state(run_id, workdir=workdir / "runs" / task_id,
                                   env_id=environment.get("env_id"), task_id=task_id,
                                   model=getattr(model, "name", None) or "candidate",
                                   seed=seed + number, user=simulated, user_rules=rules,
                                   max_turns=max_turns)
        loop.run(state, candidate, router=router)
        paths.append(str(state.path))
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
