# Build brief for the harness (read fully before writing code)

Kept as written for the first build. The paths below name the layout of that build (`src/harness/` with `builder/`, `runner/` and `shared/`); since phase 1 of the rebuild the package is `kullback/` at the repository root, split as `docs/tech/rebuild-phases.md` describes and `docs/tech/phase-1-the-move.md` records. Read the rules here, and the paths there.

This is the environment-generation harness designed in `../docs/harness-design.md`. The design is the spec; the decision log (`../docs/decision-log.md`, decisions D01 to D97) is the reasoning; `../../CONTEXT.md` is the glossary and its terms are the names used in code. When the design and this brief disagree, the design wins; say so in your return value.

## Where things are

- Package root: `monitoring-tool/harness/` (this directory). Python 3.11, `uv`, `pyproject.toml` at this root, package `harness` under `src/harness/`, tests under `tests/`.
- Layout: `src/harness/builder/` (ingest, mine, cluster, intent, compile_env, policy, user_sim, verifier, memory), `src/harness/runner/` (loop, route, verdict, judge, regrade, validate, pipeline), `src/harness/shared/` (records, canon, provider, budget, report), `src/harness/cli.py`.
- tau2-bench source, vendored, read only: `../vendor/tau2-bench/` (retail domain at `src/tau2/domains/retail/{data_model.py,tools.py,environment.py}` and `data/tau2/domains/retail/{db.json,policy.md,tasks.json}`). This is the truth for the tau2 slice: our mined `ToolSig` must match its `tools.py`, our emitted files must load in its harness.
- Raw traces, never committed, never modified: `../data/raw/*.json` (Sierra public S3, tau2 retail, 456 simulations each: `{timestamp, info, tasks, simulations}`; a simulation has `id, task_id, trial, seed, termination_reason, reward_info, messages`; messages have `role` in assistant/user/tool, `content`, `tool_calls: [{id, name, arguments, requestor}]`, tool messages have `id, content, requestor, error: bool`).
- Test fixture: `tests/fixtures/tau2_retail_small.json`, the first 3 simulations plus their 3 tasks, same shape as the raw file (built by the scaffold agent).

## Rules for every agent

1. No git commands. No edits to files another agent owns (ownership table below). If you need a change in a shared file (`records.py`, `conftest.py`, `pyproject.toml`), do not make it: return it under `needs_from_others` and write your own code against the current version.
2. No model calls in tests. `ALLOW_MODEL_REQUESTS=False` is the default; `harness.shared.provider.TestModel` (scripted replies) and `RecordedModel` (replays a stored Run) are the only models tests use. Code that calls a model takes a `Model` instance as a parameter; it never constructs one.
3. Every module is one file with one sentence of purpose at the top, and one test file `tests/test_<module>.py`. Pure functions over `records`; state lives in files under a `workdir` passed in, never in module globals. Content-addressed outputs (`harness-design.md` section 8).
4. Records are Pydantic v2 models in `src/harness/shared/records.py` (section 5 of the design, plus `TaskOverlay`, `Category`, `Task`, `UserBehaviour` stubs, the D97 sub-versions). Import them; do not redefine.
5. Raw traces are the source of truth (D66): every derived field carries `raw_ptr` (file hash, simulation index, message index). Grader fields (`reward_info`, `evaluation_criteria`, `action_checks`, `nl_assertions`, `trial`) are stripped at ingest into a separate `grader/` sidecar that only `verdict` comparison code may read (D66, D89).
6. Errors from the customer's tools keep their verbatim payload and encoding beside the D67 class (`tool_not_found, invalid_arguments, permission_denied, business_error, not_found_entity, transient, cancelled, unknown`).
7. The Runner (`loop.py`, `route.py`, `verdict.py`, `validate.py`, `budget.py`) never imports anything from `builder/` and never reads a Verifier file except `verdict.py` reading the Verifier it is given (D89, D91). `validate.py` has a test that asserts this by scanning imports.
8. No em dashes or en dashes anywhere (code, comments, docstrings, docs). Plain sentences. No emojis.
9. Keep each module in its size band from `harness-design.md` section 10; `loop.py` is 100 to 150 lines. If you cannot, say why in the return value rather than growing silently.
10. Run `uv run pytest tests/test_<yourmodule>.py -q` before returning and report the result truthfully.

## Ownership (one agent per row; shared files are the scaffold agent's)

| Module | Files |
|---|---|
| scaffold | `pyproject.toml`, `src/harness/__init__.py`, `src/harness/shared/records.py`, `src/harness/shared/provider.py` (interface, `TestModel`, `RecordedModel` only), `tests/conftest.py`, `tests/fixtures/`, `README.md` |
| ingest | `src/harness/builder/ingest.py`, `tests/test_ingest.py` |
| mine | `src/harness/builder/mine.py`, `tests/test_mine.py` |
| cluster_intent | `src/harness/builder/cluster.py`, `src/harness/builder/intent.py`, tests |
| compile_env | `src/harness/builder/compile_env.py`, `tests/test_compile_env.py` |
| policy | `src/harness/builder/policy.py`, `tests/test_policy.py` |
| user_sim | `src/harness/builder/user_sim.py`, `tests/test_user_sim.py` |
| verifier | `src/harness/builder/verifier.py`, `tests/test_verifier.py` |
| memory | `src/harness/builder/memory.py`, `tests/test_memory.py` |
| canon | `src/harness/shared/canon.py`, `tests/test_canon.py` |
| provider_budget | `src/harness/shared/provider.py` (adapters, retry; extend, keep the scaffold's interface), `src/harness/shared/budget.py`, tests |
| loop_route | `src/harness/runner/loop.py`, `src/harness/runner/route.py`, tests |
| verdict_regrade | `src/harness/runner/verdict.py`, `src/harness/runner/regrade.py`, tests |
| judge | `src/harness/runner/judge.py`, `tests/test_judge.py` |
| validate | `src/harness/runner/validate.py`, `tests/test_validate.py` |
| pipeline | `src/harness/runner/pipeline.py`, `tests/test_pipeline.py` |
| cli_report | `src/harness/cli.py`, `src/harness/shared/report.py`, tests |

## Return value (every module agent)

JSON: `{ "module": str, "files": [str], "public_api": [str signatures], "tests": {"passed": int, "failed": int, "command": str}, "needs_from_others": [str], "deviations_from_design": [str], "lines": int }`.
