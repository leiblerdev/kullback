# Developing Kullback

The Builder turns a customer's traces into an Environment (state, tools, Hard constraints, Simulated user) and the Verifiers for its Tasks; the Runner re-executes Runs with a Candidate in that Environment and computes each Verdict.

Spec: `docs/harness-design.md`. Reasoning: `docs/decision-log.md` (D01 to D97). Glossary, and the names used in code: `CONTEXT.md`. Rules for contributors, human or agent: `CONTRIBUTING.md`.

## Layout

```
src/harness/
  builder/    ingest, mine, cluster, intent, compile_env, policy, user_sim, verifier, memory
  runner/     loop, route, verdict, judge, regrade, validate, pipeline
  shared/     records, canon, provider, budget, report
  cli.py
tests/
  fixtures/   the small tau2 file and tau2-bench's retail domain files
  conftest.py shared fixtures
```

## Getting started

```
uv sync
uv run pytest -q
```

Python 3.11. Dependencies: pydantic v2, typer, httpx, pytest. Nothing heavier goes in without a reason in the pull request.

## Records

`src/harness/shared/records.py` holds every record in design section 5. Import them, do not redefine them.

- Every model subclasses `Record`, whose config is `populate_by_name=True`. Two fields are named around Python keywords: `Verdict.passed` / `GateResult.passed` carry the alias `pass`, and `ToolCallError.class_` / `ErrorShape.class_` / `Column.class_` / `Verdict.class_` carry the alias `class`. Construct them either way (`Verdict(passed=True, class_="pass")` or `Verdict(**{"pass": True, "class": "pass"})`).
- Serialize with `as_dict(record)` (which is `model_dump(mode="json", by_alias=True)`) so the aliases land in the JSON. Reading back with `Model.model_validate(payload)` accepts both spellings.
- `content_hash(obj)` is sha256 over `canonical_json(obj)` (aliases applied, keys sorted, no spare whitespace). Use it for every content-addressed output (design section 8); it accepts records, dicts, lists and scalars.
- `RawPtr(file_hash, sim_index, msg_index)` is the pointer back into the stored raw file that every derived field carries (D66).
- `RawFile.bytes` is a byte count, not the content; the content stays on disk at `RawFile.path`. That is the one place the field name in design section 5 is read as a number.
- `ALL_RECORDS` is every record class in the module, used by the round-trip test.

## Models in tests

`src/harness/shared/provider.py` defines the interface and the two offline models. `ALLOW_MODEL_REQUESTS` is `False` and `tests/conftest.py` holds it there for every test.

```python
class Model:
    def query(self, messages, tools=None, config=None) -> ModelReply: ...
```

`ModelReply` is `content, tool_calls, usage (input, output, cache_read, cache_write), model, stop_reason, raw`. `TestModel(replies)` gives scripted replies in order (a string, a dict in message shape, or a `ModelReply`). `RecordedModel(run_jsonl_path)` replays the assistant messages of a stored Run: it reads `model_call` events (`payload.reply`, or the payload itself) and plain `{"role": "assistant", ...}` lines, in file order. Real adapters live under the marked extension point at the bottom of the file and must call `require_live_calls_enabled()` first.

Code that needs a model takes one as a parameter. It never constructs one.

## Fixtures

- `tests/fixtures/tau2_retail_small.json`: the first 3 simulations of Sierra's public tau2 retail run `claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json` (Claude 3.7 Sonnet as the agent, GPT-4.1 as the user simulator) plus the 3 tasks those simulations reference. Top-level `timestamp` and `info` are kept as they are, nothing else is changed. Grader fields (`reward_info`, `trial`, `evaluation_criteria`, `action_checks`, `nl_assertions`) are still in it on purpose, so ingest can be tested stripping them (D66). Rebuild it with `uv run python tests/fixtures/make_tau2_retail_small.py [path/to/raw.json]`.
- `tests/fixtures/tau2_retail/{db.json,policy.md,tasks.json}`: copied unchanged from tau2-bench `data/tau2/domains/retail/` at commit `a2c0247`. They are the truth for the tau2 slice: what `compile_env.py` emits is checked against them, and what we emit must load in tau2's harness.
- The full raw traces under `../data/raw/` are never committed and never modified. The `raw_dir` fixture skips a test when they are absent. `scripts/fetch_tau2_traces.sh` downloads them from Sierra's public bucket.

Shape of a raw file: `{timestamp, info, tasks, simulations}`. A simulation is `{id, task_id, trial, seed, termination_reason, duration, agent_cost, user_cost, reward_info, messages}`. A message has `role` in `assistant | user | tool`, `content`, `turn_idx`, `timestamp`, and either `tool_calls: [{id, name, arguments, requestor}]` (assistant) or `id, requestor, error: bool` (tool). Note that `user` messages carry the user simulator's own cost and usage, so token counts on a `user` message belong to the simulated user, not to the agent.

## Fixtures in conftest.py

`fixtures_dir`, `tau2_small_path`, `tau2_small` (parsed), `tau2_retail_dir`, `raw_dir` (skips when missing), `workdir` (a fresh tmp directory; module state lives under a workdir, never in module globals), `make_test_model`, `test_model`, `write_run_jsonl` (write dict lines as a Run JSONL, get the path), `make_recorded_model`, and an autouse `no_live_models`.

## Mutation testing

`uv run mutmut run` mutates `src/harness/` and runs the tests against each mutant; `uv run mutmut browse` shows survivors. A survivor is one of four things: a real gap (write the test that would have caught the bug), an equivalent mutant (the change cannot be observed), unreachable code (delete it), or something we do not care about (say so). Never write a test whose only purpose is to kill a mutant.
