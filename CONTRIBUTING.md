# Contributing to Kullback

Thanks for looking. This file says how to get a change in and what a change has to respect. The design lives in `docs/harness-design.md`, the reasons in `docs/decision-log.md`, the words in `CONTEXT.md`; when this file and those disagree, those win.

## Ways to help

Open an issue for a bug, a wrong number in a report, a trace format we do not read, or a place where the design and the code disagree. Open a pull request for a fix or a small feature. For anything that changes a decision in the decision log, open an issue first and say which decision and why; the log records the alternative each decision beat, so read that entry before arguing against it.

Good first contributions: an ingest mapper for a trace format you use (OpenTelemetry GenAI, Claude Code JSONL, MCP logs), a canonicalization rule with a test, a survivor from `mutmut` turned into a real test, a gap in `SLICE_RESULTS.md` that you can measure.

## Setup

```
git clone https://github.com/leiblerdev/kullback
cd kullback
uv sync
uv run pytest
```

Python 3.11 or newer. The test suite runs in about ten seconds and never calls a model. Install the pre-commit hook once, `uv run pre-commit install`; it runs `ruff check` on the files you stage and fixes what it safely can. To run the offline slice on real data, `scripts/fetch_tau2_traces.sh` downloads Sierra's public tau2 retail runs into `data/raw/`; that directory is gitignored and stays that way.

## The workflow

1. Branch from `main`. Name it after the change, not after yourself.
2. Write the failing test first, then the code. Every module has one test file, `tests/test_<module>.py`; new behaviour goes there.
3. Run `uv run pytest` and paste the summary line in the pull request. If you touched `src/harness/runner/`, also run the import boundary check: `uv run python -c "import harness.runner.validate as v, pathlib; print(v.import_boundary_check(pathlib.Path('src')))"`.
4. Open the pull request against `main`. Say what changed, why, and what you measured. `main` is protected: every pull request needs an approving review from the maintainer (`.github/CODEOWNERS`), every conversation resolved, and no force pushes.

Mutation testing (`uv run mutmut run`) is not in CI because it takes long; run it when you touch verdict, canon, route or verifier, and mention survivors you looked at.

## Rules the code keeps

These are not style preferences. Each one holds a design decision in place, and a test enforces most of them.

The raw trace is the only source of truth. Files are stored byte for byte and hashed; every derived field carries a `raw_ptr` back to where it came from. Never modify a raw file, never derive a value you cannot point back to (D66).

No model calls in tests. `ALLOW_MODEL_REQUESTS` is `False` by default and `conftest.py` keeps it there. Code that needs a model takes a `Model` as a parameter and never constructs one; tests pass `TestModel` or `RecordedModel`.

The Runner never imports the Builder, and nothing on the Runner side ever reads a Verifier except `verdict.py` reading the one it is given. `validate.import_boundary_check` scans for this and `tests/test_e2e.py` asserts it (D89, D91).

The Verdict is code. A judge can remove a run from the bar or widen a Verifier for every candidate; it can never award a pass. Do not add a code path where a model's opinion turns into a pass.

Grader fields from benchmark traces (`reward_info`, `evaluation_criteria`, `action_checks`, `nl_assertions`, `trial`) live in the `grader/` sidecar and only Verdict comparison code reads them.

Records are Pydantic models in `src/harness/shared/records.py`. Import them; do not redefine a record elsewhere. Outputs are content-addressed; state lives in files under a `workdir` passed in, never in module globals.

Errors from a customer's tools keep their verbatim payload beside the classified error class (D67).

Every module is one file with one sentence of purpose at the top. Each has a size band in design section 10. If you cannot fit, say why in the pull request rather than growing silently.

Model-written code (tool bodies, policy predicates) runs only through the paths in `compile_env.py` and `policy.py` that gate it. Do not add an `exec` or `eval` elsewhere.

## Writing

Plain sentences. No em dashes or en dashes anywhere, in code, comments, docs or commit messages; use a comma, a colon, a period or parentheses. No emojis. Use the glossary's words (`CONTEXT.md`): a Run, a Task, a Verdict, an Environment mean specific things here and the code uses the same names.

Commit messages say what changed and why in the first line, in the imperative, under 72 characters.

## Changing the design

Small clarifications go straight into `docs/harness-design.md` in the same pull request as the code. A change of decision needs a new entry in `docs/decision-log.md` with the next D number, what it replaces, and the alternative it beat. An ADR in `docs/adr/` is only for a decision that is hard to reverse, surprising without context, and the result of a real trade-off; most changes are not that.

New numbers from an experiment go into `SLICE_RESULTS.md` or a new file beside it, with the script that produced them and the seed and held-out figures side by side. Numbers without the script do not get merged.

## Reporting problems with a customer's data

If you find that a trace format leaks something it should not, or that a record could carry a secret, open a private security advisory on GitHub rather than a public issue.

## License

Apache-2.0. By submitting a pull request you agree that your contribution is licensed under the same terms. Third-party code needs a compatible license and a note in `NOTICE`.
