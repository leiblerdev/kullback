# AGENTS.md

Rules for any agent or person changing this repo. The docs directory is [docs/README.md](docs/README.md); words
are defined in [docs/glossary.md](docs/glossary.md); the package map is [docs/architecture.md](docs/architecture.md).

## Install, test, run

```bash
uv sync
uv run pre-commit install
uv run pytest -q
uv run lint-imports
uv run python scripts/branch_checks.py origin/main HEAD
uv run kullback --help
```

In a git worktree, run pytest with `PYTHONPATH=<worktree>`, or the tests import the main checkout. The suite
never calls a model: `tests/conftest.py` keeps live requests off. For live calls copy `.env.example` to `.env`,
set `HARNESS_ALLOW_MODEL_REQUESTS=1` and the keys; the CLI reads it. Scripts load it with `set -a; . .env; set +a`.
Never print a key.

## Code

- General only: no corpus, domain, tool, table, field, Task id or model names in code, prompts or tests.
  `branch_checks.py` refuses corpus names on added lines under `kullback/` and `tests/`.
- A fix improves the harness for every Environment, or it is not a fix. Take the simple route when one exists.
- Code that needs a model takes one as a parameter. Tests use `TestModel`, `RecordedModel` or `MemoModel` and
  never mock the code under test.
- Name a test for the behaviour it states, in a sentence.
- Respect the import contracts in `pyproject.toml` (see the architecture page).

## Frozen and append-only

- Never edit: `README.md`, `docs/adr/`, `docs/founder-words.md`, `kullback/runner/`, `kullback/gates/`,
  `docs/builds/README.md`. `branch_checks.py` fails a branch that changes the first five.
- Never patch `vendor/`.
- `docs/decisions.md` changes only by a new D number; the entry it replaces moves to Retired. Numbers are never reused.

## Never committed

`data/raw/`, `vendor/`, `.env`, `.claude/` and `.work-*` directories. Customer data never enters git history.

## Writing

- No em dashes or en dashes anywhere: code, comments, docs, commit messages. No emojis.
- Plain sentences, the glossary's words, lines under 150 characters in docs.
- Say each fact once, on the page that owns it, and link to it from the others.

## Commits

Commit under the repo's git user. Subject under 72 characters. No `Co-Authored-By` or other attribution
trailers. Hooks stay on.
