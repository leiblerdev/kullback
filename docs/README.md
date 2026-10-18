# Kullback docs

Kullback rebuilds a running copy of your system from your agent's traces and grades any model, in code, on what it
changes there. This page lists every doc, one line each.

## Use it

```bash
uv sync
uv run kullback fetch leibler/retail --out env-retail
uv run kullback run --workdir env-retail --task <task id> --model provider/model
uv run kullback verdict --workdir env-retail
```

Every command takes `--help`. Contributors and agents start with [AGENTS.md](../AGENTS.md).

## Start here

- [glossary.md](glossary.md): every word the docs use, defined once.
- [vision.md](vision.md): what the product is, what it treats as true, what exists today.
- [trace-intake.md](trace-intake.md): the trace formats Kullback reads, and what to ask a customer.

## How it works

- [architecture.md](architecture.md): packages, import contracts, where each build stage lives.
- [architecture/modules.md](architecture/modules.md): one line per module.
- [prompt-caching.md](prompt-caching.md): how prompts are cached and how to check a build's cache.
- [design-philosophy.md](design-philosophy.md): the founder's narrative of the design.
- [adr/](adr/): architecture decisions that are hard to reverse.

## Plans

- [todo.md](todo.md): known work, not done.
- [scaling.md](scaling.md): more Tasks and more Environments; the synthetic Task rules.
- [training-plan.md](training-plan.md): proving an Environment by training in it.

## Reference

- [learnings.md](learnings.md): rules the build loop taught, one line each.
- [assumptions.md](assumptions.md): what the design takes as true, and how we find out.
- [benchmark-landscape.md](benchmark-landscape.md): public benchmarks that could serve as a second reference.
- [trace-sources/](trace-sources/README.md): public trace datasets, one page each.
- [papers.md](papers.md): papers and repos the founder sent, and what each fed.
- [hf/](hf/): Hugging Face cards.
- [validation/](validation/README.md): competitors and dated trust measurements.

## Records

Kept as written; not updated.

- [decision-log.md](decision-log.md): every numbered decision (D numbers), append-only.
- [founder-words.md](founder-words.md): the founder's messages, verbatim.
- [builds/README.md](builds/README.md): what each build measured.
- [reviews/](reviews/README.md): dated audits of the code and the tests.
- [overhaul/learnings-ledger.md](overhaul/learnings-ledger.md): where each earlier learning landed in the 2026-09-22
  overhaul.
- [frozen-patches/](frozen-patches/): patches waiting to be rebased.
- [archive/harness-design.md](archive/harness-design.md): the design proposal of 2026-08-27, superseded by
  ADR-0007, ADR-0011 and architecture.md.
- [archive/cross-domain-check.md](archive/cross-domain-check.md): the cross-domain audit of 2026-08-28.
- [archive/synthetic-rows.md](archive/synthetic-rows.md): the August design for growing Starting state rows; the
  current part is in scaling.md.
- [archive/company-todo.md](archive/company-todo.md): the go-to-market list of 2026-09-10.
- [live-build.md](live-build.md) and [harness-design.md](harness-design.md): pointers kept for old links.
