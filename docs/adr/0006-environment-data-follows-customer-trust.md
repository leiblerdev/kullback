# ADR-0006: Environment data follows customer trust: traces first, schema next, DB snapshot as the end goal

Status: accepted, 2026-08-26

## Context

The Environment needs rows (customers, orders, tickets, inboxes) for its tools to read and write. Three sources exist: the traces alone, the traces plus the customer's schema and tool definitions, and the traces plus a database snapshot or staging copy. Each closes more of the holes listed in `monitoring-tool/docs/eval-design.md` ("Holes in a traces-only Environment") and each asks the customer for more trust. ADR-0002 fixes that customer data is per-customer and never pooled; it does not say how much of it we may hold.

## Decision

Start with traces only. Every entity any Run in the customer's corpus read becomes a row; S0 for a Run is the corpus state at that Run's start; unseen entities return "not found" and any Candidate Run that hits one is reported "assisted" (Gate B), never verdicted. Tool definitions that already travel inside the traces (the `tools` parameter of each LLM call, MCP `tools/list`) are parsed from day one; they cost the customer nothing.

As the customer's trust increases, ask for more, in this order: schema and API definitions (DDL, OpenAPI), then policy documents and system prompts, then a database snapshot or staging copy. Traces plus a DB snapshot is the end goal: it is what makes the Environment complete enough for both evaluation and training (Scenarios, post-training).

Each rung is asked for with evidence: Gate B's "assisted" share per tool is the number that shows what the next rung would buy.

## Why

- The first customer contact must not begin with a data-sharing agreement. Traces are already leaving their system to a tracing vendor; a snapshot is a different conversation.
- tau2's own `set_state` rebuilds state from recorded writes only, so traces-only is the reference practice and is sufficient for Gate A (replay fidelity concerns recorded Steps only).
- The width problem of traces-only shows up as a measured number (assisted share), so the ask for more data is justified by evidence rather than made up front.
- The end goal is not "traces only forever": a snapshot closes the off-path holes that no amount of trace volume closes, and training on Scenarios needs a world wider than the recorded paths.

## Considered

- Ask for schema or snapshot up front: rejected, it puts the trust cost before any value is shown.
- Traces only, permanently: rejected, the assisted share on off-path Runs would cap the share of traffic that can ever get a Verdict, and Scenario generation would be confined to recorded paths.
- Synthesizing unseen rows from the schema and treating them as real: rejected as a Verdict input; allowed only as tagged synthetic rows whose Runs are reported "assisted".

## Consequences

- The Environment generator takes an optional input at each rung (tool definitions, schema, policy, snapshot) and must work with none of them.
- A snapshot, when it arrives, stays inside the customer's deployment boundary (in-VPC or equivalent) under ADR-0002; the generator must run there.
- The report shows, per tool, the assisted share, so the customer sees what the next rung would unlock.
