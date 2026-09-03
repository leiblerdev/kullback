# ADR-0006: Environment data follows customer trust: traces first, schema next, a snapshot as the end goal

Status: accepted 2026-08-26.

Context: the Environment needs rows for its tools to read and write. Traces alone, traces plus schema and tool definitions, and traces plus a database snapshot each close more holes and each ask the customer for more trust.

Decision: start from traces only. Every entity a Run read becomes a row; the starting state of a Run is the corpus state at its start; an unseen entity returns "not found" and a Run that hits one is reported assisted, never verdicted. Tool definitions already inside the traces are parsed from day one. As trust grows, ask in order for schema and API definitions, then policy documents and system prompts, then a snapshot or staging copy. Each ask is backed by the assisted share per tool, which shows what the next rung would buy.

Why: the first contact must not begin with a data-sharing agreement. Rebuilding state from recorded writes is the reference practice and is enough for replay fidelity. A snapshot closes the off-path holes no trace volume closes.

Considered: asking for a schema or snapshot up front (trust cost before value); traces only forever (caps the share of traffic that can get a Verdict); treating synthesized rows as real (allowed only as tagged synthetic rows whose Runs are reported assisted).

Consequences: the generator takes an optional input at each rung and works with none. A snapshot stays inside the customer's boundary under ADR-0002, so the generator must run there. The report shows the assisted share per tool.
