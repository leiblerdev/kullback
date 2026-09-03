# ADR-0001: Capture traces by log drain and an in-process SDK wrapper, never a proxy

Status: accepted.

Context: we need LLM traces from customer applications without adding latency or becoming a point of failure in their request path.

Decision: traces come from a log drain (the customer sends traces they already have) and later from an in-process SDK wrapper that logs fire-and-forget and forwards to the real provider. No proxy or gateway, not even as a later phase.

Why: a proxy sits in the hot path, adds a network hop and an uptime liability, and sees only flat calls. An in-process wrapper fails open and can capture the enclosing Run.

Consequence: if live routing ever ships, it runs inside the SDK wrapper, not in a network gateway.
