# Capture traces via log-drain + fail-open SDK wrapper, not a proxy

We need to collect LLM API traces from customer applications without becoming a single point of failure or adding latency to their calls. We decided to capture traces through (1) a log drain for initial validation (customers send us traces they already have) and (2) an in-process SDK wrapper that logs fire-and-forget and forwards to the real provider. We explicitly rejected a proxy/gateway, even as a later phase.

Why not a proxy: a proxy sits in the hot path, adds a network hop, reintroduces the uptime/SPOF liability, and only sees flat calls (no Run tree). An in-process SDK wrapper logs as a side effect, so logging can fail without affecting the customer's call (fail-open), and it can capture the enclosing Run context.

Consequence: if we ever ship Live Routing, it happens inside the SDK wrapper (in-process model choice), not in a network gateway, because we've committed to never being in the network path.
