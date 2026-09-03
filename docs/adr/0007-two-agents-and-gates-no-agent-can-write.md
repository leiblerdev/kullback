# ADR-0007: The Builder and the Examiner are separate agents, and the gates are a package neither can write

Status: accepted 2026-09-02.

Context: Kullback is an agent loop on a shared core, and most of the loop is repair. The agent that builds the Environment could also write its Verifiers, and the code that accepts or rejects an agent's work could live beside that agent.

Decision: the Builder builds and repairs the Environment and never writes a Verifier. The Examiner writes the Verifiers and the probes and never edits the Environment. `gates` holds every accept-or-reject check, contains no model call, is written by people, importable by both agents and writable by none (a raising tool-call hook blocks the write). The two agents share a workdir and one event stream and talk only by messages into each other's queue. The Runner is a tool of both. Two gates keep the Examiner honest: probes are monotone (every probe ever written stays in its pool and every later Verifier must fail it) and loosening is one-directional (a new Verifier may newly pass only the Reference, a frontier re-roll, or a live production Run; the pool grows from the solver, never from a model's opinion).

Why: when the same model produces and grades work, hacking arises on its own once generator and judge share context, and the fix is structural. A Builder that writes its own Verifiers can loosen an atom so a fidelity problem stops mattering, and no gate can tell a loosened Verifier from a right one. A gate inside an agent's package is code that agent can reach.

Considered: one agent with two hats (cheaper by a prompt and a session; the one place a rule cannot replace a boundary); the D79 suite inside the Examiner's package (it builds Verifiers and probes, not gates); a model evaluator for prompt edits (a judge never awards a pass).

Consequences: a second prompt, a second session and messages between two loops on one core. The Examiner writes atoms from what the Runner produced and the traces, never from tool bodies, so atoms bind to the Intent rather than the implementation. Anything a model contributes to the checking side is a probe, never a gate. See decision log D120 to D133.
