# Assumptions

What the design takes as true without proof, how strong the evidence is, and how we find out (D38). Check it
before you build on one of these; add an assumption the moment you make it, and move it to Tested with the
result once it is tested.

Evidence grades: measured (on comparable data), cited (a published number from another setting), inferred
(reasoned from nearby evidence), none (a guess we proceed on). R numbers are research notes kept outside this
repo (`../research/`).

## Traffic

| # | Assumption | Evidence | How we find out |
|---|---|---|---|
| A1 | Most chat-agent traffic is single-call; an Environment earns its cost on the multi-call tail. | cited, weak (R16, R20) | Calls per Run |
| A2 | Coding agents are not single-call (Claude Code: 6.6 model calls per human message). | measured (R13), cited (R20) | The same histogram |
| A3 | First customers run support, email, calendar and ticketing agents. | none | The first three customers |
| A4 | Tool definitions travel inside the traces; Claude Code JSONL carries none. | inferred (R13, R16) | Runs with and without tool schemas |

## Environment

| # | Assumption | Evidence | How we find out |
|---|---|---|---|
| A8 | Rebuilding the Starting state by inverse replay over Runs ordered by time is sound. | inferred | Replay fidelity itself |
| A9 | Recorded tool errors are enough to infer a tool's error paths. | none | Share of tools with a recorded error |
| A10 | Static per-tool read and write flags suffice; no per-call flag is needed. | cited (R20) | Share of calls through ambiguous tools |
| A27 | Traces hold an error per tool family, so the error encoding is copied, not guessed. | none (R21) | Errors per tool |
| A28 | An Environment built from seen Runs holds on held-out Runs and other paths (D51). | none | Seen against held-out fidelity |

## Verifier and review

| # | Assumption | Evidence | How we find out |
|---|---|---|---|
| A12 | A human audit of a sample of Verifiers, agreement rate published, earns trust. | none (R20) | Disagreement on the first customer |
| A15 | Two domain experts agree on pass or fail; agreement on the reason is lower. | cited (R18) | Audit agreement, verdict and reason apart |
| A29 | Generated Verifiers carry more defects than the 10 to 20 percent of hand-written specs. | inferred (R22) | Defects per Task |

## Simulated user

| # | Assumption | Evidence | How we find out |
|---|---|---|---|
| A22 | Replaying recorded user turns, then simulating from the Intent, is sound. | none | Simulator error on held-out Runs |
| A23 | A user conditioned on real user turns is closer to real users than a persona. | cited (R20) | A paired Turing test; abandon rate |

## Cost and effort

| # | Assumption | Evidence | How we find out |
|---|---|---|---|
| A24 | A first Environment for a customer takes 20 to 30 expert hours. | inferred (R12) | Time the first two customers |
| A25 | Rebuilding tau2 domains from their own trajectories is a fair test of the generator. | inferred | Run it on Claude Code JSONL too |

## Tested

None yet. A30 (required atoms stabilise within 10 re-runs) and A31 (two agentic judges catch most defects)
tested mechanisms that were replaced: Verifiers are now written from the Intent (D330) and judged one item at a
time (D328).
