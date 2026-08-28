# Simulated users for re-executing multi-turn agent runs

Research sweep, 2026-08-26. Source: web research agent. Topic 8.

## 1. State of the art

**tau-bench.** User is an LLM prompted with identity, intent, preferences; reward = DB state match x required outputs; pass^k for consistency. Known simulator limitations: typos, ambiguities, weak long-context memory ([arXiv 2406.12045](https://arxiv.org/abs/2406.12045)).

**tau2-bench.** GPT-4.1 user with three persona variants; in telecom the user holds its own tools so "user behavior is constrained by the available tools and the observable state": simulator error rates 40% (12% critical) retail, 47% (13% critical) airline, 16% (6% critical) telecom. **No-user mode** (agent gets a ticket and controls all tools) vs dual-control: pass^1 52% vs 34% for GPT-4.1; the gap is communication/coordination overhead ([arXiv 2506.07982](https://arxiv.org/abs/2506.07982), [evaluation.md](https://github.com/sierra-research/tau2-bench/blob/main/docs/evaluation.md)).

**APIGen-MT.** Verified task blueprints, then simulated human-agent interplay; the simulated human is persona-conditioned, "incrementally reveals task details," is unaware of APIs, stabilized with Best-of-N (N=4) plus self-critique; trajectories accepted only if final state matches ([arXiv 2504.03601](https://arxiv.org/abs/2504.03601)).

**tau-voice (Sierra, Mar 2026).** 278 tau2 tasks, voiced user, noise, full-duplex; 79-90% of failures attributed to the agent ([arXiv 2603.13686](https://arxiv.org/abs/2603.13686)).

Others: UserBench ([arXiv 2507.22034](https://arxiv.org/abs/2507.22034)), USimAgent ([arXiv 2403.09142](https://arxiv.org/abs/2403.09142)), IntellAgent ([arXiv 2501.11067](https://arxiv.org/abs/2501.11067)).

## 2. Fidelity evidence (2026 turned skeptical)

- **Lost in Simulation (Jan 2026).** Agent success moves 67.0% to 75.9% depending only on which LLM plays the user. Poor calibration vs humans (ECE 15.1). Simulated users ask questions in 18.8% of turns vs 9.8% humans; in human dialogs the user is the primary failure source 62.2% of the time ([arXiv 2601.17087](https://arxiv.org/abs/2601.17087)).
- **Mind the Sim2Real Gap (Mar 2026).** 451 humans, 165 tasks, 31 simulators. Simulators inflate success to ~77.8% vs 63.6% human. Simulators are "too uniform and cooperative," "front-load complete information," "quietly pivot rather than push back"; 29% of human turns are three words or fewer vs 1% for GPT-4o. "Higher general model capability does not necessarily yield more faithful user simulation" ([arXiv 2603.11245](https://arxiv.org/abs/2603.11245)).
- **Simulated Customers Never Walk Away (Jun 2026).** 2,790 production sales conversations; simulators inflate non-buyers (+0.40 depth bias); resistance halves. Uses a **teacher-forced probe protocol**: at 30/60/90% of real user turns, the simulator gets the identical real prefix and produces the next turn, scored against the real turn ([arXiv 2606.20708](https://arxiv.org/abs/2606.20708)).
- **Grounding in real transcripts.** RealUserSim: 7,275 behavior profiles from WildChat in "Command + Example" format; grounded personas lower agent success 3.2-3.5 points ([arXiv 2605.20204](https://arxiv.org/abs/2605.20204)). SWE-Together: a "reactive" simulator from 11,260 real coding sessions; extracts objective, constraints, and "intervention conditions grounded in the original follow-up messages," then after each agent turn decides to speak or stay silent ("trajectory-conditioned rather than scheduled") ([arXiv 2606.29957](https://arxiv.org/abs/2606.29957)). Persona Policies ([arXiv 2605.12894](https://arxiv.org/abs/2605.12894)); RUSE drops frontier agents up to 57% ([arXiv 2606.01815](https://arxiv.org/abs/2606.01815)).
- **Evaluating the simulator.** SimEval-IR: human-likeness discriminator has r=+0.09 with system-ranking validity ([arXiv 2604.27878](https://arxiv.org/abs/2604.27878)).

## 3. Replaying recorded turns, detecting divergence, bounding variance

- openevals `create_llm_simulated_user(fixed_responses=...)` returns recorded messages positionally, then hands over to the LLM; no check that the recorded turn still fits ([source](https://github.com/langchain-ai/openevals/blob/main/python/openevals/simulators/prebuilts.py)). deepeval ConversationSimulator accepts initial turns ([docs](https://deepeval.com/docs/conversation-simulator)).
- No paper names a "semantic applicability" test. Closest primitives: teacher-forced probe; SWE-Together's anchored intervention conditions ("replay the intent, not the text"); old TOD lesson that gold user turns after divergence produce rankings that disagree with human evaluation ([arXiv 2005.07362](https://arxiv.org/abs/2005.07362)).
- Bounding variance: multiple simulator LLMs with ranking-stability reporting; pass^k; Best-of-N on the user side; tool-constraining the user (13% to 6% critical errors); no-user mode as a simulator-free control.

## 4. Outcome evaluation without a simulator

tau benchmarks score observable outcomes (DB state, env assertions, communicate_info, optional NL assertions), none needing a user once the transcript exists. tau2 no-user mode is a well-defined, lower-variance measurement of reasoning and tool use.

## Recommended policy

1. **Single user turn: never simulate.** Replay the recorded message verbatim, run the candidate, score with the same outcome checks. The only fully deterministic comparison.
2. **Multi-turn: replay recorded turns while they still apply, then fork.** Positional replay up to the first agent turn whose meaning differs. Applicability gate per turn: a judge sees the original agent turn, the new agent turn, and the recorded user reply, and answers "does this reply remain plausible?" and "does it contain information the new agent did not ask for or has already received?" Log the fork index.
3. **After the fork, simulate from intent, not text.** Seed SWE-Together style: objective, constraints, known facts, disclosed facts, intervention conditions grounded in real follow-ups; withhold undisclosed facts; forbid inventing. Give the simulator tools and state where possible.
4. **Expect the simulator to be too nice.** Inflation of 9-14 points. Run at least two simulator backbones; never optimize against a single simulator.
5. **Score the outcome, not the dialogue.** pass^k over at least 4 trials.
6. **Report simulator uncertainty.** Fork index, share of turns replayed vs simulated, results per backbone, spread. Headline as an interval. Flag "simulator-sensitive" runs where ranking flips and exclude them from aggregate claims.
