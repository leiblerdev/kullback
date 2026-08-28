# 20. Retry of research coverage gaps

Second pass, 2026-08-26, on the twelve sources and questions the earlier reports could not reach. WebSearch was still unavailable (session budget exhausted before the task); everything is from direct fetches (WebFetch, curl of arXiv HTML search pages, GitHub API and raw files, PyPI wheels, vendor docs). Each section ends with a verdict against the design position it tests.

# Retry research report: traces-to-environment tool design

**WebSearch status:** did not work. The session's WebSearch budget was already exhausted (200/200) before this task; my one test query was refused. Everything below comes from WebFetch (20 calls) and curl (about 25 arXiv search-page queries, 20 arXiv abstract/full-text fetches, GitHub API and raw source files, PyPI, and roughly 40 vendor/doc pages). arXiv's API and Semantic Scholar were rate-limited, so arXiv's HTML search pages were used instead.

---

## 1. OpenHands event model

**Findings**
- V0 (Python, tag 0.40.0; the repo moved from All-Hands-AI/OpenHands to OpenHands/OpenHands and `main` no longer contains `openhands/events`): `Event` carries `id`, `timestamp`, `source` (agent/user/environment), `cause`, `timeout`, `llm_metrics`, `tool_call_metadata` (`function_name`, `tool_call_id`, `model_response`, `total_calls_in_response`), `response_id`. `Action` adds `runnable`, `ActionConfirmationStatus`, `ActionSecurityRisk` (UNKNOWN/LOW/MEDIUM/HIGH). `Observation` is just `content: str`. Sources: https://raw.githubusercontent.com/OpenHands/OpenHands/0.40.0/openhands/events/event.py, .../action/action.py, .../tool.py, .../observation/observation.py
- V1 agent SDK: `Event` = `id`, `timestamp`, `source` (agent/user/environment/hook), `parent_id` (conversation tree). `ActionEvent` = thought, reasoning_content, thinking_blocks, action, tool_name, tool_call_id, tool_call, llm_response_id, security_risk, critic_result, summary. `ObservationEvent` = observation, action_id, tool_name, tool_call_id, extended_content. `ConversationStateUpdateEvent` is key-value conversation state, not environment state. https://raw.githubusercontent.com/OpenHands/agent-sdk/main/openhands-sdk/openhands/sdk/event/base.py, https://docs.openhands.dev/sdk/arch/events.md
- Read/write semantics exist only as static MCP-style `ToolAnnotations` on the tool definition: `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`. https://raw.githubusercontent.com/OpenHands/agent-sdk/main/openhands-sdk/openhands/sdk/tool/tool.py

**Verdict: partially supports / silent.** OpenHands has a per-tool read-only and idempotency hint (static, not per step), a security-risk field, and a parent/cause link. It has no pre/post state, no entity ids, and no per-step determinism flag. Nothing contradicts the ATIF-plus-flags position; it just does not exist there.

## 2. Nomic Atlas and Lilac near-duplicate thresholds

**Findings**
- Nomic Atlas: embedding-based; each point labeled singleton, retention candidate, or deletion candidate; configurable `duplicate_cutoff`, default **0.1** ("smaller thresholds result in duplicate clusters containing datapoints that are closer to exact matches"); on by default; duplicates are flagged, not collapsed or weighted. https://docs.nomic.ai/atlas/capabilities/duplicate-detection (the /data-maps/guides/dedup URL 404s)
- Lilac (wheel 0.3.9 from PyPI; GitHub repo README is now defaced and docs.lilacml.com is down): `NearDuplicateSignal` uses MinHash LSH, same `cluster_id` if Jaccard similarity above `threshold`, default **0.85**. https://files.pythonhosted.org/packages/19/84/b046e05dff09724a3c671aae4fde6423b39b8d1bf026deb8d6eb1e3273fc/lilac-0.3.9-py3-none-any.whl (file `lilac/signals/near_dup.py`)

**Verdict: silent on weighting, supplies thresholds.** Neither tool assigns a weight; both cluster and flag. Usable defaults: 0.1 embedding cutoff (Nomic), 0.85 Jaccard on MinHash (Lilac). Magpie's min-neighbor-distance rule was not re-fetched.

## 3. W&B Weave tracing to datasets

**Findings**
- A Call captures "Input arguments, Output value, Timing and latency, Parent-child relationships (for nested calls), Any errors that occur." https://docs.wandb.ai/weave/guides/tracking/tracing (weave-docs.wandb.ai redirects here)
- Trace-to-dataset: `Dataset.from_calls([call1, call2])` in the SDK; in the UI, select calls, "Add selected rows to a dataset", map fields; separate flow "Add agent messages to a dataset" for agent turns and tool calls. https://docs.wandb.ai/weave/guides/core-types/datasets
- No mention of tool state, side effects, or environment snapshots anywhere in either page.

**Verdict: supports** the premise that mainstream trace stores hold span I/O only, so state must be inferred from traces.

## 4. Published tool-call depth per request

**Findings**
- Datadog: "59% of agentic application requests only made a single service call, while only 18% of end-to-end agentic application requests made three or more service calls." This is service calls, not tool calls. https://www.datadoghq.com/state-of-ai-engineering/
- OpenAI "AI in the enterprise" PDF: no tool-call or step-depth statistics at all (only a mention of Operator). https://cdn.openai.com/business-guides-and-resources/ai-in-the-enterprise.pdf (HTML page returns 403)
- Anthropic Economic Index (Sept 2025, Jan 2026): API records are "single input-output pairs... with no metadata linking them to prior exchanges", so no depth stats. https://www.anthropic.com/research/anthropic-economic-index-september-2025-report, https://www.anthropic.com/research/anthropic-economic-index-january-2026-report
- Anthropic "Measuring AI agent autonomy in practice": analysis done per tool call because "we have no reliable way to associate independent requests to our API into sessions"; sample of 998,481 API tool calls; 87% of tool calls on minimal-complexity tasks have human involvement vs 67% for high-complexity; Claude Code median turn ~45 s, 99.9th percentile turn rose from under 25 min to over 45 min (Oct 2025 to Jan 2026). https://www.anthropic.com/research/measuring-agent-autonomy
- Google: nothing found.

**Verdict: silent, with one weak support.** Only Datadog gives a number and it measures service calls. Anthropic explicitly cannot measure depth on the API; its Claude Code data shows a long tail of multi-step sessions, which qualifies "most traffic is single-call" for coding agents but does not contradict it for API traffic.

## 5. HUD (and Plato) environment construction

**Findings**
- HUD docs moved to docs.hud.ai (docs.hud.so returns 429). Environments are code: `Environment` with `@env.template()` generators, capabilities (ssh, mcp, cdp, rfb, robot), Docker images. Graders: `exact_match`, `contains`, `numeric_match`, `BashGrader`, `LLMJudgeGrader`, `combine`; "Grade the world... outcome verification". https://docs.hud.ai/llms.txt, https://docs.hud.ai/v6/reference/graders.md
- Trace types: `Trace.steps` of `AgentStep` / `ToolStep` (MCPToolCall paired with MCPToolResult). https://docs.hud.ai/v6/reference/types.md
- File tracking: post-setup content-hash snapshot plus sampled agent diffs (2 s interval) so the viewer can "rebuild any file at any point in a trace". https://docs.hud.ai/platform/file-tracking.md
- Verifier environments: grading runs on a separate substrate the agent never sees. https://docs.hud.ai/v6/experimental/verifier-environments.md
- "Replay every graded attempt" means trace playback, not environments built from traces. https://docs.hud.ai/platform/introduction.md
- Plato: sims are VM images of real apps (espocrm, gitea, ubuntu-vm); `reset()` starts mutation logging; `evaluate()` in MUTATION mode checks declared DB/file changes, OUTPUT mode checks a JSON schema. https://docs.plato.so/concepts/overview.md

**Verdict: silent** on traces-only environments with "not found" for unseen entities. Both build from real software, and both verdict on state mutations (supports end-state verification). HUD's snapshot-plus-diff file tracking is a pre/post-state precedent keyed on file path.

## 6. Vendor environment construction and human review rates

**Findings**
- Veris AI: "A simulated copy of your systems, data, APIs, and users"; "No production data required... Use synthetic data or bring your own." Blog (Dec 2025): "Veris platform can use the production log to reconstruct and expand that edge case into a set of similar scenarios and new targeted evaluation rubrics... without any humans in the loop." July 2026 post: verification environment where every system "is replaced by a mock that answers the way the real one would." No review rate. https://www.veris.ai/, https://www.veris.ai/blog/never-waste-a-good-failure-how-veris-ai-turns-production-incidents-into-self-improving-agents, https://www.veris.ai/blog/the-loop-is-the-easy-part
- Decagon: "auto-generate tests directly from AOPs and production conversations"; no review rate. https://decagon.ai/blog/the-next-generation-of-simulations
- Sierra: simulations of agent vs "mock user personas" with an LLM judge; "tens of thousands of conversations" daily; no review rate. https://sierra.ai/blog/simulations-the-secret-behind-every-great-agent, https://sierra.ai/blog/voice-sims-test-agents-in-real-world-conditions-before-they-talk-to-your-customers
- Snorkel: "Calibrated expert review", rubrics distilled "into programmatic graders", "Adjudication and provenance: Author, multi-reviewer"; no rate. https://snorkel.ai/
- Turing: "MCP environments with APIs, tool calls, and SME-built policies, schemas, and realistic seed data"; no rate. https://www.turing.com/frontier-ai/rl-environments
- Scale MCP-Atlas: 1,000 tasks "written and verified by human experts"; authoring reviewers used a three-way checklist; scoring is an LLM judge over claim-level rubrics; no human-vs-judge agreement in the paper text. https://scale.com/research/mcp-atlas, https://arxiv.org/abs/2602.00933
- Mechanize: environments plus graders for coding agents; no data-source detail. https://www.mechanize.work/
- Halluminate: Westworld simulated websites (Best Buy clone), sim-to-real transfer study, finance benchmarks; no rate. https://www.halluminate.ai/research
- Fleet, Matrices: JS-only marketing pages, no content. Plato: see gap 5.

**Verdict: supports** "no vendor publishes an audit rate with agreement." Note Veris and Decagon both already market building scenarios from production logs, so "from traces" is not a differentiator on its own; the audited verifier is.

## 7. TauForge

**Findings:** arXiv search: 0 results for "TauForge" and "tau forge". GitHub: only `MartynJJ/tauforge` (a crypto DeFi trading suite); "tau-forge" hits are Tauri app repos. PyPI `tauforge` and `tau-forge`: not found. tauforge.com returns 200 with empty body; tauforge.ai unreachable. https://github.com/MartynJJ/tauforge, https://pypi.org/pypi/tauforge/json

**Verdict: not found.** No AI-related TauForge exists publicly.

## 8. User simulators built from real logs

**Findings**
- Turing-RL (Jun 2026): trains user simulators against real user histories (chat and Reddit) with an LLM Turing reward; beats SFT, similarity, and log-prob baselines on LLM and human evaluation. No hand-written-persona baseline number extracted. https://arxiv.org/abs/2606.19336
- "Simulated Customers Never Walk Away" (Jun 2026): 2,790 production conversations with an LLM sales agent, 793 with verified payment outcomes. Persona-prompted simulators reproduce eventual buyers (depth bias +0.09) but push non-buyers toward purchase (+0.40, d=0.38, p<0.001), halving expressed resistance (25.1% to 13.5%) and nearly doubling deliberation (21.9% to 40.1%). Cites RealUserSim (7,275 WildChat personas, paired-trajectory Turing test) and a persistent "realism gap": simulators "too verbose, too uniformly polite, too patient." https://arxiv.org/abs/2606.20708
- TraitBasis / tau-Trait (2025, rev. 2026): steering user traits causes 2 to 30% agent degradation. https://arxiv.org/abs/2510.04491
- RecVerse (Aug 2026): shopping simulator trained on logged sessions with trajectory-level optimization. https://arxiv.org/abs/2608.20707
- PALATE (Jul 2026): per-user simulators trained from data plus personalized rubrics. https://arxiv.org/abs/2607.27816

**Verdict: supports, with a caveat.** Hand-written personas measurably diverge from real users, and 2026 work trains simulators on real logs. But no paper gives a clean "log-trained vs hand-written" fidelity number for tool-using agent evals, and 2606.20708 shows even log-grounded simulators miss decision fidelity (users who disengage).

## 9. Process reward models for evaluation

**Findings**
- AgentPRM (Choudhury, Feb 2025): training framework on ALFWorld; InversePRM from demonstrations. https://arxiv.org/abs/2502.10325
- AgentPRM (Nov 2025): PRM for Best-of-N and test-time search; results are about improving policy performance, not evaluation accuracy. https://arxiv.org/abs/2511.08325
- TRACE (Jul 2026) and TRCA (Aug 2026): turn-level credit assignment for RL, always combined with terminal outcomes. https://arxiv.org/abs/2607.13988, https://arxiv.org/abs/2608.16156
- No paper found that reports PRM plus end-state beating end-state alone as an evaluation metric.

**Verdict: supports** "end-state only for eval now, PRM later." PRMs are currently a training and search tool.

## 10. Published human-vs-generated-verifier agreement

**Findings**
- "Benchmarking the Benchmarks" (Jun 2026): 496 expert-reviewed tasks across BFCL v4, tau2-bench, LiveMCPBench, MCP-Atlas; 92 evaluator-human disagreements = **18.5% misalignment** (9.8% tau2 retail, 13.5% MCP-Atlas, 20.0% BFCL v4, 30.5% LiveMCPBench); LiveMCPBench 23 repeats range 57.9% to 76.8%. Failure modes: brittle state matching, trajectory lock-in, substring communication checks, rubric drift. Proposes Tool-Veritas: "deterministic state gates with restricted LLM fallback." https://arxiv.org/abs/2607.02577
- RubricForge (Jun 2026): rubric induced from labeled trajectories; agreement with environment reward accuracy 0.774 vs G-Eval 0.726, Cohen kappa +0.092 vs +0.026, not significant (McNemar p=0.248); 173 tau-bench and 160 WebShop trajectories. This is judge-vs-env-reward, not human-vs-verifier. https://arxiv.org/abs/2608.13564
- AutoForge, ScaleEnv, EnvFactory, AgentOmnia: full-text greps found no human verification rate; they rely on procedural tests, executable action verification, and program/solver verifiers. https://arxiv.org/abs/2512.22857, https://arxiv.org/abs/2602.06820, https://arxiv.org/abs/2605.18703, https://arxiv.org/abs/2607.23124
- Kimi K2: "An LLM-based judge evaluates each trajectory against the task rubrics"; no human rate. https://arxiv.org/abs/2507.20534
- MCP-Atlas: human-expert-verified tasks, no agreement rate (see gap 6).

**Verdict: mostly supports, with two comparators to cite.** No generator publishes an audit rate, but the 18.5% audit and RubricForge's kappa are the published baselines a 10% audit would be compared against.

## 11. Output perturbation of tool results

**Findings**
- ReliabilityBench (Jan 2026): semantically equivalent task perturbations at intensity epsilon plus fault injection ("timeouts, rate limits, partial responses, schema drift"); "action metamorphic relations that define correctness via end-state equivalence rather than text similarity"; success 96.9% at eps=0 to 88.1% at eps=0.2. https://arxiv.org/abs/2601.06112
- Judge Reliability Harness (Mar 2026): perturbs responses (formatting changes, paraphrasing, verbosity, flipped labels) to stress LLM judges; "No judge that we evaluated is uniformly reliable." https://arxiv.org/abs/2603.05399
- TraitBasis perturbs the user side (gap 8).
- Nothing found that perturbs tool-output formatting, order, or minor values specifically to test verifiers.

**Verdict: partially supports.** Adjacent work exists (schema drift, partial responses, judge-side formatting perturbation, end-state metamorphic relations), but tool-output cosmetic perturbation as a verifier robustness test appears unclaimed.

## 12. Canonicalization for replay

**Findings**
- tau2-bench `Environment.set_state(strict)`: only tools with `mutates_state=True` (inferred from `ToolType.WRITE`) are re-executed; reads and thinks are skipped "to avoid re-execution and non-deterministic output comparison issues"; outputs compared after `json.loads` (dict equality, so key order is ignored) else raw string; `strict=False` downgrades mismatch to a warning for "cosmetic drift... `25` vs `25.0`"; hallucinated tool names replay as no-ops; DB compared by hash of `model_dump` (no float or timestamp normalization). https://raw.githubusercontent.com/sierra-research/tau2-bench/main/src/tau2/environment/environment.py, .../environment/toolkit.py, .../evaluator/evaluator_env.py, https://raw.githubusercontent.com/sierra-research/tau2-bench/main/docs/evaluation.md
- BFCL multi-turn: `state_checker` compares every public attribute of each backend instance with plain equality (no normalization); `response_checker` requires ground-truth responses to be an unordered subset of the model's execution results; AST checker `standardize_string` strips spaces and punctuation and lowercases, int auto-converts to float, parallel calls matched without order. https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/multi_turn_eval/multi_turn_checker.py, .../ast_eval/ast_checker.py
- AppWorld: unit tests over DB state with `assert_plus(tolerance=pytest.approx, round_to=int or second/minute/hour/day, ignore_case, strip, merge_white_space, ignore_order)`; `answer_to_text` rounds floats to 2 decimals and sorts lists; `changed_model_names` ignores `supervisor.Task`, `admin.PaymentCard`, `amazon.BrowsedProduct`; `no_op_fail` tests catch unexpected writes. https://raw.githubusercontent.com/stonybrooknlp/appworld/main/src/appworld/common/evaluation.py, .../collections/models.py, .../evaluator.py

**Verdict: supports the canonicalization layer, contradicts "100% replay fidelity."** tau2 stopped replaying reads entirely and still needed `strict=False`; the validity audit measured 9.8 to 20% evaluator-human disagreement on these deterministic benchmarks after their normalization. No source evidences 100%.

---

## Still unreachable
- docs.hud.so (429; superseded by docs.hud.ai, which worked)
- openai.com/business/ai-in-the-enterprise HTML (403; PDF fetched instead)
- docs.lilacml.com (down) and github.com/lilacai/lilac (defaced README); Lilac source obtained from the PyPI wheel
- docs.nomic.ai/atlas/data-maps/guides/dedup (404; capabilities page used)
- plato.so, matrices.ai, fleetai.com marketing pages (JS-only or 404); Plato docs worked
- scale.com/research index and Snorkel RIFT paper page (JS-only)
- export.arxiv.org API ("Rate exceeded"), Semantic Scholar API (429), GitHub code search (auth required)
- Any OpenAI or Google statistic on tool-call depth (none found)
- Human-review rates inside Kimi K2, AutoForge, ScaleEnv, EnvFactory, AgentOmnia (not present in text)

## Where this changes the design
- Replace "100% replay fidelity after canonicalization" with a measured fidelity target; tau2's own code skips reads and tolerates cosmetic drift, and the validity audit shows deterministic checkers still disagree with humans 10 to 20% of the time. Evidence is strong.
- Publish the 10% audit against the 18.5% evaluator-human misalignment baseline (2607.02577) and RubricForge's kappa; these are the only comparators, and they make a lower misalignment rate a concrete claim. Evidence is moderate (one audit, four benchmarks).
- Steal ReliabilityBench's "action metamorphic relations" (end-state equivalence under perturbation) as the formal framing for output perturbation and for the canonicalization layer; nobody has applied it to tool-output formatting yet. Evidence is moderate.
- "Built from production traces" is not a differentiator: Veris and Decagon already market it. The differentiator is the executable environment plus audited, non-LLM verifiers. Evidence is strong for the marketing claim, weak on what they actually ship.
- Carry MCP-style `readOnlyHint` / `idempotentHint` / `destructiveHint` as the read/write and determinism flags; OpenHands and tau2 (`mutates_state`) both converge on a per-tool static flag, and interop is free. Evidence is strong.
- Simulated users: replaying recorded turns then simulating from intent is consistent with 2026 work, but 2606.20708 shows log-grounded simulators still fail on decision fidelity (users who disengage). Add a disengage/abandon behavior to the simulator and report it. Evidence is one paper on one domain.
- Adopt dedup defaults from tools that exist (Nomic 0.1 embedding cutoff, Lilac 0.85 Jaccard) rather than inventing a weight scheme; the Magpie-style weight has no tool precedent found here. Evidence is weak (two defaults, no comparison).
- The "most traffic is single-call" premise rests on one Datadog service-call figure; Anthropic's Claude Code data shows a fat tail of 25 to 45 minute turns. Keep the premise for API chat agents, drop it for coding agents. Evidence is weak.