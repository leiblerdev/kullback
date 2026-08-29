# Benchmark landscape: references beyond tau2 for measuring Environment quality

Written 2026-08-29 from three parallel research passes (AppWorld; ToolSandbox and Toolathlon; a survey of 25 other candidates) plus a look at what the first live retail build actually left behind. Purpose: decide which reference environments to add so the "high quality Environment" claim rests on more than one benchmark, without tuning the Builder to any of them (D51). Full notes per benchmark are in the session scratchpad; this file keeps the parts that bear on decisions.

## 1. What a reference has to give us

Our signal, per D62, has two halves and the harness only ingests traces:

1. Fidelity: replay the recorded calls against our generated tool bodies and against the real tool on the real seed state, compare result and state change (`scripts/env_fidelity.py`, retail floor 64.3% before the five prompt fixes).
2. Verdict agreement: our derived Verifier against the benchmark's own grader on the same Runs (`scorecard._agreement`, reads `reference_verdicts.json`).

So a reference needs, in order of how hard it is to substitute: (a) an executable environment with a resettable seed state we can call from Python, (b) a grader whose per-Run verdicts we can read, (c) published traces with tool calls and results in a shape our ingest can read, (d) ideally a simulated user, since `user_rules` and `judge_lessons` have nothing to validate against on single-instruction benchmarks.

## 2. Where the retail build actually stands on those two halves

Fidelity is measured. Verdict agreement is not, and cannot be yet:

- `.work-retail/gates.json` records `derive_verifier` as passed with `verifiers: 0`. `verifiers/` is empty and every one of the 205 tasks is `reference_confirmed: false`.
- Cause: `_verifier_stage` (`build.py:394`) reads `runs/<task.id>/*.jsonl`, which only the Runner's `_run_task` (`build.py:520`) ever writes. The Builder holds 456 Traces in `traces/` and each Task points at them through `run_ids`, but nothing converts a Trace into the Run that `derive_verifier` consumes. Every Task takes the "no paths" branch silently.
- The gate passed because it counts only Tasks that were confirmed and then failed D79. Zero confirmed means zero failures. This contradicts the standing rule at decision-log line 302 ("a miss without a reason is a gate failure"). No test drives the stage with real Tasks; `test_build.py` checks only its `code_version`.
- `intent.py` exists but no intent stage is wired; `Task.intent` is null everywhere, so the D79 leak gate runs without intent text.
- The reference side is ready: `grader/` holds the stripped tau2 evaluation criteria per trace and the raw export carries `reward_info` per simulation (retail gpt-5.2: 330 pass, 126 fail).

Any blast-radius expansion inherits this: a second benchmark would also produce fidelity only.

## 3. Candidates

Columns: state, simulated user, grader, published traces, install, licence, fit. "Traces" means agent runs with tool results, not task files.

| Benchmark | Domain | State and reset | Sim. user | Grader | Published traces | Install | Licence | Fit |
|---|---|---|---|---|---|---|---|---|
| tau2-bench (current) | retail, airline, telecom | JSON/TOML seed db, in-memory, reseed per task | yes | required actions plus end db equality, NL assertions, env assertions | yes, Sierra raw export with `reward_info` | pip, offline | MIT | in use |
| tau-bench original | retail 115, airline 50 | seed db, in-memory | yes | end db state vs goal state, pass^k | yes, `historical_trajectories/` in repo | pip, offline | MIT | traces and env |
| tau3-bench (tau2 repo v1.0.1) | tau2 domains with 75+ task fixes, plus banking (tau-Knowledge) and voice | same lineage | yes | same lineage | not confirmed; runs saved locally to `data/simulations/` | uv, Python 3.12+, offline | MIT | env only, traces unconfirmed |
| FraudBench | banking fraud, adversarial caller | tau-Knowledge banking env: 125 users, 126 accounts, 17 tools, 698 policy docs | yes, attacker | deterministic prohibited-action checks plus LLM judge over 334 hand-written assertions | task JSON with caller scripts and db init; run transcripts not verified | cloud LLMs via LiteLLM | CC BY 4.0 | env and task files; traces unverified |
| AppWorld | 9 consumer apps, 457 APIs | SQLite, 101 tables, base db plus per-task diff, frozen clock | no, one instruction plus queryable Supervisor | state diff: expected changes subset of diff subset of expected plus allowed; ORM assertions; no_op labels | yes, `appworld download experiment-outputs`, `api_calls.jsonl` per task; third-party rollouts on HF | pip, Python 3.11+, offline; tests and apps in encrypted bundles | Apache 2.0 with encrypted redistribution clause | traces and env; no user turns |
| MCPMark | Notion, GitHub, Postgres, filesystem, Playwright; 127 tasks | live backends, reset by template re-import | no | per-task `verify.py` end-state check over API | yes, HF `Jakumetsu/mcpmark-trajectory-log` | Notion workspace, GitHub tokens, Docker Postgres, Playwright; not offline | Apache 2.0 | traces and env; live services |
| chi-Bench | healthcare admin: prior auth, utilization mgmt, care mgmt | 20 simulated apps as MCP servers, 87 tools | no | LLM verifier per stage, scorecard.json | yes, `trajectory.jsonl.zst` per submission | Docker, gated HF, Anthropic key mandatory | Apache 2.0 code | traces and env; LLM grader |
| EnterpriseOps-Gym | Calendar, CSM, Drive, Email, HR, ITSM, Teams, hybrid; 1,150 tasks | Docker sandbox, 164 tables, 512 tools over MCP | no | SQL verifiers on final db, 5.3 conditions per task | not confirmed | Docker per domain | Apache 2.0 | env only |
| STATE-Bench | travel, support, shopping; 450 tasks | pre-populated dbs | yes, low-variance | deterministic state assertions plus LLM judge for procedural tasks | training-split trajectories referenced; eval traces unclear | uv, keys | MIT | env only (probably) |
| WorkBench | email, calendar, CRM, PM; 690 tasks | five CSV dbs | no | final db vs ground truth, order-independent | no by default; `--log_traces` generates; 2026 release publishes per-task verdicts | uv, offline | MIT | env only |
| ToolSandbox | phone assistant: contacts, messaging, reminders, settings | Polars DataFrames with full history, per scenario | yes, on-policy | milestone/minefield DAG matched to snapshots, similarity scored, guardrails | no (`data/` gitignored) | pip, offline mostly | Apple source-available | env only; results are stdout text |
| Toolathlon | 32 real apps over 33 MCP servers, 600+ tools | live third-party services seeded per task, cleaned after | class exists, mostly single instruction | per-task Python scripts, exit code, exact strings plus live API reads | yes, HF `hkust-nlp/Toolathlon-Verified_Trajectories`, 3,888 runs, 23 GB, no-training-use | Docker plus six live accounts | unstated; data CC BY 4.0 no-training | traces only usable; env is the internet |
| TheAgentCompany | GitLab, ownCloud, Plane, RocketChat | real self-hosted services in Docker | yes, NPC colleagues | final service state plus checkpoints | not clearly located | Docker Compose, 30+ GB | MIT | env only, heavy |
| AgentDojo | banking, Slack, workspace, travel; 97 tasks | in-repo environment objects | no | `utility()` over mutated state plus `ground_truth()` call sequence | results published; raw tool-level traces unconfirmed | pip, offline | MIT | env only, small |
| CRMArena-Pro | CRM, 19 tasks | live Salesforce org, 25 objects | yes | exact match, F1, LLM judge; not a state diff | no | Salesforce sandbox | CC BY-SA 4.0 | env only, heavy |
| BFCL v3/v4 multi-turn | 8 synthetic APIs | in-process backend classes from `initial_config` | scripted turns | backend state compare plus call-sequence subset | responses published | pip, offline | Apache 2.0 | env only, toy state |

Ruled out (do not re-research): Gecko (a method, not a benchmark), T1-Bench (unreleased), MCP-Bench, MCP-AgentBench, MCPToolBench++, ACEBench, ComplexFuncBench, Nexus, NESTFUL, MCP-Atlas (all action-matching or LLM-judge, no persistent world or read-only), OSWorld, Windows Agent Arena, HealthAdminBench (GUI grading), FHIR-AgentBench (read-only QA), E-Bench (right grader, consumer domain, no public code), "BankingBench" (does not exist; tau3-Banking and FraudBench are the real things).

## 4. What each shortlisted reference would stress in the Builder

- tau-bench original: same family as tau2; cheapest second point but the weakest test of generalization. A Builder tuned on tau2 would be expected to pass it.
- tau3 domains: the 75 task-quality fixes change some rewards relative to the tau2 export we hold. If tau3 replaces tau2 as the reference, `grader/` and `reference_verdicts` must be regenerated from tau3, not reused.
- AppWorld: relational state with foreign keys, per-task db diffs rather than one shared `db.json` (D56 assumes one), a frozen clock, integer ids, and an allowed-changes set in the grader (our D43 required/allowed atoms already have this shape). Single user turn means `user_rules` gets nothing.
- WorkBench and EnterpriseOps-Gym: closest domain match to the customer list (CRM, email, calendar, ITSM), SQL or CSV state, no published traces, so traces would be self-generated under D60 controls.
- MCPMark and Toolathlon: tool results are heterogeneous per vendor, no central db, verification means re-querying a live API. They test `ingest` and `mine` on messy results but cannot be replayed offline; our fidelity script has no real tool to call.
- ToolSandbox: results are captured stdout, not JSON; `mine` would have nothing structured to mine. Useful only for grader ideas (minefields, guardrails).

## 5. Where the sources disagree, and where they disagree with earlier positions

- The survey agent ranks MCPMark first as "the closest structural match to tau2". The ToolSandbox/Toolathlon agent says live-service worlds (Toolathlon, MCPMark-shaped) "do not match tau2's seeded-db shape our harness is built to mine" and should be stress tests, not references. Both are describing the same property (real backends, deterministic `verify.py`) and drawing opposite conclusions. The fact that decides it: our fidelity measurement needs a real tool we can call on a seed state in a subprocess. MCPMark cannot give that without live Notion and GitHub accounts.
- The survey labels FraudBench "traces and env both available". What it verified is task JSON with caller scripts and db initialization. Run transcripts with tool results were not confirmed. Treat as env plus task files until checked.
- Your framing was "benchmarks like tau2 where we have the traces and the environment". Strictly satisfied, offline, by only three: tau-bench original, AppWorld, and tau2 itself. Everything domain-shaped for the customer list (banking, CRM, ITSM, healthcare) either lacks published traces or needs live services.
- My earlier answer treated fidelity as "the signal". The build output shows fidelity is the only half that exists; verdict agreement was never produced on retail, so "how close is the environment" has been answered for tool bodies and not at all for Verifiers.
- Benchmarks disagree on what a grader is: tau2 and AppWorld use required plus allowed state diffs; ToolSandbox scores partial credit over a DAG; chi-Bench and FraudBench lean on LLM judges; Toolathlon uses exact strings. D31 says learn from every reward basis. Agreement with an LLM-judged reference is agreement with a judge, not with a state, and the scorecard does not currently distinguish the two.

## 6. Overfit boundary, as it stands after the build

D51 forbids a Builder that only works on the Runs it saw. The retail build already contains one rule written from reading tau2's source rather than from the corpus: `_ERROR_TRANSPORT_PREFIX` ("Error: ") in `compile_env.py`. The observed-rule alternative, peel any prefix that every error payload in the corpus shares, would be derivable from any customer's traces. `CONTEXT_WINDOWS` and `PRICES` are the same kind of constant but describe the model, not the customer, so they sit on the other side of the line. This boundary has not been written down as a rule anywhere in the docs.

## 7. Open questions this leaves for the design grill

1. Whether a stage gate may pass on vacuous output (zero Verifiers, zero Tasks covered), given line 302.
2. Where Reference Runs come from in the Builder: a Trace to Run conversion, or `derive_verifier` reading Traces directly.
3. Whether self-generated traces on an environment with no published traces (WorkBench, EnterpriseOps-Gym, tau3 banking) count as a reference under D60, or only as rehearsal.
4. Whether references with LLM-judged graders count toward verdict agreement, and how the scorecard labels them.
5. Which of the three offline traces-and-env references (tau-bench, AppWorld, tau2 airline and telecom) come next, and what "not overfit" means as a written rule that a code reviewer can apply.
6. What the harness assumes that AppWorld breaks: one shared `db.json` per customer (D56), string ids, a wall clock, a multi-turn user.
