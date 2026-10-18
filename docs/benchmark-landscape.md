# Benchmark landscape

Which public benchmarks could serve as a second reference beside tau2, so the claim that an Environment is good
rests on more than one benchmark, without tuning the Builder to any of them (D51). Per-source detail on traces
lives in [trace-sources/](trace-sources/README.md).

## What a reference must give

In order of how hard each is to replace:

1. An executable environment with a resettable seed state callable from Python, so replay fidelity can be measured.
2. A grader whose per-Run verdicts we can read, so our Verifiers can be compared with it.
3. Published traces with tool calls and results that our ingest can read.
4. Ideally a simulated user, so the user side has something to check against.

## Candidates

| Benchmark | Domain | Published traces | Offline | Licence | Fit |
|---|---|---|---|---|---|
| tau2-bench | retail, airline, telecom | yes, with `reward_info` | yes | MIT | in use |
| tau-bench original | retail, airline | yes, `historical_trajectories/` | yes | MIT | traces and env |
| tau3-bench (tau2 v1.0.1) | tau2 domains with 75+ task fixes, banking, voice | not confirmed | yes | MIT | env only |
| FraudBench | banking fraud, adversarial caller | not verified | no, cloud LLMs | CC BY 4.0 | env and task files |
| AppWorld | 9 consumer apps, 457 APIs | yes | yes, encrypted bundle | Apache 2.0 | traces and env, no user turns |
| MCPMark | Notion, GitHub, Postgres, filesystem, Playwright | yes | no, live services | MIT on the trace card | traces and env |
| chi-Bench | healthcare admin | yes | Docker, gated | Apache 2.0 code | traces and env, LLM grader |
| EnterpriseOps-Gym | calendar, CSM, drive, email, HR, ITSM | not confirmed | Docker | Apache 2.0 | env only |
| STATE-Bench | travel, support, shopping | unclear | yes | MIT | env only |
| WorkBench | email, calendar, CRM, PM | no | yes | MIT | env only |
| ToolSandbox | phone assistant | no | mostly | Apple source-available | env only |
| Toolathlon | 32 real apps over MCP | yes, no training use | no, live services | gated custom | traces only |
| TheAgentCompany | GitLab, ownCloud, Plane, RocketChat | not located | Docker, 30+ GB | MIT | env only, heavy |
| AgentDojo | banking, Slack, workspace, travel | unconfirmed | yes | MIT | env only, small |
| CRMArena-Pro | CRM | no | no, Salesforce | cc-by-nc-4.0 | env only, heavy |
| BFCL v3/v4 multi-turn | 8 synthetic APIs | responses only | yes | Apache 2.0 | env only, toy state |

Ruled out, do not re-research: Gecko, T1-Bench, MCP-Bench, MCP-AgentBench, MCPToolBench++, ACEBench,
ComplexFuncBench, Nexus, NESTFUL, MCP-Atlas (action matching or read-only), OSWorld, Windows Agent Arena,
HealthAdminBench (GUI grading), FHIR-AgentBench (read-only QA), E-Bench (no public code), "BankingBench" (does
not exist).

## What decides it

- Only three references give traces and an environment offline: tau-bench original, AppWorld and tau2 itself.
  Everything shaped like the customer list (banking, CRM, ITSM, healthcare) lacks published traces or needs
  live services.
- tau-bench original is the cheapest second point and the weakest test of generalisation.
- AppWorld stresses what tau2 does not: relational state with foreign keys, per-task database diffs, a frozen
  clock, integer ids, a single user turn.
- Sources disagree on MCPMark. One survey ranked it the closest structural match to tau2; another called live
  backends stress tests, not references. Fidelity needs a real tool callable on a seed state, which MCPMark
  cannot give without live accounts.
- Graders differ: tau2 and AppWorld diff state, ToolSandbox scores a DAG, chi-Bench and FraudBench lean on LLM
  judges. Agreement with an LLM-judged reference is agreement with a judge, not with a state.
