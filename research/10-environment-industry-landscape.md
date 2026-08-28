# The 2025-2026 RL environments industry: who builds what, and what to reuse

Research sweep, 2026-08-26. Source: web research agent. Topic 10. (Search budget exhausted; sources via direct fetch.)

## 1. Environment companies

- **Market**: Anthropic "discussed spending over $1 billion on RL environments over the next year"; Surge created an RL-environments org; Mercor and Scale pivoting; Ross Taylor: most public environments "require serious modification" and are prone to reward hacking ([TechCrunch](https://techcrunch.com/2025/09/21/silicon-valley-bets-big-on-environments-to-train-ai-agents/)). SemiAnalysis: "environment compute" as a scaling axis ([SemiAnalysis](https://newsletter.semianalysis.com/p/scaling-reinforcement-learning-environments-reward-hacking-agents-scaling-data)).
- **Mechanize**: environments and evals for coding agents; "replication training" (spec + reference implementation, binary behavioral equivalence); ~$480 compute per task per run; authoring should cost "a few thousand dollars per task"; procedurally generated tasks are "cheap tires on a Ferrari" ([site](https://www.mechanize.work/), [blog](https://www.mechanize.work/blog/cheap-rl-tasks-will-waste-compute/)).
- **Prime Intellect**: Environments Hub + verifiers; 500+ tasks; Prime Sandboxes ([blog](https://www.primeintellect.ai/blog/environments)).
- **Fleet**: "training gyms for agents"; docs login-gated; GitHub forks of OpenEnv, SkyRL, EnterpriseOps-Gym ([about](https://fleetai.com/about), [GitHub](https://github.com/fleet-ai)).
- **Halluminate (YC S25)**: managed replica environments (Salesforce, ticketing); Westworld diligence bench with binary and weighted verifiers; $44.21 per run ([YC](https://www.ycombinator.com/companies/halluminate), [blog](https://www.halluminate.ai/blog/due-diligence-bench)).
- **Matrices**: JS-only site; no verifiable detail ([site](https://matrices.ai/)).
- **Veris AI**: production-twin environments generated from customer tools, APIs, data, and user interactions; in-VPC; incident loop: production log to LLM-written rubric to scenario engine (n=30 variants) to simulation against simulated tools and users, held-out n=20 ([site](https://www.veris.ai/), [blog](https://www.veris.ai/blog/never-waste-a-good-failure-how-veris-ai-turns-production-incidents-into-self-improving-agents)). Sigma-rule RFT report: deterministic reward 0.1/0.3/0.7, ~12s per verification ([report](https://www.veris.ai/blog/technical-report-reinforcement-learning-fine-tuning-for-enterprise-ai-agents)). **Closest existing product to our environment generator.**
- **Surge**: MCP-native environments from a world model, entities, tools; CoreCraft (23 tools, 2,500+ entities, expert rubrics); frontier ~30-40% ([blog](https://www.surgehq.ai/blog/enterprisebench-corecraft)).
- **Mercor / Deeptune**: "recreated hundreds of enterprise applications"; recipe = replica software + tasks + verifiers ([blog](https://mercor.com/blog/mercor-to-acquire-deeptune/)).
- **HUD**: SDK for evals, environments, verifiers; QA agents that audit graders for reward hacking; $0.10 per env-hour ([site](https://www.hud.ai/)).
- **ServiceNow EnterpriseOps-Gym** (Apache-2.0): 8 domains, 512 tools over live MCP servers, 164 tables seeded from SQL snapshots, 1,150 tasks, SQL verifiers on final state ([GitHub](https://github.com/ServiceNow/EnterpriseOps-Gym)).

Who generates environments from customer traces/software? Veris and Decagon do; Mechanize, Surge, Deeptune, Halluminate build generic replicas.

## 2. Lab and agent-company practice
- **Anthropic**: clean env per trial (shared git history inflated scores); grader taxonomy code / model / human; avoid grading exact tool sequences; 20-50 tasks from real failures; "a good task is one where two domain experts would independently reach the same pass/fail verdict"; pass@k vs pass^k diverge by k=10 ([Anthropic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).
- **OpenAI**: RFT graders (string_check, text_similarity, score_model, python sandboxed); fine-tuning platform winding down; Evals platform deprecated (read-only Oct 31, 2026); Agents SDK tracing + trace grading ([RFT](https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning), [trace grading](https://developers.openai.com/api/docs/guides/trace-grading)).
- **Sierra**: Agent OS Simulations; tau2 ([platform](https://sierra.ai/platform)).
- **Decagon**: simulations auto-generated from production conversations into a golden test set; Checkpoints and Assertions; CI/CD; self-generated simulations 58% to 88% accuracy after self-critique ([blog](https://decagon.ai/blog/the-next-generation-of-simulations), [DuetBench](https://decagon.ai/blog/duetbench)).
- **Salesforce APIGen-MT**; **Meta ARE**.

## 3. Fidelity techniques and costs
Seeding via SQL snapshots per run; in-VPC data boundaries; clean env per trial; `network_mode: none` default (inspect); interception server rewriting responses to block reward hacking (verifiers v1); difficulty calibration (low pass rate for capability evals, ~100% for regression; 30-40% frontier solve by design); final-state over action-sequence verifiers everywhere. Costs: ~$480 compute/task/run (Mechanize); E2B $0.000028/s for 2 vCPU; Daytona $0.0504/vCPU-h; Modal $0.142/core-h; HUD $0.10/env-hour.

## 4. Open-source infrastructure
- **Harbor** (Terminal-Bench harness): task dir = `task.toml` + `instruction.md` + `environment/Dockerfile` + `solution/solve.sh` + `tests/test.sh` writing `/logs/verifier/reward.json`; runs on Docker, Daytona, Modal, E2B ([docs](https://harborframework.com/docs/tasks)).
- **verifiers / Prime v1**: Taskset, Harness, Trace; interception server; `HarborTaskset` ([docs](https://docs.primeintellect.ai/verifiers/v1/harbor.md)).
- **OpenEnv**: `reset/step/state`, `CallToolAction` wrapping MCP tools (RFC 003), FastAPI+WebSocket in Docker ([repo](https://github.com/meta-pytorch/OpenEnv)).
- **inspect_ai** sandboxes ([docs](https://inspect.aisi.org.uk/sandboxing.html)). **HAL harness** ([GitHub](https://github.com/princeton-pli/hal-harness)).
- **Snapshots**: E2B pause/templates ("prefer templates over snapshots" for many per-customer envs) ([docs](https://docs.e2b.dev/sandbox/persistence)); Daytona cold/hot snapshots; Modal snapshots.
- **agent-vcr**: records MCP JSON-RPC to cassettes, deterministic replay, error injection, diff ([PyPI](https://pypi.org/project/agent-vcr/)).

## 5. What a two-person startup should reuse vs build

**Reuse:**
1. **Harbor task format** as the on-disk contract. Do not invent a schema.
2. **OpenEnv `CallToolAction` over MCP** as the runtime interface (serves evals now, RL later).
3. **EnterpriseOps-Gym** as the seed for replica CRM/ITSM/Email/Calendar MCP servers; fork the SQL-snapshot-plus-SQL-verifier pattern.
4. **agent-vcr cassettes** as the trace-capture primitive.
5. **E2B templates** or Daytona warm pools for per-customer sandboxes.
6. **Anthropic's eval rules** verbatim.

**Build (this is the product):**
1. **Trace-to-replica compiler**: from cassettes plus the customer's DB export, infer tool schemas and a relational seed, emit an MCP server plus SQL snapshot. Veris does this behind enterprise sales; nobody ships it self-serve.
2. **Failure-to-taskset expansion**: APIGen-MT blueprint-committee-execute loop pointed at one production failure to produce ~30 verified variants plus a held-out set.
3. **Anonymizer that preserves joins**: deterministic PII tokenization keyed per customer so foreign keys survive.
4. **Grader audit**: a HUD-style checker that flags graders passing empty or trivial rollouts before any number reaches the customer.

**Skip:** OpenAI RFT/Evals, general-purpose replica catalogs, bespoke sandbox orchestrators.
