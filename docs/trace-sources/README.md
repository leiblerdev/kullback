# Trace sources

One file per candidate trace source Kullback could ingest from, written 2026-09-24 so every fact the
orchestrator and earlier scouts found survives in the repo. Each file has: Link and owner; What a row is;
Provenance; Counts and size; Schema; One excerpt; Tool-call shape against our Trace record; Environment and
reference verdict; Licence; How to download; Fit for Kullback; Open questions. Facts are sourced to the
URL they came from; anything not confirmed is marked unverified.

See docs/trace-sources/eligibility.md for the training-eligibility read across these and other sources
(a parallel pass, not part of this index's own research).

| Source | Link | What a row is | Produced by | Tool-call shape | Environment | Runs per task | Licence | Fit for Kullback | File |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exgentic agent-llm-traces-v2 | huggingface.co/datasets/Exgentic/agent-llm-traces-v2 | one agent session (OpenTelemetry spans) | model agent, 5 models x 5 harnesses over 6 existing benchmarks | one call per turn only in `tool_calling`/`tool_calling_with_shortlisting` harnesses (29% of rows) | none shipped, depends on the underlying benchmark | up to 15 to 24 per task on AppWorld/tau2 subsets | not stated anywhere | several runs per task fixes AppWorld/tau2's one-run gap; no env or licence | exgentic-agent-llm-traces-v2.md |
| ITBench-Trajectories | huggingface.co/datasets/ibm-research/ITBench-Trajectories | one Codex CLI session trial | model agent (ReAct, GPT-OSS-120B so far) on real Kubernetes fault-injection scenarios | one call per `function_call`/`function_call_output` pair, but a raw Codex log, not a purpose-built export | live Kubernetes cluster, not shipped, not replayable | 3 per scenario | cc-by-nc-4.0 (code repo Apache-2.0) | real tool trace, new SRE domain; only 105 trajectories, one model | itbench-trajectories.md |
| terminalbench-trajectories | huggingface.co/datasets/yoonholee/terminalbench-trajectories | one trial (agent x model x task) | model agents, 109 agent/model combos, third-party scrape of tbench.ai's public leaderboard | multiple tool calls per step possible, `"$N"` placeholder artifact seen in some `msg`/`obs` fields | none shipped, Docker-per-task in Terminal-Bench itself | typically 5 per (task, agent) | apache-2.0 | broad cross-scaffold comparison; third-party scrape, placeholder artifact, ~1/3 rows have null steps | terminalbench-trajectories.md |
| Toolathlon-Verified Trajectories | huggingface.co/datasets/hkust-nlp/Toolathlon-Verified_Trajectories | one of 3 runs per model in a per-model archive | model agent, 12 models, over 32 real apps/33 MCP servers via MCP | inferred MCP one-call-per-turn, not verified (gated, not downloaded) | live third-party services, not replayable offline | 3 per task, 108 tasks | custom no-training, no-redistribution gated policy (not CC BY 4.0, correcting an earlier note) | hardest break of any source here: no seeded env, live internet only, gated, no training use | toolathlon.md |
| AppWorld | github.com/StonyBrookNLP/appworld | one task, several trace sources (native, HALO MCP, Exgentic share) | procedurally generated world, human-written tasks, model-generated traces | native format needs call-boundary reconstruction; HALO/Exgentic tool_calling sessions match directly | offline, relational SQLite, encrypted at rest (not decrypted in this pass) | 1 (HALO, full split) up to 15 (Exgentic subset), authors' own bundle uncounted | Apache-2.0 code, encrypted data bundle with redistribution clause | offline and replayable once decrypted; relational multi-file state, frozen time, integer ids all break current harness assumptions | appworld.md |
| Real-world candidates (4 tiers) | see file | varies by tier | tier 2 is real people on live systems; tiers 3 to 4 are crowdworker, model-written or simulated | click/tap level (tier 3) or none (tiers 1, 4) | tier 2 has a live queryable API; others have none or are simulated | not applicable | varies, listed per source | tier 2 (GH Archive, Jira, Wikipedia, OSM) is the closest real match; tiers 3 to 4 are contrast cases, not real-world data | real-world-candidates.md |

## Eligibility pass additions

The eligibility Worker scored every public trace dataset it could find against the E1 to E6 test (does it
carry tool calls and results, a runnable reference environment, several runs per task, a stated licence,
real provenance, replayable state) and wrote these twelve more files. Full scoring tables and reasoning are
in eligibility.md; this is a one-line pointer to each.

| Source | Link | Verdict | Fit | File |
| --- | --- | --- | --- | --- |
| Trace source eligibility (index) | see file | n/a, this is the scoring pass itself | E1-E6 test, eligible-now/eligible-with-work/not-eligible tables, five best candidates | eligibility.md |
| AgentLogs | huggingface.co/datasets/risenlab/agentlogs | eligible with work | real GitHub Copilot cloud-agent sessions on real repos, CC BY 4.0, 307k tasks, 64M log entries; needs the `output` field's fill rate measured | agentlogs.md |
| AgentTrove | huggingface.co/datasets/open-thoughts/AgentTrove | eligible with work | 1.7M rows, Apache 2.0, teacher-model synthetic over 219 source datasets, no single reference environment; needs the empty-world-build gap closed | agenttrove.md |
| AndroidWorld trajectories | github.com/google-research/android_world (third-party trajectories: MobileJudgeBench, AndroidGen) | eligible with work | real Android apps and emulator, no canonical first-party trajectory release found; unverified third-party links | androidworld-trajectories.md |
| chi-Bench | huggingface.co/datasets/actava/chi-bench (fixtures); github.com/actava-ai/leaderboard (trajectories) | eligible with work | healthcare workflows, Apache 2.0 fixtures; trajectories live in a separate repo whose licence is unchecked | chi-bench.md |
| MCPMark | huggingface.co/datasets/Jakumetsu/mcpmark-trajectory-log | eligible with work | live MCP backends (Notion, GitHub, Postgres, filesystem, Playwright), MIT; HF viewer errors on the tool-result column, needs a direct file pull | mcpmark.md |
| OSWorld trajectories | huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs | eligible with work | Ubuntu desktop computer-use, MIT; standalone VM/container downloadability outside a paid cloud account unconfirmed | osworld-trajectories.md |
| SWE-Gym trajectories | huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories | eligible with work | 6,055 rows, real GitHub issues; licence not stated on this specific card | swe-gym-trajectories.md |
| SWE-bench trajectories (Nebius) | huggingface.co/datasets/nebius/SWE-agent-trajectories | eligible now | 80,036 trajectories, CC-BY-4.0, SWE-bench's own per-instance Docker images as reference environment | swebench-nebius-trajectories.md |
| tau-bench (original) | github.com/sierra-research/tau-bench | eligible now | predecessor of tau2-bench, our current retail/airline reference; MIT, `historical_trajectories/` published | tau-bench-original.md |
| tau3-bench | github.com/sierra-research/tau2-bench (tag v1.0.1) | eligible with work | next version of the harness we already run; no published trajectories yet, we'd generate our own | tau3-bench.md |
| WebArena trajectories | github.com/web-arena-x/webarena (third-party: HaoranLiu/WebArena, thuml/webarena-world-model-cot) | eligible with work | 794 human-gold trajectories on the official task suite; which release has full tool-call detail, and its exact licence, still needs pinning down | webarena-trajectories.md |
