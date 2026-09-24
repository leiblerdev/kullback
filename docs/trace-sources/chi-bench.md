# chi-Bench (CHI-Bench / Χ-Bench)

## Link and owner

Task fixtures: https://huggingface.co/datasets/actava/chi-bench, published by actava-ai. Trajectories: https://github.com/actava-ai/leaderboard, "Open execution logs, trajectories, and results from evaluation runs on the Χ-Bench task." Paper: arXiv 2605.16679, "CHI-Bench: Can AI Agents Automate End-to-End, Long-Horizon, Policy-Rich Healthcare Workflows?"

## What a row is

Two different things, at two different locations. The Hugging Face dataset (`actava/chi-bench`) is task fixtures, one row per task: `family`, `task_id`, `task_kind`, `task_actor`, `title`, `instruction_excerpt`, `expected_target_status`, `expected_payer_route`, `verifier_contract`, `agent_timeout_sec`, `num_fixture_files`, `num_input_pdfs`, plus URLs to the instruction and task TOML. It carries no recorded trajectories (read via WebFetch, 2026-09-24). The actual trajectories live in the separate leaderboard GitHub repo, as `benchmarks/chi-bench/submissions/<dir>/trials/<domain>/<trial_id>/agent/trajectory.jsonl.zst`, one per trial per submission (found via web search of the repo's own documentation, 2026-09-24).

## Provenance

"75 real healthcare workflows" across three domains: provider prior authorization, payer utilization management, and care management (per the paper's own framing, X post from the benchmark's author, read 2026-09-24). Submissions are agent runs (model-run against the fixtures), submitted via pull request to the leaderboard repo, with "the full audit packet (manifest, per-trial verifier evidence, and compressed trajectories) living in git so reviewers can inspect any submission directly from the PR diff."

## Counts and size

101 task rows in the `actava/chi-bench` HF dataset (test split). 75 named workflows, 20 apps, 200+ MCP tools, 1,290 skills, a 1,279-document operations handbook (distributed separately, gated, as `actava/managed-care-operations-handbook`). Trial counts per submission were not tallied this pass.

## Schema

HF task-fixture fields listed above. Trajectory files (`trajectory.jsonl.zst`) were not opened this pass; the leaderboard repo's own tooling (`zstdcat ... | jq .`) is the documented way to read one.

## One excerpt

Not obtained this pass. The leaderboard repo would need to be cloned and one submission's trajectory decompressed to get a real excerpt.

## Tool-call shape

Not directly inspected. The paper states each trial runs an agent for 60-80 steps across four to six clinical stages over 200+ MCP tools, so tool calls are expected to be MCP-shaped (structured arguments and results), consistent with the other MCP-server benchmarks already surveyed (MCPMark, EnterpriseOps-Gym).

## Environment and reference verdict

The reference environment (20 simulated apps as MCP servers) is real and Docker-based per docs/benchmark-landscape.md's earlier survey. "No agent clears 20% when the same case is run three times" (per the arXiv paper's own framing, read via web search 2026-09-24) is strong direct evidence of at least 3 independent recorded runs per case, which comfortably satisfies the held-out requirement once those runs are located in the leaderboard repo.

## Licence (quoted)

`actava/chi-bench` (task fixtures): "apache-2.0" (read from the HF card's license field via WebFetch, 2026-09-24). The `actava-ai/leaderboard` GitHub repo, which is where the actual trajectories live, was not checked for its own licence this pass. Code and data licences can differ, and this is the one that actually governs the trajectory files.

## How to download

Task fixtures: `hf download actava/chi-bench --repo-type dataset` (after Hugging Face authentication; not gated per the card read this pass, though the operations-handbook companion dataset is gated). Trajectories: `git clone https://github.com/actava-ai/leaderboard.git`, then `zstdcat benchmarks/chi-bench/submissions/<dir>/trials/<domain>/<trial_id>/agent/trajectory.jsonl.zst | jq .` per file.

## Fit for Kullback

Real domain (healthcare admin), real repeated trials per case, Apache 2.0 on the task side, MCP-shaped tool calls likely compatible with the same ingest path other MCP corpora need. The catch is that the HF dataset by itself is not a trace source at all, it is only the task definitions, so the actual eligibility rests on the leaderboard repo's licence and on confirming `trajectory.jsonl.zst`'s field shape.

## Open questions

- The `actava-ai/leaderboard` repo's own licence for the trajectory files, separate from the Apache 2.0 task-fixture licence.
- Whether every submission trial has full tool call arguments and results in `trajectory.jsonl.zst`, or only some.
- Exact trial count and how many Tasks have 2+ trials from different submissions or the same submission's repeats.
- Whether the gated `managed-care-operations-handbook` companion dataset is required to make sense of policy-grounded tool calls, and what its access terms are.
