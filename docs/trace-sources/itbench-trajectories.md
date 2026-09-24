# ITBench-Trajectories

## Link and owner

https://huggingface.co/datasets/ibm-research/ITBench-Trajectories, published by the Hugging Face org
ibm-research. Built on top of the ITBench benchmark, whose repo now lives at
https://github.com/itbench-hub/ITBench (moved from `IBM/ITBench`, confirmed by following the GitHub API
redirect on 2026-09-24) and its agent at https://github.com/itbench-hub/ITBench-SRE-Agent. Paper: "ITBench:
Evaluating AI Agents across Diverse Real-World IT Automation Tasks," arXiv 2502.05352
(https://arxiv.org/abs/2502.05352, found via the arXiv API title search).

## What a row is

Not a dataset-viewer table; the dataset is a folder tree of files, one directory per (model, domain,
scenario, run). Each leaf directory holds one trial: `session.jsonl` (the full step-by-step execution
trace), `agent_output.json` (the agent's final diagnosis), `judge_output.json` (automated evaluation
metrics against ground truth), and an optional `code_generated_by_agent/` folder of Python scripts the
agent wrote and ran during the trial. (dataset README, `Dataset Structure` section)

## Provenance

Model generated, against real IT automation scenarios. README: "This dataset contains complete execution
trajectories of LLM agents using the ITBench-SRE-Agent. It captures real agent reasoning, tool usage, and
performance across multiple state-of-the-art language models tackling Site Reliability Engineering (SRE),
Security & Compliance (CISO), and Financial Operations (FinOps) scenarios from the ITBench benchmark."
The SRE scenarios themselves are "environment snapshots capturing observability data (logs, traces,
metrics, alerts, events) from orchestrated Kubernetes environments where faults were injected"; the task is
to diagnose the injected fault from that data (README, `What Are ITBench Scenarios?`). The agent is a
ReAct-style scaffold (`ITBench-SRE-Agent`) that reasons, then acts with Bash, Python or file-operation
tools, over up to several turns per trial. Currently released trajectories are all one model, OpenAI
GPT-OSS-120B; two more models (Gemini-3-Flash-Preview, Kimi-K2-Thinking) are named in the card as "to be
released" but were not present in the file listing checked on 2026-09-24.

## Counts and size

README's own summary: "105 complete agent trajectories across 35 ITBench SRE scenarios (3 runs per
scenario)," one model (OpenAI GPT-OSS-120B) so far; CISO and FinOps trajectories are listed as future
releases, not present yet. Licence: cc-by-nc-4.0 (Hub API `cardData.license`). Size category on the card:
1K<n<10K files; not stated as total bytes, this pass did not sum the tree.

## Runs per task

3 runs per scenario, stated directly: "Currently, 3 runs per scenario are available for the OpenAI
GPT-OSS-120B model" (README, `Domains and Scenarios`). 35 scenarios x 3 runs = 105 trajectories, matching
the headline count.

## Schema

Each trial directory: `session.jsonl` (one JSON object per line, each with `timestamp`, `type`, `payload`;
`type` values observed: `session_meta`, `message`, `event_msg`, `function_call`, `function_call_output`,
`reasoning`, `token_count`, `ghost_snapshot`, `agent_message`), `agent_output.json` (final diagnosis),
`judge_output.json` (precision/recall/F1 and other metrics against ground truth), optional
`code_generated_by_agent/*.py`.

## One excerpt

From `ReAct-Agent-Trajectories/OpenAI-GPT-OSS-120B/sre/Scenario-1/1/session.jsonl` (fetched directly,
88 lines): the session format is a Codex CLI rollout log (`session_meta.originator: "codex_exec"`,
`cli_version: "0.76.0"`). A tool call and its result, trimmed:

```json
{"type": "function_call", "name": "shell",
 "arguments": "{\n  \"command\": [\"bash\", \"-lc\", \"ls -R\"],\n  \"workdir\": \"/root/projects/open_source/zero/.../Scenario-1\"\n}",
 "call_id": "functions.shell_39c6"}
{"type": "function_call_output", "call_id": "functions.shell_39c6",
 "output": "{\"output\":\"command timed out after 10152 milliseconds\\n\",\"metadata\":{\"exit_code\":124,\"duration_seconds\":10.2}}"}
```

## Tool-call shape against our Trace record

One call per `function_call` event, matched to its `function_call_output` by `call_id`: this is close to
our `ToolCall` shape (`name`, `args`, `result`) directly, since `name` is always `"shell"` here with the
real command inside a JSON-encoded `arguments` string, and the result is a JSON-encoded string with
`output`, `exit_code`, `duration_seconds`. Because this is a raw Codex CLI session log, not a purpose-built
tool-call export, it also carries Codex-specific event types (`ghost_snapshot`, `reasoning`, `token_count`)
that our `Turn`/`ToolCall` records have no place for and would need to be dropped or ignored at ingest.

## Environment and reference verdict

Environment: a live, orchestrated Kubernetes cluster with injected faults, observability tooling (logs,
traces, metrics, alerts, events) and Bash/Python access for the agent; not shipped with this dataset, only
the recorded output of a session that ran against it. Reference verdict: `judge_output.json` per trial holds
automated precision/recall/F1 scores for entity identification, propagation-chain accuracy, alert-explanation
completeness and fault localization, computed by ITBench's own judge against ground truth, not a generic
diff or exact-match check (README, `Evaluation` section).

## Licence

`cc-by-nc-4.0`, from the Hub API's `cardData.license` field and the README front matter
(`license: cc-by-nc-4.0`). The benchmark's own code repo, https://github.com/itbench-hub/ITBench, is
separately licensed Apache-2.0 (confirmed via the GitHub API license endpoint, 2026-09-24); the trajectories
dataset and the benchmark code carry different licences.

## How to download

Files are plain Hub files under a nested path, fetchable individually, for example:
`https://huggingface.co/datasets/ibm-research/ITBench-Trajectories/resolve/main/ReAct-Agent-Trajectories/OpenAI-GPT-OSS-120B/sre/Scenario-1/1/session.jsonl`
(use `-L` with curl, the Hub 307-redirects to a CDN URL). No parquet export; this is a raw file tree, so a
full download means walking `https://huggingface.co/api/datasets/ibm-research/ITBench-Trajectories` for
the `siblings` file list and fetching each path.

## Fit for Kullback

Gives a real (if narrow) tool-call trace over a live environment with fault injection and a non-trivial
automated verdict, in a domain (SRE/Kubernetes) Kullback has no reference environment for yet. Breaks: only
one model so far (105 trajectories total, thin for training); the trace is a raw Codex CLI session log with
scaffold-specific event types, not a purpose-shaped tool-call export, so ingest would need a Codex-log
parser; the live Kubernetes environment with injected faults is not redistributed, so nothing here is
replayable offline, only readable as a fixed trace, the same limitation the harness already has with
Toolathlon and MCPMark (docs/benchmark-landscape.md). One trace (trial) per task per run, several runs
(3) per scenario.

## Open questions

- Total on-disk size of the full file tree was not summed in this pass.
- CISO and FinOps trajectories are announced but not present in the file listing checked on 2026-09-24;
  unverified when or whether they will land.
- Whether the two "to be released" models (Gemini-3-Flash-Preview, Kimi-K2-Thinking) have since been added
  was not re-checked after the initial listing.
