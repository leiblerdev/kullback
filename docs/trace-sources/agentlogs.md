# AgentLogs (risenlab/agentlogs)

## Link and owner

https://huggingface.co/datasets/risenlab/agentlogs, published by RISEN Lab. Paper: arXiv 2608.29204, "AgentLogs: A Dataset for Opening the Black Box of GitHub's Cloud Agent." Schema docs, example notebooks and a sample are also on GitHub at https://github.com/risenlab/agentlogs. Found via a fresh web search this pass (2026-09-24), not in docs/benchmark-landscape.md.

## What a row is

Five related tables, joined by task and session id: `repositories` (repo metadata), `agent_tasks` (one row per task: name, request, state, creator, branch/PR identifiers), `agent_sessions` (one row per session: model, prompt, outcome, usage metrics, branch/PR identifiers), `agent_session_logs` (one row per log entry within a session: messages, tool calls, usage), and `users` (GitHub id and username only). Confirmed live via the Hugging Face datasets-server API, 2026-09-24: configs are `agent_sessions`, `agent_session_logs`, `agent_tasks`, `users` (the `repositories` config fails to parse via datasets-server, a parquet schema issue, not necessarily a data problem).

## Provenance

Real: "activity logs from GitHub's cloud agent functionality," drawn from real public GitHub repositories (over 10 stars) with genuine GitHub Copilot cloud-agent usage, "no synthetic data noted" (WebFetch of the HF card, 2026-09-24). This is the best provenance category in the eligibility test: real systems, real production agent runs, not a simulation of either.

## Counts and size

66,957,764 total records across all tables, 56.7 GB total. 549,239 agent sessions, 307,416 agent tasks, 64,255,174 session-log entries (WebFetch of the HF card, 2026-09-24). Scanned across 35,810 of 1,812,362 popular public repositories (arXiv 2608.29204 abstract, via web search).

## Schema

`agent_session_logs` row, confirmed live via the datasets-server rows API (2026-09-24): top-level fields `session`, `entry_index`, `parsed`, `data`, `raw`. The `data` object's fields include `id`, `agentId`, `arguments`, `callId`, `choices` (a chat-completion-chunk shape with `delta.tool_calls`), `content`, `created`, `error`, `model`, `object`, `output`, `parentId`, `performedBy`, `role`, `timestamp`, `toolCallId`, `toolName`, `tool_call_id`, `truncateResult`, `turn`, `type`, `usage`, plus several Copilot-specific billing and content-filter fields. `agent_tasks` row fields: `id`, `repository.full_name`, `sessions` (a list of session ids), `creator`, `collaborators`, `found`, and a `data` object with `name`, `state`, `archived`, `remote_steerable`, `sharing_status`, `created_at`, `updated_at`, `archived_at`, `custom_agent`.

## One excerpt

One real `agent_session_logs` row, pulled via the datasets-server rows API (2026-09-24), shows a tool-call chunk:
```json
{
  "choices": [{
    "delta": {
      "role": "assistant",
      "tool_calls": [{
        "function_name": "run_custom_setup_step",
        "function_arguments": "{\"name\":\"Start agent firewall\"}",
        "id": "firewall-setup"
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "object": "chat.completion.chunk"
}
```
This particular row's `arguments`, `output` and `toolName` top-level fields were null; it is a streamed chat-completion chunk carrying the call inside `choices[0].delta.tool_calls`, not the flatter `arguments`/`output` shape. A second, non-streamed row type (with `arguments` and `output` populated directly) is implied by the schema listing above but was not isolated in the single 20-row sample pulled this pass.

## Tool-call shape

Two shapes coexist in the same table per the schema: a top-level `arguments`/`toolName`/`output` triple (for `canOfferSessionApproval`, `command`, `description`, `diff`, `filePath`, `kind`, `toolName`, clearly file-edit and shell-style tool calls), and a nested `choices[0].delta.tool_calls[].function_arguments` shape (an OpenAI-style streamed chat completion). Real tool calls include file edits (with a `diff` field), git operations, and GitHub issue/PR/comment interactions, per the HF card's own description.

## Environment and reference verdict

Partly reusable. There is no single packaged "environment" the way tau2's seed database is one; the environment is the actual GitHub repository plus git plus the GitHub API, at the commit the task ran against. This is reproducible in the same spirit SWE-bench trajectories are (clone the real repo at the real commit, replay file edits and git operations against it), but it depends on the source repos staying public and the exact base commit being recoverable from the task/session metadata, neither of which was verified this pass.

## Licence (quoted)

"CC BY 4.0" with suggested attribution to "RISEN Lab" (WebFetch of the HF card, 2026-09-24). No no-training or no-redistribution clause was found on the card, unlike Toolathlon's CC BY 4.0 no-training restriction.

## How to download

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="risenlab/agentlogs", repo_type="dataset", revision="v0.2")
```
recommended for the large log tables over `load_dataset()`, per the HF card (WebFetch, 2026-09-24). Individual configs can also be sampled without a full download via the datasets-server rows API.

## Fit for Kullback

The strongest new find of this pass on scale and provenance: real systems, real production agent activity, a permissive licence with no training restriction, and tool calls that visibly include both arguments and (in at least one of the two row shapes) an `output` field. The domain (software engineering against real GitHub repos) is close to SWE-bench's shape but far larger and more organically varied. The open work is entirely about filtering: finding the subset of tasks with 2+ sessions for the held-out check, confirming what fraction of `agent_session_logs` rows have the non-streamed `output` field actually populated (only one row's shape was inspected this pass, and it happened to be the streamed-chunk kind with nulls there), and recovering the base commit per task well enough to reconstruct a starting state.

## Open questions

- What fraction of `agent_session_logs` rows use the flat `arguments`/`output` shape versus the streamed `choices[].delta` shape, and whether the flat shape's `output` field is reliably populated.
- Distribution of session count per task (mean is 1.8; the useful subset for a held-out check is tasks with 2 or more).
- Whether the base commit a task ran against is recoverable from `agent_tasks`/`agent_sessions` fields (branch/PR identifiers are present, which is a plausible path), needed to reconstruct a Starting state.
- Whether the `repositories` config's parquet schema issue (fails to parse via datasets-server) blocks anything beyond that one API path, or also affects a full download.
