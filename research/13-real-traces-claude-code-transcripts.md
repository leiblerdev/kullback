# Real agent traces: structural analysis of Claude Code session transcripts on this machine

Source: local analysis agent, 2026-08-26. Structure and statistics only; no content reproduced. Sample: 182 sessions, 10 largest analyzed in depth.


All statistics are structural. No message text, file contents, names, or business content are reproduced.

### 1. Corpus size

- 13 project directories under `/Users/krishuagarwal/.claude/projects/`, 182 top-level session transcripts, 52,889 records in those files.
- Additionally 622 sub-agent transcripts live in per-session subdirectories (`<project>/<sessionId>/subagents/agent-<id>.jsonl`, plus `subagents/workflows/wf_<id>/agent-<id>.jsonl`), each with an `agent-<id>.meta.json` sidecar. Sub-agents are NOT interleaved into the parent file.
- 10 largest transcripts: 3.5 MB to 10.3 MB each, 606 to 4,832 records each; from projects greekOCR (4), cdtm-job (4), leibler (1), intelligent-router (1).
- Per-session directories also hold `tool-results/*.txt` (persisted oversized tool outputs, referenced from the transcript by `tool_reference` blocks), and `~/.claude/file-history/<sessionId>/<hash>@vN` holds 939 plain-text file backups (11 MB total, median 5.5 KB, max 145 KB) indexed by `file-history-delta` records.

### 2. Record types and keys

Top-level `type` values seen in the 10 largest files (counts summed): `assistant` 7,091, `user` 3,738, `attachment` 3,842 (harness-injected context), `queue-operation` 1,273, `mode`/`permission-mode`/`last-prompt`/`ai-title`/`atis-latch` (one each per human turn, roughly 1,000 each), `system` 408 (subtypes: `turn_duration`, `compact_boundary`, `away_summary`, `local_command`, `informational`, `model_refusal_fallback`), `file-history-snapshot` 223, `file-history-delta` 206, `frame-link` 270, `pr-link` 19, `relocated` 68, `worktree-state` 69, `artifact-comment-monitor`, `artifact-autoreact-ledger`.

Keys on conversational records (`user`/`assistant`): `uuid`, `parentUuid`, `isSidechain`, `type`, `message`, `timestamp`, `sessionId`, `session_id`, `cwd`, `gitBranch`, `version`, `slug`, `userType`, `entrypoint`, `promptId` (user), `requestId` and `effort` (assistant), `toolUseResult` and `sourceToolAssistantUUID` (user records carrying tool results), `isMeta`, `logicalParentUuid`, `isCompactSummary`, `isVisibleInTranscriptOnly`, `toolDenialKind`, `isApiErrorMessage`, `attributionSkill`, `classifierMetaLines`, `pendingBackgroundAgentCount`. Sub-agent records add `agentId` and `attributionAgent`.

`message` keys on assistant: `role`, `content`, `model`, `id`, `type`, `stop_reason`, `stop_sequence`, `stop_details`, `usage`, `diagnostics` (occasionally `container`, `context_management`). `usage` keys: `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `cache_creation{ephemeral_5m,ephemeral_1h}`, `server_tool_use{web_search_requests,web_fetch_requests}`, `service_tier`, `inference_geo`, `iterations[]` (per-iteration token breakdown), `speed`, sometimes `output_tokens_details`. No cost or USD field anywhere. Models observed: `claude-opus-5`, `claude-fable-5`, `claude-opus-4-8`, plus `<synthetic>` for harness-generated assistant records.

Content block types: `text`, `thinking`, `tool_use`, `tool_result`; inside `tool_result.content` arrays: `text`, `image` (20 total), `tool_reference` (59 total, pointer to a persisted file). User records with plain string content are human prompts (or compaction summaries).

### 3. Per-session statistics (10 largest)

| session | records | assistant recs | API turns | human msgs | tool_use | err share | thinking share | longest tool chain | sub-agent files | result len median / p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| b5e3d048 | 4832 | 1890 | 945 | 98 | 984 | 1.4% | 31% | 286 | 21 | 437 / 2360 |
| 89b1fcfa | 3338 | 810 | 412 | 73 | 367 | 3.5% | 37% | 67 | 40 | 503 / 4328 |
| 9ae28842 | 3480 | 1251 | 495 | 49 | 485 | 2.7% | 31% | 60 | 8 | 464 / 3032 |
| 9fe9ee87 | 606 | 179 | 72 | 12 | 88 | 19.3% | 36% | 18 | 0 | 220 / 2558 |
| c7f9e74b | 1943 | 691 | 279 | 25 | 300 | 5.0% | 30% | 71 | 43 | 435 / 2974 |
| 56a585fc | 2275 | 546 | 250 | 55 | 218 | 2.3% | 43% | 29 | 21 | 780 / 3101 |
| 3476f217 | 1311 | 511 | 213 | 38 | 239 | 4.2% | 29% | 67 | 28 | 499 / 2297 |
| a16002b6 | 1629 | 585 | 244 | 40 | 299 | 6.7% | 29% | 113 | 14 | 319 / 2581 |
| 024462be | 1284 | 281 | 127 | 38 | 124 | 3.2% | 37% | 20 | 28 | 1089 / 7121 |
| 1edcab1d | 1535 | 347 | 187 | 64 | 142 | 8.5% | 33% | 16 | 1 | 129 / 1100 |

Aggregates over the 10 parents: 3,246 tool_use blocks, 100% matched to a tool_result by `tool_use_id` (no orphans), 123 flagged `is_error` (3.8%). 492 human messages vs 3,224 API turns, so about 6.6 model calls per human message. Max single tool_result 48,647 chars.

Tool name distribution (aggregate top 15): Bash 2,318, Edit 295, Read 233, Agent 133, Write 93, ToolSearch 32, SendMessage 32, Skill 31, mcp__claude-in-chrome__tabs_context_mcp 15, AskUserQuestion 14, Monitor 10, WebFetch 7, Artifact 6, TaskCreate 6, TaskUpdate 6. Bash is 71% of all calls. Grep and Glob tools were never used in these sessions (grep via Bash instead).

Parallel tool calls: the harness writes one record per content block, so naive per-record counting gives always 1. Grouping by `message.id` (one API response) gives: 1 tool call in 2,597 turns, 2 in 236, 3 in 16, 4 in 13, 5 in 5, 6 in 6, 8 in 2. So about 8.6% of tool-calling turns are parallel; up to 10 records can share one `message.id`.

Thinking: 2,316 thinking blocks across 7,091 assistant records (33%); roughly 70% of API turns start with a thinking block. Thinking is a separate record that precedes the tool_use record and shares its `message.id`.

Sub-agents: 204 sub-agent files across the 10 sessions, 38,050 records, every one flagged `isSidechain: true`, 12,173 tool calls (nearly 4x the parents' own tool calls), 6,682 thinking blocks. Linkage: the sub-agent file's first record has `parentUuid: null` and carries `agentId`; the parent's `Agent` tool_result record has `toolUseResult.agentId` (and `outputFile`, `status`, `isAsync`, `resolvedModel`, `prompt`); the `meta.json` has `agentType`, `description`, `toolUseId` (the parent's tool_use id), `spawnDepth`, `worktreeCleanlyRemoved`. About 25% of sub-agent files are not referenced by any parent `toolUseResult.agentId` (background or workflow agents); `parentUuid` never crosses files.

Latency (tool_use timestamp to tool_result timestamp): Bash median 2.2 s, p90 13.2 s; Edit 0.1 s; Read 0.0 s; Agent 2.0 s (spawn only, async); AskUserQuestion p90 1,481 s; browser MCP calls about 62 s median. Timestamps are ISO strings with ms.

State of the world: only partially recorded. `toolUseResult` for Read carries the full file slice; for Edit carries `oldString`, `newString`, `structuredPatch` (unified hunks), and `originalFile` (full pre-edit content in 206 of 374 cases, null otherwise, median 3.6 KB); for Write carries the full new `content`; for Bash carries `stdout`, `stderr`, `interrupted`, `isImage`, `noOutputExpected` only. `file-history-delta` points to on-disk backup files (pre-edit versions), `gitBranch` is stamped on every record, and a git status snapshot appears inside 37 user records (system-reminder text embedded in list content), but there is no post-step snapshot, no final repo state, and no record of Bash-caused file changes.

### 4. Read vs write, determinism

Non-Bash: read tools (Read, ToolSearch, Skill, Agent, AskUserQuestion, Monitor, WebFetch, WebSearch) 461 vs write tools (Edit, Write, TaskCreate/Update) 388.

Bash (2,318 calls) classified by command after stripping leading `cd`: read-only 1,269 (55%), mutating or effectful 827 (36%: redirects, heredocs, rm/mv/mkdir, sed -i, git commit/push/checkout, pip/uv/npm install, docker, psql, open), ambiguous script execution 222 (9%: python3, uv run, node, npx, bash, gh, curl, loops). Top Bash verbs: git 243, grep 189, echo 175, python3 160, sed 124, cat 107, uv 106, ls 70, for 66, docker 25, npm 15, psql 15, gh 11, npx 11, rm 10.

Overall read:write ratio is roughly 1.4:1 to 1.7:1 depending on how ambiguous scripts are counted. Non-deterministic categories present: WebFetch/WebSearch (8 calls), browser MCP (21 calls), and about 570 Bash calls that touch network, time, package managers, docker, databases, or subprocess scripts (roughly 25% of Bash).

### 5. One step, abstracted

```
assistant record (thinking)        {uuid:A1, parentUuid:U0, message:{id:M, content:[{type:"thinking", thinking, signature}]}, requestId, timestamp, ...}
assistant record (tool_use)        {uuid:A2, parentUuid:A1, message:{id:M, model, content:[{type:"tool_use", id:T, name:"Edit", input:{file_path, old_string, new_string, replace_all}, caller:{type}}], stop_reason, usage:{...}}, requestId, effort, cwd, gitBranch, version, sessionId}
user record (tool_result)          {uuid:U1, parentUuid:A2, sourceToolAssistantUUID:A2, promptId, message:{role:"user", content:[{type:"tool_result", tool_use_id:T, content:str|[{type:text|image|tool_reference}], is_error?}]}, toolUseResult:{filePath, oldString, newString, originalFile|null, structuredPatch[], userModified, replaceAll}, timestamp}
optional attachment record         {parentUuid:U1, attachment:{type:"edited_text_file"|"hook_success"|"task_reminder"|...}}
next assistant record              {parentUuid:U1, message:{id:M2, ...}}
```

The `parentUuid` chain is strictly linear within a file (thinking -> tool_use -> tool_result -> next), and all `parentUuid` targets resolve within the same file. Parallel calls appear as consecutive assistant records sharing `message.id`, followed by consecutive user tool_result records. The model's reasoning sits in the `thinking` block of the same API turn, one record before the tool_use, and in `text` blocks; the `tool_use.input.description` field is a second, terse rationale. `stop_reason` is on each record but reflects the whole API response. Compaction inserts a `system/compact_boundary` record (with `compactMetadata`: `preTokens`, `postTokens`, `cumulativeDroppedTokens`, preserved uuid ranges, `logicalParentUuid`) and a synthetic `user` record with `isCompactSummary: true`, so the visible history is rewritten mid-trace.

### 6. Missing for a replica environment

- Pre and post state snapshots: no filesystem or git snapshot per step; pre-edit content only for Edit/Write (55% of edits), never for Bash effects. No `git status`/diff after a step, no final state at session end.
- Tool schemas: `tools[]` and the system prompt are never stored; only tool names and inputs. Deferred-tool loading (ToolSearch) changes the tool set mid-session and is only implied.
- Bash side effects: only stdout/stderr/exit-implied; no exit code field, no list of files touched, no process/network activity, no working-directory changes beyond `cwd` stamp.
- Environment: no env vars, no installed package versions, no OS/toolchain fingerprint (only harness `version` and `cwd`).
- External service responses: WebFetch, MCP browser, database and HTTP calls inside Bash have only the text result; no raw response, headers, or replay fixture; no idempotency flag.
- Determinism markers: no seed, temperature, or sampling params; no marker distinguishing deterministic from live tools; timestamps exist but wall-clock-dependent commands are not flagged.
- Sub-agent context: sub-agent transcripts are separate files, linked only by `agentId` in a `toolUseResult`; the sub-agent's system prompt and tool set are absent; a quarter of sub-agent files have no parent reference.
- User intent and success labels: none. No task label, no outcome label, no per-step grading; the only proxies are `is_error` on tool results, `ai-title`, `turn_duration`, `pr-link`, and `toolDenialKind` (permission denials).
- Cost: token usage present per API call, no price or cost.
- History rewriting: compaction replaces dropped context with a summary; the original messages remain in the file but the model's actual input after that point must be reconstructed from `compactMetadata`.
- Records are per-block, not per-API-call; reconstruction of the true request/response needs grouping by `message.id` and `requestId`.

### 7. Trace artifacts in the leibler repo

No sample agent traces (no `.jsonl`, no captured tool_use/tool_result payloads) exist in the repo. What exists is documentation only:

- `/Users/krishuagarwal/Desktop/Programming/website/leibler/monitoring-tool/research/replay-and-trace-formats-2026-08.md`: survey of trace formats (OTel GenAI, OpenInference, Langfuse, LangSmith, raw OpenAI/Anthropic, Vercel AI SDK, agent frameworks), replay strategies, comparison ladder, and a proposed `ReplayStep` data model (ids, provenance, canonical request incl. `tools[]` and params, reference_output, observations with `idempotent` flag, episode_context with `episode_outcome` and `state_diff`, eval_metadata, raw body refs). That proposed model already covers most of the gaps in section 6.
- Other research notes referencing traces: `monitoring-tool/research/00-synthesis.md`, `03-evals-from-production-traces.md`, `04-llm-judge-for-trajectories.md`, `05-tool-call-correctness-metrics.md`, `09-environment-synthesis.md`, `11-tool-simulation-and-mocking.md`, `12-env-synthesis-pipelines-tauforge-style.md`, `landscape-2026-08.md`, `agent-eval-methods-2026-08.md`.
- `/Users/krishuagarwal/Desktop/Programming/website/leibler/docs/adr/0001-trace-capture-no-proxy.md` (capture via log-drain plus fail-open SDK wrapper) and `tech/docs/sdk-wrapper.md` (what the wrapper captures, fail-open rule, open questions).
- Hits in `dashboard/` and `deepline/data/` are unrelated word matches (CSV and lead data), and `dashboard/.next/trace` is a Next.js build trace.