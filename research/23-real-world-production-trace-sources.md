# R23. Real-world production trace sources for the first build (non-coding, tool-using agents)

Date: 2026-08-27. Method: WebSearch plus direct fetches of the Hugging Face datasets API and datasets-server, GitHub, arXiv, a public S3 bucket, and vendor docs. Extends R14 (public trajectory datasets), R15 (trace samples) and R16 (production trace formats); nothing listed there is repeated except where a new fact about it was found.

Coverage caveats:
- The Hugging Face datasets-server was flaky during this session (502 and "index is loading"); where a first-rows call failed I say so and fall back to raw file ranges.
- Several dataset viewers on HF are broken for the datasets that matter most (agent-evals/hal_traces, open-agent-leaderboard/traces, experiential-labs/wmo-crmarena-traces, danielliu99/mcp-universe-traces). Sizes and schemas come from the API and raw files, not the viewer.
- I did not log into any vendor sandbox; sandbox availability is from vendor docs as of today.
- Licenses are quoted where stated; "not stated" means the card or repo does not say. Several relevant sources are CC BY-NC, which matters for a company.

Headline: there is no public dataset of real production traces from a deployed customer-support, CRM, banking or e-commerce agent. Vendors (Intercom, Decagon, Ada, Klarna, Parloa, PolyAI, Cresta) publish none. The HF "agent traces" wave of 2026 is almost entirely coding-agent sessions (Claude Code, Codex, Pi). The closest real things are (a) benchmark runs where a real LLM agent talks to a simulated user and simulated tools, now available in bulk and in two formats (tau2 native JSON on Sierra's public S3 bucket, and OpenTelemetry GenAI spans from Exgentic), (b) benchmark runs against a real Salesforce org (CRMArena, with public org credentials), and (c) human-to-human support chats with agent actions but no tool payloads (ABCD). The practical path for the first build is: normalise (a) and (b), then generate our own traces against a real helpdesk sandbox API with an LLM agent and an LLM user.

## 1. Public datasets of multi-turn tool-calling traces in support and enterprise domains

Classification key: REAL-PROD = logged from a deployed system with real users; REAL-RUN = a real LLM agent ran, but the user and/or environment were simulated; HUMAN = human-authored or human-to-human; SYN = synthetic.

### 1.1 Sierra tau2-bench leaderboard trajectories (public S3 bucket). REAL-RUN

- Where: bucket `sierra-tau-bench-public`, prefix `submissions/<model>_<org>_<date>/trajectories/`. Listable without credentials: `https://sierra-tau-bench-public.s3.amazonaws.com/?list-type=2&prefix=submissions/`. Submission process: https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md ("Trajectory files are **not** committed to the repo — they are hosted on S3.")
- What I saw in the first 1,000 keys: 21 submission directories (2 are examples), among them `claude-3-7-sonnet_anthropic_2024-06-20`, `claude-opus-4_anthropic_2025-05-22`, `claude-opus-4-1_anthropic_2025-01-15`, `claude-opus-4-5_sierra_2026-02-26`, `claude-opus-4-6_sierra_2026-05-05`, `claude-opus-4-7_sierra_2026-05-05`, `claude-opus-4-8_sierra_2026-08-04`, `claude-opus-5_sierra_2026-08-04`, `claude-fable-5_sierra_2026-08-04`, `claude-sonnet-4-5_anthropic_2025-10-02`, `deepseek-v3.2_deepseek_2025-12-01`, `distyl-buttonagent_distyl_2026-03-25`, `gemini-2-5-pro_sierra_2026-05-05`, `gemini-3-1-flash-live-preview-thinking-high_google_2026-04-02` (voice). The listing was truncated at 1,000 keys, so more exist (search results mention GPT-5.2, Gemini 3 Pro, GLM-5, Qwen3.5).
- Size: one file per model, domain and user-simulator model, 4 trials per task. Examples: `claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json` 10.5 MB, `..._retail_...` 24.9 MB, `..._telecom_...` 41.2 MB, `claude-opus-4-5_high_banking_knowledge_gpt-5.2_4trials.json` 94.5 MB, `claude-opus-4-8 .../banking_knowledge_results.json` 268.6 MB. The trajectory files visible in the first 1,000 keys total about 2.7 GB.
- Tool results present: yes, untruncated, plus `error: true|false` per tool message. In a 2.5 MB slice of the Claude 3.7 airline file: 380 tool messages, 5 with `error: true`, e.g. `"content": "Error: Not enough seats on flight HAT229"`.
- Domains: airline, retail, telecom (dual control: the simulated user also has tools), banking_knowledge (tau3, RAG over documents).
- License: not stated in the bucket. tau2-bench code is MIT; treat the trajectories as "public, license unclear" and ask Sierra before redistribution.
- Format (verbatim head of `claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json`, then the first tool exchange):

```
{
    "timestamp": "2025-06-05T16:05:50.589090",
    "info": {
        "git_commit": "c30d59aaa71c65f9b9eb6a8f8636b48945028fcf",
        "num_trials": 4,
        "max_steps": 200,
        "max_errors": 10,
        "user_info": {
            "implementation": "user_simulator",
            "llm": "gpt-4.1-2025-04-14",
            "llm_args": {
                "temperature": 0.0
            },
            "global_simulation_guidelines": "# User Simulation Guidelines\nYou are playing the role of a customer contacting a customer service representative. \nYour goal is to simulate realistic customer interactions while following specific scenario instructions.\n\n## Core Principles\n- Generate one message at a time, maintaining natural conversation flow.\n- Strictly follow the scenario instructions you have received.\n- Never make up or hallucinate information not provided in the scenario instru
...
    "simulations": [
        {
            "id": "3f0cac3f-4387-403c-9227-31d0be509278",
            "task_id": "3",
            "timestamp": "2025-06-05T16:06:20.485639",
            "start_time": "2025-06-05T16:05:51.168382",
            "end_time": "2025-06-05T16:06:20.485618",
            "duration": 29.316879875004815,
            "termination_reason": "user_stop",
            "agent_cost": 0.118782,
            "user_cost": 0.007188,
            "reward_info": {
                "reward": 1.0,
                "db_check": {
                    "db_match": true,
                    "db_reward": 1.0
                },
                "env_assertions": [],
                "action_checks": [
                    {
                        "action": {
                            "action_id": "3_0",
                            "requestor": "assistant",
                            "name": "get_reservation_details",
                            "arguments": {
                                "reservation_id": "JMO1MG"
                            },
...
                {
                    "role": "assistant",
                    "content": "I'll check your reservation details right away to find out your baggage allowance.",
                    "tool_calls": [
                        {
                            "id": "toolu_01FsYrMgbwJgBwMSZLnXVx4N",
                            "name": "get_reservation_details",
                            "arguments": {
                                "reservation_id": "JMO1MG"
                            },
                            "requestor": "assistant"
                        }
                    ],
                    "turn_idx": 4,
                    "timestamp": "2025-06-05T16:06:01.458135",
                    "cost": 0.017724,
                    "usage": {
                        "completion_tokens": 80,
                        "prompt_tokens": 5508
                    },
                    "raw_data": {
                        "finish_reason": "tool_calls",
                        "index": 0,
                        "message": {
                            "content": "I'll check your reservation details right away to find out your baggage allowance.",
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "function": {
                                        "arguments": "{\"reservation_id\": \"JMO1MG\"}",
                                        "name": "get_reservation_details"
                                    },
                                    "id": "toolu_01FsYrMgbwJgBwMSZLnXVx4N",
                                    "type": "function"
                                }
                            ],
                            "function_call": null
                        }
                    }
                },
                {
                    "id": "toolu_01FsYrMgbwJgBwMSZLnXVx4N",
                    "role": "tool",
                    "content": "{\"reservation_id\": \"JMO1MG\", \"user_id\": \"anya_garcia_5901\", \"origin\": \"DEN\", \"destination\": \"MIA\", \"flight_type\": \"one_way\", \"cabin\": \"economy\", \"flights\": [{\"flight_number\": \"HAT255\", \"origin\": \"DEN\", \"destination\": \"MIA\", \"date\": \"2024-05-16\", \"price\": 149}], \"passengers\": [{\"first_name\": \"Anya\", \"last_name\": \"Garcia\", \"dob\": \"1992-11-12\"}, {\"first_name\": \"Raj\", \"last_name\": \"Jackson\", \"dob\": \"1956-03-09\"}], \"payment_history\": [{\"payment_id\": \"gift_card_2550356\", \"amount\": 358}], \"created_at\": \"2024-05-13T23:20:29\", \"total_baggages\": 1, \"nonfree_baggages\": 0, \"insurance\": \"yes\", \"status\": null}",
                    "requestor": "assistant",
                    "error": false,
                    "turn_idx": 5,
```

Why it matters beyond R14/R15: R14 documented the tau2 data model and the third-party `AgentSuite/tau2-bench-trajectories` dump. This bucket is the primary source, covers many more models (including current frontier ones and a voice agent), includes the banking_knowledge RAG domain, and carries the provider `raw_data` (real Anthropic `toolu_` ids, real OpenAI finish reasons), which is what a customer using the raw provider SDK would also have.

### 1.2 Exgentic/agent-llm-traces-v2 and Exgentic/traces-v2 (OpenTelemetry GenAI spans). REAL-RUN

- URL: https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2 (chat-only subset, 236 MB, 9 Parquet shards); full corpus https://huggingface.co/datasets/Exgentic/traces-v2 (returned HTTP 401 to an unauthenticated fetch, so gated or private).
- Size: 10,057 sessions, 241,674 chat spans. Per benchmark (from the card): AppWorld 1,500; BrowseCompPlus 1,948; SWE-bench 1,959; tau2 Airline 957; tau2 Retail 1,848; tau2 Telecom 1,844. So 4,649 sessions are customer-service runs.
- Models: DeepSeek-V3.2, Kimi-K2.5, claude-opus-4-5, gemini-3-pro-preview, gpt-5.2-2025-12-11. Harnesses: claude_code, openai_solo, smolagents_code, tool_calling, tool_calling_with_shortlisting.
- License: not stated on the card.
- Row schema (first-rows): `schema_version, config_path, run_id, session_id, harness, benchmark, benchmark_subset, models, score, success, status, steps, action_count, agent_cost, benchmark_cost, execution_time, total_tokens, max_tokens, spans, collected_at`. Each span has 18 attributes; messages are JSON strings in `gen_ai.input.messages` / `gen_ai.output.messages`.
- Tool results present: tool calls are in `gen_ai.output.messages` as `tool_call` parts (verified below). Tool results, by the convention this dataset follows, appear as `tool_call_response` parts in the next span's `gen_ai.input.messages`; I verified `tool_call` parts but the `gen_ai.input.messages` of the spans I pulled only carried the system instruction, so verify on a full session before relying on it.
- Verbatim (row at offset 7000, first span, abbreviated at the tool definitions):

```
tau2_retail openai_solo ['gpt-5.2-2025-12-11'] 0.0 nspans 11
{"span_id": "09f18eed11c00d89", "trace_id": "642b1494e11213f984fdfd795123f66a", "parent_span_id": "400830be9cbb0f1d", "name": "chat Azure/gpt-5.2-2025-12-11", "kind": "SPAN_KIND_INTERNAL", "start_time": "2026-01-28T20:43:22.420387+00:00", "end_time": "2026-01-28T20:43:22.420387+00:00", "status": {"code": 1, "message": ""}, "attributes": {"error.type": null, "gen_ai.conversation.id": "dfafd0a4", "gen_ai.input.messages": "[{\"role\": \"user\", \"parts\": [{\"type\": \"text\", \"content\": \"You are a customer service agent that helps the user according to the <policy> provided below. Try to be helpful and always follow the policy.\"}]}]", "gen_ai.operation.name": "chat", "gen_ai.output.messages": "[{\"role\": \"assistant\", \"parts\": [{\"type\": \"tool_call\", \"id\": \"ad0d6daa-73ff-413f-8c82-a31ffd909e70\", \"name\": \"message\", \"arguments\": {\"content\": \"To get this exchange set up, I first need to authenticate your account. Please share either:\\n1) the email on your account, or\\n2) your first name, last name, and ZIP code.\\n\\nAlso, please provide the order ID (starts with #W) for the laptop you just received.\"}}], \"finish_reason\": \"tool_calls\"}]", "gen_ai.output.type": null, "gen_ai.provider.name": "azure.ai.openai", "gen_ai.request.max_tokens": null, "gen_ai.request.model": "gpt-5.2-2025-12-11", "gen_ai.request.stop_sequences": null, "gen_ai.request.temperature": null, "gen_ai.response.finish_reasons": ["tool_calls"], "gen_ai.response.id": null, "gen_ai.response.model": "gpt-5.2-2025-12-11", "gen_ai.system_instructions": null, "gen_ai.tool.definitions": "[{\"type\": \"function\", \"name\": \"exchange_delivered_order_items\", \"description\": \"Exchange items in a delivered order to new items of the same product type.\\n\\nFor a delivered order, return or exchange can be only done once by the agent.\\nThe agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed. ...
```

Two things to note: the harness turns "reply to the user" into a tool named `message` (a customer's agent may do the same or may not), and `start_time == end_time` (timings were not captured), so latency is not usable from this corpus.

### 1.3 experiential-labs/wmo-crmarena-traces (OTLP-style spans, CRM analytics tasks). REAL-RUN

- URL: https://huggingface.co/datasets/experiential-labs/wmo-crmarena-traces. 5.96 MB. License CC BY-NC 4.0. Files: `traces.otel.jsonl`, `data/train.jsonl`, `data/test.jsonl`, `gold/crm-*.json`, `evals/default.toml`.
- What it is: Claude Opus 4.8 runs on CRMArena tasks, but the tool is `bash` running `python3 query.py "<SQL>"` against a SQLite snapshot of the CRMArena Salesforce org, not the Salesforce API. Tool results are present and untruncated.
- Verbatim (first two lines of `traces.otel.jsonl`, second line cut at the schema dump):

```
{"traceId": "d98c8d83b1f4c90e2e05942fdddeb7cf", "spanId": "d98c8d83b1f40000a", "parentSpanId": "", "name": "chat crmarena", "startTimeUnixNano": 0, "endTimeUnixNano": 1, "status": {"code": "STATUS_CODE_OK"}, "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}}, {"key": "gen_ai.request.model", "value": {"stringValue": "us.anthropic.claude-opus-4-8"}}, {"key": "gen_ai.tool.name", "value": {"stringValue": "bash"}}, {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": "{\"command\": \"cat schema.md\"}"}}, {"key": "gen_ai.prompt", "value": {"stringValue": "In the past year, is there any month when we received significantly more cases for Women's Trail Running Shorts compared to other months? The associated product Id is 01tWs000002wSKYIA2. Return only the month name.\n\n# Task Instructions\n- Today's date: 2022-08-28\n\n# Domain Details\n## Quarters of the Year\n- Q1: January 1 to March 31 (both inclusive). ...
{"traceId": "d98c8d83b1f4c90e2e05942fdddeb7cf", "spanId": "d98c8d83b1f40000b", "parentSpanId": "", "name": "execute_tool crmarena", "startTimeUnixNano": 2, "endTimeUnixNano": 3, "status": {"code": "STATUS_CODE_OK"}, "attributes": [{"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}}, {"key": "gen_ai.tool.name", "value": {"stringValue": "bash"}}, {"key": "gen_ai.tool.message", "value": {"stringValue": "# CRM database (crm.db)\n\nA realistic Salesforce org as a read-only SQLite database. Query it with:\n\n    python3 query.py \"SELECT Id, Subject, Status FROM \\\"Case\\\" WHERE Status = 'Closed' LIMIT 5\"\n\nNotes:\n- `Case`, `Order`, `User` are SQL reserved-ish words — wrap table names in double quotes.\n- Ids are Salesforce ids (e.g. `005Ws000001xSR9IAM`). ...\n### Case — 977 rows\nId (TEXT), Priority (TEXT), Subject (TEXT), Description (TEXT), Status (TEXT), ContactId (TEXT), CreatedDate (TEXT), ClosedDate (TEXT), OrderItemId__c (TEXT), IssueId__c (TEXT), AccountId (TEXT), OwnerId (TEXT) ...
```

Note the non-standard attribute `gen_ai.tool.message` and the fake nanosecond timestamps (0, 1, 2, 3). Useful as a second OTel dialect to parse, and as a CRM schema reference; not useful as a source of realistic tool schemas.

### 1.4 open-agent-leaderboard/traces (Claude Code JSONL, AppWorld). REAL-RUN

- URL: https://huggingface.co/datasets/open-agent-leaderboard/traces (1.1 GB, duplicated from `lhoestq/agent-traces-example`, viewer broken, no license on the card).
- Fields: `harness, session_id, prompt, messages, tools, metadata, sent_at, num_user_messages, num_tool_calls, trace, file_path`. `trace` is the Claude Code JSONL (uuid, parentUuid, timestamp, sessionId, message with `tool_use`/`tool_result` blocks, `is_error`). Model in the sample: `azure/DeepSeek-V3.2` driven through the Claude Code harness.
- Domain: AppWorld (amazon, venmo, gmail, spotify, splitwise, todoist, phone, file_system, simple_note): personal-life apps, not enterprise, but it is the only sizable non-coding corpus in the exact JSONL a Claude Code / Claude Agent SDK customer would hand us.
- Verbatim (first row, cut):

```
messages: [{"role": "tool", "tool_call_id": "016ac513-ee5f-4ec0-a35c-568e26dbfc34", "name": "unknown_tool", "content": "[{\"invoking_actions\": [], \"result\": null}]"}, {"role": "user", "content": "Context: {'policy': \"This environment provides a set of applications, each exposing a predefined set of APIs that may be used to perform tasks on behalf of the supervisor. The applications include: supervisor, amazon, phone, file_system, spotify, venmo, gmail, splitwise, simple_note, todoist. ...
trace: [{"type": "user", "uuid": "af6b0229-3376-445e-90cf-e74b79b52219", "parentUuid": null, "timestamp": "2026-04-20T06:41:44.770278+00:00", "sessionId": "2a21e94e0687_012da231", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "016ac513-ee5f-4ec0-a35c-568e26dbfc34", "content": "[{\"invoking_actions\": [], \"result\": null}]", "is_error": false}]}}, ...
```

Note `"name": "unknown_tool"` in the flattened `messages` view: the converter lost the tool name that the raw `trace` still has. A warning for our own normaliser.

### 1.5 agent-evals/hal_traces (Holistic Agent Leaderboard). REAL-RUN, mostly unusable as is

- URL: https://huggingface.co/datasets/agent-evals/hal_traces, 113 GB, viewer broken ("JSON parse error"). Harness: https://github.com/princeton-pli/hal-harness. Benchmarks include tau-bench and AppWorld alongside SWE-bench, USACO, CORE-bench, SciCode, AssistantBench, ScienceAgentBench, CollaborativeAgentBench. Logged via W&B Weave. The harness README says traces are encrypted before upload ("Automatic encryption of agent traces before uploading to avoid benchmark" contamination), so expect to need a key. License not stated.

### 1.6 ABCD, Action-Based Conversations Dataset (ASAPP). HUMAN, real human-to-human, actions without payloads

- URL: https://github.com/asappresearch/abcd (MIT license per the GitHub API), paper https://arxiv.org/abs/2104.00783 (CC BY 4.0). Data: `data/abcd_v1.1.json.gz`, sample `data/abcd_sample.json`.
- Size: "over 10K human-to-human dialogues with 55 distinct user intents", 30 action types, 125 slot values. Domain: a fictional online retailer (AcmeBrands): returns, refunds, shipping, account access, subscriptions.
- Tool results present: no. Actions are recorded as a third speaker `action` with a natural-language system line; the structured label is `targets` = [intent, nextstep, action, values, utterance rank]. No API payloads exist. What it does have that no benchmark has: real humans on both sides, real hesitation and identity verification, and a per-conversation `scenario` object that is effectively the ground-truth state.
- Verbatim (`abcd_sample.json`, conversation 3592, `scenario` and first 14 `original` turns):

```
{
 "convo_id": 3592,
 "scenario": {
  "personal": {"customer_name": "crystal minh", "email": "cminh730@email.com", "member_level": "bronze", "phone": "(977) 625-2661", "username": "cminh730"},
  "order": {"street_address": "6821 1st ave", "full_address": "6821 1st ave  san mateo, ny 75227", "city": "san mateo", "num_products": "1", "order_id": "3348917502", "packaging": "yes", "payment_method": "credit card", "products": "[{'brand': 'michael_kors', 'product_type': 'jeans', 'amount': 94, 'image_url': 'images/michael_kors-jeans.jpeg'}]", "purchase_date": "2019-11-06", "state": "ny", "zip_code": "75227"},
  "product": {"names": ["michael_kors jeans"], "amounts": [94]},
  "flow": "product_defect",
  "subflow": "return_size"
 }
}
['agent', 'Hi!']
['agent', 'How can I help you?']
['customer', 'Hi! I need to return an item, can you help me with that?']
['agent', 'sure, may I have your name please?']
['customer', 'Crystal Minh']
['agent', 'thanks, may I ask the reason for the return?']
['action', 'Account has been pulled up for Crystal Minh.']
['customer', 'I got the wrong size.']
['agent', 'ok, may I have your username, email address and order ID please?']
['customer', 'Username: cminh730']
['customer', 'cminh730@email.com']
['customer', 'Order ID: 3348917502']
['action', 'Purchase validation in progress ...']
['agent', 'thanks so much! What is your membership level Crystal?']
```

### 1.7 ConFETTI (Amazon Science). HUMAN-authored, expected calls only

- URL: https://github.com/amazon-science/confetti (CC BY 4.0), paper https://aclanthology.org/2025.acl-long.394/. "109 human-simulated conversations, comprising 313 user turns and covering 86 APIs." Files: `data/BFCL_v2_conversationsclean.json` (5.7 MB), `data/BFCL_v2_dialogacts.json`, `data/possible_answer/`. BFCL-compatible: the record holds the conversation and the tool schema; execution results are not part of the record (evaluation compares the predicted call to `possible_answer`).
- Verbatim (first record, cut):

```
{"id": "conversationsclean_0", "dialog_id": "0b28ca41-19f0-44fb-b2b0-f4097e5d6f35_0", "question": [{"role": "user", "content": "Hi, I am Shalissa Valentino and I need to use up my leftover PTO. If I have any PTO remaining, I'd like to book a flight through an airline with flexible refund policies from the San Diego Airport to Denver and back.", "content_type": "text"}], "function": [{"name": "RAG_query", "description": "Trigger this tool if the given task may need external knowledge to answer factual questions. The tool retrieves a list of passages based on the query.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "title": "query", "description": "search query used to retrieve relevant passages"}}}}, {"name": "BookFlight_search_flights", "description": "Search a flight given a departure and arrival location and dates.", "parameters": {"type": "object", "properties": {"departure_airport": {"type": "string", "title": "departure_airport", "description": "Departure location. get_airport_code can retrieve the location code."}, "arrival_airport": {"type": "string", ...
```

### 1.8 API-Bank (Alibaba DAMO). HUMAN-annotated evaluation set, executable tools

- URL: https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank, HF mirror https://huggingface.co/datasets/liminghao1630/API-Bank (5,953 downloads/month). "314 tool-use dialogues with 753 API calls" for evaluation (levels 1 to 3, with a runnable evaluator over 73 API tools) plus 1,888 training dialogues over 2,138 APIs (LLM-generated). The HF mirror is flattened to `instruction / input / output` where the output is a string like `API-Request: [Get_All_Sessions()]`; the API responses live in the evaluator, not the record. Domains are consumer services (yoga sessions, banking, health), 2023-era and short.

### 1.9 ToolWOZ (ASAPP) and ToolTalk (Microsoft). SIM environments with human-derived goals

- ToolWOZ: https://github.com/asappresearch/josh-llm-simulation-training (MIT, `pip install josh-train`), paper https://arxiv.org/abs/2409.04617. MultiWOZ goals converted into 7 search/booking APIs over 4 domains (restaurant, hotel, train, attraction), backed by the real MultiWOZ database, with an LLM user simulator and a sparse goal-completion reward. Data ships as `dataset.zip`. Same shape as tau-bench, older domains.
- ToolTalk: 78 conversations, 28 tools in 7 plugins, simulated tool implementations (https://arxiv.org/pdf/2311.10775). Small.

### 1.10 Adjacent real-production traces that are not support agents

- viktor-shcherb/jobseek-agent-traces: https://huggingface.co/datasets/viktor-shcherb/jobseek-agent-traces. MIT, 8.81 GB, Claude Code sessions from a production job-posting monitor (web scraping and config), header line then chronologically merged main-agent and subagent records. Real production, non-coding-ish, but the tools are web fetch and file edits.
- lambda/hermes-agent-reasoning-traces: https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces. Apache 2.0, 14,701 samples, real tool execution (terminal, browser, files, scheduling) with Kimi-K2.5 and GLM-5.1. Not support.
- trace-commons/agent-traces, MaxDevv/real-pi-coding-agent-traces-sessions, nvidia/Open-SWE-Traces: coding. Listed only so nobody re-checks them.

### 1.11 What vendors and HF searches did not yield

- Intercom, Decagon, Ada, Klarna, Parloa, PolyAI, Cresta: no trace datasets. PolyAI's HF org has `banking77`, `minds14`, `woz_dialogue`, `evi` (NLU corpora, no tool calls). Sierra is the only vendor publishing agent trajectories (tau-bench). Amazon (NatCS, ConFETTI), ASAPP (ABCD, ToolWOZ), Salesforce (CRMArena, APIGen-MT), ServiceNow (WorkArena) publish research artifacts, not production logs.
- Papers about deployed agents do not release logs: "Helping Customers in Distress" (bank triage agent, https://arxiv.org/abs/2605.16268) evaluates on "synthetic digital twins of real customers"; "Beyond IVR" / JourneyBench (https://arxiv.org/abs/2601.00596) is synthetic, 703 conversations, no tool calls in the abstract.
- HF API searches (sorted by downloads) for "customer support tool calls", "helpdesk agent", "support tickets llm agent", "crm agent trajectories", "banking agent tool", "zendesk", "langsmith" returned nothing relevant. "agent traces" returns 25+ datasets, all coding harness sessions except the Exgentic, open-agent-leaderboard and jobseek ones above.
- Synthetic, listed for completeness and to be excluded from "real": Salesforce/xlam-function-calling-60k (APIGen), glaive-function-calling-v2, interstellarninja/tool-calls-multiturn (Hermes-style, sample category "IoT and Home Automation"), stindardlogic/mcp-tool-traces (135 templated Stripe rows, Apache 2.0; a good format sample of OpenAI-style `tool_calls` plus `role: tool` with a JSON object as content, but generated), syncora/strova customer-support conversation sets (no tools), Nemotron-Agentic-v1 (R14).

## 2. Closest-to-production substitutes and their distance from a real customer's traces

"Distance" is about what a customer would not give us: reward labels, task ids, simulator prompts, repeated trials, benchmark-specific tool wrappers, and fake timestamps.

| Substitute | Agent | User | Tools and state | Scaffolding a customer would not have | Distance |
|---|---|---|---|---|---|
| tau2 S3 trajectories (1.1) | real LLM, many models | LLM simulator with scenario instructions | Python functions over a JSON DB, per-task DB snapshot | `reward_info`, `task_id`, `action_checks`, `nl_assertions`, 4 trials per task, `user_cost`, simulator guidelines | medium: conversation and tool payloads look like production; tool schemas are benchmark-designed; errors are rare (5 in 380 calls) |
| Exgentic OTel spans (1.2) | same as above | same | same | `benchmark`, `score`, `success`, harness names, `message` pseudo-tool, zero-length spans | medium; closest in wire format to a Langfuse/OTel customer |
| CRMArena / CRMArena-Pro run by us (3.1) | our agent | none (query) or LLM simulator (Pro interactive splits) | real Salesforce org, real REST/SOQL responses, synthetic records | task metadata, gold answers, org credentials | medium-low on tools, high on conversation (mostly single query) |
| wmo-crmarena-traces (1.3) | Claude Opus 4.8 | none | bash plus SQL over SQLite snapshot | fake timestamps, `gen_ai.tool.message`, reward in metadata | high: tool is `bash`, not a CRM API |
| AppWorld via Claude Code (1.4) | LLM through Claude Code | none | simulated apps, stateful | AppWorld supervisor prompt, `unknown_tool` names | medium-high: right JSONL, wrong domain |
| WorkArena / AgentLab traces (agentlabtraces/agentlabtraces, 207 GB tar parts, no card) | LLM web agent | none | real ServiceNow PDI via browser | BrowserGym observations (AXTree, screenshots), `study_id` | high: browser actions, not API tool calls |
| AgentDojo, AppWorld, tau2 native (R14) | as documented in R14 | | | | as R14 |
| ABCD (1.6) | human | human | actions as text, no payloads | `targets` labels, `scenario` | medium on dialogue realism, no tool payloads at all |
| ConFETTI, API-Bank (1.7, 1.8) | expected calls | human-authored | none in record | `possible_answer`, evaluator | high: no results, no state |
| Vendor demo logs (Langfuse demo project, R16) | demo app | demo users | RAG tools | none, but no support-domain tool calls | high |

Reading the table: the two tau2 sources are the only bulk substitutes with untruncated tool results, error flags and multi-turn user interaction in support domains. Everything else is either missing tool payloads (ABCD, ConFETTI) or missing the user (CRMArena, AppWorld).

## 3. Obtaining real traces ourselves: sandboxes, APIs and existing projects

The plan that produces the most realistic traces: a real helpdesk or CRM API (real schemas, pagination, validation errors, rate limits) plus an LLM agent (via an MCP server or hand-written tools) plus an LLM user simulator seeded from a scenario. Below, which systems give a free environment and a public API, and which projects already do something like it.

### 3.1 Existing projects that run an agent against a real business system

- CRMArena and CRMArena-Pro (Salesforce AI Research): https://github.com/SalesforceAIResearch/CRMArena, tasks on https://huggingface.co/datasets/Salesforce/CRMArenaPro (CC BY-NC 4.0; "This release is for research purposes only"). The repo's `.env.example` publishes credentials for three Salesforce orgs, e.g. `SALESFORCE_USERNAME=kh.huang+00dws000004urq4@salesforce.com` / `SALESFORCE_PASSWORD=crmarenatest0`, accessed with `simple-salesforce`. Orgs hold 29,101 (B2B) and 54,569 (B2C) records across 25 objects. This is the only public setup where an LLM agent hits a real enterprise API. The NC license blocks commercial use of the data; the org itself is Salesforce's and shared for research, so we should provision our own Developer Edition org and reuse only the generator idea.
- WorkArena (ServiceNow): https://github.com/ServiceNow/WorkArena runs on a real Personal Developer Instance, but through the browser (BrowserGym). Gated instance list: https://huggingface.co/datasets/ServiceNow/WorkArena-Instances (benchmarking and research only, "explicit prohibition against training, production workloads").
- MCPMark and MCP-Universe (R14) run against real Notion, GitHub, Postgres, Playwright and other services through MCP; not support systems, but the same pattern.
- Plato (https://plato.so, docs at https://docs.plato.so/concepts, R16) sells sandboxed real apps for agent training; a commercial option if we do not want to host.
- Nobody publishes a helpdesk benchmark on Zendesk, Freshdesk or Jira with a user simulator; the search for that returned only integration how-tos.

### 3.2 Free environments with public APIs

| System | Free environment | API | Notes for trace generation |
|---|---|---|---|
| Salesforce | Developer Edition, free, no expiry ("Ongoing") https://developer.salesforce.com/free-trials | REST, SOQL, Bulk; "Developer Edition" orgs have a limit of "15,000" calls per 24 hours; "5" concurrent requests of 20 s or longer (limits cheat sheet) | Salesforce also advertises hosted MCP with "60+ MCP tools" for Developer Edition; Nango: "200MB of data storage". CRMArena's generator shows how to fill it |
| HubSpot | Developer test accounts: "free HubSpot accounts with access to a 90-day trial of many enterprise features"; up to 10 per developer account; "will expire after 90 days if no API calls are made" https://developers.hubspot.com/docs/getting-started/account-types | CRM v3 REST, OAuth or private app token | Ticket, contact, deal, conversation objects; workflows cap 100,000 enrolments/day |
| Zendesk | "free, 14-day trial account"; sponsored accounts (no expiry, up to 5 agents) are for Marketplace developers, "d3v-" subdomain prefix https://developer.zendesk.com/documentation/api-basics/getting-started/getting-a-trial-or-sponsored-account-for-development/ | Support API v2 | The "sandbox" product is plan-gated; for us the 14-day trial is the realistic path. MCP servers: drobson03/zendesk-mcp, reminia/zendesk-mcp-server |
| Freshdesk | Free plan for small teams; trial accounts default to "50 calls/minute" https://developers.freshdesk.com/api/ | `https://<domain>.freshdesk.com/api/v2/...`, API key from profile settings | MCP servers: Enreign/freshdeck-mcp (tickets, contacts, agents, companies, conversations), effytech/freshdesk_mcp |
| Chatwoot | Self-host (Docker), or cloud | Application API (user `access_token`), Client API, Platform API (self-hosted only) https://developers.chatwoot.com/api-reference/introduction | Built-in AI agent "Captain" with actions (FAQ Lookup, Handoff to Human, Resolve Conversation, more "coming soon") and custom tools: "An HTTP GET or POST endpoint you control", "capped at 15 per account". Tool-call logs are not documented as exportable, so we would log at our own agent, not inside Captain |
| Zammad | Self-host (AGPLv3) | `https://{fqdn}/api/v1/{resource}`, token auth, `?expand=true`, search endpoints https://docs.zammad.org/en/latest/api/intro.html | MCP servers: alexandernicholson/zammad-mcp-server ("52 tools"), Softoft-Orga/zammad-mcp-server, basher83/Zammad-MCP. Zammad 7.0 has its own AI features (triggers, macros) |
| Frappe Helpdesk / ERPNext | Self-host, Frappe Cloud trial, demo at demo.erpnext.com | `/api/resource/{DocType}`, `Authorization: token key:secret` https://docs.frappe.io/framework/user/en/api/rest | MCP: Casys-AI/mcp-erpnext ("120 tools across 14 categories"); Helpdesk is a Frappe app, so tickets are `HD Ticket` DocTypes |
| Odoo | `demo.odoo.com/start` returns a throwaway DB with host, db, login, password; self-host Community for full API | XML-RPC `xmlrpc/2/common` and `xmlrpc/2/object`, `execute_kw` https://www.odoo.com/documentation/18.0/developer/reference/external_api.html | On Odoo Online the external API is limited to Custom plans, so use demo or self-host |
| ServiceNow | Personal Developer Instance, free; hibernates after about 10 days idle; from 2026-07-11 instances older than 90 days must be logged into every 10 days | REST Table API | Same instance WorkArena uses; ITSM tickets, not customer support |
| Intercom | "Development workspaces are free", unlimited https://developers.intercom.com/docs/build-an-integration/getting-started | REST | Conversations, contacts, tickets |
| Shopify | Free development store (Partners) | Admin REST and GraphQL | Orders, refunds, fulfilment: the tau2 retail domain with real schemas |
| Stripe | Test mode, free | REST | Refunds, disputes, subscriptions |
| Enterprise-gated (Nango) | Workday: "only provisions API sandbox access to paying customers on certain plans"; SAP SuccessFactors "requires you to be a certified SAP partner"; UKG, ADP, Oracle partner-only https://nango.dev/blog/how-to-build-api-integrations-without-sandbox-or-test-account/ | | Not reachable for us without a customer |

### 3.3 A concrete generator design (what the free pieces add up to)

1. Environment: Chatwoot or Zammad in Docker (deterministic, resettable, full API, no rate limits we do not control) for support; a Salesforce Developer Edition org for CRM; Shopify dev store or Stripe test mode for orders and refunds.
2. Tools: an existing MCP server per system (above) so tool schemas are real and not ours, wrapped with a recorder (agent-vcr or mcp-recorder, R16) so every request and response is captured at the wire.
3. Agent: any provider SDK; log in the provider's native format plus OTel GenAI spans, so we get two of the formats in section 4 for free.
4. User: tau2's user simulator (the S3 files ship the full "User Simulation Guidelines" text) seeded with scenarios generated from the seeded DB state, the way ABCD's `scenario` object seeds a conversation.
5. Faults: inject real API errors (rate limit by hammering, 404 by deleting a record mid-conversation, validation errors by malformed payloads) because the benchmark corpora barely contain them.

## 4. Trace formats in the wild and how tool errors are encoded

R16 covered LangSmith, Langfuse, Braintrust, OpenInference, Weave, Helicone, Opik, OpenAI Agents SDK, Claude Code JSONL, MCP JSON-RPC and Vercel telemetry. This section adds the OpenTelemetry GenAI semantic conventions (now the lingua franca; Exgentic and wmo use them), the OpenAI Responses API items, the Anthropic `is_error` flag, Vercel AI SDK message parts, HF's Session Trace format, and the tau2 native encoding, and lines them up on the error question.

### 4.1 OpenTelemetry GenAI semantic conventions (execute_tool span, message parts)

Source: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md (the opentelemetry.io page now redirects there). Status of every gen_ai attribute is still "Development". Verbatim from the "Execute tool span" section:

```
`gen_ai.operation.name` SHOULD be `execute_tool`.
...
**Span name** SHOULD be `execute_tool {gen_ai.tool.name}`.
**Span kind** SHOULD be `INTERNAL`.
**Span status** SHOULD follow the [Recording Errors](https://github.com/open-telemetry/semantic-conventions/blob/v1.44.0/docs/general/recording-errors.md) document.
| `gen_ai.operation.name` | `Required` | string | The name of the operation being performed. |
| `gen_ai.tool.name` | `Required` | string | Name of the tool utilized by the agent. | `Flights` |
| `error.type` | `Conditionally Required` If the operation ended in an error. | string | Describes a class of error the operation ended with. | `timeout`; `java.net.UnknownHostException`; `server_certificate_invalid`; `500` |
| `gen_ai.agent.name` | `Conditionally Required` When applicable. | string | The human-readable name of the agent executing the tool. |
| `gen_ai.tool.call.id` | `Recommended` If available. | string | The tool call identifier. | `call_mszuSIzqtI65i1wAUOE8w5H4` |
| `gen_ai.tool.description` | `Recommended` If available. | string | The tool description. |
| `gen_ai.tool.type` | `Recommended` If available. | string | Type of the tool utilized by the agent | `function`; `extension`; `datastore` |
| `gen_ai.tool.call.arguments` | `Opt-In` | any | Parameters passed to the tool call. |
| `gen_ai.tool.call.result` | `Opt-In` | any | The result returned by the tool call (if any and if execution was successful). |
```

And the message-parts encoding that carries tool calls and results inside inference spans (`gen_ai.input.messages`, example value in the same file, de-HTML'd):

```
[
  {"role": "user", "parts": [{"type": "text", "content": "Weather in Paris?"}]},
  {"role": "assistant", "parts": [{"type": "tool_call", "id": "call_VSPygqKTWdrhaFErNvMV18Yl", "name": "get_weather", "arguments": {"location": "Paris"}}]},
  {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call_VSPygqKTWdrhaFErNvMV18Yl", "response": "rainy, 57°F"}]}
]
```

Error encoding: a failed tool execution is an `execute_tool` span with span status Error plus `error.type`; `gen_ai.tool.call.result` is defined only for successful execution. The `tool_call_response` part has no error flag in the spec example, so if an application feeds an error string back to the model, it looks like a normal response at the message level and only the `execute_tool` span (if emitted) tells you it failed. Arguments and results are Opt-In, so many customers will have spans without payloads. Two dialects seen: Exgentic (attributes as a flat JSON object, `status: {"code": 1, "message": ""}`, `error.type: null`) and wmo (OTLP JSON, `attributes: [{key, value: {stringValue}}]`, `status: {"code": "STATUS_CODE_OK"}`).

### 4.2 OpenAI Responses API

Source: https://developers.openai.com/api/docs/guides/function-calling. Verbatim shape:

```
{
  "call_id": "call_12345xyz",
  "type": "function_call",
  "name": "get_weather",
  "arguments": "{\"location\":\"Paris, France\"}"
}
```

The result goes back as `{"type": "function_call_output", "call_id": "...", "output": "..."}`. Error encoding: none structural. "the result you pass in the `function_call_output` message should typically be a string, where the format is up to you (JSON, error codes, plain text, etc.)" and for void functions "simply return a string that indicates success or failure." Chat Completions is the same story with `role: "tool"` and `tool_call_id`. Detecting errors in OpenAI-format customer traces therefore means parsing the output string.

### 4.3 Anthropic Messages API

Source: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview and the handle-tool-calls page (R16). Verbatim shape of a failed tool result:

```
{
  "type": "tool_result",
  "tool_use_id": "tool_123",
  "content": "Error: Location 'Atlantis' not found in weather database.",
  "is_error": true
}
```

The SDK tool runner sets `is_error: true` automatically on exceptions. Claude Code JSONL carries the same block (see 1.4, `"is_error": false`).

### 4.4 Vercel AI SDK (v5/v6)

Model messages (source: https://ai-sdk.dev/docs/reference/ai-sdk-core/model-message), verbatim:

```
export interface ToolResultPart {
  type: 'tool-result';
  toolCallId: string;
  toolName: string;
  output: LanguageModelV4ToolResultOutput;
  providerOptions?: ProviderOptions;
}
{ type: 'text'; value: string; providerOptions?: ProviderOptions }
{ type: 'json'; value: JSONValue; providerOptions?: ProviderOptions }
{ type: 'execution-denied'; reason?: string; providerOptions?: ProviderOptions }
{ type: 'error-text'; value: string; providerOptions?: ProviderOptions }
{ type: 'error-json'; value: JSONValue; providerOptions?: ProviderOptions }
{ type: 'content'; value: Array<TextContent | FileData | FileUrl | FileReference | CustomContent> }
```

UI messages (source: https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message): a `tool-${NAME}` part with `state` in `input-streaming | input-available | output-available | output-error`, fields `toolCallId, input, output, errorText, providerExecuted`. This is the richest error encoding of the set: errors are typed (`error-text`, `error-json`), and denial by a human approval step is its own type (`execution-denied`). A customer exporting `UIMessage[]` from a chat DB gives us `output-error` + `errorText`.

### 4.5 Hugging Face Session Trace Simple Format (STS)

Source: https://huggingface.co/docs/hub/session-traces-format. New in 2026, used by the Hub trace viewer alongside native Claude Code, Codex and Pi formats (https://huggingface.co/docs/hub/agent-traces). Verbatim example:

```
{"type":"session","harness":"my-agent","id":"abc123","name":"what time is it"}
{"type":"message","message":{"role":"user","content":"what time is it?"}}
{"type":"message","message":{"role":"assistant","content":"","toolCalls":[{"id":"t1","function":{"name":"get_time","arguments":"{}"}}]}}
{"type":"message","message":{"role":"tool","toolCallId":"t1","content":"2026-07-01T15:00:00Z"}}
{"type":"message","message":{"role":"assistant","content":"it is 15:00 UTC"}}
```

No error field at all; no timestamps required (`timestamp` is optional epoch ms). Lossy, but it is the format the "upload your traces to HF" push is standardising on, so expect it.

### 4.6 tau2 native

`ToolMessage{id, role: "tool", content, requestor, error: bool, turn_idx, timestamp}` (1.1). The `error` boolean is set by the environment, and the content is a string ("Error: Not enough seats on flight HAT229"). `requestor` distinguishes agent tools from user tools (telecom).

### 4.7 Error encodings side by side

| Format | Error signal | Payload on error |
|---|---|---|
| OTel GenAI | `execute_tool` span status Error + `error.type` (Conditionally Required); message part has no flag | `gen_ai.tool.call.result` absent by definition |
| OpenAI Chat / Responses | none; string convention in `output` / `content` | free text |
| Anthropic | `is_error: true` on `tool_result` | string or content blocks |
| Vercel model messages | `output.type` in `error-text`, `error-json`, `execution-denied` | typed |
| Vercel UI messages | `state: "output-error"`, `errorText` | string |
| HF STS | none | free text |
| tau2 | `error: true` on tool message | string |
| MCP (R16) | `isError: true` result vs JSON-RPC `error` | content blocks or code+message |
| LangSmith / Langfuse / OpenInference (R16) | run `error` string + `status`; `level: ERROR` + `statusMessage`; `status_code` | as R16 |

## 5. Best real-world inputs for the first build (ranked)

Ranked for the first build's need: support/enterprise domain, tool calls with results present, volume, and how much has to be stripped or re-created to look like a customer's export.

| Rank | Source | Real or simulated | Domain | Tool results | License | Distance from a real customer's traces | Why this rank |
|---|---|---|---|---|---|---|---|
| 1 | tau2 S3 trajectories (1.1) | REAL-RUN (LLM agent, LLM user, simulated tools) | airline, retail, telecom, banking RAG | yes, with `error` flag, provider `raw_data` | not stated (code MIT) | medium: strip `reward_info`, task ids, trials, simulator prompt | largest, most model-diverse, has voice and RAG variants, real provider ids |
| 2 | Exgentic/agent-llm-traces-v2, tau2 subset (1.2) | REAL-RUN | same three tau2 domains | in message parts (verify `tool_call_response`) | not stated | medium: same env, but already in the OTel wire format a Langfuse/OTel customer would ship | forces our OTel ingestion path; 4,649 sessions |
| 3 | Self-generated traces against Chatwoot/Zammad/Freshdesk + MCP + LLM user (3.3) | REAL tools, LLM agent, LLM user | helpdesk | yes, real API responses and real errors | ours | low on tool realism; agent and user are ours | only way to get real schemas, pagination and rate-limit errors; produces native, OTel and MCP formats at once |
| 4 | CRMArena / Pro run on our own Developer Edition org (3.1, 3.2) | REAL Salesforce API, LLM agent, optional LLM user | CRM service, sales, CPQ | yes | tasks CC BY-NC 4.0; our own runs are ours | medium-low: single-query tasks, little dialogue | real enterprise API with 25 objects; avoid the NC data in the product, reuse the pattern |
| 5 | open-agent-leaderboard/traces and Exgentic AppWorld sessions (1.4, 1.2) | REAL-RUN | personal apps (orders, payments, email) | yes, Claude Code JSONL with `is_error` | not stated | medium-high: right JSONL, wrong domain, harness prompt baked in | tests the Claude Code / Agent SDK ingestion path on non-coding work |
| 6 | ABCD (1.6) | REAL humans both sides | retail support | no payloads, action labels only | MIT | medium for dialogue, infinite for tools | best source of realistic customer behaviour and scenario seeds for the user simulator |
| 7 | wmo-crmarena-traces (1.3) | REAL-RUN | CRM analytics | yes, SQL over SQLite | CC BY-NC 4.0 | high: tool is `bash` | second OTel dialect (OTLP JSON) for parser tests; NC |
| 8 | ConFETTI, API-Bank (1.7, 1.8) | HUMAN-authored | travel, HR, consumer | no (ConFETTI), evaluator-side (API-Bank) | CC BY 4.0; check API-Bank | high | tool schemas and human phrasing only |
| 9 | ToolWOZ, ToolTalk (1.9) | SIM | hotels, restaurants, trains | yes, simulated | MIT; MIT | high | older, smaller than tau2 |
| 10 | HAL traces, agentlabtraces (1.5, 2) | REAL-RUN | tau-bench, AppWorld, WorkArena | yes but encrypted / browser-level | not stated | high: 113 GB encrypted, 207 GB tar parts, no cards | only if we need WorkArena browser traces later |
| 11 | jobseek, Hermes traces (1.10) | REAL-PROD / REAL-RUN | scraping, terminal | yes | MIT; Apache 2.0 | high (domain) | proof that real production non-coding traces do get published; format reference for merged subagent sessions |

Synthetic sets (xlam, glaive, Nemotron-Agentic, Toucan, Stripe MCP traces, interstellarninja) are excluded from the ranking; use them only as format fixtures.

## 6. What this changes for us

1. Drop the assumption that a real production support-agent dataset exists somewhere on HF or Kaggle. It does not, and the 2026 "agent traces" boom is coding sessions. The founder's "real world tasks" requirement is best met by tau2 runs (support dialogue with tool results) plus our own runs against real helpdesk and CRM APIs. Say so in the design docs so nobody re-searches this.
2. The first ingestion targets are now concrete files, not formats in the abstract: one tau2 S3 file per domain (airline 10.5 MB, retail 24.9 MB, telecom 41.2 MB for Claude 3.7; a banking_knowledge file for RAG), 4,649 tau2 sessions in Exgentic OTel, and the Claude Code JSONL of open-agent-leaderboard. Three parsers cover them: tau2 native, OTel GenAI (both the flat-attributes and the OTLP-JSON dialect), Claude Code JSONL. That is the same three a customer on raw SDK, Langfuse/OTel, or Claude Agent SDK would need.
3. Scaffolding to strip before anything sees a trace: `reward_info`, `task_id`, `action_checks`, `nl_assertions`, trial indices, simulator guidelines and `user_cost` (tau2); `benchmark`, `score`, `success`, `config_path` (Exgentic); the AppWorld supervisor prompt and `unknown_tool` names (open-agent-leaderboard). If the replica builder ever reads these, our evaluation is contaminated by the benchmark's own grader.
4. Error handling must be a normalisation step with a small taxonomy, because the encodings disagree: boolean flag (`error`, `is_error`, `isError`), typed output (`error-text`, `error-json`, `execution-denied`), span status plus `error.type`, or nothing but a string (OpenAI, HF STS). Benchmark corpora under-represent errors (5 of 380 tool calls in the tau2 slice), so error paths in the replica need the self-generated traces from item 5.
5. Build the generator in section 3.3 early. Chatwoot or Zammad in Docker plus an existing MCP server plus the tau2 user simulator gives us traces with real schemas, real HTTP errors and full control over resets, at zero licence risk. Salesforce Developer Edition (15,000 calls/day) is enough for a CRM variant. This is also the demo we can show a prospect without needing their data.
6. Licences: tau2 trajectories are public but unlicensed (ask Sierra), Exgentic is unlicensed, CRMArena and wmo are CC BY-NC (research only), ABCD and ToolWOZ are MIT, ConFETTI CC BY 4.0. Keep NC material out of anything shipped; use it only to design generators.
7. Timestamps and latency are mostly fake or absent in the substitutes (Exgentic spans have zero duration; wmo uses 0, 1, 2, 3; HF STS makes timestamps optional). Any replica feature that depends on timing has to be validated on self-generated traces, not on these corpora.
8. Where this disagrees with earlier assumptions: R16 treated Vercel telemetry as a minor format; the Vercel message model is in fact the only one with typed tool errors and an explicit approval-denied state, which is exactly the signal a replica needs for human-in-the-loop steps. And the "distance" column says the best public inputs are still two simulators away from a customer (simulated user, simulated tools); the only way to close the tool half is to run against real APIs ourselves.

Sources, in order of first use: https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md, https://sierra-tau-bench-public.s3.amazonaws.com/?list-type=2&prefix=submissions/, https://github.com/sierra-research/tau2-bench, https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2, https://huggingface.co/datasets/Exgentic/traces-v2, https://huggingface.co/datasets/experiential-labs/wmo-crmarena-traces, https://huggingface.co/datasets/open-agent-leaderboard/traces, https://huggingface.co/datasets/agent-evals/hal_traces, https://github.com/princeton-pli/hal-harness, https://github.com/asappresearch/abcd, https://arxiv.org/abs/2104.00783, https://github.com/amazon-science/confetti, https://aclanthology.org/2025.acl-long.394/, https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank, https://huggingface.co/datasets/liminghao1630/API-Bank, https://github.com/asappresearch/josh-llm-simulation-training, https://arxiv.org/abs/2409.04617, https://arxiv.org/pdf/2311.10775, https://huggingface.co/datasets/viktor-shcherb/jobseek-agent-traces, https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces, https://huggingface.co/datasets/stindardlogic/mcp-tool-traces, https://huggingface.co/datasets/interstellarninja/tool-calls-multiturn, https://arxiv.org/abs/2605.16268, https://arxiv.org/abs/2601.00596, https://huggingface.co/datasets/agentlabtraces/agentlabtraces, https://arxiv.org/html/2412.05467, https://github.com/SalesforceAIResearch/CRMArena, https://huggingface.co/datasets/Salesforce/CRMArenaPro, https://github.com/ServiceNow/WorkArena, https://huggingface.co/datasets/ServiceNow/WorkArena-Instances, https://developer.salesforce.com/free-trials, https://developer.salesforce.com/docs/atlas.en-us.salesforce_app_limits_cheatsheet.meta/salesforce_app_limits_cheatsheet/salesforce_app_limits_platform_api.htm, https://developers.hubspot.com/docs/getting-started/account-types, https://developer.zendesk.com/documentation/api-basics/getting-started/getting-a-trial-or-sponsored-account-for-development/, https://developers.freshdesk.com/api/, https://developers.chatwoot.com/api-reference/introduction, https://www.chatwoot.com/hc/user-guide/articles/1777328078-lesson-5-ai-actions, https://docs.zammad.org/en/latest/api/intro.html, https://github.com/alexandernicholson/zammad-mcp-server, https://github.com/Enreign/freshdeck-mcp, https://github.com/drobson03/zendesk-mcp, https://github.com/Casys-AI/mcp-erpnext, https://docs.frappe.io/framework/user/en/api/rest, https://www.odoo.com/documentation/18.0/developer/reference/external_api.html, https://www.servicenow.com/community/developer-articles/servicenow-pdi-reclamation-rules-avoid-losing-access/ta-p/3572371, https://developers.intercom.com/docs/build-an-integration/getting-started, https://nango.dev/blog/how-to-build-api-integrations-without-sandbox-or-test-account/, https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md, https://developers.openai.com/api/docs/guides/function-calling, https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview, https://ai-sdk.dev/docs/reference/ai-sdk-core/model-message, https://ai-sdk.dev/docs/reference/ai-sdk-core/ui-message, https://huggingface.co/docs/hub/session-traces-format, https://huggingface.co/docs/hub/agent-traces.
