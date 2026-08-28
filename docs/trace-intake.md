# Customer trace intake

Questions to answer when a customer's traces arrive (first: the vendor export expected in early September 2026, D56). Fill in the answers here; the ingestion todo is written around them.

| # | Question | Answer | Why it matters |
|---|----------|--------|----------------|
| 1 | Source: whose agent produced the traces (own agent, design partner, vendor export) and which domain (support, ticketing, sales, ops)? | vendor export; domain unknown | Domain decides which tool families and policy shapes to expect. |
| 2 | Format: raw model API request/response logs (OpenAI or Anthropic message JSON), a tracing export (LangSmith, Langfuse, OpenTelemetry GenAI spans), or the agent framework's own log? One example file answers this. | | Ingestion parser; how tool calls, results and errors are encoded (D45, R23 section 4). |
| 3 | Tool results: kept in full or truncated? | | Truncated results block S0 reconstruction for reads (D39, D40). |
| 4 | Tool definitions: is the `tools` list sent to the model present in the log? | | Day-one contracts for unseen tools (ADR-0006 rung 2). |
| 5 | Grouping: are the turns of one conversation linked by a session or thread id? | | Runs cannot be assembled without it (R18 filters). |
| 6 | Errors: do failed tool calls appear with their error text? How many per tool? | | The Environment copies the real error encoding; none seen means a guess (A27). |
| 7 | System prompt: present per Call? | | Policy lines compile to Hard constraints from it (D43 case 3). |
| 8 | Volume: conversations, period, distinct tools, share of multi-turn Runs. | | Task floor and CI (D36); Simulated user hold-out (D44). |
| 9 | Outcome signals: CSAT, escalation, reopen, refund reversal, anything the customer records after the Run? | | Reference confirmation (step 5) and audit calibration. |
| 10 | Labels: can a domain expert mark 20 to 50 conversations pass or fail? | | D50 proof 3; D48 check 1 second pair of eyes. |
| 11 | Policy or knowledge documents available beyond the system prompt? | | ADR-0006 rung 4. |
| 12 | Where the traces live: gitignored folder in `monitoring-tool/`, a folder outside the repository, or a bucket. Never committed. | | Customer data must not enter git history. |
| 13 | Retention and sharing rules the customer set (what may leave their boundary, for how long). | | ADR-0002 and rung 6 (snapshot inside their boundary). |
