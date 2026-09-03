# ADR-0002: Customer data is per customer and never pooled

Status: accepted.

Context: models trained or optimized from traces could be stronger if traces from many customers were pooled.

Decision: each customer's traces produce a model only that customer runs. Traces are never pooled across customers.

Why: the target customers treat prompts and outputs as core IP, and "your data, your model" is the trust story that survives first contact. Pooling is a hard trust and legal line for them.

Consequence: the optimization layer is a per-customer service, not a data moat. A weaker model per customer is the accepted price, permanently.
