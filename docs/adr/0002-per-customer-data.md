# Customer data is per-customer and never pooled

When we train or optimize models from customer traces, each customer's data produces a model that only that customer runs and benefits from. We never pool traces across customers to train a shared model.

Why: our target customers are AI-native companies whose prompts and outputs are core IP, and the trust story that survives first contact is "your data → your model." Pooling would produce a stronger, cheaper model (more training data), but it is a hard trust/legal red line for this audience and would block adoption at the trust gate (H3).

Consequence: the optimization layer is a per-customer service, not a data moat that compounds across customers. We accept a weaker model per customer in exchange for trust. This is permanent, not a temporary validation-time stance.
