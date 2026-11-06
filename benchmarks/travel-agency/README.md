# Travel Agency

65 tasks for a support agent at TripDesk, a travel agency that sold a trip it does not operate. Every task is a
broken trip from a real public complaint, graded in code on what the agent changed. [card.md](card.md) has the full
description, scoring and judge setup, and [RESULTS.md](RESULTS.md) has the host check.

The tasks share one environment, so it is kept once:

- `environment/`: the database, tools, runtime, Docker files, policy and code guards
- `tasks/`: one file per task with only what differs (the customer's message, the overlay, the expected end state, the
  judge questions and the right walks)
- `export.py`: writes the Harbor tasks, standard library only

```bash
python export.py --out harbor
harbor run -p harbor -a oracle -n 4
```

The same tasks, already exported, are on Hugging Face as
[leibler/travel-agency](https://huggingface.co/datasets/leibler/travel-agency).
