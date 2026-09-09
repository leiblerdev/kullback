"""Builder modules: traces in, Environment, Tasks, References and re-rolls out. The pipeline scheduler
and the search-memo cache live here because build.py is their only caller; the worker pool moved to
kullback.runner.parallel when the Examiner's derivation began using it too (D163).
The Verifiers are the Examiner's (kullback.examiner, D123); the derivation moved there in phase 5."""
