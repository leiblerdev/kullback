"""Builder modules: traces in, Environment, Tasks, References and re-rolls out. The pipeline scheduler,
the parallel worker pool and the search-memo cache live here because build.py is their only caller.
The Verifiers are the Examiner's (kullback.examiner, D123); the derivation moved there in phase 5."""
