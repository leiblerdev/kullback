"""Publishing an Environment: the package a build leaves, its card, and the fetch that brings it back (D221).

`package.py` turns a workdir into a self-contained Environment package with a manifest and a leak
scan; `card.py` renders the dataset card and the organisation card from that manifest; `client.py`
is the whole surface a hosting API is spoken to through, so the rest of the harness never imports a
hosting library and a test drives an in-memory stand-in; `publish.py` joins them into the three
commands the CLI exposes.

Nothing is re-exported here on purpose: `publish`, `export` and `fetch` are the names of both a
submodule and a function, and a package that binds the function to the name leaves nobody able to
import the module. Callers name the module they want.
"""
