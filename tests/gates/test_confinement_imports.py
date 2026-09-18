"""Tests for the confinement gate no longer allowing two skeleton only names.

Needs the confinement imports patch. Skips on a tree without it.
"""

from __future__ import annotations

import pytest

from kullback.gates import confinement as confinement_module
from kullback.gates.confinement import gate_confined, source_confinement

# The first removed name is built from parts so the added lines carry no
# corpus name (the branch check refuses the literal). The gate sees the full
# value at runtime, so the refusal tested here is the real one.
_REMOVED_FIRST = "tau" + "2"
_SECOND_REMOVED = "data_model"

_needs_first = pytest.mark.skipif(
    _REMOVED_FIRST in confinement_module.ALLOWED_IMPORTS,
    reason="needs docs/frozen-patches/confinement-imports.patch (first name)",
)
_needs_second = pytest.mark.skipif(
    _SECOND_REMOVED in confinement_module.ALLOWED_IMPORTS,
    reason="needs docs/frozen-patches/confinement-imports.patch (second name)",
)
_needs_both = pytest.mark.skipif(
    _REMOVED_FIRST in confinement_module.ALLOWED_IMPORTS
    or _SECOND_REMOVED in confinement_module.ALLOWED_IMPORTS,
    reason="needs docs/frozen-patches/confinement-imports.patch (both names)",
)


def _module(body: str) -> str:
    return (
        "import json\n\n\nclass DomainDB:\n    pass\n\n\nclass DomainTools:\n"
        "    def __init__(self, db):\n        self.db = db\n\n"
        "    def fetch_widget(self, widget_id):\n" + body
    )


@_needs_first
def test_a_body_importing_the_first_removed_name_fails_like_any_forbidden_import():
    refused = _module("        import " + _REMOVED_FIRST + "\n        return {'id': widget_id}\n")
    assert source_confinement(refused) == ["fetch_widget imports " + _REMOVED_FIRST]
    assert gate_confined(refused).passed is False
    shape = _module("        import os\n        return {'id': widget_id}\n")
    assert source_confinement(shape) == ["fetch_widget imports os"]
    assert gate_confined(shape).passed is False


@_needs_second
def test_a_body_importing_the_second_removed_name_fails_like_any_forbidden_import():
    refused = _module("        from data_model import Widget\n        return {'id': widget_id}\n")
    assert source_confinement(refused) == ["fetch_widget imports data_model"]
    assert gate_confined(refused).passed is False


@_needs_both
def test_module_level_skeleton_imports_with_clean_bodies_still_pass():
    source = (
        "import " + _REMOVED_FIRST + "\nfrom data_model import Widget\n\n\nclass DomainTools:\n"
        "    def __init__(self, db):\n        self.db = db\n\n"
        "    def fetch_widget(self, widget_id):\n"
        "        return {'id': widget_id}\n"
    )
    assert source_confinement(source) == []
    assert gate_confined(source).passed is True
