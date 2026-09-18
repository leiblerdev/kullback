"""Tests for the confinement gate no longer allowing two skeleton only names.

Needs the confinement imports patch. Skips on a tree without it.
"""

from __future__ import annotations

import pytest

from kullback.gates import confinement as confinement_module
from kullback.gates.confinement import gate_confined, source_confinement

pytestmark = pytest.mark.skipif(
    "tau2" in confinement_module.ALLOWED_IMPORTS
    or "data_model" in confinement_module.ALLOWED_IMPORTS,
    reason="needs docs/frozen-patches/confinement-imports.patch",
)


def _module(body: str) -> str:
    return (
        "import json\n\n\nclass DomainDB:\n    pass\n\n\nclass DomainTools:\n"
        "    def __init__(self, db):\n        self.db = db\n\n"
        "    def fetch_widget(self, widget_id):\n" + body
    )


def test_a_body_importing_the_first_removed_name_fails_like_any_forbidden_import():
    refused = _module("        import tau2\n        return {'id': widget_id}\n")
    assert source_confinement(refused) == ["fetch_widget imports tau2"]
    assert gate_confined(refused).passed is False
    shape = _module("        import os\n        return {'id': widget_id}\n")
    assert source_confinement(shape) == ["fetch_widget imports os"]
    assert gate_confined(shape).passed is False


def test_a_body_importing_the_second_removed_name_fails_like_any_forbidden_import():
    refused = _module("        from data_model import Widget\n        return {'id': widget_id}\n")
    assert source_confinement(refused) == ["fetch_widget imports data_model"]
    assert gate_confined(refused).passed is False


def test_module_level_skeleton_imports_with_clean_bodies_still_pass():
    source = (
        "import tau2\nfrom data_model import Widget\n\n\nclass DomainTools:\n"
        "    def __init__(self, db):\n        self.db = db\n\n"
        "    def fetch_widget(self, widget_id):\n"
        "        return {'id': widget_id}\n"
    )
    assert source_confinement(source) == []
    assert gate_confined(source).passed is True
