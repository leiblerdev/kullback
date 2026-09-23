"""The world build hashes the import closure of what it delegates to (G29).

The stage walk this file once pinned died with the stage graph; what remains is the
closure half of `kullback.builder.cache_reach`: `closure_hash` over `world_tools.py`
and what it imports, so a cached world is never served entries from before a change
in anything it can call.
"""

from __future__ import annotations

from pathlib import Path

from kullback.builder import cache_reach as reach


def test_module_hash_covers_the_closure():
    """The key hash moves with closure bytes, deterministically, honoring exemptions."""
    from kullback.builder import mine, world_tools
    from kullback.runner.records import content_hash

    assert reach.import_closure("kullback.builder.mine"), "pin needs a nonempty closure"
    plain = content_hash(Path(mine.__file__).read_bytes())[:16]
    assert world_tools._module_hash(mine) != plain
    assert world_tools._module_hash(mine) == world_tools._module_hash(mine)
    assert "kullback.runner.canon" in reach.import_closure("kullback.builder.mine")
    assert reach.closure_hash(mine, exempt=frozenset({"kullback.runner.canon"})) != \
        reach.closure_hash(mine)
    assert "kullback.builder.__init__" in reach.closure_files("kullback.builder.world_tools"), \
        "importing a submodule executes its package init"


def test_two_modules_that_import_each_other_keep_their_own_hashes():
    """A cycle gives both modules one closure, so only the module's own name tells them apart.

    The tool compiler imports the sandbox, and the sandbox reaches the compiler again through
    the effects module, so their closures are the same 42 files. A key that delegates to one of
    them has to say which, or an edit that should rebuild one stage rebuilds the other as well
    and the key stops naming the module it carries.
    """
    from kullback.builder import compile_env, sandbox

    assert reach.closure_files("kullback.builder.compile_env") == \
        reach.closure_files("kullback.builder.sandbox"), "pin needs the cycle to still be there"
    assert reach.closure_hash(compile_env) != reach.closure_hash(sandbox)
