"""The rule-driven Simulated user, which now lives in `kullback.user.rules` (D214).

D214 made the Simulated user a first-class agent of the harness with its own package, so the module
moved and this name stayed: every caller written against `kullback.builder.user_sim` (D44, D77,
D115, D196, D210) keeps working, and the Builder still reaches the rules through the Builder. The
re-export is a star import on purpose, so a name added to the rules is here the day it is added and
this file never has to be edited again; it goes away when the next decision moves the callers.
"""

from __future__ import annotations

from kullback.user.rules import *  # noqa: F401,F403
from kullback.user.rules import (  # noqa: F401  - the underscored helpers a caller or a test reads
    _bare_value,
    _closes,
    _field_of,
    _flattened,
    _matching_rows,
    _names_change,
    _norm,
    _request_sentences,
    _row_value,
    _sentences,
    _spoken_args,
    _stated_in,
    _words,
)
