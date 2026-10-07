"""The names the value strip (D196) and the Intent record (D123) were always imported under.

The Builder no longer writes Intents: the Spec writer does. What stays here is a re-export, so a
caller that loads `kullback.builder.intent` by name, as the user factory does, keeps reading the
strip from `kullback.user.value_strip` and the Intent record from `kullback.runner.records`.
"""

from __future__ import annotations

from kullback.runner.records import Intent as Intent
from kullback.runner.records import apply_intent as apply_intent
from kullback.user.value_strip import strip_intent as strip_intent
from kullback.user.value_strip import value_strip as value_strip
