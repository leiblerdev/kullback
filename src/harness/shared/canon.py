"""The canonicalizer (D39): rules as data turn a value into one canonical string, and equality by column class (D73), with a judged pair cached in an equivalence table a human can overturn (D84)."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, getcontext, localcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from harness.shared.records import ClassifiedBy, ColumnClass, content_hash

STRING_TAG = "\x00"  # a canonical string a typed value can never produce (D39)
EXEMPT = STRING_TAG + "<exempt>"  # the tag keeps it out of reach of any string
COLUMN_CLASSES = ("exempt", "hard", "semantic")
NUMERIC = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
LEADING_ZERO = re.compile(r"[+-]?0\d+(\.\d*)?")
THOUSANDS = re.compile(r"[+-]?\d{1,3}(,\d{3})+(\.\d+)?")
DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")
# Extended ISO only: a date with dashes, optionally a time with colons. A digit run is an id.
ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?)?\s?(Z|z|[+-]\d{2}:?\d{2})?"
)
RESERVED = ("null", "true", "false")
# The judge's D84 vocabulary (judge.py's `equivalence` use); None means it did not decide.
EQUIVALENCE_VERDICTS: dict[str, Optional[bool]] = {
    "equivalent": True, "not_equivalent": False, "abstain": None,
}
CompareRoute = Literal["exempt", "canon", "cache", "judge", "unresolved"]


class CanonRules(BaseModel):
    """The canonicalizer's rules, as data: learned per customer, reviewed in the setup check (D39)."""
    model_config = ConfigDict(frozen=False)
    lowercase: bool = True
    collapse_whitespace: bool = True
    numbers: bool = True
    number_precision: Optional[int] = None
    currency: bool = True
    currency_symbols: list[str] = Field(default_factory=lambda: ["$", "€", "£", "¥"])
    currency_codes: list[str] = Field(default_factory=lambda: ["USD", "EUR", "GBP", "JPY"])
    currency_symbol_codes: dict[str, str] = Field(
        default_factory=lambda: {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
    )
    timestamps: bool = True
    timestamp_formats: list[str] = Field(default_factory=list)
    assume_utc: bool = True
    id_patterns: dict[str, str] = Field(default_factory=dict)
    id_upper: bool = True
    id_strip_chars: str = ""
    unordered_lists: list[str] = Field(default_factory=list)
    unordered_all: bool = False
    case_sensitive_paths: list[str] = Field(default_factory=list)
    default_class: ColumnClass = "hard"


class Comparison(BaseModel):
    """One comparison of two values: the canonical strings, the answer and how it was reached."""
    equal: bool
    route: CompareRoute
    a: str
    b: str
    key: Optional[str] = None
    judge_used: bool = False
    judge_called: bool = False
    classified_by: Optional[ClassifiedBy] = None
    note: Optional[str] = None


class EquivalenceEntry(BaseModel):
    """One cached (column, canonical a, canonical b) verdict; a human entry outranks a judged one."""
    key: str
    column: str
    a: str
    b: str
    equal: bool
    classified_by: ClassifiedBy = "llm"
    judge_version: str = "0"
    note: Optional[str] = None


class EquivalenceTable(BaseModel):
    """The per-customer equivalence table (D84): a file the review and the audit can open."""
    version: str = "0"
    entries: list[EquivalenceEntry] = Field(default_factory=list)


DEFAULT_RULES = CanonRules()


def _rules(rules: Optional[CanonRules]) -> CanonRules:
    return DEFAULT_RULES if rules is None else rules


# --- canonical values ---

def canon_value(
    value: Any, column_class: ColumnClass = "hard", rules: Optional[CanonRules] = None, path: str = ""
) -> str:
    """The canonical string for a value; exempt columns collapse to one sentinel so they never hash."""
    if column_class not in COLUMN_CLASSES:
        raise ValueError(f"unknown column class: {column_class!r}")
    if column_class == "exempt":
        return EXEMPT
    return _canon(value, _rules(rules), path)


def _canon(value: Any, rules: CanonRules, path: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float, Decimal)):
        return _number(value if isinstance(value, Decimal) else Decimal(str(value)), rules)
    if isinstance(value, (list, tuple, set, frozenset)):
        parts = [_canon(item, rules, path) for item in value]
        if isinstance(value, (set, frozenset)) or rules.unordered_all or _named(path, rules.unordered_lists):
            parts = sorted(parts)
        return json.dumps(parts, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, dict):
        out = {str(k): _canon(v, rules, _join(path, str(k))) for k, v in value.items()}
        return json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if isinstance(value, (datetime, date)):
        return _canon_str(value.isoformat(), rules, path)
    if isinstance(value, str):
        return _canon_str(value, rules, path)
    raise TypeError(f"canon has no rule for a {type(value).__name__}; give it a JSON value")


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _named(path: str, names: list[str]) -> bool:
    return path in names or (bool(path) and path.rsplit(".", 1)[-1] in names)


def _escape(text: str) -> str:
    """A string that spells a typed value or a container is tagged, so the two never collide."""
    if text.lower() in RESERVED or text == EXEMPT:
        return STRING_TAG + text
    if text[:1] in ("[", "{"):
        try:
            json.loads(text)
        except ValueError:
            return text
        return STRING_TAG + text
    return text


def _canon_str(value: str, rules: CanonRules, path: str = "") -> str:
    text = value.strip()
    cased = _named(path, rules.case_sensitive_paths)
    as_id = _as_id(text, rules, cased)
    if as_id is not None:
        return _escape(as_id)
    as_time = _as_timestamp(text, rules)
    if as_time is not None:
        return as_time
    as_number = _as_number(text, rules)
    if as_number is not None:
        return as_number
    if rules.collapse_whitespace:
        text = re.sub(r"\s+", " ", text)
    return _escape(text.lower() if rules.lowercase and not cased else text)


def _as_id(text: str, rules: CanonRules, cased: bool = False) -> Optional[str]:
    if not any(re.fullmatch(pattern, text) for pattern in rules.id_patterns.values()):
        return None
    for char in rules.id_strip_chars:
        text = text.replace(char, "")
    return text.upper() if rules.id_upper and not cased else text


def _as_timestamp(text: str, rules: CanonRules) -> Optional[str]:
    if not rules.timestamps or len(text) < 8:
        return None
    if DATE_ONLY.fullmatch(text):
        return text
    moment = None
    if ISO_TIMESTAMP.fullmatch(text):
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00").replace("z", "+00:00"))
        except ValueError:
            moment = None
    if moment is None:
        for fmt in rules.timestamp_formats:
            try:
                moment = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if moment is None:
        return None
    if moment.tzinfo is None:
        if not rules.assume_utc:
            return moment.isoformat()
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    shape = "%Y-%m-%dT%H:%M:%S.%fZ" if moment.microsecond else "%Y-%m-%dT%H:%M:%SZ"
    return moment.strftime(shape)


def _as_number(text: str, rules: CanonRules) -> Optional[str]:
    if not rules.numbers or not text:
        return None
    code = None
    if rules.currency:
        text, code = _strip_currency(text, rules)
    if THOUSANDS.fullmatch(text):
        text = text.replace(",", "")
    if not NUMERIC.fullmatch(text) or LEADING_ZERO.fullmatch(text):
        return None
    try:
        number = _number(Decimal(text), rules)
    except InvalidOperation:
        return None
    return number if code is None else f"{number} {code}"


def _strip_currency(text: str, rules: CanonRules) -> tuple[str, Optional[str]]:
    """The amount without its currency, and the currency itself: 25 USD is not 25 EUR (D39)."""
    code = None
    for symbol in rules.currency_symbols:
        if symbol and symbol in text:
            text = text.replace(symbol, "")
            code = rules.currency_symbol_codes.get(symbol, symbol)
            break
    for name in rules.currency_codes:
        pattern = rf"\b{re.escape(name)}\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            code = name
            break
    return text.strip(), code.lower() if code else None


def _number(value: Decimal, rules: CanonRules) -> str:
    """The number as text, with no digit rounded away: two long ids must not meet (D39)."""
    if not value.is_finite():
        return str(value).lower()
    if rules.number_precision is not None:
        step = Decimal(1).scaleb(-rules.number_precision)
        with localcontext() as context:
            context.prec = _wide_precision(value, rules.number_precision)
            value = value.quantize(step, rounding=ROUND_HALF_UP)
    if value == 0:
        return "0"
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _wide_precision(value: Decimal, places: int) -> int:
    """Enough significant digits for this value at this many decimal places, never fewer than 28."""
    digits = len(value.as_tuple().digits)
    return max(getcontext().prec, abs(value.adjusted()) + abs(places) + digits + 10)


# --- the one name every other module calls ---

def canonicalize(value: Any, rules: Optional[CanonRules] = None, path: str = "") -> str:
    """The canonical form of a whole value, containers included, under the hard-column rules.

    route.py keys its recordings with this, verdict.py compares with it and compile_env.py
    compares replayed results with it, so the three agree by construction (D39).
    """
    return canon_value(value, "hard", rules, path)


def canonical_args(args: Any, rules: Optional[CanonRules] = None, path: str = "") -> str:
    """The canonical form of a tool call's arguments; the name route.py looks for."""
    return canonicalize(args, rules, path)


# --- rows ---

def canon_record(
    row: dict, schema: Any = None, table: Optional[str] = None, rules: Optional[CanonRules] = None
) -> dict[str, str]:
    """A row as canonical strings, one class per column from the EntitySchema (D73)."""
    active = _rules(rules)
    return {
        str(key): canon_value(
            value, _class_of(schema, table, str(key), active), active, _join(table or "", str(key))
        )
        for key, value in row.items()
    }


def record_hash(
    row: dict, schema: Any = None, table: Optional[str] = None, rules: Optional[CanonRules] = None
) -> str:
    """Content hash of a canonicalized row; exempt columns cannot move it."""
    return content_hash(canon_record(row, schema, table, rules))


def _class_of(schema: Any, table: Optional[str], name: str, rules: CanonRules) -> ColumnClass:
    if schema is None:
        return rules.default_class
    columns = getattr(schema, "columns", None)
    if columns is not None:
        for column in columns:
            if column.name == name and (table is None or column.table == table):
                return column.class_
        return rules.default_class
    if isinstance(schema, dict):
        scope = schema.get(table) if table and isinstance(schema.get(table), dict) else schema
        found = scope.get(name)
        if isinstance(found, str):
            return found  # type: ignore[return-value]
    return rules.default_class


# --- equality by column class ---

def compare(
    a: Any,
    b: Any,
    column_class: ColumnClass = "hard",
    rules: Optional[CanonRules] = None,
    judge: Optional[Callable[[str, str, str], Any]] = None,
    table: Union[EquivalenceTable, str, Path, None] = None,
    column: str = "",
    judge_version: str = "0",
) -> Comparison:
    """Compare two values for one column: exempt always equal, hard by string, semantic by judge (D84)."""
    if column_class not in COLUMN_CLASSES:
        raise ValueError(f"unknown column class: {column_class!r}")
    if column_class == "exempt":
        return Comparison(equal=True, route="exempt", a=EXEMPT, b=EXEMPT, classified_by="code")
    active = _rules(rules)
    left = _canon(a, active, column)
    right = _canon(b, active, column)
    if left == right:
        return Comparison(equal=True, route="canon", a=left, b=right, classified_by="code")
    if column_class == "hard":
        return Comparison(equal=False, route="canon", a=left, b=right, classified_by="code")

    key = pair_key(column, left, right)
    path = None if isinstance(table, EquivalenceTable) else table
    loaded = table if isinstance(table, EquivalenceTable) else load_table(path) if path else None
    if loaded is not None:
        cached = lookup(loaded, column, left, right)
        if cached is not None:
            return Comparison(
                equal=cached.equal, route="cache", a=left, b=right, key=key,
                judge_used=cached.classified_by == "llm", classified_by=cached.classified_by,
                note=cached.note,
            )
    if judge is None:
        return Comparison(equal=False, route="unresolved", a=left, b=right, key=key,
                          note="a semantic column with no judge")

    try:
        answer = judge(column, left, right)
    except Exception as error:  # one failed judge call leaves this pair unresolved, not the Verdict
        return Comparison(equal=False, route="unresolved", a=left, b=right, key=key,
                          judge_called=True,
                          note=f"the judge raised {type(error).__name__}: {error}")
    verdict, note = read_judge_answer(answer)
    if verdict is None:
        return Comparison(equal=False, route="unresolved", a=left, b=right, key=key,
                          judge_called=True, note=note or "the judge did not decide")
    if loaded is not None:
        put(loaded, column, left, right, verdict, "llm", judge_version, note)
        if path is not None:
            save_table(loaded, path)
    return Comparison(
        equal=verdict, route="judge", a=left, b=right, key=key,
        judge_used=True, judge_called=True, classified_by="llm", note=note,
    )


def read_judge_answer(answer: Any) -> tuple[Optional[bool], Optional[str]]:
    """The judge's answer as (verdict, note); None is an abstain. Never coerced with bool() (D84).

    judge.py answers with a JudgeResult whose verdict is one of equivalent, not_equivalent, abstain.
    A bool or a dict carrying a bool `equal` is accepted too. Anything else is a bug in the caller,
    and raising is the only safe answer: coercion turned every unreadable answer into a silent pass.
    """
    if isinstance(answer, bool):
        return answer, None
    if isinstance(answer, dict):
        note = answer.get("reason") or answer.get("note")
        if "equal" in answer:
            if not isinstance(answer["equal"], bool):
                raise TypeError(f"the judge's 'equal' must be a bool, not {answer['equal']!r}")
            return answer["equal"], note
        verdict = answer.get("verdict")
        refused = bool(answer.get("refused"))
    else:
        note = getattr(answer, "reason", None)
        verdict = getattr(answer, "verdict", None)
        refused = bool(getattr(answer, "refused", False))
    if refused:
        return None, note or "the judge refused"
    if isinstance(verdict, str) and verdict in EQUIVALENCE_VERDICTS:
        return EQUIVALENCE_VERDICTS[verdict], note
    if isinstance(verdict, str):
        raise TypeError(f"unknown equivalence verdict from the judge: {verdict!r}")
    raise TypeError(
        "the judge must answer with a bool, a dict with a bool 'equal', or an equivalence "
        f"verdict; got {type(answer).__name__}"
    )


def equal(
    a: Any,
    b: Any,
    column_class: ColumnClass = "hard",
    rules: Optional[CanonRules] = None,
    judge: Optional[Callable[[str, str, str], Any]] = None,
    table: Union[EquivalenceTable, str, Path, None] = None,
    column: str = "",
    judge_version: str = "0",
) -> bool:
    """True when the two values are the same for this column class."""
    return compare(a, b, column_class, rules, judge, table, column, judge_version).equal


# --- the equivalence table as a file ---

def pair_key(column: str, a: str, b: str) -> str:
    """The cache key for a pair; equality is symmetric, so the pair is sorted first."""
    return content_hash([column, *sorted([a, b])])


def lookup(table: EquivalenceTable, column: str, a: str, b: str) -> Optional[EquivalenceEntry]:
    """The cached entry for a canonical pair, or None."""
    key = pair_key(column, a, b)
    for entry in table.entries:
        if entry.key == key:
            return entry
    return None


def put(
    table: EquivalenceTable, column: str, a: str, b: str, verdict: bool,
    classified_by: ClassifiedBy = "llm", judge_version: str = "0", note: Optional[str] = None,
) -> EquivalenceEntry:
    """Write a pair into the table; a human entry is never overwritten by a machine one."""
    key = pair_key(column, a, b)
    entry = EquivalenceEntry(
        key=key, column=column, a=a, b=b, equal=verdict,
        classified_by=classified_by, judge_version=judge_version, note=note,
    )
    for index, existing in enumerate(table.entries):
        if existing.key != key:
            continue
        if existing.classified_by == "human" and classified_by != "human":
            return existing
        table.entries[index] = entry
        return entry
    table.entries.append(entry)
    return entry


def overturn(
    table: EquivalenceTable, column: str, a: str, b: str, verdict: bool, note: Optional[str] = None,
    workdir: Union[str, Path, None] = None,
) -> EquivalenceEntry:
    """A person overrules a cached pair (D84); every Verdict that used it is queued for a regrade.

    `workdir` is where `record_use` logged which Runs rested on which pair. Given one, the Runs that
    used this pair go into the regrade queue that `regrade.py`'s batch entrypoint consumes; without
    one the table is still corrected, and the caller is on its own for the regrade.
    """
    entry = put(table, column, a, b, verdict, "human", "0", note)
    if workdir is not None:
        queue_regrade(workdir, entry.key, note or f"a person overturned {column}")
    return entry


# --- which Verdicts rest on which cached pair (D84) ---

USES_FILE = "equivalence_uses.jsonl"
QUEUE_FILE = "regrade_queue.jsonl"


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record_use(workdir: Union[str, Path], comparison: Comparison, run_id: str, task_id: str = "") -> None:
    """Note that one Run's Verdict rested on one cached or judged pair, so an overturn can find it."""
    if comparison.key is None:
        return
    _append(Path(workdir) / USES_FILE,
            {"key": comparison.key, "run_id": run_id, "task_id": task_id, "route": comparison.route})


def queue_regrade(workdir: Union[str, Path], key: str, reason: str) -> list[str]:
    """Put every Run that used this pair into the regrade queue; returns the run ids queued."""
    run_ids = sorted({row["run_id"] for row in _rows(Path(workdir) / USES_FILE) if row.get("key") == key})
    for run_id in run_ids:
        _append(Path(workdir) / QUEUE_FILE, {"run_id": run_id, "key": key, "reason": reason})
    return run_ids


def queued_regrades(workdir: Union[str, Path]) -> list[str]:
    """The Runs waiting to be re-scored, in the order they were queued, each once."""
    out: list[str] = []
    for row in _rows(Path(workdir) / QUEUE_FILE):
        if row.get("run_id") and row["run_id"] not in out:
            out.append(row["run_id"])
    return out


def clear_regrade_queue(workdir: Union[str, Path]) -> None:
    """Empty the queue once the Runs in it have been re-scored."""
    path = Path(workdir) / QUEUE_FILE
    if path.is_file():
        path.unlink()


def load_table(path: Union[str, Path]) -> EquivalenceTable:
    """Read the table from a workdir file; a missing file is an empty table."""
    file = Path(path)
    if not file.is_file():
        return EquivalenceTable()
    return EquivalenceTable.model_validate_json(file.read_text(encoding="utf-8"))


def save_table(table: EquivalenceTable, path: Union[str, Path]) -> Path:
    """Write the table, entries in key order so the file diffs cleanly for a reviewer."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    ordered = table.model_copy(update={"entries": sorted(table.entries, key=lambda e: e.key)})
    file.write_text(json.dumps(ordered.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return file


def learn_rules(
    schema: Any = None,
    rows: Optional[Iterable[dict]] = None,
    base: Optional[CanonRules] = None,
    table: Optional[str] = None,
) -> CanonRules:
    """The rules for one customer, learned from the mined schema and the observed rows (D39).

    The id patterns are the ones `mine.py` already found, and every id column is marked case
    sensitive so two ids that differ only in case stay two ids. A list column the traces show in
    two orders is unordered. Save the result with `save_rules` and pass it to every caller; the
    defaults are a starting point, not the customer's rules.
    """
    rules = (base or CanonRules()).model_copy(deep=True)
    patterns = dict(getattr(schema, "id_patterns", None) or {})
    rules.id_patterns = {**rules.id_patterns, **patterns}
    cased = list(rules.case_sensitive_paths)
    for column in getattr(schema, "columns", None) or []:
        name = getattr(column, "name", "")
        if not _looks_like_id(name, getattr(column, "samples", None) or [], rules.id_patterns):
            continue
        for path in (_join(str(getattr(column, "table", "") or ""), str(name)), str(name)):
            if path and path not in cased:
                cased.append(path)
    rules.case_sensitive_paths = cased
    unordered = list(rules.unordered_lists)
    for name in _unordered_columns(rows or []):
        path = _join(table or "", name)
        if path not in unordered:
            unordered.append(path)
    rules.unordered_lists = unordered
    return rules


def _looks_like_id(name: str, samples: Iterable[Any], patterns: dict[str, str]) -> bool:
    if name == "id" or name.endswith("_id"):
        return True
    return any(
        isinstance(sample, str) and any(re.fullmatch(p, sample) for p in patterns.values())
        for sample in samples
    )


def _unordered_columns(rows: Iterable[dict]) -> list[str]:
    """A column the traces show holding one multiset in two orders is order free (D39)."""
    seen: dict[str, list[list[str]]] = {}
    for row in rows:
        for key, value in (row or {}).items():
            if isinstance(value, (list, tuple)):
                seen.setdefault(str(key), []).append([canonicalize(item) for item in value])
    out = []
    for name, observed in seen.items():
        pairs = [(sorted(v), v) for v in observed]
        if any(a[0] == b[0] and a[1] != b[1] for a in pairs for b in pairs):
            out.append(name)
    return out


def load_rules(path: Union[str, Path]) -> CanonRules:
    """Read the customer's canonicalizer rules; a missing file means the defaults."""
    file = Path(path)
    if not file.is_file():
        return CanonRules()
    return CanonRules.model_validate_json(file.read_text(encoding="utf-8"))


def save_rules(rules: CanonRules, path: Union[str, Path]) -> Path:
    """Write the rules as data, for the setup review and for memory.py's Builder tree."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(rules.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return file
