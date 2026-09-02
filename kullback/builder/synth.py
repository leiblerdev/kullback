"""Grow the Starting state past the ids the traces named (D40, D107).

The rows the traces showed are the only evidence of the customer's world, so every synthetic row
is composed from them. A new row starts as a bootstrap of one observed row, which keeps the
co-occurrences a column-by-column sampler loses (a cancelled order carries a cancel reason, a
pending one has no fulfillment), and the rules mined below replace the parts that have to be new.
Every rule is structural. None names a table, a column or a value, which is the overfit check the
next corpus has to pass (D51).

  template          a string leaf is runs of letters, digits and punctuation; a run that always
                    equals another leaf of the same row is a reference, a run that never varies is
                    a literal, a digit run is redrawn, the rest is vocabulary from the observed runs
  identity          a leaf is redrawn rather than copied when it is the id, is unique across the
                    observed rows, or is what another leaf's template refers to
  foreign key       a leaf whose values match another table's ids
  back reference    a list of foreign keys that holds the rows pointing back at this one
  embedded copy     an element of a list that carries a foreign key and repeats fields of the row
                    it names (an order line repeats the item's price)
  mirror            a list of strings that is one field of a list of dicts in the same row
  keyed collection  a dict keyed by each value's own id column (mine.nested_rows' home rule)
  nested key        a leaf whose values are keys of a keyed collection on the row a foreign key
                    names (a payment method id under the order's user)
  sum               a number that equals the sum of one field over a list in the same row

Observed rows are never edited: a synthetic child hangs off a synthetic parent, so a recorded
result stays replayable on the grown database. Generated ids join `EntitySchema.synthetic_rows`,
which is what marks a Run that reads one as assisted (D49). Nothing here calls a model.
"""
from __future__ import annotations

import copy
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from kullback.builder.mine import _entity_of, _plural
from kullback.builder.sandbox import id_field, id_pattern_for
from kullback.runner.records import EntitySchema, canonical_json

# A rule holds when at least this share of the rows that could show it do show it.
RULE_SHARE = 0.8
# A leaf is an identity when it is unique across the observed rows: this share of its values are
# distinct, over at least IDENTITY_EVIDENCE of them (below that, four values that differ are
# not evidence of uniqueness; a product option seen four times is a category).
IDENTITY_SHARE = 0.9
IDENTITY_EVIDENCE = 10
MIN_EVIDENCE = 3
SUM_TOLERANCE = 0.011
_RUN = re.compile(r"[^\W\d_]+|\d+|[\W_]+")


@dataclass
class Rules:
    """Everything mined from one table's observed rows, as the report writes it."""
    table: str
    id_column: Optional[str]
    observed: int
    templates: dict[str, list[dict]] = field(default_factory=dict)
    identity: list[str] = field(default_factory=list)
    id_values: list[str] = field(default_factory=list)
    foreign_keys: dict[str, str] = field(default_factory=dict)
    back_refs: dict[str, str] = field(default_factory=dict)
    embedded: dict[str, dict[str, str]] = field(default_factory=dict)
    mirrors: dict[str, tuple[str, str]] = field(default_factory=dict)
    collections: dict[str, Optional[str]] = field(default_factory=dict)
    collection_keys: dict[str, str] = field(default_factory=dict)
    nested_keys: dict[str, tuple[str, str]] = field(default_factory=dict)
    sums: dict[str, tuple[str, str]] = field(default_factory=dict)
    list_sizes: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class Grown:
    """What `grow` added and what the checks found."""
    added: dict[str, list[str]] = field(default_factory=dict)
    implied: dict[str, int] = field(default_factory=dict)
    rules: dict[str, Rules] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def ids(self) -> list[str]:
        return sorted({row_id for ids in self.added.values() for row_id in ids})


# --- reading rows ---

def table_rows(db: dict, schema: EntitySchema, table: str) -> dict[str, dict]:
    """A table's rows wherever the corpus keeps them: its own table and its home (schema.homes)."""
    rows = {k: v for k, v in (db.get(table) or {}).items() if isinstance(v, dict)}
    home = (schema.homes or {}).get(table)
    if home:
        parent, column = home.split(".", 1)
        for parent_row in (db.get(parent) or {}).values():
            nest = parent_row.get(column) if isinstance(parent_row, dict) else None
            if isinstance(nest, dict):
                rows.update({k: v for k, v in nest.items() if isinstance(v, dict)})
    return rows


def _leaves(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """(path, scalar) pairs of a row; a list adds `[]`, a keyed collection adds `{}`."""
    if isinstance(value, dict):
        if _is_collection(value):
            return [pair for v in value.values() for pair in _leaves(v, prefix + "{}.")]
        return [pair for k, v in value.items() for pair in _leaves(v, f"{prefix}{k}.")]
    if isinstance(value, list):
        return [pair for v in value for pair in _leaves(v, prefix + "[].")]
    return [(prefix.rstrip("."), value)]


def _is_collection(value: dict) -> bool:
    """A dict of dicts where some key is the value's own field: keyed by id, not by column name."""
    if not value or not all(isinstance(v, dict) for v in value.values()):
        return False
    return any(_own_key(key, row) for key, row in value.items())


def _own_key(key: Any, row: dict) -> Optional[str]:
    for name, inner in row.items():
        if isinstance(inner, str) and inner == key and isinstance(name, str):
            return name
    return None


def _scalars(row: dict) -> dict[str, Any]:
    """Scalar leaves of a record dict, nested records included, collections and lists left out."""
    out = {}
    for key, value in row.items():
        if isinstance(value, dict) and not _is_collection(value):
            out.update({f"{key}.{k}": v for k, v in _scalars(value).items()})
        elif not isinstance(value, (dict, list)):
            out[str(key)] = value
    return out


def _get(row: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(row, dict) or part not in row:
            return None
        row = row[part]
    return row


def _set(row: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        row = row.setdefault(part, {})
    row[parts[-1]] = value


def _is_list(path: str) -> bool:
    return path.endswith(".[]")


def _record_path(path: str) -> bool:
    """A path that sits in the row itself, not inside a list or a keyed collection."""
    return ".[]" not in path and ".{}" not in path


def _relative(path: str) -> str:
    """The path inside its innermost list or collection element."""
    return re.split(r"\.\[\]\.|\.\{\}\.", path)[-1]


# --- templates ---

def _runs(text: str) -> list[tuple[str, str]]:
    return [("a" if m.group()[0].isalpha() else "d" if m.group()[0].isdigit() else "o", m.group())
            for m in _RUN.finditer(text)]


def _shape(text: str) -> tuple:
    return tuple((kind, run if kind == "o" else None) for kind, run in _runs(text))


_CASES = {"same": lambda s: s, "lower": str.lower, "upper": str.upper, "title": str.title,
          "capitalize": str.capitalize}


def _template(pairs: list[tuple[dict, dict, str]], path: str) -> Optional[list[dict]]:
    """Mine one leaf's template from (root row, element, value) triples; None without a shape.

    A reference is looked for among the element's own scalar leaves (relative path) and the root
    row's (path with a leading `/`), so an id inside a keyed collection can name its own `source`
    and a root email can name `name.first_name`. A leaf never refers to itself.
    """
    texts = [v for _, _, v in pairs if isinstance(v, str) and v]
    if len(texts) < MIN_EVIDENCE:
        return None
    shape, share = Counter(_shape(t) for t in texts).most_common(1)[0]
    if share / len(texts) < RULE_SHARE or not shape:
        return None
    kept = [(root, elem, v) for root, elem, v in pairs if isinstance(v, str) and _shape(v) == shape]
    slots: list[dict] = []
    for index, (kind, literal) in enumerate(shape):
        runs = [_runs(v)[index][1] for _, _, v in kept]
        if kind == "o":
            slots.append({"kind": "lit", "text": literal})
        elif len(set(runs)) == 1:
            slots.append({"kind": "lit", "text": runs[0]})
        elif kind == "d":
            # Drawn inside the observed range per length, so a month stays a month and a house
            # number a house number; a run that spans its whole range is still every digit.
            by_len: dict[int, list[int]] = {}
            for run in runs:
                by_len.setdefault(len(run), []).append(int(run))
            slots.append({"kind": "digits", "lengths": sorted(Counter(len(r) for r in runs).elements()),
                          "ranges": {str(n): [min(v), max(v)] for n, v in by_len.items()}})
        else:
            ref = _reference(kept, runs, {f"/{path}", _relative(path)})
            slots.append(ref or {"kind": "vocab", "choices": sorted(Counter(runs).elements())})
    return slots


def _reference(kept: list[tuple[dict, dict, str]], runs: list[str], own: set[str]) -> Optional[dict]:
    candidates: Counter = Counter()
    for (root, elem, _), run in zip(kept, runs, strict=True):
        for ref_path, value in _scope(root, elem).items():
            if ref_path in own or not isinstance(value, str) or not value:
                continue
            for case, fn in _CASES.items():
                if fn(value) == run:
                    candidates[(ref_path, case)] += 1
                    break
    if not candidates:
        return None
    (ref_path, case), hits = candidates.most_common(1)[0]
    return {"kind": "ref", "path": ref_path, "case": case} if hits / len(kept) >= RULE_SHARE else None


def _scope(root: dict, elem: dict) -> dict[str, Any]:
    scope = {f"/{k}": v for k, v in _scalars(root).items()}
    if elem is not root:
        scope.update(_scalars(elem))
    return scope


def _render(slots: list[dict], root: dict, elem: dict, rng: random.Random,
            widen: bool = False) -> Optional[str]:
    """`widen` ignores the observed digit ranges: sequential ids fill their range and need room."""
    out = []
    for slot in slots:
        if slot["kind"] == "lit":
            out.append(slot["text"])
        elif slot["kind"] == "digits":
            length = rng.choice(slot["lengths"])
            low, high = [0, 10 ** length - 1] if widen else \
                (slot.get("ranges") or {}).get(str(length), [0, 10 ** length - 1])
            out.append(f"{rng.randint(low, high):0{length}d}")
        elif slot["kind"] == "vocab":
            out.append(rng.choice(slot["choices"]))
        else:
            value = _scope(root, elem).get(slot["path"])
            if not isinstance(value, str) or not value:
                return None
            out.append(_CASES[slot["case"]](value))
    return "".join(out)


def _by_position(texts: list[str], rng: random.Random) -> Optional[str]:
    """A fresh value of the same length, each position drawn from what was seen there.

    The fallback for an id with no run shape in common (airline's `OWZ5XL`, six characters that
    are letters or digits in no fixed order): the alphabet per position is the observed one.
    """
    lengths = {len(t) for t in texts}
    if len(lengths) != 1 or len(texts) < MIN_EVIDENCE:
        return None
    return "".join(rng.choice(sorted({t[i] for t in texts})) for i in range(lengths.pop()))


def _mutate_digits(text: str, rng: random.Random) -> str:
    """The fallback for an identity-bearing string with no template: new digits, same everything else."""
    return "".join("".join(rng.choice("0123456789") for _ in run) if kind == "d" else run
                   for kind, run in _runs(text))


# --- mining ---

def mine_rules(db: dict, schema: EntitySchema) -> dict[str, Rules]:
    """Every structural rule the observed rows show, per table."""
    rows_by_table = {table: table_rows(db, schema, table) for table in schema.tables}
    ids_by_table = {table: set(rows) for table, rows in rows_by_table.items()}
    out: dict[str, Rules] = {}
    for table, rows in rows_by_table.items():
        rules = Rules(table=table, id_column=id_field(schema, table), observed=len(rows),
                      id_values=sorted(rows))
        if not rows:
            out[table] = rules
            continue
        values = _leaf_values(rows)
        rules.list_sizes = _list_sizes(rows)
        rules.collections = _collections(rows, schema, table)
        rules.collection_keys = _collection_keys(rows)
        rules.foreign_keys = _foreign_keys(table, values, ids_by_table, schema, rules.id_column,
                                           rules.collection_keys)
        _mine_identity(rules, rows, values)
        rules.embedded = _embedded(rows, rules.foreign_keys, rows_by_table, schema)
        rules.mirrors = _mirrors(rows, values)
        rules.sums = _sums(rows, values)
        out[table] = rules
    for table, rules in out.items():
        rules.back_refs = _back_refs(table, rules, out, rows_by_table)
        rules.nested_keys = _nested_keys(table, rules, out, rows_by_table)
    return out


def _leaf_values(rows: dict[str, dict]) -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = {}
    for row in rows.values():
        for path, value in _leaves(row):
            values.setdefault(path, []).append(value)
    return values


def _unique(texts: list[str]) -> bool:
    return len(texts) >= IDENTITY_EVIDENCE and len(set(texts)) / len(texts) >= IDENTITY_SHARE


def _mine_identity(rules: Rules, rows: dict[str, dict], values: dict[str, list[Any]]) -> None:
    """Identity leaves and their templates, in passes: the unique ones, then what they refer to."""
    texts_by_path = {p: [v for v in seen if isinstance(v, str) and v] for p, seen in values.items()}
    identity = {p for p, texts in texts_by_path.items() if texts and p not in rules.foreign_keys
                and (p == rules.id_column or _unique(texts))}
    for _ in range(3):
        for path in sorted(identity):
            if path not in rules.templates:
                template = _template(_triples(rows, path), path)
                if template:
                    rules.templates[path] = template
        referred = set()
        for path, template in rules.templates.items():
            for slot in template:
                if slot["kind"] == "ref":
                    referred.add(_absolute(slot["path"], path))
        new = {p for p in referred if p in texts_by_path and p not in identity and p not in rules.foreign_keys}
        if not new:
            break
        identity |= new
    rules.identity = sorted(identity)


def _absolute(ref: str, path: str) -> str:
    """The leaf a template slot refers to, as a path of the table: root refs drop the `/`."""
    if ref.startswith("/"):
        return ref[1:]
    prefix = path[: len(path) - len(_relative(path))]
    return prefix + ref


def _list_sizes(rows: dict[str, dict]) -> dict[str, list[int]]:
    sizes: dict[str, list[int]] = {}

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            if _is_collection(value):
                sizes.setdefault(prefix, []).append(len(value))
                for v in value.values():
                    walk(v, prefix + ".{}")
            else:
                for k, v in value.items():
                    walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(value, list):
            sizes.setdefault(prefix, []).append(len(value))
            for v in value:
                walk(v, prefix + ".[]")

    for row in rows.values():
        walk(row, "")
    return {k: sorted(v) for k, v in sizes.items()}


def _collections(rows: dict[str, dict], schema: EntitySchema, table: str) -> dict[str, Optional[str]]:
    """Keyed-collection paths of this table -> the table stored there (a home), if any."""
    homes = {home: child for child, home in (schema.homes or {}).items()}
    out: dict[str, Optional[str]] = {}
    for row in rows.values():
        for key, value in row.items():
            if isinstance(value, dict) and _is_collection(value):
                out.setdefault(key, homes.get(f"{table}.{key}"))
    return out


def _collection_keys(rows: dict[str, dict]) -> dict[str, str]:
    """Keyed-collection path -> the entry field its keys repeat."""
    out: dict[str, str] = {}
    for row in rows.values():
        for key, value in row.items():
            if isinstance(value, dict) and _is_collection(value):
                for k, entry in value.items():
                    own = _own_key(k, entry)
                    if own:
                        out.setdefault(key, own)
                        break
    return out


def _triples(rows: dict[str, dict], path: str) -> list[tuple[dict, dict, str]]:
    """(root, element, value) for a leaf path, the element being the innermost list or collection member."""
    return [(root, elem, value) for root in rows.values() for elem, value in _walk_path(root, path)]


def _walk_path(node: Any, path: str, elem: Optional[dict] = None):
    elem = node if elem is None else elem
    if not path:
        yield elem, node
        return
    head, _, rest = path.partition(".")
    if head == "[]":
        for item in node if isinstance(node, list) else []:
            yield from _walk_path(item, rest, item if isinstance(item, dict) else elem)
    elif head == "{}":
        for item in node.values() if isinstance(node, dict) else []:
            yield from _walk_path(item, rest, item if isinstance(item, dict) else elem)
    elif isinstance(node, dict) and head in node:
        yield from _walk_path(node[head], rest, elem)


def _foreign_keys(table: str, values: dict[str, list[Any]], ids_by_table: dict[str, set],
                  schema: EntitySchema, id_column: Optional[str],
                  collection_keys: dict[str, str]) -> dict[str, str]:
    """A leaf is a foreign key when its values sit among another table's ids, or when it carries
    that table's own id column name (the customer's naming, the same rule `mine._is_id` trusts) and
    its values match the id shape: the traces show a sample of the world, so the rows a key names
    are often missing and membership alone would miss the key."""
    own_ids = {f"{coll}.{{}}.{key}" for coll, key in collection_keys.items()}
    out = {}
    for path, seen in values.items():
        texts = [v for v in seen if isinstance(v, str) and v]
        if len(texts) < MIN_EVIDENCE or path == id_column or path in own_ids:
            continue
        best: Optional[tuple[float, str]] = None
        for target, ids in ids_by_table.items():
            if target == table or not ids:
                continue
            pattern = id_pattern_for(schema, target, id_field(schema, target))
            if pattern and not all(re.fullmatch(pattern, t) for t in texts[:50]):
                continue
            share = sum(1 for t in texts if t in ids) / len(texts)
            leaf = _relative(path[:-3] if _is_list(path) else path).split(".")[-1]
            leaf = leaf[:-1] if leaf.endswith("s") and _entity_of(leaf[:-1]) != leaf[:-1] else leaf
            named = pattern is not None and (leaf == id_field(schema, target)
                                             or _plural(_entity_of(leaf)) == target)
            if (share >= RULE_SHARE or named) and (best is None or share > best[0]):
                best = (share, target)
        if best:
            out[path] = best[1]
    return out


def _embedded(rows: dict[str, dict], foreign_keys: dict[str, str], rows_by_table: dict[str, dict],
              schema: EntitySchema) -> dict[str, dict[str, str]]:
    """fk path -> {element key: 'row' | 'parent'} for fields copied from the row the key names."""
    out: dict[str, dict[str, str]] = {}
    parents = _parents(rows_by_table, schema)
    for fk_path, target in foreign_keys.items():
        if fk_path.count(".[].") != 1 or ".{}" in fk_path:
            continue
        key = fk_path.split(".[].", 1)[1]
        if "." in key:
            continue
        hits: dict[str, Counter] = {}
        for _, elem, value in _triples(rows, fk_path):
            referenced = rows_by_table[target].get(value)
            if referenced is None:
                continue
            parent = parents.get((target, value))
            for name, seen in elem.items():
                if name == key:
                    continue
                counts = hits.setdefault(name, Counter())
                counts["n"] += 1
                if name in referenced and canonical_json(referenced[name]) == canonical_json(seen):
                    counts["row"] += 1
                elif parent and name in parent and canonical_json(parent[name]) == canonical_json(seen):
                    counts["parent"] += 1
        copied = {name: ("row" if c["row"] >= c["parent"] else "parent")
                  for name, c in hits.items()
                  if c["n"] >= MIN_EVIDENCE and max(c["row"], c["parent"]) / c["n"] >= RULE_SHARE}
        if copied:
            out[fk_path] = copied
    return out


def _parents(rows_by_table: dict[str, dict], schema: EntitySchema) -> dict[tuple[str, str], dict]:
    """(homed table, id) -> the parent row it sits in."""
    out = {}
    for child, home in (schema.homes or {}).items():
        parent, column = home.split(".", 1)
        for parent_row in rows_by_table.get(parent, {}).values():
            nest = parent_row.get(column)
            if isinstance(nest, dict):
                out.update({(child, k): parent_row for k in nest})
    return out


def _mirrors(rows: dict[str, dict], values: dict[str, list[Any]]) -> dict[str, tuple[str, str]]:
    """list-of-strings path -> (list-of-dicts path, key) whose values it repeats, set for set."""
    string_lists = [p for p, seen in values.items() if _is_list(p)
                    and all(isinstance(v, str) for v in seen) and len(seen) >= MIN_EVIDENCE]
    keyed = [(p.split(".[].", 1)[0], p.split(".[].", 1)[1]) for p in sorted(values)
             if p.count(".[].") == 1 and "." not in p.split(".[].", 1)[1]
             and all(isinstance(v, str) for v in values[p])]
    out = {}
    for path in string_lists:
        for list_path, key in keyed:
            if path.startswith(list_path + ".["):
                continue
            n = hit = 0
            for root in rows.values():
                for elem, _ in _walk_path(root, path[:-3]):
                    mine = _get(elem, _relative(path[:-3]))
                    source = _get(elem, list_path)
                    if source is None:
                        source = _get(root, list_path)
                    if not isinstance(mine, list) or not isinstance(source, list) or not source:
                        continue
                    n += 1
                    hit += set(mine) == {e.get(key) for e in source if isinstance(e, dict)}
            if n >= MIN_EVIDENCE and hit / n >= RULE_SHARE and path not in out:
                out[path] = (list_path, key)
    return out


def _sums(rows: dict[str, dict], values: dict[str, list[Any]]) -> dict[str, tuple[str, str]]:
    """number path -> (list path, field) where the number is the sum of the field over the list."""
    numeric = {p for p, seen in values.items()
               if seen and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in seen)}
    lists = {p.split(".[].", 1)[0]: p.split(".[].", 1)[1] for p in numeric
             if p.count(".[].") == 1 and "." not in p.split(".[].", 1)[1]}
    out = {}
    for path in numeric:
        for list_path, fld in lists.items():
            if path.startswith(list_path + ".["):
                continue
            n = hit = 0
            for root, _, value in _triples(rows, path):
                items = _get(root, list_path)
                if not isinstance(items, list) or not items:
                    continue
                total = sum(i.get(fld, 0) for i in items if isinstance(i, dict))
                n += 1
                hit += abs(total - value) <= SUM_TOLERANCE
            if n >= MIN_EVIDENCE and hit / n >= RULE_SHARE:
                out[path] = (list_path, fld)
    return out


def _back_refs(table: str, rules: Rules, all_rules: dict[str, Rules],
               rows_by_table: dict[str, dict]) -> dict[str, str]:
    """list-of-fk path -> the child's scalar fk path that points back at this table."""
    out = {}
    for path, target in rules.foreign_keys.items():
        if not _is_list(path) or not _record_path(path[:-3]):
            continue
        child = all_rules.get(target)
        if child is None or child.id_column is None:
            continue
        for child_path, child_target in child.foreign_keys.items():
            if child_target != table or not _record_path(child_path):
                continue
            n = hit = 0
            for row_id, row in rows_by_table[table].items():
                listed = set(_get(row, path[:-3]) or [])
                for child_id, child_row in rows_by_table[target].items():
                    if _get(child_row, child_path) == row_id:
                        n += 1
                        hit += child_id in listed
            if n >= MIN_EVIDENCE and hit / n >= RULE_SHARE:
                out[path] = child_path
    return out


def _nested_keys(table: str, rules: Rules, all_rules: dict[str, Rules],
                 rows_by_table: dict[str, dict]) -> dict[str, tuple[str, str]]:
    """string path -> (fk path, collection on the referenced row) whose keys the values come from."""
    out = {}
    scalar_fks = {p: t for p, t in rules.foreign_keys.items() if _record_path(p)}
    rows = rows_by_table[table]
    for path in _leaf_values(rows):
        if path in rules.foreign_keys or path == rules.id_column:
            continue
        for fk_path, target in scalar_fks.items():
            for coll in all_rules.get(target, Rules(target, None, 0)).collections:
                n = hit = 0
                for root, _, value in _triples(rows, path):
                    if not isinstance(value, str):
                        continue
                    referenced = rows_by_table[target].get(_get(root, fk_path))
                    if not isinstance(referenced, dict):
                        continue
                    n += 1
                    hit += value in (referenced.get(coll) or {})
                if n >= MIN_EVIDENCE and hit / n >= RULE_SHARE:
                    out[path] = (fk_path, coll)
    return out


# --- generating ---

def grow(db: dict, schema: EntitySchema, targets: dict[str, int], seed: int = 0) -> Grown:
    """Add rows until each named table holds `targets[table]` rows, observed ones included.

    A table whose rows point at a parent with a back reference gets synthetic parents only, so
    observed rows stay as the traces showed them; the parents this implies beyond the target are
    counted in `Grown.implied`. A homed table grows with its parent's collections and cannot be
    targeted on its own.
    """
    rng = random.Random(seed)
    rules = mine_rules(db, schema)
    grown = Grown(rules=rules)
    homed = set(schema.homes or {})
    for table in targets:
        if table not in rules:
            raise ValueError(f"{table} is not a table of this schema")
        if table in homed:
            raise ValueError(f"{table} lives under {schema.homes[table]}; grow its parent instead")
    known = {table: set(table_rows(db, schema, table)) for table in rules}
    observed = {table: dict(table_rows(db, schema, table)) for table in rules}
    counts = _resolve_counts(rules, observed, targets, grown)
    synthetic: dict[str, dict[str, dict]] = {t: {} for t in rules}
    for table in _generation_order(rules):
        for _ in range(counts.get(table, 0)):
            row = _generate(table, rules, observed, synthetic, known, schema, rng)
            if row is None:
                break
            row_id = row[rules[table].id_column]
            synthetic[table][row_id] = row
            known[table].add(row_id)
            db.setdefault(table, {})[row_id] = row
            grown.added.setdefault(table, []).append(row_id)
            for child, child_ids in _collection_ids(row, rules[table]).items():
                grown.added.setdefault(child, []).extend(child_ids)
                known[child] |= set(child_ids)
    _fill_back_refs(rules, synthetic)
    grown.checks = verify(db, schema, rules, observed, grown)
    return grown


def _generation_order(rules: dict[str, Rules]) -> list[str]:
    """Parents before children, by scalar foreign keys; ties by name."""
    pending = sorted(rules)
    done: list[str] = []
    while pending:
        progressed = False
        for table in list(pending):
            deps = {t for p, t in rules[table].foreign_keys.items()
                    if _record_path(p) and t in pending and t != table}
            if not deps:
                done.append(table)
                pending.remove(table)
                progressed = True
        if not progressed:  # a cycle: take them in name order
            done.extend(pending)
            break
    return done


def _resolve_counts(rules: dict[str, Rules], observed: dict[str, dict], targets: dict[str, int],
                    grown: Grown) -> dict[str, int]:
    counts = {t: max(0, n - len(observed[t])) for t, n in targets.items()}
    for table, n in list(counts.items()):
        if n <= 0:
            continue
        for path, target in rules[table].foreign_keys.items():
            if not _record_path(path):
                continue
            parent = rules.get(target)
            if parent and path in parent.back_refs.values() and counts.get(target, 0) <= 0:
                ratio = _children_per_parent(parent, path)
                counts[target] = math.ceil(n / ratio)
                grown.implied[target] = counts[target]
    return counts


def _children_per_parent(parent: Rules, child_path: str) -> float:
    for list_path, fk in parent.back_refs.items():
        if fk == child_path:
            sizes = parent.list_sizes.get(list_path[:-3]) or [1]
            return max(1.0, sum(sizes) / len(sizes))
    return 1.0


def _generate(table: str, rules: dict[str, Rules], observed: dict[str, dict],
              synthetic: dict[str, dict[str, dict]], known: dict[str, set], schema: EntitySchema,
              rng: random.Random) -> Optional[dict]:
    rule = rules[table]
    if not observed[table] or rule.id_column is None:
        return None
    row = copy.deepcopy(rng.choice(list(observed[table].values())))
    _redraw_identity(row, row, rule, rng, known.get(table, set()), rule.id_column)
    pools = {t: {**observed[t], **synthetic[t]} for t in rules}
    for path, target in rule.foreign_keys.items():
        if not _record_path(path):
            continue
        pool = synthetic[target] if _has_back_ref(rules, target, path) else pools[target]
        if pool:
            _set(row, path, rng.choice(sorted(pool)))
    for path, (fk_path, coll) in rule.nested_keys.items():
        parent_row = pools[rule.foreign_keys[fk_path]].get(_get(row, fk_path))
        keys = sorted((parent_row or {}).get(coll) or {})
        if keys:
            for elem, _ in _walk_path(row, path):
                _set(elem, _relative(path), rng.choice(keys))
    embedded_lists = sorted({p.split(".[].", 1)[0] for p in rule.embedded})
    for list_path in embedded_lists:
        _rebuild_list(row, list_path, rule, rules, observed, pools, schema, rng)
    for path, target in rule.foreign_keys.items():
        if _is_list(path) and _record_path(path[:-3]) and path not in rule.back_refs \
                and path not in rule.mirrors:
            pool = sorted(pools[target])
            size = rng.choice(rule.list_sizes.get(path[:-3]) or [1])
            _set(row, path[:-3], rng.sample(pool, min(size, len(pool))) if pool else [])
    for coll_path, child in rule.collections.items():
        _rebuild_collection(row, coll_path, child, rule, rules, known, rng)
    for path, (list_path, key) in rule.mirrors.items():
        for elem, _ in _walk_path(row, path[:-3]):
            source = _get(elem, list_path)
            if source is None:
                source = _get(row, list_path)
            _set(elem, _relative(path[:-3]), [e.get(key) for e in source or [] if isinstance(e, dict)])
    for path, (list_path, fld) in rule.sums.items():
        items = _get(row, list_path) or []
        total = round(sum(i.get(fld, 0) for i in items if isinstance(i, dict)), 2)
        for elem, _ in _walk_path(row, path):
            _set(elem, _relative(path), total)
    for list_path in rule.back_refs:
        _set(row, list_path[:-3], [])
    return row


def _has_back_ref(rules: dict[str, Rules], target: str, child_path: str) -> bool:
    return child_path in rules.get(target, Rules(target, None, 0)).back_refs.values()


def _redraw_identity(root: dict, elem: dict, rule: Rules, rng: random.Random, taken: set,
                     id_column: Optional[str], prefix: str = "") -> None:
    """Every identity-bearing scalar of a record gets a new value, references resolved first."""
    pending = [p for p in rule.identity if p.startswith(prefix) and _record_path(p[len(prefix):])
               and p not in rule.foreign_keys]
    for _ in range(len(pending) + 1):
        progress = False
        for path in list(pending):
            template = rule.templates.get(path)
            if any(_absolute(s["path"], path) in pending for s in template or [] if s["kind"] == "ref"):
                continue
            relative = path[len(prefix):]
            current = _get(elem, relative)
            value = _draw(template, root, elem, current, rng)
            if path == id_column and not template:
                # No run shape in common: the per-position alphabet is the only evidence there is.
                value = _by_position(rule.id_values, rng) or value
            if value is not None:
                if path == id_column:
                    for attempt in range(100):
                        if value not in taken:
                            break
                        value = ((_draw(template, root, elem, current, rng, widen=attempt >= 50) if template
                                  else _by_position(rule.id_values, rng))
                                 or _draw(template, root, elem, current, rng, widen=True))
                    else:
                        value = f"{value}{rng.randrange(10**6):06d}"
                _set(elem, relative, value)
            pending.remove(path)
            progress = True
        if not progress:
            break


def _draw(template: Optional[list[dict]], root: dict, elem: dict, current: Any,
          rng: random.Random, widen: bool = False) -> Optional[str]:
    if template:
        rendered = _render(template, root, elem, rng, widen)
        if rendered is not None:
            return rendered
    if isinstance(current, str) and any(k == "d" for k, _ in _runs(current)):
        return _mutate_digits(current, rng)
    return None


def _rebuild_list(row: dict, list_path: str, rule: Rules, rules: dict[str, Rules],
                  observed: dict[str, dict], pools: dict[str, dict], schema: EntitySchema,
                  rng: random.Random) -> None:
    """A list whose elements name rows of other tables is resampled from those tables.

    The element's leading key is the one that copies the most fields (the order line's item);
    a second key whose table is the leading row's parent takes that parent (the item's product),
    so one element never names an item under one product and a different product beside it.
    """
    keys = {k.split(".[].", 1)[1]: t for k, t in rule.foreign_keys.items()
            if k.startswith(list_path + ".[].") and "." not in k.split(".[].", 1)[1]}
    copied_by_key = {k.split(".[].", 1)[1]: c for k, c in rule.embedded.items()
                     if k.startswith(list_path + ".[].")}
    lead = max(keys, key=lambda k: (len(copied_by_key.get(k, {})), k))
    pool = pools[keys[lead]]
    if not pool:
        return
    parents = _parents(pools, schema)
    elements = [e for r in observed[rule.table].values() for e in (_get(r, list_path) or [])
                if isinstance(e, dict)]
    # Every known row can be named, and one the observed lists named often is named more often:
    # the popularity the traces showed, with one count of smoothing so unseen rows appear at all.
    seen = Counter(e.get(lead) for e in elements)
    ids = sorted(pool)
    weights = [1 + seen.get(i, 0) for i in ids]
    size = rng.choice(rule.list_sizes.get(list_path) or [1])
    out = []
    for _ in range(size):
        ref_id = rng.choices(ids, weights)[0]
        referenced, parent = pool[ref_id], parents.get((keys[lead], ref_id)) or {}
        elem = copy.deepcopy(rng.choice(elements)) if elements else {}
        elem[lead] = ref_id
        for key, target in keys.items():
            if key == lead:
                continue
            parent_id = parent.get(id_field(schema, target) or "")
            elem[key] = parent_id if isinstance(parent_id, str) else rng.choice(sorted(pools[target]) or [None])
        for key in keys:
            for name, source in copied_by_key.get(key, {}).items():
                src = (referenced if key == lead else pools[keys[key]].get(elem.get(key)) or {}) \
                    if source == "row" else (parent if key == lead else {})
                if name in src:
                    elem[name] = copy.deepcopy(src[name])
        out.append(elem)
    _set(row, list_path, out)


def _rebuild_collection(row: dict, coll_path: str, child: Optional[str], rule: Rules,
                        rules: dict[str, Rules], known: dict[str, set], rng: random.Random) -> None:
    """The bootstrap row's own entries, each re-keyed by a fresh own id; the entries stay coherent."""
    entries = row.get(coll_path) or {}
    if not isinstance(entries, dict) or not entries:
        return
    taken = known.setdefault(child, set()) if child else set()
    child_rule = rules.get(child) if child else None
    out: dict[str, dict] = {}
    for key, sample in entries.items():
        if not isinstance(sample, dict):
            continue
        entry = copy.deepcopy(sample)
        key_field = _own_key(key, sample)
        if child_rule is not None:
            _redraw_identity(entry, entry, child_rule, rng, taken | set(out), child_rule.id_column)
        else:
            _redraw_identity(row, entry, rule, rng, taken | set(out), None, prefix=coll_path + ".{}.")
        new_key = entry.get(key_field) if key_field else None
        if not isinstance(new_key, str) or new_key in out or new_key in taken:
            new_key = _mutate_digits(str(new_key or key), rng)
            if key_field:
                entry[key_field] = new_key
        out[new_key] = entry
        taken.add(new_key)
    row[coll_path] = out


def _collection_ids(row: dict, rule: Rules) -> dict[str, list[str]]:
    return {child: sorted(row.get(coll_path) or {}) for coll_path, child in rule.collections.items() if child}


def _fill_back_refs(rules: dict[str, Rules], synthetic: dict[str, dict[str, dict]]) -> None:
    for table, rule in rules.items():
        for list_path, child_path in rule.back_refs.items():
            target = rule.foreign_keys[list_path]
            for row_id, row in synthetic[table].items():
                _set(row, list_path[:-3], sorted(cid for cid, c in synthetic[target].items()
                                                 if _get(c, child_path) == row_id))


# --- checks (docs/synthetic-rows.md section 3, practice 6) ---

def verify(db: dict, schema: EntitySchema, rules: dict[str, Rules], observed: dict[str, dict],
           grown: Grown) -> dict[str, Any]:
    """Id uniqueness, foreign key closure, id shape, no synthetic twin of an observed row, marginals."""
    checks: dict[str, Any] = {"dangling": [], "bad_ids": [], "twins": [], "duplicate_ids": [],
                              "marginals": {}, "warnings": []}
    for table in grown.added:
        if rules[table].observed < IDENTITY_EVIDENCE:
            checks["warnings"].append(f"{table} was grown from {rules[table].observed} observed rows, "
                                      f"below the {IDENTITY_EVIDENCE} a rule needs; its rows are copies "
                                      "with new digits and its id pattern was read off too few values")
    all_rows = {t: table_rows(db, schema, t) for t in rules}
    seen_ids: dict[str, str] = {}
    for table, rows in all_rows.items():
        for row_id in rows:
            if row_id in seen_ids and seen_ids[row_id] != table:
                checks["duplicate_ids"].append(f"{row_id} in {seen_ids[row_id]} and {table}")
            seen_ids[row_id] = table
    for table, ids in grown.added.items():
        rule = rules[table]
        pattern = id_pattern_for(schema, table, rule.id_column)
        rows = all_rows[table]
        fingerprints = {_fingerprint(r, rule) for r in observed[table].values()}
        for row_id in ids:
            row = rows.get(row_id)
            if row is None:
                checks["dangling"].append(f"{table} {row_id} was added but is not in the database")
                continue
            if pattern and not re.fullmatch(pattern, row_id):
                checks["bad_ids"].append(f"{table} {row_id} does not match {pattern}")
            if fingerprints and _fingerprint(row, rule) in fingerprints:
                checks["twins"].append(f"{table} {row_id} repeats an observed row's identity")
            for path, target in rule.foreign_keys.items():
                for _, value in _walk_path(row, path):
                    if isinstance(value, str) and value not in all_rows[target]:
                        checks["dangling"].append(f"{table} {row_id} {path} -> {target} {value}")
        skip = set(rule.identity) | set(rule.foreign_keys)
        checks["marginals"][table] = _marginals(observed[table], {i: rows[i] for i in ids if i in rows}, skip)
    checks["ok"] = not (checks["dangling"] or checks["bad_ids"] or checks["twins"] or checks["duplicate_ids"])
    return checks


def _fingerprint(row: dict, rule: Rules) -> str:
    return canonical_json({p: _get(row, p) for p in rule.identity if _record_path(p)})


def _marginals(observed: dict[str, dict], added: dict[str, dict], skip: set[str]) -> dict[str, dict[str, float]]:
    """Per leaf: total variation for categories, a two-sample KS statistic for numbers."""
    before, after = _leaf_values(observed), _leaf_values(added)
    out: dict[str, dict[str, float]] = {}
    for path in sorted((set(before) & set(after)) - skip):
        a, b = before[path], after[path]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in a + b):
            out[path] = {"ks": round(_ks(a, b), 3)}
        elif len(set(map(canonical_json, a))) <= 40:
            out[path] = {"tvd": round(_tvd(a, b), 3)}
    return out


def _tvd(a: list, b: list) -> float:
    ca, cb = Counter(map(canonical_json, a)), Counter(map(canonical_json, b))
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))


def _ks(a: list, b: list) -> float:
    sa, sb = sorted(a), sorted(b)
    best, i, j = 0.0, 0, 0
    for x in sorted(set(a) | set(b)):
        while i < len(sa) and sa[i] <= x:
            i += 1
        while j < len(sb) and sb[j] <= x:
            j += 1
        best = max(best, abs(i / len(sa) - j / len(sb)))
    return best


def report(grown: Grown) -> dict[str, Any]:
    """The JSON the build writes as synthetic.json: what was added, the rules, the checks."""
    return {
        "added": {t: len(ids) for t, ids in sorted(grown.added.items())},
        "implied_parents": dict(grown.implied),
        "rules": {t: {"observed": r.observed, "identity": r.identity, "templates": r.templates,
                      "foreign_keys": r.foreign_keys, "back_refs": r.back_refs, "embedded": r.embedded,
                      "mirrors": {k: list(v) for k, v in r.mirrors.items()},
                      "collections": r.collections, "collection_keys": r.collection_keys,
                      "nested_keys": {k: list(v) for k, v in r.nested_keys.items()},
                      "sums": {k: list(v) for k, v in r.sums.items()}}
                  for t, r in sorted(grown.rules.items())},
        "checks": grown.checks,
    }
