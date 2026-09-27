"""Add per-unit shown-id sets: ids the reference reads reach once rebound
to the unit's new rows (the new primary row, its contained rows, the owning
actor). Replacement catalog ids are never shown: no read reaches them.
Tables come from schema.json; actor bodies are followed as references."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import units
from units import leaves_of

R = units.R
OUT = R.parent


def shown_for(bind: units.Binding, db: dict, actor: tuple[str, str],
              rows: list[str]) -> list[str]:
    shown = set()
    for r in rows:
        for table in bind.tables:
            body = (db.get(table) or {}).get(r)
            if body is not None:
                shown |= leaves_of(body)
                shown.add(r)
    ubody = (db.get(actor[0]) or {}).get(actor[1])
    if ubody is not None:
        shown |= leaves_of(ubody)
        shown.add(actor[1])
    return sorted(shown)


def main(workdir: Path = R, outdir: Path | None = None) -> None:
    od = outdir or OUT
    bind = units.Binding(workdir)
    db = json.loads((Path(workdir) / "db.json").read_text())
    y = json.loads((od / "yields.json").read_text())

    def actor_of(u: dict) -> tuple[str, str]:
        a = u.get("actor") or {}
        return (str(a.get("table") or ""), str(a.get("id") or ""))

    for u in y["do_units"]:
        u["shown"] = shown_for(bind, db, actor_of(u), u.get("rows", []))
    for u in y["decline_units"]:
        u["shown"] = shown_for(bind, db, actor_of(u), u.get("rows", []))
    (od / "yields.json").write_text(json.dumps(y, indent=1))
    print("shown sets added")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else R,
         Path(sys.argv[2]) if len(sys.argv) > 2 else None)
