"""Run the real tau2 tools in their own process, one JSON call per line.

This runs under tau2's dependencies, not the harness's. It is a subprocess on purpose: tau2 pulls
in litellm, pandas and the rest, and none of that belongs in an environment whose whole claim is
that it needs three packages. Read a line of {"tool": ..., "args": {...}}, run it against a fresh
copy of the seed database, and write back what came out and what changed.
"""

import copy
import json
import sys


def load(domain):
    if domain == "retail":
        from tau2.domains.retail.data_model import RetailDB as DB
        from tau2.domains.retail.tools import RetailTools as Tools
        from tau2.domains.retail.utils import RETAIL_DB_PATH as PATH
    elif domain == "airline":
        from tau2.domains.airline.data_model import FlightDB as DB
        from tau2.domains.airline.tools import AirlineTools as Tools
        from tau2.domains.airline.utils import AIRLINE_DB_PATH as PATH
    elif domain == "telecom":
        from tau2.domains.telecom.data_model import TelecomDB as DB
        from tau2.domains.telecom.tools import TelecomTools as Tools
        from tau2.domains.telecom.utils import TELECOM_DB_PATH as PATH
    else:
        raise SystemExit(f"unknown domain {domain}")
    return DB, Tools, PATH


def plain(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


def main():
    domain = sys.argv[1]
    DB, Tools, path = load(domain)
    seed = json.loads(open(path, encoding="utf-8").read()) if str(path).endswith(".json") else None
    if seed is None:  # telecom ships toml
        import tomllib
        seed = tomllib.load(open(path, "rb"))
    print(json.dumps({"ready": True, "tables": sorted(seed)}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        ask = json.loads(line)
        db = DB(**copy.deepcopy(seed))
        tools = Tools(db)
        before = json.dumps(plain(db.model_dump(mode="json")), sort_keys=True, default=str)
        out = {"tool": ask["tool"]}
        try:
            fn = getattr(tools, ask["tool"])
            out["result"] = plain(fn(**ask.get("args") or {}))
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        after = json.dumps(plain(db.model_dump(mode="json")), sort_keys=True, default=str)
        out["changed"] = before != after
        out["db_after"] = json.loads(after) if out["changed"] else None
        print(json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main()
