"""Coherent binding copied from the arm A scratch work (e1_binding fix1+fix2,
e1_coherent inserts, retry on refusal), plus one arm D rule: a write that
lands on a member row (a row the cluster's own failures touched) is refused
and rebound, so new Tasks exercise the skill on other rows.

No names are typed here. Everything comes from the graph, the signatures
and the world.
"""
import collections

from kullback import graph as graph_mod
from kullback import synthesise as S
from kullback.gates.tool_runs import CRASH_ERRORS
from kullback.sampling import sample_key

K = 8
MAX_INSERT = 3


def make_bind():
    def bind(world, step, answers, observed, sigs, ident):
        tool = str(step.get("tool"))
        properties, required = S.declared(sigs.get(tool))
        if not properties:
            return None
        args = {}
        for binding in step.get("bindings") or ():
            head = str(binding["arg"]).split(".")[0].replace("[]", "")
            if head not in properties:
                continue
            found = graph_mod.values_at(answers[int(binding["from"])],
                                        str(binding["column"]))
            if not found:
                continue
            S._set_path(args, str(binding["arg"]),
                        found[sample_key("synth-edge-value",
                                         f"{ident}:{binding['arg']}", "") % len(found)])
        owed = [name for name in required if name not in args]
        seen = {}
        for answer in reversed(answers):
            for name, vals in S.named_leaves(answer).items():
                seen.setdefault(name, vals)
        row = S._row_source(world, [name for name in owed if name not in seen],
                            ident)
        for name in owed:
            spec = properties.get(name) or {}
            rec = observed.get(tool) or {}
            obs = (rec.get(name) or {}).get("values") or []
            if not obs:
                obs = (rec.get(name + "[]") or {}).get("values") or []
            pool = seen.get(name) or row.get(name) or list(obs)
            if not pool:
                return None
            args[name] = S._typed(spec, pool[sample_key(
                "synth-arg", f"{ident}:{tool}:{name}", "") % len(pool)])
        return {name: value for name, value in args.items() if name in properties}
    return bind


def seen_names(answers):
    seen = {}
    for answer in reversed(answers):
        for name, value in S.named_leaves(answer).items():
            seen.setdefault(name, value)
    return seen


def missing(world, step, answers, sigs):
    props, req = S.declared(sigs.get(str(step["tool"])))
    bound = set()
    for binding in step.get("bindings") or ():
        if graph_mod.values_at(answers[int(binding["from"])],
                               str(binding["column"])):
            bound.add(str(binding["arg"]).split(".")[0].replace("[]", ""))
    seen = seen_names(answers)
    return [name for name in req if name not in bound and name not in seen
            and (name.endswith("_id") or name.endswith("_ids"))]


def supplier(graph, write, arg, writes, used):
    best = None
    for edge in graph["edges"]:
        if edge["to"] == write and str(edge["arg"]).split(".")[0].replace("[]", "") == arg \
                and edge["from"] not in writes and edge["from"] != write \
                and edge["from"] not in used:
            if best is None or edge["weight"] > best["weight"]:
                best = edge
    return best


def hits_excluded(args: dict, excluded: set[str]) -> bool:
    if not excluded:
        return False
    stack = [args]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
        elif str(value) in excluded:
            return True
    return False


def make_run_walk(graph, bind, excluded=(), k=K):
    writes = graph_mod.write_names(graph)
    stats = collections.Counter()

    def run_walk(world, steps, observed, ident):
        sigs = {sig.name: sig for sig in world.sigs}
        answers, calls = [], []
        for index, step in enumerate(steps):
            name = str(step["tool"])
            result = None
            for trial in range(k):
                sub = f"{ident}:{index}" if trial == 0 else f"{ident}:{index}:r{trial}"
                toolkit = world.toolkit(world.fresh_db())
                for done in calls:
                    getattr(toolkit, done["name"])(**done["args"])
                local_answers, local_calls = list(answers), []
                step_now = dict(steps[index])
                step_now["bindings"] = list(step_now.get("bindings") or ())
                if name in writes:
                    used = set()
                    for _ in range(MAX_INSERT):
                        miss = missing(world, step_now, local_answers, sigs)
                        if not miss:
                            break
                        edge = supplier(graph, name, miss[0], writes, used)
                        if edge is None:
                            break
                        used.add(edge["from"])
                        names = ([call["name"] for call in calls]
                                 + [call["name"] for call in local_calls])
                        read_step = {"tool": edge["from"], "bindings": [
                            {"arg": e2["arg"], "column": e2["column"], "from": i}
                            for e2 in graph["edges"]
                            if e2["to"] == edge["from"] and e2["kind"] == "value"
                            for i, nm in enumerate(names) if nm == e2["from"]]}
                        read_args = bind(world, read_step, local_answers, observed,
                                         sigs, f"{sub}:ins{len(used)}")
                        if read_args is None:
                            break
                        try:
                            read_answer = getattr(toolkit, edge["from"])(**read_args)
                        except Exception:
                            break
                        local_answers.append(read_answer)
                        step_now["bindings"].append(
                            {"arg": edge["arg"], "column": edge["column"],
                             "from": len(local_answers) - 1})
                        local_calls.append({"name": edge["from"], "args": read_args,
                                            "result": read_answer})
                        stats["inserted"] += 1
                    if missing(world, step_now, local_answers, sigs):
                        result = S.UNBOUND
                        continue
                shifted = dict(step_now)
                args = bind(world, shifted, local_answers, observed, sigs, sub)
                if args is None:
                    result = S.UNBOUND
                    continue
                if name in writes and hits_excluded(args, set(excluded)):
                    result = S.REFUSED
                    stats["excluded"] += 1
                    continue
                try:
                    answer = getattr(toolkit, name)(**args)
                except Exception as error:
                    if type(error).__name__ in CRASH_ERRORS:
                        return calls, S.CRASHED
                    result = S.REFUSED
                    continue
                for call in local_calls:
                    calls.append({"id": f"{ident}-{len(calls)}", "name": call["name"],
                                  "args": call["args"], "requestor": "assistant",
                                  "turn": len(calls), "result": call["result"],
                                  "error": None})
                answers = local_answers + [answer]
                calls.append({"id": f"{ident}-{len(calls)}", "name": name,
                              "args": args, "requestor": "assistant",
                              "turn": len(calls), "result": answer, "error": None})
                result = None
                break
            if result is not None:
                return calls, result
        return calls, ""
    run_walk.stats = stats
    return run_walk
