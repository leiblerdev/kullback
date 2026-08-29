"""synth.py grows the Starting state from the rows the traces showed (D40, D107).

Every fixture below is a small world in the retail shape, built by hand: users with nested
names and a keyed collection of payment methods, orders that name a user and repeat item fields,
products that keep their items under `variants`. No corpus, no model, no benchmark file.
"""
from __future__ import annotations

import copy
import random

import pytest

from harness.builder import synth
from harness.shared.records import Column, EntitySchema

FIRST = ["Ava", "Liam", "Mia", "Noah", "Zoe", "Eli", "Ivy", "Max", "Uma", "Kai", "Lea", "Tom"]
LAST = ["Chen", "Diaz", "Khan", "Lee", "Moss", "Nair", "Ortiz", "Park", "Quinn", "Reyes", "Sato", "Voss"]


def _schema() -> EntitySchema:
    ids = {"users": "user_id", "orders": "order_id", "products": "product_id", "items": "item_id"}
    return EntitySchema(
        tables=["users", "orders", "products", "items"],
        columns=[Column(table=t, name=n, **{"class": "hard"}) for t, n in ids.items()],
        id_patterns={"users.user_id": r"^[a-z]+_[a-z]+_\d{4}$", "orders.order_id": r"^#W\d{7}$",
                     "products.product_id": r"^\d{10}$", "items.item_id": r"^\d{10}$"},
        homes={"items": "products.variants"},
    )


def _world(n_users: int = 12, orders_per_user: int = 2) -> dict:
    rng = random.Random(7)
    products, items = {}, {}
    for p in range(4):
        pid = f"{1000000000 + p:010d}"
        variants = {}
        for v in range(3):
            iid = f"{2000000000 + p * 10 + v:010d}"
            variants[iid] = {"item_id": iid, "price": round(10.0 + p * 5 + v, 2), "available": v != 2,
                             "options": {"color": ["red", "blue", "green"][v], "size": f"{v + 1}L"}}
            items[iid] = (pid, variants[iid])
        products[pid] = {"product_id": pid, "name": f"Product {p}", "variants": variants}
    users, orders = {}, {}
    for u in range(n_users):
        first, last = FIRST[u], LAST[u]
        uid = f"{first.lower()}_{last.lower()}_{1000 + u:04d}"
        methods = {f"gift_card_{5000000 + u}": {"id": f"gift_card_{5000000 + u}", "source": "gift_card",
                                                "balance": float(20 + u)}}
        if u % 2:
            methods[f"paypal_{6000000 + u}"] = {"id": f"paypal_{6000000 + u}", "source": "paypal"}
        user = {"user_id": uid, "name": {"first_name": first, "last_name": last},
                "email": f"{first.lower()}.{last.lower()}{1000 + u}@example.com",
                "address": {"address1": f"{100 + u} Elm Street", "city": ["Austin", "Boston"][u % 2],
                            "zip": f"{70000 + u:05d}"},
                "payment_methods": methods, "orders": []}
        for o in range(orders_per_user):
            oid = f"#W{3000000 + u * 10 + o:07d}"
            chosen = rng.sample(sorted(items), 2)
            lines = [{"item_id": i, "product_id": items[i][0], "name": products[items[i][0]]["name"],
                      "price": items[i][1]["price"], "options": items[i][1]["options"]} for i in chosen]
            total = round(sum(line["price"] for line in lines), 2)
            status = ["pending", "delivered", "cancelled"][o % 3] if u % 3 else "pending"
            orders[oid] = {"order_id": oid, "user_id": uid, "items": lines, "status": status,
                           "cancel_reason": "no longer needed" if status == "cancelled" else None,
                           "fulfillments": [{"item_ids": chosen, "tracking_id": [f"{700000000000 + u:012d}"]}]
                           if status == "delivered" else [],
                           "payment_history": [{"amount": total, "payment_method_id": sorted(methods)[0],
                                                "transaction_type": "payment"}]}
            user["orders"].append(oid)
        users[uid] = user
    return {"users": users, "orders": orders, "products": products, "items": {}}


# --- mining ---

def test_rules_are_read_off_the_observed_rows():
    rules = synth.mine_rules(_world(), _schema())
    users, orders, products = rules["users"], rules["orders"], rules["products"]
    assert orders.foreign_keys["user_id"] == "users"
    assert orders.foreign_keys["items.[].item_id"] == "items"
    assert orders.foreign_keys["items.[].product_id"] == "products"
    assert users.foreign_keys["orders.[]"] == "orders"
    assert users.back_refs == {"orders.[]": "user_id"}
    assert orders.embedded["items.[].item_id"] == {"price": "row", "options": "row",
                                                    "name": "parent", "product_id": "parent"}
    assert orders.mirrors == {"fulfillments.[].item_ids.[]": ("items", "item_id")}
    assert orders.nested_keys == {"payment_history.[].payment_method_id": ("user_id", "payment_methods")}
    assert orders.sums == {"payment_history.[].amount": ("items", "price")}
    assert products.collections == {"variants": "items"}
    assert users.collections == {"payment_methods": None}
    assert users.collection_keys == {"payment_methods": "id"}


def test_templates_refer_to_other_leaves_of_the_row_and_never_to_themselves():
    rules = synth.mine_rules(_world(), _schema())
    email = rules["users"].templates["email"]
    kinds = [(s["kind"], s.get("path") or s.get("text")) for s in email]
    assert kinds[0] == ("ref", "/name.first_name") and kinds[2] == ("ref", "/name.last_name")
    assert email[0]["case"] == "lower"
    assert any(s["kind"] == "digits" for s in email)
    assert rules["users"].templates["user_id"][0] == {"kind": "ref", "path": "/name.first_name", "case": "lower"}
    # what a template refers to becomes an identity leaf of its own, drawn from the vocabulary
    assert "name.first_name" in rules["users"].identity
    assert rules["users"].templates["name.first_name"][0]["kind"] == "vocab"


def test_a_category_with_few_observations_is_not_an_identity():
    """Three variants with three distinct sizes are a category, not three identities."""
    rules = synth.mine_rules(_world(), _schema())
    assert "options.size" not in rules["items"].identity
    assert "address.city" not in rules["users"].identity
    assert "address.zip" in rules["users"].identity


@pytest.mark.parametrize("text, shape", [
    ("#W1234567", [("o", "#"), ("a", None), ("d", None)]),
    ("ava.chen1000@example.com", [("a", None), ("o", "."), ("a", None), ("d", None), ("o", "@"),
                                  ("a", None), ("o", "."), ("a", None)]),
    ("Suite 12", [("a", None), ("o", " "), ("d", None)]),
])
def test_a_string_has_a_shape_of_runs(text, shape):
    assert list(synth._shape(text)) == shape


# --- growing ---

def test_grow_reaches_the_target_and_the_checks_pass():
    db, schema = _world(), _schema()
    before = copy.deepcopy(db)
    grown = synth.grow(db, schema, {"users": 30, "orders": 60, "products": 6}, seed=3)
    assert len(db["users"]) == 30 and len(db["orders"]) == 60 and len(db["products"]) == 6
    assert grown.checks["ok"], grown.checks
    assert len(grown.added["users"]) == 18 and len(grown.added["orders"]) == 36
    assert len(grown.added["items"]) == 6  # two new products, three variants each
    # observed rows are exactly as they were
    for table in before:
        for row_id, row in before[table].items():
            assert db[table][row_id] == row


def test_a_synthetic_order_hangs_off_a_synthetic_user_and_the_user_lists_it_back():
    db, schema = _world(), _schema()
    grown = synth.grow(db, schema, {"orders": 40}, seed=1)
    assert grown.implied["users"] >= 1
    for oid in grown.added["orders"]:
        order = db["orders"][oid]
        assert order["user_id"] in grown.added["users"]
        assert oid in db["users"][order["user_id"]]["orders"]
        methods = db["users"][order["user_id"]]["payment_methods"]
        assert order["payment_history"][0]["payment_method_id"] in methods


def test_an_order_line_repeats_the_item_it_names_and_the_total_is_the_sum():
    db, schema = _world(), _schema()
    grown = synth.grow(db, schema, {"orders": 40}, seed=2)
    items = synth.table_rows(db, schema, "items")
    for oid in grown.added["orders"]:
        order = db["orders"][oid]
        for line in order["items"]:
            item = items[line["item_id"]]
            assert line["price"] == item["price"] and line["options"] == item["options"]
            assert line["product_id"] in db["products"]
            assert line["item_id"] in db["products"][line["product_id"]]["variants"]
            assert line["name"] == db["products"][line["product_id"]]["name"]
        assert order["payment_history"][0]["amount"] == round(sum(line["price"] for line in order["items"]), 2)
        for fulfillment in order["fulfillments"]:
            assert fulfillment["item_ids"] == [line["item_id"] for line in order["items"]]


def test_a_synthetic_user_has_a_fresh_identity_built_from_its_own_name():
    db, schema = _world(), _schema()
    observed = set(db["users"])
    grown = synth.grow(db, schema, {"users": 24}, seed=5)
    for uid in grown.added["users"]:
        user = db["users"][uid]
        first, last = user["name"]["first_name"], user["name"]["last_name"]
        assert uid.startswith(f"{first.lower()}_{last.lower()}_") and uid not in observed
        assert user["email"].startswith(f"{first.lower()}.{last.lower()}")
        assert user["email"].endswith("@example.com")
        for key, method in user["payment_methods"].items():
            assert method["id"] == key
        assert all(m["id"] not in {k for u in observed for k in db["users"][u]["payment_methods"]}
                   for m in user["payment_methods"].values())


def test_a_homed_table_grows_with_its_parent_and_cannot_be_targeted_alone():
    db, schema = _world(), _schema()
    with pytest.raises(ValueError, match="grow its parent"):
        synth.grow(db, schema, {"items": 50})
    with pytest.raises(ValueError, match="not a table"):
        synth.grow(db, schema, {"carts": 5})
    grown = synth.grow(db, schema, {"products": 5}, seed=0)
    assert len(grown.added["products"]) == 1 and len(grown.added["items"]) == 3
    new = db["products"][grown.added["products"][0]]
    assert set(new["variants"]) == set(grown.added["items"])
    assert all(v["item_id"] == k for k, v in new["variants"].items())


def test_grow_is_deterministic_under_a_seed():
    a, b = _world(), _world()
    synth.grow(a, _schema(), {"users": 20, "orders": 30}, seed=9)
    synth.grow(b, _schema(), {"users": 20, "orders": 30}, seed=9)
    assert a == b


def test_the_report_names_what_was_added_the_rules_and_the_checks():
    db = _world()
    grown = synth.grow(db, _schema(), {"users": 14}, seed=1)
    out = synth.report(grown)
    assert out["added"] == {"users": 2}
    assert out["rules"]["orders"]["sums"] == {"payment_history.[].amount": ["items", "price"]}
    assert out["checks"]["ok"] is True
    assert "address.city" in out["checks"]["marginals"]["users"]


def test_the_checks_catch_a_dangling_key_and_a_twin():
    db, schema = _world(), _schema()
    rules = synth.mine_rules(db, schema)
    observed = {t: dict(synth.table_rows(db, schema, t)) for t in rules}
    twin = copy.deepcopy(db["users"]["ava_chen_1000"])
    twin["user_id"] = "ava_chen_1000x"  # same identity, new id: a twin, and a bad id shape
    db["users"]["ava_chen_1000x"] = twin
    bad = copy.deepcopy(db["orders"]["#W3000000"])
    bad["order_id"], bad["user_id"] = "#W9999999", "nobody_here_0000"
    db["orders"]["#W9999999"] = bad
    grown = synth.Grown(added={"users": ["ava_chen_1000x"], "orders": ["#W9999999"]}, rules=rules)
    checks = synth.verify(db, schema, rules, observed, grown)
    assert not checks["ok"]
    assert any("nobody_here_0000" in d for d in checks["dangling"])
    assert checks["bad_ids"] and "ava_chen_1000x" in checks["bad_ids"][0]


def test_marginal_statistics():
    assert synth._tvd(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(1 / 3)
    assert synth._tvd(["a"], ["a"]) == 0
    assert synth._ks([1, 2, 3], [1, 2, 3]) == 0
    assert synth._ks([1, 2, 3], [4, 5, 6]) == 1


# --- the rules on other shapes: airline's ids, dates, lists named after the table (D107) ---

def _airline_like() -> tuple[dict, EntitySchema]:
    """Six-character ids with no run shape in common, a date, and a `reservations` list of keys."""
    rng = random.Random(3)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    users, reservations = {}, {}
    for u in range(12):
        uid = f"{FIRST[u].lower()}_{LAST[u].lower()}_{1000 + u}"
        users[uid] = {"user_id": uid, "name": {"first_name": FIRST[u], "last_name": LAST[u]},
                      "dob": f"{1960 + u * 3}-{(u % 12) + 1:02d}-{(u % 27) + 1:02d}", "reservations": []}
        for _ in range(2):
            rid = "".join(rng.choice(alphabet) for _ in range(6))
            reservations[rid] = {"reservation_id": rid, "user_id": uid, "cabin": ["economy", "business"][u % 2],
                                 "created_at": f"2024-05-{(u % 28) + 1:02d}T{u:02d}:15:00"}
            users[uid]["reservations"].append(rid)
    schema = EntitySchema(
        tables=["users", "reservations"],
        columns=[Column(table="users", name="user_id", **{"class": "hard"}),
                 Column(table="reservations", name="reservation_id", **{"class": "hard"})],
        id_patterns={"users.user_id": r"^[a-z]+_[a-z]+_\d{4}$", "reservations.reservation_id": r"^.{6}$"})
    return {"users": users, "reservations": reservations}, schema


def test_a_list_named_after_the_table_is_a_foreign_key_and_a_back_reference():
    db, schema = _airline_like()
    rules = synth.mine_rules(db, schema)
    assert rules["users"].foreign_keys == {"reservations.[]": "reservations"}
    assert rules["users"].back_refs == {"reservations.[]": "user_id"}


def test_an_id_with_no_run_shape_is_drawn_per_position_and_keeps_its_length():
    db, schema = _airline_like()
    observed = set(db["reservations"])
    grown = synth.grow(db, schema, {"reservations": 60}, seed=2)
    assert grown.checks["ok"], grown.checks
    for rid in grown.added["reservations"]:
        assert len(rid) == 6 and rid not in observed
        assert rid in db["users"][db["reservations"][rid]["user_id"]]["reservations"]


def test_a_digit_run_is_drawn_inside_its_observed_range_so_a_date_stays_a_date():
    db, schema = _airline_like()
    grown = synth.grow(db, schema, {"users": 40, "reservations": 80}, seed=4)
    for uid in grown.added["users"]:
        year, month, day = db["users"][uid]["dob"].split("-")
        assert 1960 <= int(year) <= 1993 and 1 <= int(month) <= 12 and 1 <= int(day) <= 27
    for rid in grown.added["reservations"]:
        stamp = db["reservations"][rid]["created_at"]
        assert stamp.startswith("2024-05-") and 1 <= int(stamp[8:10]) <= 28


def test_sequential_ids_widen_past_their_observed_range_instead_of_colliding():
    db, schema = _world(), _schema()
    grown = synth.grow(db, schema, {"products": 40}, seed=1)
    assert grown.checks["bad_ids"] == []
    assert all(len(pid) == 10 for pid in grown.added["products"])


def test_a_list_of_keys_is_drawn_without_repeats():
    db, schema = _world(), _schema()
    rules = synth.mine_rules(db, schema)
    rules["users"].back_refs = {}  # as if the back reference were unknown: a plain list of keys
    rng = random.Random(0)
    observed = {t: dict(synth.table_rows(db, schema, t)) for t in rules}
    synthetic = {t: {} for t in rules}
    known = {t: set(observed[t]) for t in rules}
    for _ in range(20):
        row = synth._generate("users", rules, observed, synthetic, known, schema, rng)
        assert len(row["orders"]) == len(set(row["orders"]))


def test_a_table_grown_from_too_few_rows_carries_a_warning():
    db, schema = _world(), _schema()
    grown = synth.grow(db, schema, {"products": 6}, seed=1)
    assert any(w.startswith("products was grown from 4 observed rows") for w in grown.checks["warnings"])
    assert not any(w.startswith("users") for w in grown.checks["warnings"])
