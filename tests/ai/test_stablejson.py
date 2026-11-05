from kullback.ai import stablejson


def test_two_dicts_with_the_same_items_in_another_insertion_order_give_identical_text():
    one = {"a": 1, "b": [1, 2], "c": "x"}
    two = {"c": "x", "b": [1, 2], "a": 1}
    assert stablejson.dumps(one) == stablejson.dumps(two) == '{"a":1,"b":[1,2],"c":"x"}'


def test_nested_dicts_are_sorted_too():
    one = {"outer": {"z": 1, "a": {"y": 2, "b": 3}}}
    two = {"outer": {"a": {"b": 3, "y": 2}, "z": 1}}
    assert stablejson.dumps(one) == stablejson.dumps(two)
    assert stablejson.dumps(one) == '{"outer":{"a":{"b":3,"y":2},"z":1}}'


def test_lists_keep_their_order():
    assert stablejson.dumps([3, 1, 2]) == "[3,1,2]"
    assert stablejson.dumps({"k": [{"b": 1, "a": 2}, {"a": 0}]}) == '{"k":[{"a":2,"b":1},{"a":0}]}'


def test_non_ascii_text_stays_as_it_is():
    assert stablejson.dumps({"name": "Zoë 日本"}) == '{"name":"Zoë 日本"}'


def test_an_indent_gives_fixed_separators_and_sorted_keys():
    assert stablejson.dumps({"b": 1, "a": [1]}, indent=1) == '{\n "a": [\n  1\n ],\n "b": 1\n}'


def test_the_posted_body_is_the_same_bytes_whatever_order_its_keys_were_built_in():
    from kullback.ai import provider as pv

    model = pv.model_for("openai/model-a", env={})
    one = {"model": "m", "tools": [{"type": "function", "function": {"name": "t", "parameters": {"b": 1, "a": 2}}}]}
    two = {"tools": [{"function": {"parameters": {"a": 2, "b": 1}, "name": "t"}, "type": "function"}], "model": "m"}
    assert model.encode_body(one) == model.encode_body(two)
