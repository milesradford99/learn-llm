from learn_llm.checking import Case


def _t(id, *threads):
    return {"id": id, "threads": list(threads)}


def _twice(f):
    store = {}
    return f(store, _t("a", "pr-1"), True), f(store, _t("a", "pr-1"), True), len(store)


def _filtered(f):
    store = {}
    return f(store, _t("b", "chat-9"), False), "b" in store


def _none(f):
    try:
        f({}, _t("c"), True)
    except ValueError:
        return "raised"
    return "no error"


SUBS = {"pr-1": ["T1"], "issue-4": ["T1", "T2"]}

EXERCISES = {
    "receive": [
        Case("ch37.receive.twice", _twice, ("stored", "duplicate", 1), hint="ch37.receive.hint"),
        Case("ch37.receive.filtered", _filtered, ("filtered", True), hint="ch37.receive.hint"),
        Case("ch37.receive.none", _none, "raised", hint="ch37.receive.hint"),
    ],
    "deliver": [
        Case("ch37.deliver.new", lambda f: f(_t("x", "chat-1"), SUBS), "router", hint="ch37.deliver.hint"),
        Case("ch37.deliver.one", lambda f: f(_t("x", "pr-1"), SUBS), ["T1"], hint="ch37.deliver.hint"),
        Case("ch37.deliver.many", lambda f: f(_t("x", "pr-1", "issue-4"), SUBS), ["T1", "T2"], hint="ch37.deliver.hint"),
    ],
}
