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
        Case("ch47.receive.twice", _twice, ("stored", "duplicate", 1), hint="ch47.receive.hint"),
        Case("ch47.receive.filtered", _filtered, ("filtered", True), hint="ch47.receive.hint"),
        Case("ch47.receive.none", _none, "raised", hint="ch47.receive.hint"),
    ],
    "deliver": [
        Case("ch47.deliver.new", lambda f: f(_t("x", "chat-1"), SUBS), "router", hint="ch47.deliver.hint"),
        Case("ch47.deliver.one", lambda f: f(_t("x", "pr-1"), SUBS), ["T1"], hint="ch47.deliver.hint"),
        Case("ch47.deliver.many", lambda f: f(_t("x", "pr-1", "issue-4"), SUBS), ["T1", "T2"], hint="ch47.deliver.hint"),
    ],
}
