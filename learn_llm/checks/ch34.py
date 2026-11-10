from learn_llm.agents import say, trace, user
from learn_llm.checking import Case


def msg(message):
    return {"type": "message", "message": message}


def _new(f, messages):
    session = {"entries": [], "leaf": None}
    for m in messages:
        f(session, msg(m))
    return session


def _ids(f):
    session = _new(f, [user("a"), say("b")])
    return [(e["id"], e["parentId"]) for e in session["entries"]], session["leaf"]


def _given(*links, leaf):
    """A session from (id, parent, text) triples."""
    return {"entries": [{"id": i, "parentId": p, "type": "message", "message": user(t)} for i, p, t in links], "leaf": leaf}


TREE = _given(("e0", None, "a"), ("e1", "e0", "b"), ("e2", "e1", "c"), ("e3", "e1", "d"), leaf="e3")


def _branch(f):
    session = _given(("e0", None, "a"), ("e1", "e0", "b"), leaf="e1")
    f(session, "e0")
    return session["leaf"], len(session["entries"])


def _unknown(f):
    try:
        f(_given(("e0", None, "a"), leaf="e0"), "zz")
    except ValueError:
        return "raised"
    return "no error"


def _with_setting():
    session = _given(("e0", None, "a"), ("e2", "e1", "b"), leaf="e2")
    session["entries"].insert(1, {"id": "e1", "parentId": "e0", "type": "model_change", "modelId": "x"})
    return session


EXERCISES = {
    "append": [
        Case("ch34.append.ids", _ids, ([("e0", None), ("e1", "e0")], "e1"), hint="ch34.append.hint"),
        Case("ch34.append.returns", lambda f: f({"entries": [], "leaf": None}, msg(user("a"))), "e0", hint="ch34.append.hint"),
    ],
    "active_path": [
        Case("ch34.active_path.empty", lambda f: f({"entries": [], "leaf": None}), [], hint="ch34.active_path.hint"),
        Case("ch34.active_path.branch", lambda f: [e["id"] for e in f(TREE)], ["e0", "e1", "e3"], hint="ch34.active_path.hint"),
    ],
    "branch": [
        Case("ch34.branch.moves", _branch, ("e0", 2), hint="ch34.branch.hint"),
        Case("ch34.branch.unknown", _unknown, "raised", hint="ch34.branch.hint"),
    ],
    "context": [
        Case("ch34.context.path", lambda f: trace(f(TREE)), ["user: a", "user: b", "user: d"], hint="ch34.context.hint"),
        Case("ch34.context.skip", lambda f: trace(f(_with_setting())), ["user: a", "user: b"], hint="ch34.context.hint"),
    ],
    "to_jsonl": [
        Case("ch34.to_jsonl.lines", lambda f: len(f(TREE).split("\n")), 4, hint="ch34.to_jsonl.hint"),
        Case("ch34.to_jsonl.first", lambda f: f(_given(("e0", None, "a"), leaf="e0")), '{"id": "e0", "parentId": null, "type": "message", "message": {"role": "user", "content": "a"}}', hint="ch34.to_jsonl.hint"),
    ],
    "from_jsonl": [
        Case("ch34.from_jsonl.leaf", lambda f: f('{"id": "e0", "parentId": null}\n{"id": "e1", "parentId": "e0"}')["leaf"], "e1", hint="ch34.from_jsonl.hint"),
        Case("ch34.from_jsonl.count", lambda f: len(f('{"id": "e0", "parentId": null}')["entries"]), 1, hint="ch34.from_jsonl.hint"),
    ],
}
