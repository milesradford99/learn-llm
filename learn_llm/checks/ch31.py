from learn_llm.agents import FakeModel, ask, call, result, say, trace, user
from learn_llm.checking import Case


def _run_call(c):
    return result(c["id"], c["name"], "ok")


def _loop(f, replies, steering=(), follow_ups=()):
    return trace(f(FakeModel(replies), [user("go")], _run_call, [user(s) for s in steering], [user(s) for s in follow_ups]))


def _pops(f):
    queue = ["a", "b"]
    return f(queue), queue


def _steered(f):
    """A model that queues a steering message while its first tool runs."""
    steering = []

    def run_call(c):
        if c["id"] == "a":
            steering.append(user("stop that"))
        return result(c["id"], c["name"], "ok")

    return trace(f(FakeModel([ask(call("read", id="a"), call("edit", id="b")), say("ok")]), [user("go")], run_call, steering, []))


EXERCISES = {
    "take_one": [
        Case("ch31.take_one.empty", lambda f: f([]), [], hint="ch31.take_one.hint"),
        Case("ch31.take_one.oldest", _pops, (["a"], ["b"]), hint="ch31.take_one.hint"),
    ],
    "can_continue": [
        Case("ch31.can_continue.empty", lambda f: f([]), False, hint="ch31.can_continue.hint"),
        Case("ch31.can_continue.user", lambda f: f([user("hi")]), True, hint="ch31.can_continue.hint"),
        Case("ch31.can_continue.tool", lambda f: f([result("a", "read", "x")]), True, hint="ch31.can_continue.hint"),
        Case("ch31.can_continue.assistant", lambda f: f([user("hi"), say("yo")]), False, hint="ch31.can_continue.hint"),
    ],
    "truncated_result": [
        Case("ch31.truncated_result.error", lambda f: f(call("write", {}, "a"))["isError"], True, hint="ch31.truncated_result.hint"),
        Case("ch31.truncated_result.id", lambda f: f(call("write", {}, "a"))["toolCallId"], "a", hint="ch31.truncated_result.hint"),
        Case("ch31.truncated_result.says", lambda f: "token limit" in f(call("write", {}, "a"))["content"], True, hint="ch31.truncated_result.hint"),
    ],
    "run_loop": [
        Case("ch31.run_loop.plain", lambda f: _loop(f, [ask(call("read")), say("done")]),
             ["user: go", "assistant: [read]", "tool read ok: ok", "assistant: done"], hint="ch31.run_loop.hint"),
        Case("ch31.run_loop.steer_start", lambda f: _loop(f, [say("ok")], steering=["wait"]), ["user: go", "user: wait", "assistant: ok"], hint="ch31.run_loop.hint"),
        Case("ch31.run_loop.steer_mid", _steered,
             ["user: go", "assistant: [read] [edit]", "tool read ok: ok", "tool edit ok: ok", "user: stop that", "assistant: ok"], hint="ch31.run_loop.hint"),
        Case("ch31.run_loop.follow", lambda f: _loop(f, [say("one"), say("two")], follow_ups=["next"]),
             ["user: go", "assistant: one", "user: next", "assistant: two"], hint="ch31.run_loop.hint"),
        Case("ch31.run_loop.error", lambda f: _loop(f, [say("bad", stop="error")], follow_ups=["next"]), ["user: go", "assistant: bad"], hint="ch31.run_loop.exit_hint"),
        Case("ch31.run_loop.length", lambda f: _loop(f, [ask(call("write"), stop="length"), say("sorry")])[2].startswith("tool write error"), True, hint="ch31.run_loop.length_hint"),
    ],
}
