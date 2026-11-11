from learn_llm.agents import FakeModel, ask, call, result, say, trace, user
from learn_llm.checking import Case


def _boom(arguments):
    raise ValueError("disk full")


TOOLS = {"add": lambda a: str(a["x"] + a["y"]), "boom": _boom}
TWO = ask(call("add", {"x": 1, "y": 2}, "a"), call("add", {"x": 3, "y": 4}, "b"))


def _loop(f, replies, max_turns=10):
    return trace(f(FakeModel(replies), [user("go")], TOOLS, max_turns))


EXERCISES = {
    "tool_calls": [
        Case("ch40.tool_calls.none", lambda f: f(say("hi")), [], hint="ch40.tool_calls.hint"),
        Case("ch40.tool_calls.two", lambda f: [c["id"] for c in f(TWO)], ["a", "b"], hint="ch40.tool_calls.hint"),
    ],
    "run_tool": [
        Case("ch40.run_tool.ok", lambda f: f(TOOLS, call("add", {"x": 1, "y": 2}, "a")), result("a", "add", "3"), hint="ch40.run_tool.hint"),
        Case("ch40.run_tool.unknown", lambda f: f(TOOLS, call("nope", {}, "b")), result("b", "nope", "Tool nope not found", True), hint="ch40.run_tool.hint"),
        Case("ch40.run_tool.raises", lambda f: f(TOOLS, call("boom", {}, "c")), result("c", "boom", "disk full", True), hint="ch40.run_tool.hint"),
    ],
    "agent_loop": [
        Case("ch40.agent_loop.talk", lambda f: _loop(f, [say("hello")]), ["user: go", "assistant: hello"], hint="ch40.agent_loop.hint"),
        Case("ch40.agent_loop.tool", lambda f: _loop(f, [ask(call("add", {"x": 1, "y": 2})), say("3")]),
             ["user: go", "assistant: [add]", "tool add ok: 3", "assistant: 3"], hint="ch40.agent_loop.hint"),
        Case("ch40.agent_loop.two", lambda f: _loop(f, [TWO, say("done")]),
             ["user: go", "assistant: [add] [add]", "tool add ok: 3", "tool add ok: 7", "assistant: done"], hint="ch40.agent_loop.hint"),
        Case("ch40.agent_loop.error", lambda f: _loop(f, [ask(call("boom")), say("sorry")])[2], "tool boom error: disk full", hint="ch40.agent_loop.hint"),
        Case("ch40.agent_loop.limit", lambda f: len(_loop(f, [ask(call("add", {"x": 1, "y": 1}))] * 50, 3)), 7, hint="ch40.agent_loop.limit_hint"),
    ],
}
