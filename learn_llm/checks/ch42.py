from learn_llm.agents import ask, call, result, say, user
from learn_llm.checking import Case

STREAM = [
    {"type": "start"},
    {"type": "text_start", "contentIndex": 0},
    {"type": "text_delta", "contentIndex": 0, "delta": "Hel"},
    {"type": "text_delta", "contentIndex": 0, "delta": "lo"},
    {"type": "text_end", "contentIndex": 0},
    {"type": "text_start", "contentIndex": 1},
    {"type": "text_delta", "contentIndex": 1, "delta": "Bye"},
]
ASK = ask(call("add", {"x": 1}, "a"))
BOTH = {"role": "assistant", "content": [{"type": "text", "text": "Adding."}, call("add", {"x": 1}, "a")], "stopReason": "toolUse"}
TOOL_A, TOOL_B = result("a", "add", "1"), result("b", "add", "2", True)


EXERCISES = {
    "collect_text": [
        Case("ch42.collect_text.none", lambda f: f([{"type": "start"}]), {}, hint="ch42.collect_text.hint"),
        Case("ch42.collect_text.two", lambda f: f(STREAM), {0: "Hello", 1: "Bye"}, hint="ch42.collect_text.hint"),
    ],
    "final_message": [
        Case("ch42.final_message.done", lambda f: f([{"type": "start"}, {"type": "done", "reason": "stop", "message": say("hi")}]), say("hi"), hint="ch42.final_message.hint"),
        Case("ch42.final_message.error", lambda f: f([{"type": "error", "reason": "error", "error": say("no", stop="error")}]), say("no", stop="error"), hint="ch42.final_message.hint"),
        Case("ch42.final_message.cut", lambda f: f(STREAM), None, hint="ch42.final_message.hint"),
    ],
    "to_openai": [
        Case("ch42.to_openai.user", lambda f: f(user("hi")), {"role": "user", "content": "hi"}, hint="ch42.to_openai.hint"),
        Case("ch42.to_openai.text", lambda f: f(say("yo")), {"role": "assistant", "content": "yo"}, hint="ch42.to_openai.hint"),
        Case("ch42.to_openai.call", lambda f: f(BOTH)["tool_calls"], [{"id": "a", "type": "function", "function": {"name": "add", "arguments": '{"x": 1}'}}], hint="ch42.to_openai.hint"),
        Case("ch42.to_openai.tool", lambda f: f(TOOL_A), {"role": "tool", "tool_call_id": "a", "content": "1"}, hint="ch42.to_openai.hint"),
    ],
    "to_anthropic": [
        Case("ch42.to_anthropic.call", lambda f: f([ASK])[0]["content"], [{"type": "tool_use", "id": "a", "name": "add", "input": {"x": 1}}], hint="ch42.to_anthropic.hint"),
        Case("ch42.to_anthropic.one", lambda f: f([TOOL_A]), [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "1", "is_error": False}]}], hint="ch42.to_anthropic.hint"),
        Case("ch42.to_anthropic.merge", lambda f: [len(m["content"]) for m in f([ASK, TOOL_A, TOOL_B])], [1, 2], hint="ch42.to_anthropic.merge_hint"),
        Case("ch42.to_anthropic.user", lambda f: [m["role"] for m in f([user("a"), TOOL_A])], ["user", "user"], hint="ch42.to_anthropic.merge_hint"),
    ],
    "tool_changes": [
        Case("ch42.tool_changes.same", lambda f: f({"read": 1}, {"read": 1}), ([], []), hint="ch42.tool_changes.hint"),
        Case("ch42.tool_changes.add", lambda f: f({"read": 1}, {"read": 1, "edit": 2}), (["edit"], []), hint="ch42.tool_changes.hint"),
        Case("ch42.tool_changes.remove", lambda f: f({"read": 1, "edit": 2}, {"read": 1}), ([], ["edit"]), hint="ch42.tool_changes.hint"),
        Case("ch42.tool_changes.changed", lambda f: f({"read": 1}, {"read": 2}), (["read"], ["read"]), hint="ch42.tool_changes.hint"),
    ],
}
