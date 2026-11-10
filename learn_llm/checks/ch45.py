from learn_llm.agents import ask, call, result, say, user
from learn_llm.checking import Case


def _words(n):
    return "x" * (n * 4)


LONG = [user(_words(100)), ask(call("read", {"p": "a"})), result("call_1", "read", _words(100)), say(_words(100)), user(_words(10))]
TOOL_HEAVY = [user("go"), ask(call("read", {"p": "a"})), result("call_1", "read", _words(1000))]


EXERCISES = {
    "estimate_tokens": [
        Case("ch45.estimate_tokens.user", lambda f: f(user("abcd")), 1, hint="ch45.estimate_tokens.hint"),
        Case("ch45.estimate_tokens.round", lambda f: f(user("abcde")), 2, hint="ch45.estimate_tokens.hint"),
        Case("ch45.estimate_tokens.text", lambda f: f(say("abcdefgh")), 2, hint="ch45.estimate_tokens.hint"),
        Case("ch45.estimate_tokens.call", lambda f: f(ask(call("read", {"p": "a"}))), 4, hint="ch45.estimate_tokens.hint"),
        Case("ch45.estimate_tokens.tool", lambda f: f(result("a", "read", "abcdefgh")), 2, hint="ch45.estimate_tokens.hint"),
    ],
    "should_compact": [
        Case("ch45.should_compact.below", lambda f: f(100, 1000, 200), False, hint="ch45.should_compact.hint"),
        Case("ch45.should_compact.edge", lambda f: f(800, 1000, 200), False, hint="ch45.should_compact.hint"),
        Case("ch45.should_compact.over", lambda f: f(801, 1000, 200), True, hint="ch45.should_compact.hint"),
    ],
    "find_cut_point": [
        Case("ch45.find_cut_point.keep", lambda f: f(LONG, 105), 3, hint="ch45.find_cut_point.hint"),
        Case("ch45.find_cut_point.tool", lambda f: f(LONG, 200), 3, hint="ch45.find_cut_point.hint"),
        Case("ch45.find_cut_point.more", lambda f: f(LONG, 211), 1, hint="ch45.find_cut_point.hint"),
        Case("ch45.find_cut_point.all", lambda f: f(LONG, 100000), 0, hint="ch45.find_cut_point.hint"),
        Case("ch45.find_cut_point.trailing", lambda f: f(TOOL_HEAVY, 10), 1, hint="ch45.find_cut_point.tail_hint"),
    ],
    "retry_delay": [
        Case("ch45.retry_delay.first", lambda f: f(1, 2000, 60000), 2000, hint="ch45.retry_delay.hint"),
        Case("ch45.retry_delay.third", lambda f: f(3, 2000, 60000), 8000, hint="ch45.retry_delay.hint"),
        Case("ch45.retry_delay.cap", lambda f: f(10, 2000, 60000), 60000, hint="ch45.retry_delay.hint"),
    ],
    "next_step": [
        Case("ch45.next_step.done", lambda f: f(say("ok"), False, False, 0, 3), "done", hint="ch45.next_step.hint"),
        Case("ch45.next_step.compact", lambda f: f(say("too long", stop="error"), True, False, 0, 3), "compact", hint="ch45.next_step.hint"),
        Case("ch45.next_step.once", lambda f: f(say("too long", stop="error"), True, True, 0, 3), "give up", hint="ch45.next_step.hint"),
        Case("ch45.next_step.retry", lambda f: f(say("503", stop="error"), False, False, 2, 3), "retry", hint="ch45.next_step.hint"),
        Case("ch45.next_step.spent", lambda f: f(say("503", stop="error"), False, False, 3, 3), "give up", hint="ch45.next_step.hint"),
    ],
}
