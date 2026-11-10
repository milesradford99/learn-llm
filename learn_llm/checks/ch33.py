from learn_llm.agents import call, result
from learn_llm.checking import Case

CODE = "def a():\n    return 1\n\ndef b():\n    return 1\n"


def _raises(f, *args):
    try:
        f(*args)
    except ValueError as error:
        return str(error).split(" ")[0]
    return "no error"


EXERCISES = {
    "truncate_head": [
        Case("ch33.truncate_head.fits", lambda f: f("a\nb", 5, 100), ("a\nb", False), hint="ch33.truncate_head.hint"),
        Case("ch33.truncate_head.lines", lambda f: f("a\nb\nc", 2, 100), ("a\nb", True), hint="ch33.truncate_head.hint"),
        Case("ch33.truncate_head.bytes", lambda f: f("aaaa\nbbbb\ncc", 100, 9), ("aaaa\nbbbb", True), hint="ch33.truncate_head.hint"),
        Case("ch33.truncate_head.whole", lambda f: f("aaaa\nbbbb", 100, 8), ("aaaa", True), hint="ch33.truncate_head.hint"),
        Case("ch33.truncate_head.utf8", lambda f: f("你好\n世界", 100, 7), ("你好", True), hint="ch33.truncate_head.hint"),
        Case("ch33.truncate_head.huge", lambda f: f("x" * 50, 100, 10), ("", True), hint="ch33.truncate_head.hint"),
    ],
    "fuzzy": [
        Case("ch33.fuzzy.trailing", lambda f: f("a  \nb\t\nc"), "a\nb\nc", hint="ch33.fuzzy.hint"),
        Case("ch33.fuzzy.leading", lambda f: f("    a\n"), "    a\n", hint="ch33.fuzzy.hint"),
    ],
    "apply_edit": [
        Case("ch33.apply_edit.one", lambda f: f("x = 1\ny = 2\n", "y = 2", "y = 3"), "x = 1\ny = 3\n", hint="ch33.apply_edit.hint"),
        Case("ch33.apply_edit.missing", lambda f: _raises(f, CODE, "return 2", "x"), "Could", hint="ch33.apply_edit.hint"),
        Case("ch33.apply_edit.twice", lambda f: _raises(f, CODE, "return 1", "x"), "Found", hint="ch33.apply_edit.hint"),
        Case("ch33.apply_edit.empty", lambda f: _raises(f, CODE, "", "x"), "oldText", hint="ch33.apply_edit.hint"),
        Case("ch33.apply_edit.spaces", lambda f: f("a  \nb\n", "a\nb", "c"), "c\n", hint="ch33.apply_edit.fuzzy_hint"),
        Case("ch33.apply_edit.untouched", lambda f: f("x  \na  \nb\nc  \n", "a\nb", "A"), "x  \nA\nc  \n", hint="ch33.apply_edit.fuzzy_hint"),
    ],
    "in_source_order": [
        Case("ch33.in_source_order.flip", lambda f: [r["toolCallId"] for r in f([call("a", id="1"), call("b", id="2")], [result("2", "b", "y"), result("1", "a", "x")])], ["1", "2"], hint="ch33.in_source_order.hint"),
        Case("ch33.in_source_order.same", lambda f: [r["toolCallId"] for r in f([call("a", id="1")], [result("1", "a", "x")])], ["1"], hint="ch33.in_source_order.hint"),
    ],
}
