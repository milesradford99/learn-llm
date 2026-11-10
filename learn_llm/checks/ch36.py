from learn_llm.checking import Case

ONE = {"name": "pdf", "description": "Read PDFs", "filePath": "/s/pdf/SKILL.md", "disableModelInvocation": False}
HIDDEN = {"name": "deploy", "description": "Ship it", "filePath": "/s/d/SKILL.md", "disableModelInvocation": True}
TRICKY = {"name": "a", "description": "x < y & z", "filePath": "/a", "disableModelInvocation": False}


def _blocker(event):
    return {"block": True, "reason": "no"}


def _note(event):
    return {"note": event["name"]}


def _silent(event):
    return None


def _never(event):
    raise ValueError("should not run")


EXERCISES = {
    "build_prompt": [
        Case("ch36.build_prompt.preamble", lambda f: f({"preamble": "You help."}), "You help.", hint="ch36.build_prompt.hint"),
        Case("ch36.build_prompt.tag", lambda f: f({"preamble": "Hi", "cwd": "/app"}), "Hi\n\n<cwd>\n/app\n</cwd>", hint="ch36.build_prompt.hint"),
        Case("ch36.build_prompt.empty", lambda f: f({"preamble": "Hi", "skills": ""}), "Hi", hint="ch36.build_prompt.hint"),
    ],
    "escape_xml": [
        Case("ch36.escape_xml.all", lambda f: f("a<b>&c"), "a&lt;b&gt;&amp;c", hint="ch36.escape_xml.hint"),
        Case("ch36.escape_xml.twice", lambda f: f("<"), "&lt;", hint="ch36.escape_xml.hint"),
    ],
    "format_skills": [
        Case("ch36.format_skills.none", lambda f: f([HIDDEN]), "", hint="ch36.format_skills.hint"),
        Case("ch36.format_skills.one", lambda f: f([ONE, HIDDEN]), "<available_skills>\n  <skill>\n    <name>pdf</name>\n    <description>Read PDFs</description>\n    <location>/s/pdf/SKILL.md</location>\n  </skill>\n</available_skills>", hint="ch36.format_skills.hint"),
        Case("ch36.format_skills.escape", lambda f: "x &lt; y &amp; z" in f([TRICKY]), True, hint="ch36.format_skills.hint"),
    ],
    "emit_tool_call": [
        Case("ch36.emit_tool_call.none", lambda f: f([_silent], {"name": "read"}), None, hint="ch36.emit_tool_call.hint"),
        Case("ch36.emit_tool_call.block", lambda f: f([_silent, _blocker, _never], {"name": "bash"}), {"block": True, "reason": "no"}, hint="ch36.emit_tool_call.hint"),
        Case("ch36.emit_tool_call.last", lambda f: f([_note, _silent], {"name": "read"}), {"note": "read"}, hint="ch36.emit_tool_call.hint"),
    ],
}
