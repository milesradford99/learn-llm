"""Reference solutions for chapter 46. Peek only after trying!"""

def build_prompt(sections):
    parts = []
    for name, text in sections.items():
        if text == "":
            continue
        if name == "preamble":
            parts.append(text)
        else:
            parts.append(f"<{name}>\n{text}\n</{name}>")
    return "\n\n".join(parts)


def escape_xml(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_skills(skills):
    visible = [skill for skill in skills if not skill["disableModelInvocation"]]
    if len(visible) == 0:
        return ""
    lines = ["<available_skills>"]
    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{escape_xml(skill['name'])}</name>")
        lines.append(f"    <description>{escape_xml(skill['description'])}</description>")
        lines.append(f"    <location>{escape_xml(skill['filePath'])}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)


def emit_tool_call(handlers, event):
    outcome = None
    for handler in handlers:
        answer = handler(event)
        if answer is not None:
            outcome = answer
            if answer.get("block"):
                return outcome
    return outcome
