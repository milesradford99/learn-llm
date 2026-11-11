"""Reference solutions for chapter 43. Peek only after trying!"""

def truncate_head(text, max_lines, max_bytes):
    kept = []
    size = 0
    for line in text.split("\n"):
        cost = len(line.encode("utf-8"))
        if len(kept) > 0:
            cost = cost + 1
        if len(kept) == max_lines or size + cost > max_bytes:
            return "\n".join(kept), True
        kept.append(line)
        size = size + cost
    return text, False


def fuzzy(text):
    return "\n".join([line.rstrip() for line in text.split("\n")])


def apply_edit(content, old, new):
    if old == "":
        raise ValueError("oldText must not be empty.")
    matches = fuzzy(content).count(fuzzy(old))
    if matches == 0:
        raise ValueError("Could not find the text. It must match exactly.")
    if matches > 1:
        raise ValueError(f"Found {matches} occurrences of the text. It must be unique.")
    if old in content:
        return content.replace(old, new)
    return rewrite_touched_lines(content, old, new)


def rewrite_touched_lines(content, old, new):
    base = fuzzy(content)
    first = base[:base.index(fuzzy(old))].count("\n")
    last = first + fuzzy(old).count("\n")
    rewritten = base.replace(fuzzy(old), new).split("\n")
    lines = content.split("\n")
    middle = rewritten[first:len(rewritten) - (len(lines) - last - 1)]
    return "\n".join(lines[:first] + middle + lines[last + 1:])


def in_source_order(calls, finished):
    by_id = {}
    for message in finished:
        by_id[message["toolCallId"]] = message
    return [by_id[call["id"]] for call in calls]
