"""Reference solutions for chapter 48. Peek only after trying!"""

def admission_cell(listed, addressed):
    if listed:
        if addressed:
            return None
        return "listedUnaddressed"
    if addressed:
        return "unlistedAddressed"
    return "unlistedUnaddressed"


def gate(listed, addressed, admission, follow_up, mode):
    if listed and addressed:
        return "explicit"
    if admission == "explicit":
        return "explicit"
    if follow_up and (listed or mode != "off"):
        if listed:
            return "explicit"
        return "inferred"
    if admission == "inferred" and mode != "off":
        return "inferred"
    return "not admitted"


def outcome(decision, matched, strength, mode):
    if decision != "accept":
        return decision
    if len(matched) == 0:
        return "ignore"
    if strength == "inferred" and mode != "auto":
        return "would accept"
    return "accept"


def on_deadline(expired):
    expired = expired + 1
    if expired > 2:
        return "unavailable"
    return "requeue"
