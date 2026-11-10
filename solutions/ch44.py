"""Reference solutions for chapter 44. Peek only after trying!"""

import json


def append(session, entry):
    entry["id"] = f"e{len(session['entries'])}"
    entry["parentId"] = session["leaf"]
    session["entries"].append(entry)
    session["leaf"] = entry["id"]
    return entry["id"]


def active_path(session):
    by_id = {}
    for entry in session["entries"]:
        by_id[entry["id"]] = entry
    path = []
    current = session["leaf"]
    while current is not None:
        entry = by_id[current]
        path.append(entry)
        current = entry["parentId"]
    path.reverse()
    return path


def branch(session, entry_id):
    ids = [entry["id"] for entry in session["entries"]]
    if entry_id not in ids:
        raise ValueError(f"Entry {entry_id} not found")
    session["leaf"] = entry_id


def context(session):
    return [entry["message"] for entry in active_path(session) if entry["type"] == "message"]


def to_jsonl(session):
    return "\n".join([json.dumps(entry) for entry in session["entries"]])


def from_jsonl(text):
    entries = [json.loads(line) for line in text.split("\n")]
    return {"entries": entries, "leaf": entries[-1]["id"]}
