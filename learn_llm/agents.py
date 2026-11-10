"""Offline stand-ins for the agents chapters (30 onward): messages, a scripted model, a trace.

Nothing here calls a real model, so the chapters run with no key, no network and no cost,
and give the same result every time. Messages follow the shape Pi uses in its transcripts.
"""

def user(text):
    return {"role": "user", "content": text}


def text_block(text):
    return {"type": "text", "text": text}


def call(name, arguments=None, id="call_1"):
    """One tool call block, as it sits inside an assistant message."""
    return {"type": "toolCall", "id": id, "name": name, "arguments": arguments or {}}


def say(text, stop="stop"):
    """An assistant message that only talks."""
    return {"role": "assistant", "content": [text_block(text)], "stopReason": stop}


def ask(*calls, stop="toolUse"):
    """An assistant message that asks for one or more tools to run."""
    return {"role": "assistant", "content": list(calls), "stopReason": stop}


def result(call_id, name, text, is_error=False):
    """A tool result message, which answers one tool call by its id."""
    return {"role": "toolResult", "toolCallId": call_id, "toolName": name, "content": text, "isError": is_error}


class FakeModel:
    """Plays back a fixed list of replies, and remembers what it was shown each time."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages):
        self.seen.append(list(messages))
        if not self.replies:
            return say("(the script has no more replies)")
        return self.replies.pop(0)


def trace(messages):
    """A readable one-line-per-message view of a conversation."""
    lines = []
    for message in messages:
        role = message["role"]
        if role == "toolResult":
            mark = "error" if message["isError"] else "ok"
            lines.append(f"tool {message['toolName']} {mark}: {message['content']}")
        elif role == "assistant":
            parts = [b["text"] if b["type"] == "text" else f"[{b['name']}]" for b in message["content"]]
            lines.append("assistant: " + " ".join(parts))
        else:
            lines.append(f"{role}: {message['content']}")
    return lines


def attempt(fn, *args):
    """Run an exercise for a demo cell; until it's written, say so instead of showing an error."""
    try:
        return fn(*args)
    except NotImplementedError:
        return "(finish the exercise above to see this run)"


def watch(fn, *args):
    """Like `attempt`, for a function that gives back messages: show them as a trace."""
    outcome = attempt(fn, *args)
    return outcome if isinstance(outcome, str) else trace(outcome)
