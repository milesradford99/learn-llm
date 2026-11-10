"""Reference solutions for chapter 35. Peek only after trying!"""

import json
import math


def estimate_tokens(message):
    if message["role"] == "assistant":
        chars = 0
        for block in message["content"]:
            if block["type"] == "text":
                chars = chars + len(block["text"])
            else:
                chars = chars + len(block["name"]) + len(json.dumps(block["arguments"], separators=(",", ":")))
    else:
        chars = len(message["content"])
    return math.ceil(chars / 4)


def should_compact(context_tokens, context_window, reserve):
    return context_tokens > context_window - reserve


def find_cut_point(messages, keep):
    cuts = [i for i in range(len(messages)) if messages[i]["role"] != "toolResult"]
    total = 0
    for i in reversed(range(len(messages))):
        total = total + estimate_tokens(messages[i])
        if total >= keep:
            later = [cut for cut in cuts if cut >= i]
            if len(later) > 0:
                return later[0]
            return cuts[-1]
    return cuts[0]


def retry_delay(attempt, base, cap):
    return min(base * 2 ** max(0, attempt - 1), cap)


def next_step(message, overflowed, compacted_already, retries, max_retries):
    if overflowed:
        if compacted_already:
            return "give up"
        return "compact"
    if message["stopReason"] == "error":
        if retries < max_retries:
            return "retry"
        return "give up"
    return "done"
