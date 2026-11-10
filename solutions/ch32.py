"""Reference solutions for chapter 32. Peek only after trying!"""

import json


def collect_text(events):
    texts = {}
    for event in events:
        if event["type"] == "text_delta":
            index = event["contentIndex"]
            texts[index] = texts.get(index, "") + event["delta"]
    return texts


def final_message(events):
    last = events[-1]
    if last["type"] == "done":
        return last["message"]
    if last["type"] == "error":
        return last["error"]
    return None


def to_openai(message):
    if message["role"] == "user":
        return {"role": "user", "content": message["content"]}
    if message["role"] == "toolResult":
        return {"role": "tool", "tool_call_id": message["toolCallId"], "content": message["content"]}
    texts = [block["text"] for block in message["content"] if block["type"] == "text"]
    out = {"role": "assistant", "content": "".join(texts)}
    calls = []
    for block in message["content"]:
        if block["type"] == "toolCall":
            function = {"name": block["name"], "arguments": json.dumps(block["arguments"])}
            calls.append({"id": block["id"], "type": "function", "function": function})
    if len(calls) > 0:
        out["tool_calls"] = calls
    return out


def to_anthropic(messages):
    out = []
    for message in messages:
        if message["role"] == "user":
            out.append({"role": "user", "content": message["content"]})
        elif message["role"] == "assistant":
            blocks = []
            for block in message["content"]:
                if block["type"] == "text":
                    blocks.append({"type": "text", "text": block["text"]})
                else:
                    blocks.append({"type": "tool_use", "id": block["id"], "name": block["name"], "input": block["arguments"]})
            out.append({"role": "assistant", "content": blocks})
        else:
            block = {"type": "tool_result", "tool_use_id": message["toolCallId"], "content": message["content"], "is_error": message["isError"]}
            if len(out) > 0 and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


def tool_changes(declared, current):
    added = [name for name in current if name not in declared or declared[name] != current[name]]
    removed = [name for name in declared if name not in current or declared[name] != current[name]]
    return added, removed
