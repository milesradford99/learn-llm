"""Reference solutions for chapter 30. Peek only after trying!"""

from learn_llm.agents import result


def tool_calls(message):
    return [block for block in message["content"] if block["type"] == "toolCall"]


def run_tool(tools, call):
    name = call["name"]
    if name not in tools:
        return result(call["id"], name, f"Tool {name} not found", True)
    try:
        text = tools[name](call["arguments"])
    except Exception as error:
        return result(call["id"], name, str(error), True)
    return result(call["id"], name, text)


def agent_loop(model, messages, tools, max_turns):
    turns = 0
    while turns < max_turns:
        reply = model(messages)
        messages.append(reply)
        turns = turns + 1
        calls = tool_calls(reply)
        if len(calls) == 0:
            return messages
        for call in calls:
            messages.append(run_tool(tools, call))
    return messages
