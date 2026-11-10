"""Reference solutions for chapter 31. Peek only after trying!"""

from learn_llm.agents import result


def take_one(queue):
    if len(queue) == 0:
        return []
    return [queue.pop(0)]


def can_continue(messages):
    if len(messages) == 0:
        return False
    return messages[-1]["role"] != "assistant"


def truncated_result(call):
    text = f"Tool call \"{call['name']}\" was not executed: the response hit the output token limit."
    return result(call["id"], call["name"], text, True)


def run_loop(model, messages, run_call, steering, follow_ups):
    pending = take_one(steering)
    while True:
        more_calls = True
        while more_calls or len(pending) > 0:
            for message in pending:
                messages.append(message)
            pending = []
            reply = model(messages)
            messages.append(reply)
            if reply["stopReason"] == "error" or reply["stopReason"] == "aborted":
                return messages
            calls = [block for block in reply["content"] if block["type"] == "toolCall"]
            more_calls = len(calls) > 0
            for call in calls:
                if reply["stopReason"] == "length":
                    messages.append(truncated_result(call))
                else:
                    messages.append(run_call(call))
            pending = take_one(steering)
        extra = take_one(follow_ups)
        if len(extra) == 0:
            return messages
        pending = extra
