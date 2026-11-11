"""Reference solutions for chapter 51. Peek only after trying!"""

import re


def prompt_order(role):
    order = ["shared and role", "team sections", "subscriptions and replies"]
    if role == "orchestrator":
        order.append("dispatch")
    else:
        order.append("work")
    if role == "orchestrator":
        order.append("admission")
    order = order + ["handoff", "environment"]
    if role == "orchestrator":
        order.append("policy")
    return order


def preview(text):
    return re.sub(r"\s+", " ", text)[:160]


def exam():
    return {
        "q1": "b", "q2": "c", "q3": "a", "q4": "b", "q5": "c", "q6": "a",
        "q7": "c", "q8": "b", "q9": "a", "q10": "c", "q11": "b", "q12": "a",
    }
