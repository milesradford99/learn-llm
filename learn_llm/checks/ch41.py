from learn_llm.checking import Case

ANSWERS = ["b", "c", "a", "b", "c", "a", "c", "b", "a", "c", "b", "a"]


def _q(number):
    return Case(f"ch41.exam.q{number}", lambda f: f()[f"q{number}"], ANSWERS[number - 1], hint="ch41.exam.hint")


EXERCISES = {
    "prompt_order": [
        Case("ch41.prompt_order.router", lambda f: f("orchestrator"), ["shared and role", "team sections", "subscriptions and replies", "dispatch", "admission", "handoff", "environment", "policy"], hint="ch41.prompt_order.hint"),
        Case("ch41.prompt_order.worker", lambda f: f("worker"), ["shared and role", "team sections", "subscriptions and replies", "work", "handoff", "environment"], hint="ch41.prompt_order.hint"),
    ],
    "preview": [
        Case("ch41.preview.spaces", lambda f: f("fix\n\n  the   button"), "fix the button", hint="ch41.preview.hint"),
        Case("ch41.preview.cut", lambda f: len(f("a" * 500)), 160, hint="ch41.preview.hint"),
        Case("ch41.preview.edge", lambda f: f("\thi "), " hi ", hint="ch41.preview.hint"),
    ],
    "exam": [_q(n) for n in range(1, 13)],
}
