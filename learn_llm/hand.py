"""Exercises done by hand: number boxes checked on the page, and hints a learner opens.

    boxes = hand.boxes(["p", "g"])                      # in one cell, returned
    hand.marked(boxes, {"p": 200, "g": 140}, hints, LOCALE)   # in the next cell

marimo re-runs the second cell whenever a box changes, so each answer is checked as it's typed.
`hints` is lesson text, so it lives in the lesson, in that locale's words.
"""

import marimo as mo

from learn_llm.i18n import t


def boxes(names, step=1):
    """One number box per name. Create it in its own cell and return it."""
    return mo.ui.dictionary({name: mo.ui.number(label=name, step=step) for name in names})


def right(value, answer, tolerance=1e-6) -> bool:
    return value is not None and abs(value - answer) <= tolerance


def done(boxes, answers, tolerance=1e-6) -> bool:
    return all(right(boxes.value[name], answer, tolerance) for name, answer in answers.items())


def marked(boxes, answers, hints, locale, tolerance=1e-6, per_row=5):
    """The boxes, in rows, over a list saying which answers are right and a hint for the others."""
    lines = []
    for name, answer in answers.items():
        value = boxes.value[name]
        if value is None:
            lines.append(t("hand.empty", locale, name=name, hint=hints[name]))
        elif right(value, answer, tolerance):
            lines.append(t("hand.right", locale, name=name, value=f"{value:g}"))
        else:
            lines.append(t("hand.wrong", locale, name=name, value=f"{value:g}", hint=hints[name]))
    names = list(answers)
    rows = [
        mo.hstack([boxes[name] for name in names[start:start + per_row]], justify="start", wrap=True)
        for start in range(0, len(names), per_row)
    ]
    kind = "success" if done(boxes, answers, tolerance) else "neutral"
    return mo.vstack([*rows, mo.callout(mo.md("\n".join(lines)), kind=kind)])


def stuck(hints, locale):
    """Hints the learner opens one at a time, from the gentlest to the most direct."""
    return mo.accordion({t("hand.hint", locale, n=i + 1): mo.md(hint) for i, hint in enumerate(hints)})
