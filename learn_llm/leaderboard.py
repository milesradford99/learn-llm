"""The leaderboard: one perplexity per model, on the same held-out stories, from chapter 12 on.

Each learner's own scores live in `my-work/.leaderboard.json`, so they carry across chapters.
"""

import json

from learn_llm.course import WORK
from learn_llm.i18n import t

BOARD = WORK / ".leaderboard.json"


def _read() -> dict[str, float]:
    try:
        return json.loads(BOARD.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def record(model: str, score: float) -> None:
    """Store `score` for `model`, replacing any earlier one."""
    board = {**_read(), model: round(float(score), 1)}
    try:
        BOARD.parent.mkdir(exist_ok=True)
        BOARD.write_text(json.dumps(board), encoding="utf-8")
    except OSError:  # a read-only folder costs the learner the table, not the lesson
        pass


def show(locale: str):
    """The board as a table on the page, best score first."""
    import marimo as mo

    rows = sorted(_read().items(), key=lambda item: item[1])
    lines = [f"| {t('board.model', locale)} | {t('board.perplexity', locale)} |", "|---|---:|"]
    lines += [f"| {t('board.' + model, locale)} | {score:g} |" for model, score in rows]
    return mo.md("\n".join(lines))
