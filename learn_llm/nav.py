"""Links between chapters, shown at the bottom of every lesson."""

from learn_llm.course import LESSONS, chapters
from learn_llm.i18n import t


def links(number: str, locale: str) -> list[tuple[str, str]]:
    """(label, href) for the previous chapter, home and the next chapter, in that order."""
    order = chapters(locale)
    numbers = [ch.number for ch in order]
    i = numbers.index(number)
    found = []
    if i > 0:
        prev = order[i - 1]
        found.append((t("nav.previous", locale, title=prev.title), _href(prev.path)))
    found.append((t("nav.home", locale), "?file=home.py"))
    if i + 1 < len(order):
        nxt = order[i + 1]
        found.append((t("nav.next", locale, title=nxt.title), _href(nxt.path)))
    else:
        found.append((t("nav.last", locale), ""))
    return found


def _href(path) -> str:
    # The server serves my-work/, which mirrors lessons/: same relative paths.
    return f"?file={path.relative_to(LESSONS).as_posix()}"


def nav(number: str, locale: str):
    """The previous / home / next bar for chapter `number`."""
    import marimo as mo

    parts = [f"[{label}]({href})" if href else f"*{label}*" for label, href in links(number, locale)]
    return mo.md("---\n\n" + "  ·  ".join(parts))
