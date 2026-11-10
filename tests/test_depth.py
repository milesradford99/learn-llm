"""Even depth: no chapter may dictate its exercises or pack in too many ideas.

`WAITING` lists the chapters written before these limits (docs/design/depth-plan.md).
A chapter leaves the list when it is reworked; a new chapter never joins it.
"""

import pytest

from learn_llm.depth import ideas, printed
from learn_llm.ledger import load

MOST_PRINTED = 0.30  # share of a chapter's solution lines its text may print word for word
MOST_IDEAS = 12  # new ideas in a chapter, not counting its ledger `toolbox` list

WAITING = {
    "printed": {"03", "04", "05", "06", "08", "09", "10", "11", "12", "13", "14", "16",
                "18", "19", "20", "22", "23", "40", "41"},
    "ideas": {"00", "05", "06", "09", "10", "11", "12", "13", "14", "16", "18", "19", "20",
              "22", "23", "40"},
}
CHAPTERS = sorted(load()["chapters"])


def _too_printed(number):
    found, total = printed(number)
    return total > 0 and found / total > MOST_PRINTED


def _too_many(number):
    return ideas(number) > MOST_IDEAS


@pytest.mark.parametrize("number", CHAPTERS)
@pytest.mark.parametrize("limit, over", [("printed", _too_printed), ("ideas", _too_many)])
def test_chapter_keeps_an_even_depth(number, limit, over):
    if number in WAITING[limit]:
        assert over(number), f"ch{number} now meets the '{limit}' limit: take it off WAITING"
    else:
        assert not over(number), f"ch{number} is over the '{limit}' limit: see `python -m learn_llm.depth`"


def test_toolbox_ideas_are_introduced_by_the_same_chapter():
    for number, entry in load()["chapters"].items():
        assert set(entry.get("toolbox", [])) <= set(entry.get("introduces", [])), number
