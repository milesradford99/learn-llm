"""Even depth: two numbers per chapter that show when a lesson rushes or dictates.

- `printed`: the share of the reference solution's lines that the lesson text prints word
  for word. High means the exercises can be passed by copying.
- `ideas`: concepts the chapter introduces in `lessons/ledger.toml`, not counting those the
  chapter lists under `toolbox` (small tools taught in a Toolbox box).

    python -m learn_llm.depth        # the table for every chapter

Limits and the chapters still waiting for a rework are in `tests/test_depth.py`.
"""

import ast
import re

from learn_llm.course import LESSONS, ROOT
from learn_llm.i18n import SOURCE_LOCALE
from learn_llm.ledger import load

SOLUTIONS = ROOT / "solutions"
SHORTEST = 12  # shorter lines (`out = []`, `return x`) match by accident


def _squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _notebook(number: str):
    return next((LESSONS / SOURCE_LOCALE).glob(f"{number}-*.py"))


def prose(number: str) -> str:
    """All the lesson text of a chapter: the string inside every `mo.md(...)`."""
    tree = ast.parse(_notebook(number).read_text(encoding="utf-8"))
    parts = []
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "md":
            parts += [n.value for n in ast.walk(call) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return "\n".join(parts)


def stubs(number: str) -> set[str]:
    """Names of the exercise functions: the ones whose body still calls `todo()`."""
    tree = ast.parse(_notebook(number).read_text(encoding="utf-8"))
    return {
        f.name for f in tree.body if isinstance(f, ast.FunctionDef)
        and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "todo" for c in ast.walk(f))
    }


def printed(number: str) -> tuple[int, int]:
    """(solution lines printed in the lesson text, solution lines) over a chapter's exercises."""
    path = SOLUTIONS / f"ch{number}.py"
    if not path.exists():
        return 0, 0
    source = path.read_text(encoding="utf-8")
    lines, text, names = source.splitlines(), _squeeze(prose(number)), stubs(number)
    found = total = 0
    for f in ast.parse(source).body:
        if isinstance(f, ast.FunctionDef) and f.name in names:
            body = [_squeeze(line) for line in lines[f.body[0].lineno - 1:f.end_lineno]]
            body = [line for line in body if len(line) >= SHORTEST and not line.startswith(("#", '"""'))]
            total += len(body)
            found += sum(line in text for line in body)
    return found, total


def ideas(number: str) -> int:
    entry = load()["chapters"][number]
    return len(set(entry.get("introduces", [])) - set(entry.get("toolbox", [])))


def words(number: str) -> int:
    return len(re.findall(r"\w+", prose(number)))


if __name__ == "__main__":
    print("ch  words  ideas  words/idea  printed")
    for number in sorted(load()["chapters"]):
        found, total = printed(number)
        share = f"{100 * found / total:.0f}%" if total else "-"
        print(f"{number}  {words(number):5}  {ideas(number):5}  {words(number) / max(ideas(number), 1):10.0f}  {share:>7}")
