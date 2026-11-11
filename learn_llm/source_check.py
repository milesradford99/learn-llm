"""Check that the source excerpts quoted in the agents chapters match the pinned projects.

An excerpt is a fenced block right after a `<!-- source pi path -->` comment. Every line of it
must appear in that file, in order; a line holding only `…` marks a cut. Run this after
re-pinning, with fresh clones of both projects at the commits in `learn_llm/data/AGENT-SOURCES.md`:

    python -m learn_llm.source_check --pi ../pi-mono --dsh ../dsh-superagent-prototype
"""

import argparse
import re
import sys
from pathlib import Path

from learn_llm.course import LESSONS

EXCERPT = re.compile(r"<!-- source (\w+) (\S+) -->\s*```\w*\n(.*?)\n\s*```", re.DOTALL)


def excerpts(lesson: Path) -> list[tuple[str, str, list[str]]]:
    """Each excerpt in a lesson: (project, path, its lines without their indentation)."""
    found = []
    for project, path, body in EXCERPT.findall(lesson.read_text(encoding="utf-8")):
        lines = [line.strip() for line in body.splitlines()]
        found.append((project, path, [line for line in lines if line and line != "…"]))
    return found


def missing_line(quoted: list[str], source: str) -> str | None:
    """The first quoted line not found in `source` after the line before it, if any."""
    lines = [line.strip() for line in source.splitlines()]
    position = 0
    for line in quoted:
        try:
            position = lines.index(line, position) + 1
        except ValueError:
            return line
    return None


def problems(roots: dict[str, Path]) -> list[str]:
    found = []
    for lesson in sorted(LESSONS.rglob("3[0-9]-*.py")) + sorted(LESSONS.rglob("4[0-9]-*.py")):
        for project, path, quoted in excerpts(lesson):
            file = roots[project] / path
            if not file.exists():
                found.append(f"{lesson.name}: {project} has no {path}")
                continue
            line = missing_line(quoted, file.read_text(encoding="utf-8"))
            if line is not None:
                found.append(f"{lesson.name}: {path}: not found in order: {line}")
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m learn_llm.source_check")
    parser.add_argument("--pi", type=Path, required=True)
    parser.add_argument("--dsh", type=Path, required=True)
    args = parser.parse_args(argv)
    found = problems({"pi": args.pi, "dsh": args.dsh})
    print("\n".join(found) or "Every excerpt matches.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
