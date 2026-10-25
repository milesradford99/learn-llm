import ast
import re
import string
import subprocess
import sys

import pytest

from learn_llm.checking import PASS, TODO, run
from learn_llm.checks import exercises
from learn_llm.course import LESSONS, SOLUTIONS, chapters, grade, load
from learn_llm.i18n import LOCALES, SOURCE_LOCALE, catalog
from learn_llm.i18n_stamp import stale

CHAPTERS = [ch.number for ch in chapters(SOURCE_LOCALE)]
NOTEBOOKS = [ch.path for loc in LOCALES for ch in chapters(loc) if not ch.fallback]


def ids(paths):
    return [str(p.relative_to(LESSONS)) for p in paths]


@pytest.mark.parametrize("number", CHAPTERS)
def test_solutions_pass(number):
    for name, results in grade(number, SOLUTIONS / f"ch{number}.py").items():
        assert all(r.status == PASS for r in results), name


@pytest.mark.parametrize("path", NOTEBOOKS, ids=ids(NOTEBOOKS))
def test_stubs_are_not_started(path):
    """Untouched exercises must not pass, or the checks prove nothing."""
    for name, results in grade(path.name[:2], path).items():
        assert all(r.status == TODO for r in results), name


@pytest.mark.parametrize("path", NOTEBOOKS, ids=ids(NOTEBOOKS))
def test_code_is_the_same_in_every_language(path):
    """Only prose and comments are translated; exercise code must match the English notebook."""
    def functions(p):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != "_"}

    assert functions(path) == functions(LESSONS / SOURCE_LOCALE / path.name)


@pytest.mark.parametrize("locale", [loc for loc in LOCALES if loc != SOURCE_LOCALE])
def test_translations_are_complete_and_current(locale):
    assert not [ch.path.name for ch in chapters(locale) if ch.fallback], "missing translations"
    assert not stale(locale), "English changed since translation; update it, then `make i18n-stamp`"


@pytest.mark.parametrize("locale", LOCALES)
def test_catalogs_match_english(locale):
    def fields(text):
        return {f for _, f, _, _ in string.Formatter().parse(text) if f}

    english, other = catalog(SOURCE_LOCALE), catalog(locale)
    assert set(other) == set(english)
    for key, text in other.items():
        assert fields(text) == fields(english[key]), key


@pytest.mark.parametrize("number", CHAPTERS)
def test_every_check_has_a_message(number):
    keys = {case.key for cases in exercises(number).values() for case in cases}
    assert keys <= set(catalog(SOURCE_LOCALE))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=ids(NOTEBOOKS))
def test_no_required_external_links(path):
    """Everything needed lives in the course; links only in the optional "go deeper" part."""
    required = path.read_text(encoding="utf-8").split("<!-- go-deeper -->")[0]
    assert not re.findall(r'(?<!xmlns=")https?://', required)  # SVG namespaces are not links


def test_marimo_check_is_clean():
    out = subprocess.run(
        [sys.executable, "-m", "marimo", "check", "--strict", str(LESSONS)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_checks_report_common_mistakes():
    def forgot_return(numbers):
        sum(numbers)

    def crashes(numbers):
        return numbers[10]

    cases = exercises("00")["total"]
    assert {r.status for r in run(cases, forgot_return)} == {"fail"}
    assert all(r.got is None for r in run(cases, forgot_return))
    assert {r.status for r in run(cases, crashes)} == {"error"}


def test_notebooks_import_without_running_cells():
    module = load(LESSONS / SOURCE_LOCALE / "00-first-program.py")
    assert module.double(4) == 8


def test_unfinished_helper_is_reported_not_blamed_on_the_caller():
    from learn_llm import todo
    from learn_llm.checking import BLOCKED, explain

    def total(numbers):
        todo()

    def average(numbers):
        return total(numbers) / len(numbers)

    results = run(exercises("00")["average"], average)
    assert {r.status for r in results} == {BLOCKED}
    assert "`total`" in explain(results[0], "en")


def test_common_errors_get_a_hint():
    from learn_llm.checking import explain

    def uses_wrong_name(celsius):
        return x * 9 / 5 + 32  # noqa: F821

    result = run(exercises("00")["to_fahrenheit"], uses_wrong_name)[0]
    assert "`def` line" in explain(result, "en")  # points at the input name


def test_concept_ledger_has_no_gaps():
    from learn_llm.ledger import problems

    assert problems() == []


def test_a_list_counts_as_a_pair():
    cases = exercises("03")["day_slopes"]

    def as_list(w, b, t, s):
        e = w * t + b - s
        return [2 * e * t, 2 * e]

    assert {r.status for r in run(cases, as_list)} == {"pass"}


def test_engine_mistakes_read_plainly():
    from learn_llm.checking import explain
    from learn_llm.checks.ch04 import Value

    def add_without_return(a, b, book):
        out = Value(a.data + b.data)
        out.parents, out.local_slopes = [a, b], [1, 1]
        book.append(out)

    def square_twice(a, book):
        out = Value(a.data * a.data)
        out.parents, out.local_slopes = [a, a], [2 * a.data]
        book.append(out)
        return out

    assert all(r.got is None for r in run(exercises("04")["add"], add_without_return))
    parents = run(exercises("04")["square"], square_twice)[2]
    assert "got `[a, a]`, expected `[a]`" in explain(parents, "en")

    def add_boxes_directly(a, b, book):
        return a + b

    error = run(exercises("04")["add"], add_boxes_directly)[0]
    assert "`add(x, y, book)`" in explain(error, "en")


def test_errors_keep_the_exercise_tip():
    from learn_llm.checking import explain

    def append_returns_nothing(rows, ws, b):
        guesses = []
        for row in rows:
            guesses = guesses.append(b)
        return guesses

    line = explain(run(exercises("05")["predict_all"], append_returns_nothing)[0], "en")
    assert "on its own line" in line  # the specific AttributeError hint
    assert "Tip:" in line  # and the exercise's own tip


@pytest.mark.parametrize("path", NOTEBOOKS, ids=ids(NOTEBOOKS))
def test_every_lesson_links_to_its_neighbours(path):
    assert f'nav("{path.name[:2]}", LOCALE)' in path.read_text(encoding="utf-8")


def test_chapter_links():
    from learn_llm.nav import links

    first, last = links(CHAPTERS[0], "en"), links(CHAPTERS[-1], "en")
    assert [href for _, href in first][0] == "?file=home.py"  # no previous chapter
    assert first[-1][1] == f"?file={chapters('en')[1].path.relative_to(LESSONS).as_posix()}"
    assert last[-1][1] == ""  # nothing after the last chapter yet
