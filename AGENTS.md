# AGENTS.md

Course repo: lessons are marimo notebooks (`lessons/<locale>/NN-slug.py`); shared code,
checks and UI strings live in `learn_llm/`. Scope, chapter map and decisions: `SCOPE.md`.

## Commands
- `make test`: everything CI runs except the smoke test. `make lint`: `marimo check`.
- `make check-solutions`: reference solutions in `solutions/chNN.py` against the checks.
- `make smoke`: starts the real server via `start.sh --smoke` and stops it.
- `uv run marimo edit lessons/en/00-first-program.py`: edit a lesson. This edits the
  source lesson, not a learner copy; learners get copies in `my-work/` (gitignored).

## Gotchas
- Exercise functions must be `@app.function` so `learn_llm.course.load` can import them
  without running cells. Stubs call `todo()`; a passing stub fails `test_stubs_are_not_started`.
- A marimo variable can be defined in only one cell. Use `_name` for cell-local names.
- Translations: code in every locale must be identical to `en` (checked via AST; comments
  are free). After editing an `en` lesson, its translations go stale; after updating a
  translation, run `uv run python -m learn_llm.i18n_stamp zh-CN/<file>`.
- New UI strings go in every `learn_llm/locales/*.json` with identical keys and `{fields}`.
- Page styling: `learn_llm/course.css`, loaded on every page via `custom_css` in
  `pyproject.toml` (not per lesson). marimo reads it at server start: restart to see edits.
  Visual rules (colours, type, radii) come from `DESIGN.md`; read it before changing UI.
- Lesson text: no required external links. Links go after `<!-- go-deeper -->`.
- Lesson text is original. Never copy course notes, slides or assignment text.
