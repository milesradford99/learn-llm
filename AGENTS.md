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
- Lessons using PyTorch call `laptop_speed()` in their setup (fewer threads, no denormals).
- A class used by other cells goes in a regular cell that returns it; `@app.class_definition`
  didn't reach cells when a class subclassed `nn.Module` and a lesson ran as a script.
- Math in lesson text and check messages is LaTeX: `$...$` inline, `> $$...$$` for a formula on its own line, `\text{...}` for words inside one. Backticks are only for Python a learner could type. Use a raw `mo.md(r"...")`; in a non-raw or f-string write `\\times`. Chart and diagram labels are SVG text, so they keep Unicode.
- Exercises (`docs/design/depth-plan.md`, enforced by `tests/test_depth.py`): give the name, inputs, shapes and one worked example, and formulas as math; don't print the solution's lines. New syntax goes in a Toolbox box on a different example. The chapter's main idea is done by hand first (`learn_llm.hand.boxes` / `marked`), and hints sit in `hand.stuck([...])` under the check. At most 12 new ideas a chapter; small tools listed under `toolbox` in the ledger don't count. `python -m learn_llm.depth` prints both numbers.
- Lesson text: no required external links. Links go after `<!-- go-deeper -->`.
- Lesson text is original. Never copy course notes, slides or assignment text.
- Chapters 40+ are the optional agents track (30–41 before the 2026-11-09 renumbering). They quote pinned source (`learn_llm/data/AGENT-SOURCES.md`); after changing an excerpt or re-pinning, run `python -m learn_llm.source_check --pi <clone> --dsh <clone>`. A cell's text can't contain `""`: marimo's formatter corrupts it, so write `''` and quote docstrings without their quote marks.
