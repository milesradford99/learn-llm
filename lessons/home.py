import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="learn-llm")

with app.setup(hide_code=True):
    import json

    import marimo as mo

    from learn_llm.course import DONE, IN_PROGRESS, WORK, chapters, progress
    from learn_llm.i18n import LOCALE_NAMES, SOURCE_LOCALE, t

    SETTINGS = WORK / ".settings.json"


@app.cell(hide_code=True)
def _():
    try:
        saved = json.loads(SETTINGS.read_text(encoding="utf-8")).get("locale", SOURCE_LOCALE)
    except (OSError, ValueError):
        saved = SOURCE_LOCALE
    language = mo.ui.dropdown(
        options={name: code for code, name in LOCALE_NAMES.items()},
        value=LOCALE_NAMES.get(saved, LOCALE_NAMES[SOURCE_LOCALE]),
        label="🌐",
    )
    return (language,)


@app.cell(hide_code=True)
def _(language):
    _locale = language.value
    SETTINGS.write_text(json.dumps({"locale": _locale}), encoding="utf-8")

    _icons = {DONE: "✅", IN_PROGRESS: "🟡"}
    _rows = []
    for _ch in chapters(_locale, WORK):
        _status = progress(_ch.number, _ch.path)
        _link = f"[{_ch.title}](?file={_ch.path.relative_to(WORK).as_posix()})"
        _note = f" {t('home.fallback', _locale)}" if _ch.fallback else ""
        _rows.append(f"| {_icons.get(_status, '⬜')} | {_link}{_note} | {t('home.' + _status, _locale)} |")

    mo.vstack([
        mo.hstack([mo.md(f"# {t('home.title', _locale)}"), language], justify="space-between"),
        mo.md(f"## {t('home.chapters', _locale)}\n\n| | | |\n|---|:---|---|\n" + "\n".join(_rows)),
    ])
    return


if __name__ == "__main__":
    app.run()
