# Novice-reader review (a gate for every chapter)

Every chapter is read by a fresh reviewer that has seen nothing but the chapter itself and
the list of ideas earlier chapters taught. It plays a specific beginner and hunts for the
places where a real beginner would get lost. A chapter isn't done while a *blocker* remains.

## Prompt

> You are reviewing one lesson of a course that promises a complete beginner can follow
> it with no outside help. Play this reader, strictly:
>
> {PERSONA}
>
> You know only what the reader knows plus the ideas listed under "Already taught".
> Read the lesson file at {PATH} top to bottom, as the reader would see it on the page
> (text cells render as formatted text; code cells show code and their result). Do every
> exercise in your head, as the reader would attempt it, including the likely wrong attempts.
>
> Report, in order of where they occur:
> 1. **Blockers:** a place where this reader cannot continue without help: a word, symbol,
>    idea or step used before it is explained; a leap between two steps; an exercise they
>    cannot do with what the page taught; an instruction that doesn't match what they'll see.
> 2. **Shaky spots:** the reader could continue, but would be guessing or memorising
>    rather than understanding. Say what they'd likely misunderstand.
> 3. **Wrong attempts:** for each exercise, the 3 most likely wrong answers this reader would
>    write, and whether the page's feedback would get them unstuck.
> 4. **Cut candidates:** anything the reader doesn't need here.
> 5. **Dictation:** an exercise this reader could pass by copying lines the page prints,
>    without understanding them. Say what the page should give instead (a worked example,
>    the shapes, the formula as math).
> 6. **Unread code:** a finished code cell of more than about 10 lines that the reader has no
>    reason to read: nothing to write in it, nothing to predict from it.
>
> Quote the exact text for each finding and propose a concrete fix. Don't praise; don't
> summarise the lesson. If you find no blockers, say so plainly.
>
> Already taught (from `python -m learn_llm.ledger NN`):
> {LEDGER}

## Personas

- **P1 The careful adult beginner.** 35, office job, high-school arithmetic long forgotten,
  has never written code or used a terminal. Reads every word, takes things literally,
  gets anxious when something on screen differs from the text.
- **P2 The impatient skimmer.** 19, uses apps all day, has never coded. Skips paragraphs,
  jumps to the exercise, tries things by trial and error, quits after two confusing failures.
- **P3 The Chinese-only reader** (for zh-CN). Reads only the 简体中文 version; English
  limited to code keywords. Also flags any sentence that reads as translated rather than
  written in natural Chinese, and any term that differs from `i18n/glossary.zh-CN.md`.
