import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="43 · Tools that survive the real world")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import attempt, call, result

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 43 · Tools that survive the real world

    Pi's whole default toolset is `read`, `bash`, `edit` and `write`, with `grep`, `find`, `ls` and `powershell` available when you turn them on. They live in `packages/coding-agent/src/core/tools/`. (`bash` runs a shell command, the kind you'd type in a terminal.) Four tools is a statement: `bash` can do almost anything, so most special-purpose tools only add things for the model to read.

    The tools are short. What's worth studying is how each one meets a specific way that an agent goes wrong.

    ## Too much output

    The command `cat`, which prints a file, could return a million lines from a large log. That output would go into the conversation and be sent again on *every later turn* (chapter 40), until the model can't read any more. So nothing reaches the model untrimmed:

    <!-- source pi packages/coding-agent/src/core/tools/truncate.ts -->
    ```ts
     * Truncation is based on two independent limits - whichever is hit first wins:
     * - Line limit (default: 2000 lines)
     * - Byte limit (default: 50KB)
     *
     * Never returns partial lines (except bash tail truncation edge case).
    …
    export const DEFAULT_MAX_LINES = 2000;
    export const DEFAULT_MAX_BYTES = 50 * 1024; // 50KB
    ```

    Three details matter. The limit is in bytes as well as lines, because one line of minified code can be 10 MB. Whole lines only, because half a line misleads. And the direction depends on the tool: `read` keeps the **head** of a file and tells the model the next `offset` to ask for, while `bash` keeps the **tail**, because the end of a command's output is where the error is, and saves the full text in a temporary file so the model can look there if it needs to.

    The model has to be told, too. A silent cut looks like a short file. The tool's text says what was cut and how to get the rest.

    Recall from chapter 19: `text.encode("utf-8")` gives the bytes, and a Chinese character takes 3. `"\n"` is the newline character, which is 1 byte. `text.split("\n")` cuts text into lines, and `"\n".join(lines)` puts them back.

    ## Exercise 1: truncate_head

    Write `truncate_head(text, max_lines, max_bytes)`, giving a tuple: the text kept, and whether anything was cut.

    1. `kept = []` and `size = 0`
    2. For each line of `text.split("\n")`: `cost` is its length in bytes, plus 1 for the newline if `kept` isn't empty
    3. If `len(kept) == max_lines` or `size + cost > max_bytes`, return `"\n".join(kept), True`
    4. Otherwise add it to `kept`, and add `cost` to `size`
    5. After the loop, return `text, False`
    """)
    return


@app.function
def truncate_head(text, max_lines, max_bytes):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("43", "truncate_head", truncate_head, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Editing without destroying

    The obvious way to edit a file is to have the model send the whole new file. That's slow, expensive, and a chance to quietly lose lines it didn't feel like repeating. Line numbers are no better: they go stale when an earlier edit shifts the file.

    Pi's `edit` takes pairs of `oldText` and `newText`, and the rule is simple: **the old text must appear exactly once.** Zero matches means the model's picture of the file is wrong; two means the edit is ambiguous. Both become error results, so the model reads the file and tries again (rule one). The messages say what to do next:

    <!-- source pi packages/coding-agent/src/core/tools/edit-diff.ts -->
    ```ts
    `Found ${occurrences} occurrences of the text in ${path}. The text must be unique. Please provide more context to make it unique.`,
    ```

    Models also get invisible details wrong: a trailing space, a curly quote, a special dash. So the match is **fuzzy** in a narrow way: try exactly, and if that fails, compare after normalising. The first step of `normalizeForFuzzyMatch` strips trailing whitespace from each line:

    <!-- source pi packages/coding-agent/src/core/tools/edit-diff.ts -->
    ```ts
    export function normalizeForFuzzyMatch(text: string): string {
    …
          // Strip trailing whitespace per line
          .split("\n")
          .map((line) => line.trimEnd())
          .join("\n")
    ```

    Those three lines chain onto `text`: split it, run a function on each line (`.map` is a list comprehension in another form), and join it back.

    One detail shows how much care this takes. When the fuzzy match is the one that succeeds, the replacement has to happen in the normalised text. Done naively, that would strip trailing spaces from the whole file, and a one-line edit would show up as a huge *diff* (the list of changed lines). So `applyReplacementsPreservingUnchangedLines` rewrites only the lines the edit touched, and copies every other line back from the original. We've written that part for you below, as `rewrite_touched_lines`.

    ## Toolbox: raising, and text tools

    `raise ValueError("message")` raises an exception yourself. In chapter 40, `run_tool` turned exceptions into error results, so a tool can just raise.

    `line.rstrip()` removes spaces and tabs from the end of a line. `text.count(part)` counts how often `part` appears, `text.replace(a, b)` replaces it, and `a in text` tests whether `a` appears at all.

    ## Exercise 2: fuzzy

    Write `fuzzy(text)`, which strips trailing whitespace from every line and keeps everything else, indentation included.
    """)
    return


@app.function
def fuzzy(text):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("43", "fuzzy", fuzzy, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's `rewrite_touched_lines`, ready to call:
    """)
    return


@app.function
def rewrite_touched_lines(content, old, new):
    base = fuzzy(content)
    first = base[:base.index(fuzzy(old))].count("\n")
    last = first + fuzzy(old).count("\n")
    rewritten = base.replace(fuzzy(old), new).split("\n")
    lines = content.split("\n")
    middle = rewritten[first:len(rewritten) - (len(lines) - last - 1)]
    return "\n".join(lines[:first] + middle + lines[last + 1:])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: apply_edit

    Write `apply_edit(content, old, new)`, giving back the new file text:

    1. If `old == ''`: `raise ValueError("oldText must not be empty.")`
    2. `matches = fuzzy(content).count(fuzzy(old))`
    3. If 0: `raise ValueError("Could not find the text. It must match exactly.")`
    4. If more than 1: `raise ValueError(f"Found {matches} occurrences of the text. It must be unique.")`
    5. If `old in content`, return `content.replace(old, new)`
    6. Otherwise return `rewrite_touched_lines(content, old, new)`
    """)
    return


@app.function
def apply_edit(content, old, new):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("43", "apply_edit", apply_edit, LOCALE)
    return


@app.cell
def _():
    # The edit touches lines 1 and 2, and line 3 keeps its trailing spaces.
    attempt(apply_edit, "a  \nb\nc  \n", "a\nb", "A\nB")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When it passes, this shows `'A\nB\nc  \n'`.

    ## Many tools at once

    A model may ask for five `read`s in one reply. Running them one by one wastes time, so by default Pi runs them **in parallel**, in three stages:

    1. **Prepare, one at a time.** Find the tool, validate the arguments, run `beforeToolCall`. This is sequential on purpose: hooks may ask the user a question, or touch shared state.
    2. **Execute, together.**
    3. **Report, in the original order.** The model must get results in the order it asked, whatever order they finished in. `Promise.all` gives exactly that: it waits for all of them, and keeps the list's order. (In the code below, an entry is either a finished result or a function still to run, which is what the `typeof` test sorts out.)

    <!-- source pi packages/agent/src/agent-loop.ts -->
    ```ts
    const orderedFinalizedCalls = await Promise.all(
      finalizedCalls.map((entry) => (typeof entry === "function" ? entry() : Promise.resolve(entry))),
    );
    const messages: ToolResultMessage[] = [];
    for (const finalized of orderedFinalizedCalls) {
      const toolResultMessage = createToolResultMessage(finalized);
      await emitToolResultMessage(toolResultMessage, emit);
      messages.push(toolResultMessage);
    }
    ```

    A tool can declare `executionMode: "sequential"`, and one such tool makes the whole batch sequential.

    ## Exercise 4: in_source_order

    Write `in_source_order(calls, finished)`. `calls` are the tool calls as the model asked for them, and `finished` the result messages in the order they completed. Give back the results in `calls` order: build a dictionary from each result's `toolCallId` to the result, then use a list comprehension over `calls`.
    """)
    return


@app.function
def in_source_order(calls, finished):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("43", "in_source_order", in_source_order, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Two edits, one file

    Parallel raises a new danger. Two `edit`s to one file would each read it, change it and write it back, and the second write would erase the first. Pi's answer is small: a queue per file, keyed by its real path, so that symlinks (files that point at another file) to one file share a queue. In the signature below, `<T>` just means "any type".

    <!-- source pi packages/coding-agent/src/core/tools/file-mutation-queue.ts -->
    ```ts
     * Serialize file mutation operations targeting the same file.
     * Operations for different files still run in parallel.
     */
    export async function withFileMutationQueue<T>(filePath: string, fn: () => Promise<T>): Promise<T> {
    ```

    Each new operation waits for the previous one on that file, and different files don't wait for each other. It's the smallest thing that's correct, which is a good description of Pi in general.

    ## What Pi leaves out

    Just as worth knowing: `bash` has no permission prompts and no sandbox. Pi's security guide states it directly: it can read, change and run things with the permissions of the account that started it, and watching the transcript isn't a security boundary. If you want isolation, you run all of Pi inside a container. Chapter 46 returns to this, and chapter 51 shows what Super-agent says about it.

    ## What you can do now

    - explain why output must be trimmed, why by bytes too, and why `read` and `bash` cut opposite ends
    - explain why the edit tool uses unique text rather than line numbers or whole files
    - describe the fuzzy match, and why it rewrites only the lines it touches
    - describe the three stages of parallel tool calls, and why results keep their order
    - explain the file mutation queue, and why it's keyed by real path

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** read `tools/edit.ts` and `tools/edit-diff.ts` together, then `tools/output-accumulator.ts` for how `bash` output is held and spilled to a file while a command runs. To try it for real, install Pi from [pi.dev](https://pi.dev) and run `pi --help`.
    """)
    return


if __name__ == "__main__":
    app.run()
