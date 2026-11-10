import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="35 · Running out of room")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import ask, attempt, call, result, say, user

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 35 · Running out of room

    Every long agent run ends up with the same problem: the conversation grows until it no longer fits in the model's context window. You can't just drop the old messages, because they hold what the agent has learned and decided.

    **Compaction** replaces the old part of the conversation with a summary written by the model. The idea is one sentence, and every line of the hard part is about which messages can't be separated. This is a study of `packages/coding-agent/src/core/compaction/compaction.ts`, and the file's first comment gives away its design: "Pure functions for compaction logic. The session manager handles I/O".

    ## When: counting tokens cheaply

    You need a size for each message, and a real tokenizer (chapter 17) differs for every model. So Pi uses the rule of thumb that a token is about 4 characters:

    <!-- source pi packages/coding-agent/src/core/compaction/compaction.ts -->
    ```ts
      if (block.type === "text") {
        chars += block.text.length;
    …
      } else if (block.type === "toolCall") {
        chars += block.name.length + JSON.stringify(block.arguments).length;
      }
    }
    return Math.ceil(chars / 4);
    ```

    Where it can, Pi prefers the real count: each assistant reply reports the `usage` of its request, so the true size of the context up to that reply is known, and only the messages after it need estimating. (That's `estimateContextTokens`.)

    ## Toolbox: rounding up

    `math.ceil(x)` rounds up (`import math`): `math.ceil(1.25)` is 2. `json.dumps(x, separators=(",", ":"))` writes JSON with no spaces, matching JavaScript's `JSON.stringify`.

    ## Exercise 1: estimate_tokens

    Write `estimate_tokens(message)`. Put `import json` and `import math` as the first lines inside the function. Count the characters, then give back `math.ceil(chars / 4)`.

    - If `message["role"] == "assistant"`, go through `message["content"]` and add up: `len(block["text"])` when `block["type"] == "text"`, and otherwise (a tool call) `len(block["name"]) + len(json.dumps(block["arguments"], separators=(",", ":")))`
    - For anything else, `len(message["content"])`, since user and tool messages hold plain text here
    """)
    return


@app.function
def estimate_tokens(message):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("35", "estimate_tokens", estimate_tokens, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The decision itself is one line. `reserveTokens` holds back room for the model's next reply and defaults to 16,384, so compaction starts before the window is completely full:

    <!-- source pi packages/coding-agent/src/core/compaction/compaction.ts -->
    ```ts
    export function shouldCompact(contextTokens: number, contextWindow: number, settings: CompactionSettings): boolean {
      if (!settings.enabled) return false;
      return contextTokens > contextWindow - settings.reserveTokens;
    }
    ```

    ## Exercise 2: should_compact

    Write `should_compact(context_tokens, context_window, reserve)`. Notice that it's a strict `>`.
    """)
    return


@app.function
def should_compact(context_tokens, context_window, reserve):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("35", "should_compact", should_compact, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where to cut

    Keep the recent part (`keepRecentTokens`, 20,000 by default) and summarise the rest. To find the line, walk backwards from the newest message adding up sizes, and stop at the budget.

    But you can't cut just anywhere. A tool result must follow its tool call, so **a cut can never land on a tool result**: if it did, the kept part would begin with an answer to a question that's gone, and a provider would reject it (chapter 32).

    <!-- source pi packages/coding-agent/src/core/compaction/compaction.ts -->
    ```ts
    // Walk backwards from newest, accumulating estimated message sizes
    let accumulatedTokens = 0;
    let cutIndex = cutPoints[0]; // Default: keep from first message (not header)

    for (let i = endIndex - 1; i >= startIndex; i--) {
    …
      accumulatedTokens += messageTokens;
    …
      if (accumulatedTokens >= keepRecentTokens) {
        // Prefer the closest valid cut point at or after this entry. If trailing
        // tool results exceed the budget by themselves, keep their preceding
        // assistant tool call instead of falling back to the first message.
        cutIndex = cutPoints.find((candidate) => candidate >= i) ?? cutPoints[cutPoints.length - 1];
        break;
      }
    }
    ```

    (`for (let i = endIndex - 1; i >= startIndex; i--)` counts `i` down from the last index to the first.)

    That comment records a bug that was fixed. If one huge tool result fills the budget on its own, no valid cut point exists at or after it, and the naive fallback would be the very first message, meaning no compaction at all. So the fallback is the *last* valid cut point instead.

    ## Exercise 3: find_cut_point

    Write `find_cut_point(messages, keep)`, giving the index of the first message to keep.

    1. `cuts` = every index `i` whose `messages[i]["role"]` isn't `"toolResult"`
    2. `total = 0`, then for `i` going backwards through the indices (`reversed(range(len(messages)))`), add `estimate_tokens(messages[i])`
    3. If `total >= keep` (`>=`, not `>`): `later` = the cuts that are `>= i`. If there are any, return the first; otherwise return `cuts[-1]`
    4. After the loop (the budget was never reached), return `cuts[0]`
    """)
    return


@app.function
def find_cut_point(messages, keep):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("35", "find_cut_point", find_cut_point, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A summary that can be worked from

    An ordinary summary won't do. The next model has to carry on the work, so Pi asks for a fixed form, with sections for the goal, constraints, progress (done, in progress, blocked), key decisions, next steps and critical context:

    <!-- source pi packages/coding-agent/src/core/compaction/compaction.ts -->
    ```ts
    const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.
    …
    Keep each section concise. Preserve exact file paths, function names, and error messages.`;
    ```

    Some more choices in the same file:

    - **Summaries build on each other.** With an earlier compaction, Pi doesn't summarise the summary from scratch. It sends the previous one and asks for an update: "PRESERVE all existing information".
    - **File lists are tracked by code.** Read and modified files are pulled out of the tool calls, carried across compactions in `details`, and appended exactly, so they can't be lost to a model's paraphrase.
    - **Cutting inside a turn.** One user request may have produced a hundred tool calls, so the cut can land in the middle of a turn. `isSplitTurn` flags that, and Pi summarises the cut-off prefix of that turn separately.
    - **A cut-off summary is a failure.** `getSummarizationFailure` refuses a reply with a length stop: half a checkpoint must never become the checkpoint.
    - **Nothing is deleted.** It's chapter 34's log. A `compaction` entry records the summary and `firstKeptEntryId`; the projection starts from there, and the original messages are still in the file.

    ## When things fail

    Models fail in two different ways, and the harness layer treats them differently (`agent-session.ts`). This is what the loop's clean exit in chapter 31 was for.

    **Passing errors** (rate limits, overloaded servers) get retried, by default up to 3 times, waiting longer each time from a 2-second start, up to a cap:

    <!-- source pi packages/ai/src/utils/retry.ts -->
    ```ts
    const delay = policy.baseDelayMs * 2 ** Math.max(0, attempt - 1);
    ```

    **Context overflow** will fail again if it's retried as it is, so `_isRetryableError` excludes it, with the comment "Context overflow is handled by compaction, not retry." Pi compacts, then retries, **once**. If it overflows again, Pi stops, since another try could loop forever: "Context overflow recovery failed after one compact-and-retry attempt."

    A failed attempt stays in the log, and is left out of what the model sees (`_omitRecoveryAttempt`), the same log-versus-projection split as before.

    ## Exercise 4: retry_delay

    Write `retry_delay(attempt, base, cap)`, using `min`, `max` and `**` (chapter 00) as in the line above, with the result capped at `cap`.
    """)
    return


@app.function
def retry_delay(attempt, base, cap):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("35", "retry_delay", retry_delay, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 5: next_step

    Write `next_step(message, overflowed, compacted_already, retries, max_retries)`, giving one of `"compact"`, `"retry"`, `"give up"` or `"done"`:

    1. If `overflowed`: `"give up"` if `compacted_already`, otherwise `"compact"`
    2. If the message's `stopReason` is `"error"`: `"retry"` if `retries < max_retries`, otherwise `"give up"`
    3. Otherwise `"done"`
    """)
    return


@app.function
def next_step(message, overflowed, compacted_already, retries, max_retries):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("35", "next_step", next_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where Pi doesn't use `transformContext`

    You might expect compaction to live in the loop's `transformContext` hook. It doesn't. Compaction runs in the session layer after a run ends or between turns, and the session is reloaded. So a compaction is a durable event in the log, not a per-request illusion, and the file on disk always agrees with what the model saw. The `prepareNextTurn` hook from chapter 31 exists partly for this. Its comment gives compaction as an example of long-running preparation.

    Keep that design in mind: in chapter 40, Super-agent also stores what it will do before doing it.

    ## What you can do now

    - estimate tokens the way Pi does, and say when it uses real counts instead
    - explain the reserve and the keep budget
    - find a valid cut point, and explain the trailing-result bug
    - list what makes Pi's summaries good to work from, and why file lists bypass the model
    - explain why overflow compacts once rather than retrying, and why compaction isn't a per-request hook

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** `packages/coding-agent/docs/compaction.md` is an excellent reference. Then read `_checkCompaction` in `agent-session.ts`, including what counts as a recoverable length stop.
    """)
    return


if __name__ == "__main__":
    app.run()
