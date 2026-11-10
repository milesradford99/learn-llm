import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="32 · One API for every model")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import ask, attempt, call, result, say, user

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 32 · One API for every model

    In chapters 30 and 31 the model was a function that gave back a finished message. A real one is neither finished nor uniform: replies arrive a few characters at a time, and every provider has its own format. Pi's `ai` package hides both, and it's one of the reasons the rest of Pi can stay small. It speaks the APIs of Anthropic, OpenAI, Google, Bedrock, Mistral and others from `packages/ai/src/api/`.

    ## Why stream?

    A long reply can take a minute. If you wait for all of it, the screen stays blank, and a dropped connection loses everything. So providers send a **stream** of small **events**. Pi turns each provider's stream into one shared list of events:

    <!-- source pi packages/ai/src/types.ts -->
    ```ts
    export type AssistantMessageEvent =
      | { type: "start"; partial: AssistantMessage }
      | { type: "text_start"; contentIndex: number; partial: AssistantMessage }
      | { type: "text_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
      | { type: "text_end"; contentIndex: number; content: string; partial: AssistantMessage }
    …
      | { type: "toolcall_start"; contentIndex: number; partial: AssistantMessage }
      | { type: "toolcall_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
      | { type: "toolcall_end"; contentIndex: number; toolCall: ToolCall; partial: AssistantMessage }
    ```

    Reading the type: an event is one of these shapes, told apart by `type`. A **delta** is a small piece of new content, and `contentIndex` says which block it belongs to, because a reply can hold text, then a tool call, then more text. Thinking blocks work the same way. `partial` is the message so far, so a listener doesn't have to rebuild it.

    In Python, we'll write an event as a dictionary, such as `{"type": "text_delta", "contentIndex": 0, "delta": "Hel"}`.

    ## Exercise 1: collect_text

    Do what a listener would. Write `collect_text(events)`, giving a dictionary from `contentIndex` to the text received so far. Start with `texts = {}`. Look only at `text_delta` events, set `index = event["contentIndex"]`, and use `.get` as in chapter 11:

    ```
    texts[index] = texts.get(index, '') + event["delta"]
    ```
    """)
    return


@app.function
def collect_text(events):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("32", "collect_text", collect_text, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A stream that never throws

    A stream ends in one of two ways: `{"type": "done", "message": ...}`, with the finished message, or `{"type": "error", "error": ...}`, with a message too. That second part is the important design choice. The rule is written at the top of the stream function's type:

    <!-- source pi packages/agent/src/types.ts -->
    ```ts
    * Contract:
    * - Must not throw or return a rejected promise for request/model/runtime failures.
    * - Must return an AssistantMessageEventStream.
    * - Failures must be encoded in the returned stream via protocol events and a
    *   final AssistantMessage with stopReason "error" or "aborted" and errorMessage.
    ```

    (A *promise* is TypeScript's word for a result that arrives later.) Now you can see why the loop in chapter 31 could simply test `stopReason`. A network failure isn't an exception that jumps out through the loop. It's an ordinary message, which can be stored, shown and retried. Rule one from chapter 30 applied to the model itself: **failures are data.**

    ## Exercise 2: final_message

    Write `final_message(events)`, looking only at the last event, `events[-1]` (the list is never empty): for `done` give its `"message"`, for `error` give its `"error"`, otherwise (the stream got cut off) give `None`.
    """)
    return


@app.function
def final_message(events):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("32", "final_message", final_message, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The same message, three ways

    Here's why you need a translator. The same exchange, in the format Pi stores and in two providers' formats:

    | | Pi | OpenAI chat | Anthropic |
    |---|---|---|---|
    | Assistant text | block `text` | field `content` | block `text` |
    | Tool call | block `toolCall` | list `tool_calls`, nested under `function`, arguments as **JSON text** | block `tool_use`, arguments as an object |
    | Tool result | role `toolResult` | role `tool` | a `tool_result` block inside a **`user`** message |

    The conversion lives in each provider's file. In `openai-completions.ts`, a result becomes a message with `role: "tool"`; in `anthropic-messages.ts`, a `tool_result` block. Pi stores its own format and converts at the last moment, so a session can switch model halfway through.

    ## Toolbox: JSON

    **JSON** is the standard text format for structured data, and it looks like a Python dictionary. `import json`, then `json.dumps(value)` makes the text and `json.loads(text)` reads it back.

    `''.join(pieces)` glues a list of strings together, and `isinstance(x, list)` tests whether `x` is a list.
    """)
    return


@app.cell
def _():
    import json as _json

    _text = _json.dumps({"x": 1})
    _text, _json.loads(_text)["x"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: to_openai

    Write `to_openai(message)`, converting one message. (The hidden setup doesn't import `json`, so start your function with `import json`.)

    1. `user`: `{"role": "user", "content": message["content"]}`
    2. `toolResult`: `{"role": "tool", "tool_call_id": message["toolCallId"], "content": message["content"]}`
    3. `assistant`: `{"role": "assistant", "content": ...}` where the content is the `block["text"]` of all its text blocks, joined
    4. If it has tool calls, also set `out["tool_calls"]` to a list of `{"id": ..., "type": "function", "function": {"name": ..., "arguments": json.dumps(...)}}`
    """)
    return


@app.function
def to_openai(message):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("32", "to_openai", to_openai, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: to_anthropic

    Anthropic is stricter: all results for one reply must share one `user` message. So this converter takes the whole list. Write `to_anthropic(messages)`, building a list `out`:

    1. `user`: append `{"role": "user", "content": message["content"]}`
    2. `assistant`: append `{"role": "assistant", "content": blocks}`, where each text block becomes `{"type": "text", "text": ...}` and each tool call becomes `{"type": "tool_use", "id": ..., "name": ..., "input": block["arguments"]}`
    3. `toolResult`: make `{"type": "tool_result", "tool_use_id": ..., "content": ..., "is_error": message["isError"]}`. If the last item of `out` is a `user` message whose content is a list, append the block to that list. Otherwise append a new `user` message holding `[block]`.
    """)
    return


@app.function
def to_anthropic(messages):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("32", "to_anthropic", to_anthropic, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Repairing the conversation

    Since a stored conversation can be replayed to any model, `packages/ai/src/api/transform-messages.ts` cleans it up before sending. It removes assistant messages that ended in `error` or `aborted`, because they may be half-finished, and it adds a result for any tool call that doesn't have one, so a strict provider won't refuse the request:

    <!-- source pi packages/ai/src/api/transform-messages.ts -->
    ```ts
    // Second pass: insert synthetic empty tool results for orphaned tool calls
    ```

    Remember this one: every harness has to keep calls and results paired, in some layer.

    ## Tools and prompts as part of the transcript

    Most frameworks keep "the system prompt" and "the tools" as settings outside the conversation. Current Pi doesn't. The prompt and the tool declarations are **system messages inside the transcript**, and the loop's comment states the contract: "`context.tools` is what the runtime can execute; the transcript's system messages declare what the model may call."

    When an extension adds a tool mid-session, Pi doesn't rewrite history. It appends a system message that says what changed, with `toolsAdded` and `toolsRemoved`. So a replay of the transcript always gives exactly what the model knew at each point, and the earlier text of the request stays the same, which keeps provider caching useful (chapter 41 explains that).

    The difference is computed like this. Note what happens to a tool whose declaration changed: it's listed as both removed and added.

    <!-- source pi packages/ai/src/utils/transcript.ts -->
    ```ts
    toolsAdded: current
      .filter((tool) => {
        const previousTool = previousTools.get(tool.name);
        return previousTool === undefined || !declarationsEqual(previousTool, tool);
      })
    ```

    ## Exercise 5: tool_changes

    Write `tool_changes(declared, current)`. Both are dictionaries from tool name to its declaration. Give back a tuple of two lists, `(added, removed)`, each in its dictionary's own order:

    - `added`: names in `current` that aren't in `declared`, or whose declaration differs
    - `removed`: names in `declared` that aren't in `current`, or whose declaration differs
    """)
    return


@app.function
def tool_changes(declared, current):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("32", "tool_changes", tool_changes, LOCALE)
    return


@app.cell
def _():
    attempt(tool_changes, {"read": "v1", "bash": "v1"}, {"read": "v2", "edit": "v1"})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When it passes, this shows `(['read', 'edit'], ['read', 'bash'])`. `read` changed, so it's in both lists.

    ## What you can do now

    - explain what a stream is, and read Pi's event list
    - explain why Pi's streams turn failures into messages, and what that does to the loop
    - convert one conversation to two providers' formats, and explain Anthropic's grouping rule
    - say what `transform-messages.ts` repairs, and why
    - explain how Pi puts tools and prompts inside the transcript, and how it announces a change

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** read `packages/ai/src/types.ts`, then one provider from end to end: `packages/ai/src/api/anthropic-messages.ts` is a good choice. `packages/ai/src/utils/transcript.ts` holds the transcript rules. Pi's `packages/coding-agent/docs/custom-provider.md` shows how to add your own provider.
    """)
    return


if __name__ == "__main__":
    app.run()
