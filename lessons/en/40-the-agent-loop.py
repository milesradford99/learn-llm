import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="40 · The agent loop")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import FakeModel, ask, call, result, say, trace, user, watch

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 40 · The agent loop

    This chapter starts an optional track that sits beside the main course. It's about **agents**, and it has one goal: when you finish, you'll understand two real open-source agents, **Pi** and **DeepSeek's Super-agent**, down to their source code, well enough to explain, change and defend their designs.

    You need Python up to chapter 19 (`while`, dictionaries, classes). You don't need a GPU, an API key or a network connection: a scripted stand-in plays the model, so every exercise runs here, for free, with the same result every time.

    Each chapter works the same way. We read the real code, quoted on the page, then you rebuild the mechanism in Python and the checks grade it. Rebuilding is what turns "I read it" into "I know why it's written that way."

    ## From a model to an agent

    Everything you trained in the main course does one thing: given some text, it produces more text. A chat model is a function:

    ```
    reply = model(messages)
    ```

    It can't open a file. It can't run code. It doesn't remember anything between calls, either: the only reason it seems to is that we send the whole conversation again each time.

    So how does a model "fix a bug"? The trick is small. We tell the model about some **tools**, and it may reply with a request instead of an answer: "please run `read` on `main.py`." The model doesn't run anything. *Our program* runs the tool, adds the outcome to the conversation, and calls the model again. Repeat until the model answers without asking for a tool.

    That's an **agent**: a model, some tools, and a loop. The program around the model, the part that runs the loop and everything it needs, is called a **harness**. Pi calls itself a "minimal, extensible agent harness", and DeepSeek's Super-agent is built as a bundle on a harness, DeepSeek Harness. By the end you'll see that the two are very different answers to the same question.

    ## The messages

    A conversation is a list of dictionaries, in the shape Pi uses. Each has a `role`, which is one of `"user"`, `"assistant"` or `"toolResult"`. Here's a short one. `trace` is a helper that shows each message on one line, and `user`, `ask`, `call`, `result` and `say` build the messages for us:
    """)
    return


@app.cell
def _():
    trace([
        user("What's 2 + 3?"),
        ask(call("add", {"x": 2, "y": 3}, id="call_1")),
        result("call_1", "add", "5"),
        say("It's 5."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's the second one in full. An assistant message holds a list of **blocks**, and a `stopReason` that says why the model stopped: `"toolUse"` means "I'm waiting for a tool", `"stop"` means "I'm finished".
    """)
    return


@app.cell
def _():
    ask(call("add", {"x": 2, "y": 3}, id="call_1"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice the `id`. A model may ask for several tools at once, and each result names the call it answers through `toolCallId`. Real model providers reject a conversation where a call has no matching result, which will cause us real trouble in later chapters.

    Here's the stand-in model. `FakeModel` plays back the replies you give it, and remembers what it was shown.
    """)
    return


@app.cell
def _():
    _model = FakeModel([ask(call("add", {"x": 2, "y": 3})), say("It's 5.")])
    _first = _model([user("What's 2 + 3?")])
    _first["stopReason"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: tool_calls

    Write `tool_calls(message)`, the list of `toolCall` blocks in an assistant message, in order. It's one list comprehension (chapter 12) with an `if` at the end:

    ```
    [block for block in message["content"] if block["type"] == "toolCall"]
    ```
    """)
    return


@app.function
def tool_calls(message):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("40", "tool_calls", tool_calls, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Toolbox: when things go wrong

    A tool can fail: the file doesn't exist, the command crashes. Python signals that by **raising an exception**, which normally stops the program with a red error box. To catch one:

    ```
    try:
        risky()
    except Exception as error:
        print(str(error))
    ```

    The lines under `try:` run as usual. If one raises, Python jumps to `except`, and `error` holds what happened. `str(error)` is its message. To raise one yourself, write `raise ValueError("what went wrong")`.

    ## Toolbox: text with values in it

    Put an `f` before the quotes, and anything in `{ }` is replaced by its value: `f"Tool {name} not found"` with `name = "read"` gives `Tool read not found`. Inside one, use the other kind of quote for a dictionary key: `f"{call['name']}"`.

    ## Rule one: a failure is information

    When a tool fails, an agent must **not** crash. It hands the failure to the model as a normal result, marked `isError`, because the model can often fix it: read the error, correct the path, try again. This is the biggest difference between agent code and ordinary code: an error is something to *show the model*.

    The same goes for a tool that doesn't exist. Models sometimes invent tool names. This is what Pi does:

    <!-- source pi packages/agent/src/agent-loop.ts -->
    ```ts
    const tool = tools.find((t) => t.name === toolCall.name);
    if (!tool) {
    …
        result: createErrorToolResult(`Tool ${toolCall.name} not found`),
        isError: true,
    ```

    That's TypeScript, which chapter 41 teaches properly. For now, read it as: if no tool has that name, make an error result. `…` marks lines we've cut.

    ## Exercise 2: run_tool

    `tools` is a dictionary from a tool's name to a function, and each function takes the arguments dictionary and gives back text. Write `run_tool(tools, call)`, which gives back a result message, always:

    1. `name = call["name"]`. If `name` isn't in `tools`, give back `result(call["id"], name, f"Tool {name} not found", True)`
    2. Otherwise, in a `try:`, set `text = tools[name](call["arguments"])`. `tools[name]` is a function, so the second pair of parentheses calls it
    3. If that raises, give back a result holding `str(error)`, with `True` at the end
    4. If it worked, give back `result(call["id"], name, text)`
    """)
    return


@app.function
def run_tool(tools, call):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("40", "run_tool", run_tool, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The loop

    Now the whole thing, in plain words:

    1. Show the model the conversation; add its reply to the conversation.
    2. If the reply has no tool calls, stop.
    3. Run every tool call; add each result to the conversation.
    4. Go to 1.

    A model can loop forever, so a harness also needs a limit. Each time round is a **turn**.

    ## Exercise 3: agent_loop

    Write `agent_loop(model, messages, tools, max_turns)`. It changes `messages` in place and gives it back.

    1. `turns = 0`, then `while turns < max_turns:`
    2. Inside: `reply = model(messages)`, then `messages.append(reply)`, then `turns = turns + 1`
    3. `calls = tool_calls(reply)`; if `len(calls) == 0`, `return messages`
    4. `for call in calls:` append `run_tool(tools, call)`
    5. After the loop, `return messages`
    """)
    return


@app.function
def agent_loop(model, messages, tools, max_turns):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("40", "agent_loop", agent_loop, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here it is at work. `read_file` below is a tool that only knows one file, and `watch` runs your loop and shows the trace. (Until exercise 3 passes, the cell just says so.) The model fails, reads the error, and tries again:
    """)
    return


@app.function
def read_file(arguments):
    if arguments["path"] != "notes.txt":
        raise ValueError(f"No such file: {arguments['path']}")
    return "buy milk"


@app.cell
def _():
    _model = FakeModel([
        ask(call("read", {"path": "note.txt"})),
        ask(call("read", {"path": "notes.txt"})),
        say("Your note says: buy milk."),
    ])
    watch(agent_loop, _model, [user("What's in my notes?")], {"read": read_file}, 10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You've just built an agent, and nothing about it is magic. Everything else in this track is what happens when you take this loop seriously.

    ## Why the model's input keeps growing

    Because the model remembers nothing, each turn sends the whole list again. If each turn adds about the same amount of text, the *total* you send grows with the square of the number of turns: turn 1 sends 1 part, turn 2 sends 2, and turn 50 sends 50, so a 50-turn run sends 1,275 parts. Providers charge for all of it, and a model can only read so much at once (its **context window**).
    """)
    return


@app.cell
def _():
    _turns = 50
    sum(range(1, _turns + 1))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What's left to solve

    Your 12-line loop works for a demo and would fall over in real use. Here are the problems, and where each one gets solved. Pi is the depth-first study of one loop; Super-agent is the study of many loops working for a whole team.

    | Problem | Chapter | Project |
    |---|---|---|
    | The user wants to interrupt; a reply gets cut off | 31 | Pi |
    | Dozens of model providers, each with its own format; replies arrive piece by piece | 32 | Pi |
    | A tool returns a million lines; two tools edit one file | 33 | Pi |
    | A crash loses the conversation; you want to try another path | 34 | Pi |
    | The conversation outgrows the context window | 35 | Pi |
    | Everyone wants different behaviour | 36 | Pi |
    | One person can only watch a few agents at once | 37 | Super-agent |
    | Anyone in a chat can ask; who may start work? | 38 | Super-agent |
    | Later messages must reach the right work, without re-asking | 39 | Super-agent |
    | Many workers, restarts, nothing lost | 40 | Super-agent |
    | Prompts, context and talking to people | 41 | Super-agent |

    ## What you can do now

    - say what an agent, a tool call and a harness are
    - read the message shapes, and explain what ids are for
    - explain why a tool failure goes back to the model instead of stopping the program
    - write an agent loop with a turn limit, from memory
    - explain why the cost of a long run grows faster than its length

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** the code in this track is pinned to Pi at commit `6291b9b` (2026-11-01) and DeepSeek's `dsh-superagent-prototype` at `8c32a9d` (2026-10-31). Both change daily, so a newer checkout may differ: see [badlogic/pi-mono](https://github.com/badlogic/pi-mono) and [deepseek-ai/dsh-superagent-prototype](https://github.com/deepseek-ai/dsh-superagent-prototype). Excerpts are quoted under their MIT licences; `learn_llm/data/AGENT-SOURCES.md` has the notices.
    """)
    return


if __name__ == "__main__":
    app.run()
