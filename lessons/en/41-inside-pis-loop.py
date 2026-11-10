import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="41 · Inside Pi's loop")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import FakeModel, ask, call, result, say, user, watch

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 41 · Inside Pi's loop

    Chapter 40's loop was 12 lines. Pi's is about 170, in `packages/agent/src/agent-loop.ts`, and its core is the shape you already know. In this chapter you'll learn to read it, and find out what the extra lines are for.

    First, a map. Pi is a **monorepo**: one repository holding several packages, which build on each other.

    | Package | What it is |
    |---|---|
    | `ai` | One way to talk to every model provider (chapter 42) |
    | `agent` | The loop, in one file under 1,000 lines. Knows nothing about files or terminals |
    | `coding-agent` | The `pi` command: tools, sessions, compaction, extensions (chapters 43 to 46) |
    | `tui` | The terminal interface library |
    | `mcp`, `codemode`, `durable`, `chord`, `schema`, `env`, `evals`, `telemetry` | Newer building blocks (chapter 46) |

    That split is the first thing to remember about Pi: `agent` is a small library with no opinions, and everything that makes it a coding tool is plugged in from outside.

    ## Toolbox: reading TypeScript

    Pi is written in **TypeScript**, which is JavaScript with types. You only need to read it. Almost all of it maps directly onto Python:

    | TypeScript | Python | Meaning |
    |---|---|---|
    | `const x = 1`, `let y = 2` | `x = 1` | `const` never changes; `let` can |
    | `x: string` | | a type label; it doesn't change what runs |
    | `(a, b) => a + b` | `lambda a, b: a + b` | a small function |
    | `x?.y` | | `x.y`, or nothing if `x` is missing |
    | `a ?? b` | | `a`, unless it's missing; then `b` |
    | `{ ...a, b: 1 }` | | a copy of dictionary `a` with one field changed |
    | `[...a, b]` | | a copy of list `a` with one more item |
    | `a \|\| b` | `a or b` | `a`, or `b` if `a` is empty |
    | `list.push(x)` | `list.append(x)` | add to the end |
    | `list.filter((c) => test)` | `[c for c in list if test]` | keep some items |
    | `` `Tool ${name}` `` | `f"Tool {name}"` | text with a value in it |
    | `===` | `==` | equal |
    | `async`, `await` | | "this may wait for something; let other work go on meanwhile" |

    The last row matters: a model call takes seconds, and `await` waits for one without freezing everything else. Read `await f()` as just `f()`.

    The table's `lambda` is Python's way to write a small nameless function: `lambda a, b: a + b` is a `def` with one `return`. And in Python, `x or y` holds if either test does, like `||` in TypeScript.

    ## The two loops

    Here's the heart of Pi's `runLoop`, with the details cut (`…`). Read it slowly. You'll recognise chapter 40's loop as the *inner* `while`.

    <!-- source pi packages/agent/src/agent-loop.ts -->
    ```ts
    let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];
    …
    while (true) {
      let hasMoreToolCalls = true;
    …
      while (hasMoreToolCalls || pendingMessages.length > 0) {
    …
        const message = await streamAssistantResponse(currentContext, config, signal, emit, streamFunction);
        newMessages.push(message);
    …
        if (message.stopReason === "error" || message.stopReason === "aborted") {
    …
          await emit({ type: "agent_end", messages: newMessages });
          return;
        }
    …
        const toolCalls = message.content.filter((c) => c.type === "toolCall");
    …
        hasMoreToolCalls = false;
        if (toolCalls.length > 0) {
    …
        pendingMessages = (await config.getSteeringMessages?.()) || [];
    …
        }
    …
      const followUpMessages = (await config.getFollowUpMessages?.()) || [];
      if (followUpMessages.length > 0) {
    …
        pendingMessages = followUpMessages;
        continue;
      }
    …
      break;
    }
    ```

    Three reading notes. `(await config.getSteeringMessages?.()) || []` means "call this hook if there is one, and if that gives nothing, use an empty list". `hasMoreToolCalls` is set back to true inside the `if`, in a line we cut. And `continue` and `break` work as they do in Python.

    Why two loops? Because of a problem chapter 40 ignored: **the user can type while the agent works**, and two different things could be meant.

    - **Steering:** "stop, wrong file!" This should land as soon as it safely can.
    - **Follow-up:** "and when you're done, run the tests." This should wait until the agent has finished.

    So the inner loop runs while there are tool calls *or* steering messages, and the outer loop picks up follow-ups when the inner one runs dry.

    ## When is "as soon as safely possible"?

    Pi drains steering **after the whole reply's tools have run**, never between them. If a reply asks for `read` and `edit` and you steer during `read`, `edit` still runs. Why not cancel it? Because every tool call must get a result (chapter 40). Pi's own comments on the hook put it plainly: messages are added before the next model call, and tool calls from the current assistant message are not skipped.

    Queues drain one message at a time by default. If you type three messages, the model sees them one per turn, so it can react to each. That's what `take_one` below does.

    ## Toolbox: list.pop

    `queue.pop(0)` removes the first item of a list and gives it to you. `messages[-1]` (with no pop) reads the last item and leaves it there.

    ## Exercise 1: take_one

    Write `take_one(queue)`: if the queue is empty, give back `[]`; otherwise a one-item list holding the oldest entry, removed from the queue. The list form lets the loop treat "nothing" and "something" the same way.
    """)
    return


@app.function
def take_one(queue):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "take_one", take_one, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exit rules

    Three more details from the source, each one the scar of a real failure.

    **Errors end the run at once.** If the reply has `stopReason` `"error"` or `"aborted"`, the loop returns without looking at the queues. The harness above it (chapter 45) decides whether to retry.

    **You can't continue from an assistant message.** Pi can pick up a stopped run with `agentLoopContinue`, but only if the last message is one the model can answer:

    <!-- source pi packages/agent/src/agent-loop.ts -->
    ```ts
    if (context.messages[context.messages.length - 1].role === "assistant") {
      throw new Error("Cannot continue from message role: assistant");
    }
    ```

    ## Exercise 2: can_continue

    Write `can_continue(messages)`: `False` for an empty list, `False` if the last message's role is `"assistant"`, otherwise `True`.
    """)
    return


@app.function
def can_continue(messages):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "can_continue", can_continue, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Cut-off replies can lie.** A reply can hit the output limit partway through. Its `stopReason` is `"length"`, and a tool call in it may be incomplete. The danger is subtle. Pi repairs broken JSON from a stream as best it can, so a cut-off `write` call may parse and validate perfectly while holding half of a file. Running it would silently destroy data. So Pi runs none of that reply's tools:

    <!-- source pi packages/agent/src/agent-loop.ts -->
    ```ts
    // A "length" stop means the output was cut off by the token limit, so
    // every tool call in the message may carry truncated arguments. Fail
    // them all instead of executing potentially borked calls.
    const executedToolBatch =
      message.stopReason === "length"
        ? await failToolCallsFromTruncatedMessage(toolCalls, emit)
        : await executeToolCalls(currentContext, message, config, signal, emit);
    ```

    ## Exercise 3: truncated_result

    Write `truncated_result(call)`, the error result for one such call. Give back `result(call["id"], call["name"], text, True)`, with this `text` (a shorter form of Pi's; `\"` puts a quote mark inside a string):

    ```
    f"Tool call \"{call['name']}\" was not executed: the response hit the output token limit."
    ```
    """)
    return


@app.function
def truncated_result(call):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "truncated_result", truncated_result, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: run_loop

    Now put it together. Write `run_loop(model, messages, run_call, steering, follow_ups)`. `run_call(call)` gives back a result message; `steering` and `follow_ups` are lists used as queues; it changes `messages` and gives it back.

    1. `pending = take_one(steering)`, then `while True:`
    2. Inside: `more_calls = True`, then `while more_calls or len(pending) > 0:`
    3. Inside that: append everything in `pending` to `messages`, then set `pending = []`
    4. `reply = model(messages)`; append it
    5. If `reply["stopReason"]` is `"error"` or `"aborted"`, `return messages`
    6. `calls = [block for block in reply["content"] if block["type"] == "toolCall"]`, as in chapter 40; `more_calls = len(calls) > 0`
    7. For each call: if the stop reason is `"length"`, append `truncated_result(call)`, else append `run_call(call)`
    8. `pending = take_one(steering)` (the end of the inner loop)
    9. Back in the outer loop: `extra = take_one(follow_ups)`. If it's empty, `return messages`; otherwise `pending = extra`
    """)
    return


@app.function
def run_loop(model, messages, run_call, steering, follow_ups):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "run_loop", run_loop, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's a run with one follow-up waiting, and one steering message arriving while the first tool runs. Once `run_loop` passes, you should see the steering message right after the tool's result, and the follow-up only after the model's reply to it:
    """)
    return


@app.function
def demo_steering():
    steering = []

    def run_call(call):
        steering.append(user("Actually, skip the tests."))
        return result(call["id"], call["name"], "42 files")

    model = FakeModel([ask(call("ls")), say("Skipping them."), say("Committed.")])
    return run_loop(model, [user("Run the tests")], run_call, steering, [user("Then commit")])


@app.cell
def _():
    watch(demo_steering)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The hooks: how a small loop stays small

    What about permissions, context trimming, cost limits? None of that lives in the loop. Instead, `AgentLoopConfig` offers places to plug code in:

    | Hook | When it runs | Used for |
    |---|---|---|
    | `transformContext` | before each model call | trimming or adding context |
    | `convertToLlm` | before each model call | turning app-only messages into what a model understands |
    | `prepareRequest` | before each model call | swapping the model or context for this request |
    | `beforeToolCall` | after arguments validate, before the tool runs | blocking a call |
    | `afterToolCall` | after the tool runs | rewriting its result |
    | `finishTurn`, `prepareNextTurn` | around the end of a turn | ending early; compaction between turns |
    | `getSteeringMessages`, `getFollowUpMessages` | at the two drain points | the queues you just used |

    Notice `convertToLlm`. The loop works on `AgentMessage`, which is wider than what a model understands: an app can keep its own kinds of message in the transcript (a UI notification, a shell command the user ran), and they're only turned into model messages at the last moment. It's a small line with a large effect: the transcript belongs to the app, not to the provider.

    The loop also **emits events** as it goes: `agent_start`, `turn_start`, `message_start`, `message_update`, `message_end`, `tool_execution_start`, `tool_execution_update`, `tool_execution_end`, `turn_end`, `agent_end`. It doesn't draw anything. The terminal interface, the JSON mode and the RPC mode are all just listeners, which is why they all behave the same.

    ## What you can do now

    - name Pi's packages, and say what each is responsible for
    - read TypeScript well enough to follow the loop
    - explain why the loop has two levels, and the difference between steering and follow-ups
    - say exactly when a steering message takes effect, and why not sooner
    - explain why a `length` stop fails every tool call in the reply
    - list the loop's hooks, and what each is for

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** read all of `packages/agent/src/agent-loop.ts`, then `agent.ts`, which wraps the loop in a stateful class with the two queues. Search for `declareToolChanges` (chapter 42) and `executeToolCallsParallel` (chapter 43).
    """)
    return


if __name__ == "__main__":
    app.run()
