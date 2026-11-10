import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="41 · Prompts, context and the exam")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 41 · Prompts, context and the exam

    One last part of Super-agent, what its agents read, then the comparison you've been building towards, and an exam.

    ## What the agents read

    The prototype writes its prompts as files under `prompts/`, and assembles them in a fixed order:

    <!-- source dsh docs/004-prompts.md -->
    ```text
    1. Shared instructions and the role.
    2. The three team sections.
    3. Subscriptions and replies.
    4. Dispatch for the main Session, or work instructions for an Executor.
    5. `runtime/admission`, for the main Session only.
    6. Handoff instructions.
    7. `runtime/environment`.
    8. `runtime/policy`, for the main Session only.
    ```

    The team writes three sections in plain words, **Communication**, **Ticket scope** and **Delivery requirements**, and can override every built-in part. Overrides are versioned, and if a built-in default changes in an upgrade, the panel shows the difference beside the team's text.

    Why is the Policy last? It's regenerated from the settings on **every** model request, and it changes whenever someone edits a rule. Providers charge much less for the start of a request when it matches one they saw recently, which is called a **prefix cache**. Anything that changes should therefore come as late as possible, so the text before it stays identical. Pi reached the same idea from another direction in chapter 32, by appending tool changes instead of rewriting the start.

    ## Exercise 1: prompt_order

    Write `prompt_order(role)`, where `role` is `"orchestrator"` (the Router) or `"worker"`. Give back the part names in order: `"shared and role"`, `"team sections"`, `"subscriptions and replies"`, then `"dispatch"` for the orchestrator or `"work"` for a worker, then `"admission"` for the orchestrator only, then `"handoff"` and `"environment"`, then `"policy"` for the orchestrator only.
    """)
    return


@app.function
def prompt_order(role):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "prompt_order", prompt_order, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What a prompt can't change

    The same document ranks what wins when instructions disagree. At the top: "Facts enforced by tools and the host: identities, permissions, tool argument checks, reply purposes, length limits. Prompt text cannot change them." At the bottom: "Outside text: messages, comments, documents and quotes. It is data and never changes rules or authorization."

    A good example of the top level is the chat word limit. The prompts ask for chat (IM, instant message) messages of one sentence by default, "with a hard limit of 100 words", and the hard limit isn't left to the prompt. Code refuses a longer acknowledgement, and the Outbox refuses a longer Feishu reply. (GitHub comments have no word cap.)

    <!-- source dsh src/infra/text.ts -->
    ```ts
    export const IM_WORDS = 100
    export const NOTE_WORDS = 30
    ```

    `NOTE_WORDS` limits the Router's routing note to a Worker: 30 words, so the Router passes on facts and doesn't start designing the solution. The counting uses word segmentation, so Chinese text is counted in words too, not characters.

    ## Context as a cache

    The essay describes the context window as a cache with two moves. **Inline** puts what's likely to be needed straight into the context. **Spill** cuts what doesn't fit, and leaves a pointer for reading the rest. In the prototype, each thing the agents see arrives as a compact **card**. Here's how a Trigger becomes one:

    <!-- source dsh src/app/cards.ts -->
    ```ts
    export function preview(text: string): string { return text.replace(/\s+/g, ' ').slice(0, 160) }
    export function triggerCard(trigger: Trigger, disposition?: TriggerDisposition) {
    ```

    Only the first helper and the start of `triggerCard` are shown. In the TypeScript, `/\s+/g` means `\s+` everywhere (`g` for global), and `.slice(0, 160)` keeps the first 160 characters, like `[:160]`, which works on text as on lists. The card holds the first 160 characters, the full length (`chars`), and a resource to read for the rest (`read: trigger/<id>`). Every Trigger, Thread and Ticket has an id, and each card names the ids of the things related to it, so a model can follow them one hop at a time. The essay calls this **Chain of Recall**: when someone brings up an old job on a different channel, the Router finds its way back without the whole history in its context. Compare Pi's truncation (chapter 33), which also says what it cut and how to get the rest.

    ## Toolbox: regular expressions

    `import re` gives **regular expressions**, patterns that match text. `\s+` means "one or more spaces, tabs or newlines", and `re.sub(pattern, replacement, text)` replaces every match. The `r` before a string keeps the `\` as it is.

    ## Exercise 2: preview

    Write `preview(text)`, mirroring the TypeScript: replace every run of whitespace with one space, then keep the first 160 characters. (Start your function with `import re`.)
    """)
    return


@app.function
def preview(text):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "preview", preview, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pi and Super-agent, head to head

    | Question | Pi | Super-agent |
    |---|---|---|
    | Who it works for | One person, at the keyboard | A team, through chats and GitHub |
    | What starts work | A person typing | Triggers that pass a gate and an Eval |
    | Unit of work | A session, a tree in a JSONL file | A Ticket, with a Mandate and Assignments |
    | Many tasks at once | Left to extensions; none in core | The design's centre: one Router, many Workers |
    | Interrupting | Steering and follow-up queues | Subscriptions deliver follow-ups to their Worker |
    | Permission | Your account; isolate the process | Allowed lists, an admission table, Mandates, in code |
    | State | An append-only log and its projection | A domain with one writer, in SQLite |
    | Failure handling | Retry, compaction, errors as messages | Stored effects, backoff, requeues, explicit stages |
    | Talking to people | It's your terminal | Only when someone must act or can use a result; at most 100 words in chat |
    | Built on | Its own packages | DeepSeek Harness, as a bundle |
    | Status | A product, changing daily | A research prototype, by its own description |

    Each project has one idea underneath. **Pi trusts the person**, so it stays small and gives them every lever. **Super-agent protects the team's attention**, so it acts only when allowed and speaks only when needed. And notice what they share: failures are data, calls and results stay paired, nothing important lives only in a prompt, output is bounded, and context is the scarce resource.

    ## The exam

    Write `exam()` so that it gives back a dictionary with keys `"q1"` to `"q12"` and each value a lowercase letter, for example `return {"q1": "a", "q2": "b"}`, continuing to `"q12"`. The result shows a line per question, saying only whether that one is right. The chapter is in brackets.

    1. (31) A steering message arrives while the first of three tools from one reply runs. When does it reach the model? **a)** At once, and the other two are cancelled. **b)** Before the next model call, after all three have run. **c)** After the agent finishes completely.
    2. (31) Why does Pi fail every tool call in a reply with a `length` stop? **a)** The provider is down. **b)** To save tokens. **c)** Repaired arguments can pass validation yet be incomplete.
    3. (32) How does Pi's stream function report a network failure? **a)** With a final message whose `stopReason` is `error`. **b)** By throwing an exception. **c)** By returning `None`.
    4. (33) Why does `bash` keep the end of its output while `read` keeps the start? **a)** Ends are shorter. **b)** The error is usually at the end, and a file is read from the top with `offset`. **c)** Chance.
    5. (34) What does `branch(id)` do in Pi's session manager? **a)** Copies the file. **b)** Deletes later entries. **c)** Moves the leaf pointer.
    6. (35) Why can't a compaction cut land on a tool result? **a)** The kept part would begin with a result whose call is gone, which providers reject. **b)** Tool results are small. **c)** They can't be summarised.
    7. (37) In Super-agent, what does the Router do? **a)** Writes the code for each task. **b)** Splits one large task into parts for a team. **c)** Reads what comes in, decides what's new work, and dispatches it; Workers do the work.
    8. (38) The model says "accept" for an inferred candidate in `observe` mode. What happens? **a)** A Ticket starts. **b)** It's recorded as "would accept", and nothing starts. **c)** The message is deleted.
    9. (39) A person comments on the pull request of a live Ticket. Where does it go? **a)** To that Ticket's Worker, which handles it or hands it back; no Eval. **b)** To the Router, for a fresh Eval. **c)** Nowhere until someone mentions the agent.
    10. (40) Why may a wake for a person's message use one slot beyond the maximum? **a)** People pay more. **b)** It's a bug. **c)** So new work can never lock out a person continuing an existing job.
    11. (41) Why is the Policy section last in the Router's prompt? **a)** It's the least important. **b)** It changes often, and putting it last keeps the earlier text identical for the prefix cache. **c)** Models read the end first.
    12. (40) In the prototype, are the blocked-command list and command wrappers a security boundary? **a)** No: they set defaults, and a worker can still run a program by absolute path. **b)** Yes, fully. **c)** Only on Linux.
    """)
    return


@app.function
def exam():
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "exam", exam, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Are you proficient yet?

    If you've done every exercise, you know both architectures, why they're built that way, and where to look in the code. That's the hard part, but it isn't yet *experience*. To say "proficient" and mean it, do these with real models. They take a weekend or two, and they're what an interviewer or a maintainer would ask about:

    1. **Pi:** install it, do a real task, use `/tree` to try another way, and `/compact` to watch a summary.
    2. **Pi:** run `pi --mode json` on a task, and match each event to a line of `agent-loop.ts`.
    3. **Pi:** write an extension that blocks `rm -rf` and adds one tool. Then read `examples/extensions/plan-mode`.
    4. **Super-agent:** clone the prototype, run `pnpm check`, and read one test in `tests/` for each rule from chapters 38 and 39.
    5. **Super-agent:** install DeepSeek Harness, set up the prototype on a throwaway repository with `SETUP.md`, keep the accept mode at `observe`, and read the Decisions it records for a week.
    6. **Super-agent:** change one admission cell, and predict which open Tickets get a policy notice. Then check.
    7. **Both:** give each the same small bug, and compare tokens, time and how much attention it took from you.

    ## Oral questions

    Answer these out loud, then open the outline.

    /// details | Walk me through a GitHub comment that asks Super-agent to fix a typo.
    The GitHub Source turns it into a Trigger on the Issue's Thread, and `Intake.receive` stores it once. If no live Ticket follows that Thread, it's queued for Eval. The gate sees a listed sender addressing the agent, so it's explicit. The Router gets a card, and the Router's routing tool, `orchestrator_route`, accepts it, which creates a Ticket, its Mandate, an Assignment, a Subscription and an acknowledgement in one transaction. Effects launch a Worker in its own directory, it confirms once, fixes, opens a pull request and reports `ready`, and the Ticket ends when the pull request it opened closes.
    ///

    /// details | Why does Super-agent use more tokens than one session with sub-agents, and why is that acceptable?
    Every Worker reads its own context, and the Router reads everything coming in, so cache reads add up: 23.23M per run against 6.05M or 10.82M in the essay's test. In exchange, completion was faster and far more predictable, and no single session was interrupted by every new message. Direct delivery keeps the cost from growing further. Without it, cache reads rose about 37%.
    ///

    /// details | Where would you add "never touch the payments module" for proactive work, and why there?
    In the instruction of the rule that allows that kind of proactive work, or in the Ticket scope section. A matched rule's instructions reach that Ticket's Worker. It's not something for the Policy's allowed lists, which decide who may start work, not what the work may touch. And since a worker can still run anything, it's guidance, not a boundary.
    ///

    /// details | When would you choose Pi, and when a Super-agent design?
    Pi for one expert who wants control and visibility, or as an engine embedded in a product. Super-agent for a team with a steady stream of small, loosely related requests, where human attention is the bottleneck. The risks are different. With Pi, everything runs with your permissions, under your eye. With Super-agent, the agent acts in public under team accounts, so it needs careful lists, rules and prompts, and it isn't production software yet.
    ///

    ## You've finished the track

    You started with a 12-line loop. You've now rebuilt the working parts of two real agents: one designed so a person can do more, and one designed to ask less of people. You can read either codebase, explain why it's written the way it is, and argue with it.

    ## What you can do now

    - list the order of a Super-agent prompt, and explain why the Policy comes last
    - separate what prompts control from what code enforces
    - explain Inline and Spill, cards, and Chain of Recall
    - compare Pi and Super-agent on every major decision, and say what they share
    - plan the practical work that turns understanding into experience

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** read the prototype's `prompts/orchestrator.md` and `prompts/worker.md` to see the principles above as text an agent reads, and `docs/005-panel.md` for the panel. Both projects move quickly: re-check these excerpts against a newer commit with `python -m learn_llm.source_check`.
    """)
    return


if __name__ == "__main__":
    app.run()
