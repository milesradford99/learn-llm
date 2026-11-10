import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="50 · An engine that never loses work")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 50 · An engine that never loses work

    Pi runs in your terminal, and when it stops you notice. Super-agent runs unattended for days, with many Workers, and it gets restarted for upgrades. A message it forgot to send, or one it sent twice, costs a team's attention (chapter 47). This chapter is about the machinery that makes it dependable, and nearly all of it is ordinary engineering, done carefully. That's the lesson.

    ## One writer, and a small core

    Everything in the domain, meaning Triggers, Threads, Decisions, Tickets and Assignments, has exactly **one writer**: the `Engine` in `src/core/engine.ts`. The architecture document states the rules plainly:

    <!-- source dsh docs/007-architecture.md -->
    ```text
    All Engine methods are synchronous. Each command changes its aggregates in one transaction. `evaluateDue` uses a separate transaction for each Eval or Ticket it changes. Ports enqueue external work inside the transaction; adapters perform it after commit. Invalid requests throw `UserError`.
    ```

    In plain words: Engine methods finish before they return (they're *synchronous*), each command changes its data in one transaction, and outside work is only queued inside it, then done by other parts once the changes are saved (*after commit*). A **transaction** is a group of changes to stored data that happen all together or not at all. And `src/core` does no input or output at all, no network, no files, no model calls. It decides what should happen and *records* the work to do. Other parts, called adapters, do it later. Two more rules from the same document:

    - Every caller gets only its role's slice of the Engine (`EngineForSources`, `EngineForMainSession`, `EngineForExecutor` and so on), so a GitHub adapter *can't* stop a Ticket.
    - Names tell you which way things flow. A request is an imperative that may be refused, like `stopTicket`. A fact an adapter reports is in the past tense, like `assignmentStarted`.

    Compare Pi: its loop also does no I/O itself, and calls hooks (chapter 41). Super-agent goes further, with a whole domain that can be tested without any model or network.

    ## Store first, act after

    Here's the core idea. Sending a chat reply, launching a Worker, delivering a message to it: each is an **effect**. An effect is stored in the same transaction as the decision that caused it, and done afterwards by a durable queue (`Effects`, in `src/app/effects.ts`). If the process dies between deciding and doing, the stored effect is still there at the next start. That's called an **outbox**. Compare chapter 44, where Pi writes every entry to its log before it matters.

    The queue has one more trick. Each kind of effect plus a key gives one row. Enqueuing the same pair again doesn't add a second row. It replaces the payload and makes it due at once, while keeping its place in the queue. So if one Ticket's status changes five times while its Worker is busy, the Worker gets one delivery, not five.

    ## Exercise 1: enqueue

    `queue` is a list of rows, each `{"kind", "key", "payload", "due"}`. Write `enqueue(queue, kind, key, payload, now)`, giving back the queue:

    1. For each row: if its `"kind"` and `"key"` both match, set its `"payload"` to `payload` and its `"due"` to `now`, and give back the queue
    2. After the loop, append `{"kind": kind, "key": key, "payload": payload, "due": now}` and give back the queue
    """)
    return


@app.function
def enqueue(queue, kind, key, payload, now):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("50", "enqueue", enqueue, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Retrying, and knowing when to stop

    A failed effect is tried again with **exponential backoff**, which you met in Pi's retries (chapter 45):

    <!-- source dsh src/app/effects.ts -->
    ```ts
    const delay = Math.min(handler.retry[1], handler.retry[0] * 2 ** Math.min(row.attempts, 16))
    ```

    `handler.retry[0]` is the first delay and `handler.retry[1]` the cap. Each kind has its own. Delivering to a Worker starts at 2 seconds and stops growing at 60. A chat reply starts at 5 and stops at 300. The `Math.min(row.attempts, 16)` stops the number itself from growing without limit. Once one effect has failed three times in a row, its kind shows as a problem on the panel, and only then does a person hear about it. The exception is a failed delivery to a Worker, which shows on its Ticket at once.

    Some failures aren't failures. When there's no Worker slot free, the handler throws `Later`, and the row is put back without counting a failure.

    ## Exercise 2: retry_delay

    Write `retry_delay(attempts, first, cap)`, mirroring the line above.
    """)
    return


@app.function
def retry_delay(attempts, first, cap):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("50", "retry_delay", retry_delay, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Slots, and why a person goes first

    Running Workers costs money and machine time, so a team sets a maximum. `WorkerSlots`, in `src/executor/dsh.ts`, hands out the places. Look at the condition:

    <!-- source dsh src/executor/dsh.ts -->
    ```ts
    reserve(ticketId: string, by: 'launch' | 'wake' | 'person'): void {
      if (this.starting.has(ticketId)) return
      if (this.used() >= this.max() + (by === 'person' ? 1 : 0)) throw new Later('No worker slot is free')
      this.starting.set(ticketId, by)
    }
    ```

    A `launch` starts a new Ticket's Worker, a `wake` restarts an idle Worker for an automatic message such as a CI result, and `person` is a wake for a person's message. A wake for a **person's message** may use **one extra slot** beyond the maximum, and wakes are served before new launches. The reason: if ten new jobs filled every slot, a person replying to an existing job would wait behind all of them. New work must never lock out a person who is already engaged. That's the attention principle, written as a single `+ 1`.

    ## Exercise 3: reserve

    Write `reserve(used, max_workers, by)`, giving `True` if a slot may be taken. The extra is 1 when `by == "person"`, otherwise 0, and a slot is free when `used < max_workers + extra`.
    """)
    return


@app.function
def reserve(used, max_workers, by):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("50", "reserve", reserve, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A loop that doesn't block

    In chapter 40's loop, a slow tool blocks everything. Nothing else can happen until it returns. For a Worker that must hear a person's follow-up *while* it builds and tests, that's no good. The prototype's *presets* (DeepSeek Harness's named sets of tools and settings for a session) move a foreground shell command into a **background job after 5 seconds**: the command keeps running, the Worker gets on with its turn, and the job's result arrives later as a new message. The Router has a stricter guard, which refuses `sleep` and long waits outright, because a blocked Router delays every new request.

    The essay describes the general form: long commands and sub-tasks become asynchronous `start()` and `stop()` calls that return at once, with results delivered back as events. It says this is implemented in DeepSeek Harness but **off by default**. So the 5-second rule is what you'll actually find in the prototype.

    When a turn ends with background work still running, the Ticket keeps its slot. After 30 minutes, the leftover jobs are killed, and the Ticket is reported waiting.

    ## Restarts

    On restart, a running Assignment ends with "host restarted", and a new one starts with a **recovery note**: check your directory, git history and GitHub *before* repeating any outside action. Work that had already reported waiting, ready, blocked or failed is left alone, unless a message reached it after that report, in which case it starts again too. The stored record of which messages each Worker actually took into its history decides what to deliver again, so a message isn't lost and isn't shown twice.

    ## What the prototype admits

    The documents are candid about limits, and you should be too:

    <!-- source dsh docs/003-execution.md -->
    ```text
    These wrappers and directories establish defaults, not a security boundary. A worker can still execute a program by absolute path.
    ```

    Workers run commands **without approval**, as the operator's own user. The list of blocked commands and the command wrappers guide the agent; they don't contain it. That's close to Pi's honesty in chapter 43. The essay goes further: if the cloud loop it describes were compromised, every connected machine could be controlled, and the team still needs a way for the agent to *use* credentials without ever seeing them. Its long-term design, with the loop in stateless cloud containers and only a small daemon on each machine, is proposed, not shipped.

    ## What you can do now

    - explain the Engine's one-writer rule, its transactions, and the role slices callers get
    - explain store-first effects and why they survive a crash
    - coalesce effects by kind and key
    - compute a capped backoff, and say when a failure reaches a person
    - explain the person's extra worker slot, and why it exists
    - explain the 5-second background rule and the Router's guard, and separate them from the essay's proposals
    - say what's a security boundary in the prototype and what isn't

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** `docs/007-architecture.md` is the map: the Effects table, the dependency rules and the one-writer table. Then read `DshExecutor` in `src/executor/dsh.ts`, and `cordis.patch.yml` for the two agent presets.
    """)
    return


if __name__ == "__main__":
    app.run()
