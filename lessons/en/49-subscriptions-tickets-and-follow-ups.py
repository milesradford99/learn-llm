import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="49 · Subscriptions, Tickets and follow-ups")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 49 · Subscriptions, Tickets and follow-ups

    Chapter 47's `deliver` was the essay's idea in its simplest form: later messages on a Thread go straight to the Ticket that follows it. The prototype's real rules are stricter, and each one closes a specific hole. This chapter is about `src/core/intake.ts`, `src/core/tickets.ts` and `src/core/stage.ts`.

    ## Who a message reaches

    A Ticket can be **live** (open) or **ended**. When it ends, it keeps its Subscriptions. That's deliberate: a reviewer's comment on a merged pull request a week later should still reach the work it's about. But not everything should wake ended work. Here's the rule, inside the fan-out loop of `Intake.receive` (the loop that hands a Trigger to each subscribed Ticket). `!!x` turns `x` into true or false, and `.includes(x)` is Python's `x in`:

    <!-- source dsh src/core/intake.ts -->
    ```ts
    // Only a person's message on a non-context thread resumes an ended ticket; bots and CI do not (the Executor re-reads
    // threads when it resumes), and a human-stopped ticket receives nothing because its Mandate was withdrawn.
    const followed = !!thread.context?.includes(ticketId)
    const resumes = trigger.human && ticket?.ended?.by !== 'human' && !followed
    if (!ticket || (!liveTicket(ticket) && !resumes)) continue
    ```

    Each part of `resumes` is a decision:

    - **Only a person.** If CI results woke ended Tickets, every nightly build would restart old work. When a person *does* wake one, its Worker reads its Threads again anyway, so it sees what the bots said.
    - **Not if a person stopped it.** A stop withdraws the Mandate (chapter 48). Work a person stopped stays stopped.
    - **Not through a context Thread.** A Worker can follow a related Issue only to *read* it (`followForContext`). That gives it information, not authority, so it can't be woken from there.

    ## Exercise 1: reaches

    `ticket` is a dictionary with `"live"` (true or false) and `"ended_by"` (`None`, `"human"`, or who else ended it). Write `reaches(ticket, human, context)`: `human` says whether a person wrote the message, and `context` whether the Ticket follows this Thread only for context. Here's the logic, Python's version of the TypeScript above (`not x` is `!x`). Type it as the body of your function:

    ```
    resumes = human and ticket["ended_by"] != "human" and not context
    return ticket["live"] or resumes
    ```
    """)
    return


@app.function
def reaches(ticket, human, context):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("49", "reaches", reaches, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Follow-ups belong to their Worker

    Now the subtle part. A person's message that reached a Ticket this way **doesn't go to Eval at all**. Its Worker decides it:

    <!-- source dsh docs/002-authorization.md -->
    ```text
    A person's message on a Ticket's own Thread belongs to that Ticket's Executor to decide. The same applies when the main Session attaches a message to a Ticket. While an Executor owns the message, it takes no Eval and gets no Decision. Each Executor either handles the request within its Ticket's scope or hands it back for routing elsewhere.
    ```

    Why? Think about "also fix the footer while you're there". Only the Worker that's in the middle of the job knows whether that fits. The Router would have to load the whole job to judge it. So the Worker handles it within its Mandate's scope, or **hands it back** with a reason (`handback`), and only then does it become a fresh candidate for the Router. Meanwhile, the Router may only *stop* that work, at a person's request.

    There's one more safety net. When a Ticket ends, any follow-ups its Worker never read go back to Eval, so a message can't fall into a crack between "the Worker had it" and "the Worker finished".

    ## What a Ticket waits for

    The panel's rule is the attention principle again: show only what needs a person. So every open Ticket has exactly one **stage**, worked out from stored facts and never stored itself. The table is in `docs/003-execution.md`, and the code follows it row by row:

    <!-- source dsh src/core/stage.ts -->
    ```ts
    if (ticket.ended) return { stage: 'ended' }
    if (ticket.delivery) return { stage: 'person', action: 'delivery' }
    if (ticket.attention) return { stage: 'person', action: 'answer' }
    const assignment = latestAssignment(ticket)
    const report = assignment?.report?.state
    if (report === 'ready') return { stage: 'person', action: 'review' }
    // A human follow-up after a blocked or failed report returns the ticket to the executor.
    if ((report === 'blocked' || report === 'failed') && waitsForPerson(ticket)) return { stage: 'person', action: report }
    if (!assignment || assignment.state === 'pending') return { stage: 'working', action: 'queued' }
    if (report === 'waiting' || assignment.state === 'ended') return { stage: 'idle' }
    return { stage: 'working' }
    ```

    *Delivery* here means handing a message to the Worker's session; if that keeps failing, a person must look. A **Decision** is what an Eval ends with: accept, stop, ignore or unavailable. A Worker reports one of `progress`, `waiting`, `ready`, `blocked`, `failed` or `dropped`. `waitsForPerson` is true for a `blocked` or `failed` report with no person's message since. Once someone writes, the Worker has something to go on, so the Ticket is working again. A question for a person isn't hidden in a long report: it goes in its own structured field, `human_action`, so it reaches the person (the "answer" stage).

    ## Exercise 2: ticket_stage

    `ticket` is a dictionary with `"ended"`, `"delivery_failed"`, `"question"` (`None` or text), `"report"` (`None` or a state), `"person_since"` (whether a person wrote since that report) and `"assignment"` (`None`, `"pending"`, `"running"` or `"ended"`). Write `ticket_stage(ticket)`, testing the rows in this order and giving back the first that holds:

    | Holds when | Give back |
    |---|---|
    | `ended` | `"ended"` |
    | `delivery_failed` | `"person/delivery"` |
    | `question` | `"person/answer"` |
    | report is `"ready"` | `"person/review"` |
    | report is `"blocked"` or `"failed"`, and not `person_since` | `"person/"` + the report |
    | assignment is `None` or `"pending"` | `"working/queued"` |
    | report is `"waiting"`, or assignment is `"ended"` | `"idle"` |
    | anything else | `"working"` |
    """)
    return


@app.function
def ticket_stage(ticket):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("49", "ticket_stage", ticket_stage, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## When is a pull request job done?

    A Worker saying "ready" isn't the end when there's a pull request: a person still has to review it, and it may get merged, closed, or reopened. So a Ticket **follows** pull requests, and its end is decided by them. When it opened some pull requests itself, only those count. Otherwise every pull request it follows must close.

    <!-- source dsh src/core/tickets.ts -->
    ```ts
    followsDone(ticket: Ticket): Follow['closed'] {
      const follows = ticket.follows ?? []
      const own = follows.filter(item => item.own)
      const counted = own.length ? own : follows
      if (!counted.length || counted.some(item => !item.closed)) return undefined
      return counted.map(item => item.closed!).sort((a, b) => a.at.localeCompare(b.at)).at(-1)
    }
    ```

    It ends as the *latest* close did: merged, or closed. And it still keeps its Subscriptions, so a later comment from a person resumes it under the same Mandate.

    ## Exercise 3: follows_done

    Each item of `follows` is `{"own": True or False, "closed": None or (time, outcome)}`. Times are text that sorts in order. Write `follows_done(follows)`:

    1. `own` is the items with `"own"` true; `counted` is `own` if it isn't empty, otherwise all of `follows`
    2. If `counted` is empty, or any item's `"closed"` is `None`, give back `None`
    3. Otherwise give back the last of `sorted(...)` of their `"closed"` values

    (Tuples sort by their first item, so the latest time comes last.)
    """)
    return


@app.function
def follows_done(follows):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("49", "follows_done", follows_done, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compared with Pi

    In Pi, the person *is* the scheduler. They type, steer and branch, and work lives in one session tree (chapter 44). In Super-agent, there's no one watching each Worker, so the harness has to keep the state of every job itself, and say which ones need a person. That's why it has so many explicit states: Ticket stages, Report states, Assignment states, Decisions. Each one exists so that a question like "why didn't the agent answer?" or "is this stuck?" has an answer in stored facts, not in a model's memory.

    ## What you can do now

    - say which messages reach a live Ticket and which wake an ended one, and why
    - explain why a follow-up goes to its Worker instead of Eval, and what a hand-back is
    - explain what happens to unread follow-ups when a Ticket ends
    - work out a Ticket's stage from its facts, in table order
    - decide when a pull-request Ticket is done, and how it ends
    - contrast Super-agent's explicit states with Pi's single session tree

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** `docs/003-execution.md` has the stage table and the full story of retries, restarts and pull requests. In `src/core/model.ts`, read `followupOwners` and `waitsForPerson`, which several rules share.
    """)
    return


if __name__ == "__main__":
    app.run()
