import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="47 · Super-agent: one Router, many Workers")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 47 · Super-agent: one Router, many Workers

    Pi is one agent, working for one person who watches it. The second project asks a different question. What if an agent works for a **whole team**, listening to their chats and their code reviews, and moving many pieces of work forward at once?

    On 31 October 2026 the DeepSeek Harness team published a design essay, *Super-agent Design Exploration*, and an MIT-licensed reference implementation, **`dsh-superagent-prototype`**. It runs as a bundle on **DeepSeek Harness** (`dsh`), DeepSeek's own open-source agent harness. Keep three things apart:

    | | What it is | Where |
    |---|---|---|
    | The essay | Ideas, diagrams and experiments | DeepSeek's write-up (in Chinese) |
    | The prototype | Real code: about 15,000 lines of TypeScript in `src/`, as many again in tests, and 9 design documents | `deepseek-ai/dsh-superagent-prototype` |
    | DeepSeek Harness | The harness the prototype plugs into: sessions, tools, presets | `deepseek-ai/deepseek-harness` |

    The prototype's README is clear that it "differs from both the internal version and the essay", and that some of the essay's designs aren't implemented. So when this track says *the code does X*, that's the prototype. When it says *the essay proposes X*, that's not something you can find in the code.

    ## The problem: attention

    The essay's starting point isn't the model at all. It's people. The team observes that one person can follow only 3 to 5 active agent sessions. Past that, the model decides how well one task goes, but human attention decides how many tasks move at all.

    The prototype's first document turns this into a rule that every design choice is judged by:

    <!-- source dsh docs/000-philosophy.md -->
    ```text
    **Take as little human attention as possible, and give as much value as possible.**

    Every part of the project is judged by the attention it costs. A message, a notification, a question, a setting, a panel element and a paragraph of documentation each cost someone's attention, and each has to give more value than that, or it is removed.
    ```

    Hold on to that sentence. Many details in the next four chapters, such as the 100-word limit on chat replies and the panel that shows only what needs a person, come straight from it.

    ## Two layers

    The essay starts from two extremes. Send every event into one session, and the context is perfectly continuous, but one loop works one thing at a time. Open a new session per event, and everything runs at once, but the work is fragmented, and several sessions may answer the same person.

    So the work is split across two layers:

    - **The Router** is one resident session. It reads everything that comes in, decides what's new work and what isn't, and starts Workers. It does no technical work itself, so it can run a lighter model with fewer tools. In the code it's the **main Session**, and its prompt calls it the `orchestrator`.
    - **Workers** each get one piece of work, with a fresh context, the work tools (many more than the Router has) and their own directory. The essay says they usually run the strongest model; in the prototype, the team configures each role's model. A Worker acknowledges a request itself and handles pull-request review threads, but results, answers and questions for a person go back through the Router. In the code each is an **Executor Session**, named `worker-<ticket id>`.

    Compare Pi, which has no sub-agents (agents another agent starts like a tool) in its core at all (chapter 46). And this isn't a "team lead" splitting one big task into parts. The Router only deals with how separate tasks relate to each other; splitting a task up stays inside its Worker.

    ## The words the code uses

    The prototype has a short, strict vocabulary, and the code, prompts and panel use no other words. Here's how work flows:

    <!-- source dsh docs/001-domain.md -->
    ```text
    Work moves from Trigger to Eval to execution. A Source produces Triggers. A Trigger fans out as context to the Tickets subscribed to its Threads. A person's message that reached a Ticket this way stays with that Ticket; every other Trigger enters Eval, which decides whether it asks for new work. When Eval accepts, it creates a Ticket with a Mandate. An Assignment hands the Ticket to an Executor, and results go back to the Source through a Channel.
    ```

    | Word | Meaning | In everyday terms |
    |---|---|---|
    | Source | A stream of events: GitHub, or the Feishu chat app | Where messages come from |
    | Trigger | One event, which never changes | A message, a comment, a CI result |
    | Thread | Triggers in time order | A chat topic, or one pull request's discussion |
    | Eval | The judgement on a batch of Triggers | "Is this a request for work?" |
    | Ticket | A unit of authorized work | One job |
    | Mandate | What the acceptance allowed, and why | The job's permission slip |
    | Subscription | A Ticket listening to a Thread | "Send me what's said here" |
    | Assignment | A Ticket handed to one Executor | A Worker on the job |
    | Channel | The write side of a Thread | Where replies go |

    Notice what's *not* a concept: a person. People are just a field of a Trigger (its actor). The design document lists such **rejected concepts** with a reason for each, which is unusual and worth copying.

    ## First stop: taking a Trigger in

    Here's the start of `Intake.receive`, in `src/core/intake.ts`. It's TypeScript, which chapter 41 taught you to read. Two new pieces: `throw new Error(...)` is Python's `raise`, and `this.repo` is the store the Engine was given.

    <!-- source dsh src/core/intake.ts -->
    ```ts
    if (!trigger.threads.length) throw new Error('A trigger must belong to at least one thread')
    if (this.repo.triggers.get(trigger.id)) return 'duplicate'
    const at = this.now()
    const filter = this.options.scope.scope(trigger)
    if (!filter.listened) {
      this.repo.triggers.save(trigger); this.repo.triggers.saveDisposition(trigger.id, { filtered: { code: filter.reason }, fanout: [], at })
      return 'filtered'
    }
    this.repo.triggers.save(trigger)
    ```

    Two ideas here. A Trigger is stored **once**, keyed by its id. GitHub and Feishu can both deliver the same event twice, and after a restart a Source may read again from where it was. If each copy made new work, the agent would answer twice. Being safe to repeat like this is called **idempotence**. And a Trigger that nobody listens to is still **stored**, with a record of why it was dropped. Nothing is silently lost, so the panel can always answer "why didn't the agent respond?".

    ## Exercise 1: receive

    `store` is a dictionary from Trigger id to Trigger, and `trigger` has an `"id"` and a list of `"threads"`. Write `receive(store, trigger, listened)`:

    1. If `trigger["threads"]` is empty, `raise ValueError("A trigger must belong to at least one thread")`
    2. If its id is already in `store`, give back `"duplicate"`
    3. Store it: `store[trigger["id"]] = trigger`
    4. Give back `"filtered"` if not `listened`, otherwise `"stored"`
    """)
    return


@app.function
def receive(store, trigger, listened):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("47", "receive", receive, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Direct delivery

    What happens after a Trigger is stored is the essay's key idea. The first message about some work goes through the Router, which starts a Worker. Accepting the work **subscribes** the new Ticket to that message's Thread, so later messages there go **straight to its Worker**, without asking the Router again. A follow-up comment on a pull request reaches the Worker that opened the pull request.

    The essay measured what this saves. In a synthetic test with about 300 messages per run (about 20 of them real requests), turning direct delivery off barely changed completion time, but the Router had to make about 30% more decisions, and the tokens read from the provider's cache rose about 37%. (Providers can re-use the repeated start of a request from a cache, and charge less for it; *cache reads* count those tokens. Chapter 51 explains why that matters.)

    Subscriptions are **many-to-many**: two Workers can follow one pull request, and both hear new comments there, so they can notice that they conflict without messaging each other. Chapter 49 has the real rules, which are stricter. Here's the simple version.

    ## Exercise 2: deliver

    `subscriptions` is a dictionary from a Thread to a list of Ticket ids. Write `deliver(trigger, subscriptions)`, giving back the Tickets subscribed to any of the Trigger's Threads, in order and with no repeats, or `"router"` if there are none.

    1. `tickets = []`
    2. For each thread of the trigger, for each ticket in `subscriptions.get(thread, [])`: append it if it isn't in `tickets` yet
    3. Give back `"router"` if `tickets` is empty, otherwise `tickets`
    """)
    return


@app.function
def deliver(trigger, subscriptions):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("47", "deliver", deliver, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What the experiments show, and don't

    | Setup | Done in, typical / slow (min) | Cache reads per run |
    |---|---|---|
    | Super-agent | 12.4 / 15.7 | 23.23M |
    | Super-agent, no direct delivery | 12.6 / 15.5 | 31.77M |
    | One session with a sub-agent tool, neutral prompt | 20.4 / 29.5 | 6.05M |
    | One session with a sub-agent tool, "delegate first" prompt | 13.8 / 29.0 | 10.82M |

    (*Typical* is the median, p50, and *slow* the 95th percentile, p95: 1 run in 20 is slower. M means millions of tokens.)

    Read it as an expert would. Super-agent was faster and much **more predictable**, and it used **more** tokens, not fewer. With a neutral prompt, the single session never used its sub-agent tool at all. The essay is careful to call all this preliminary: the load was synthetic, coding was replaced by a tool that just waits, timing was scaled, and each setup ran 5 times. Don't repeat these numbers as a benchmark of code quality, because they don't measure it.

    ## Where things are in the code

    | Folder | Holds |
    |---|---|
    | `src/core/` | The domain: `Engine` and its parts `Intake`, `Evals`, `Tickets`, `Assignments`, `Replies`. No input or output at all |
    | `src/policy.ts`, `src/admission.ts` | Who may ask for work (chapter 48) |
    | `src/executor/` | `DshExecutor`: one Executor Session and directory per Ticket (chapter 50) |
    | `src/app/` | Runtime, storage, durable side effects, prompts, sessions, the management API |
    | `src/sources/` | GitHub and Feishu, turned into Triggers |
    | `prompts/`, `skills/` | What the agents read (chapter 51) |
    | `docs/000` to `docs/008` | Design documents, each building only on the ones before |

    That last row is worth noticing. The documents are written in a fixed order where no idea is used before it's introduced, which is the same rule this course follows.

    ## What you can do now

    - separate the essay, the prototype and DeepSeek Harness, and say which one a claim comes from
    - state the attention principle, and why it's the starting point
    - explain the Router and Worker layers, their models and tools, and their names in the code
    - trace a Trigger through Source, Thread, Eval, Ticket, Assignment and Channel
    - explain idempotent intake, and why filtered Triggers are still stored
    - explain direct delivery, and read the experiment table without over-claiming

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** read the prototype's `docs/000-philosophy.md` and `docs/001-domain.md` in full: they're short, and the table of rejected concepts is a lesson in itself. The repository is [deepseek-ai/dsh-superagent-prototype](https://github.com/deepseek-ai/dsh-superagent-prototype), and the harness is [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness).
    """)
    return


if __name__ == "__main__":
    app.run()
