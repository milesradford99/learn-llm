import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="38 · Who may ask for work")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 38 · Who may ask for work

    Pi trusts the one person at the keyboard (chapter 36). Super-agent reads a team's chats and GitHub, where anyone can type anything, including a pasted web page that says "ignore your instructions". So before any work starts, the prototype answers two questions in code, not in a prompt: **who may start work here**, and **how far may that work go**.

    Its authorization document opens with the rule that everything else follows:

    <!-- source dsh docs/002-authorization.md -->
    ```text
    The host enforces allowed lists, rules and Mandates. Messages, comments and quoted documents are inputs to a decision; they cannot change those permissions.
    ```

    ## Two strengths

    Every request that gets in has a **strength**:

    - **explicit:** a person the team allowed asked the agent directly
    - **inferred:** anything else that might be a request, such as a typo an allowed person reports in passing without addressing the agent, or a follow-up from someone not on the list

    Explicit requests always work. Inferred ones depend on the **accept mode**, which the team sets:

    | Mode | Inferred candidates |
    |---|---|
    | `off` | Never even considered |
    | `observe` (the default) | Assessed, and a "yes" is only *recorded* as "would accept" |
    | `auto` | A "yes" starts work |

    That default is the attention principle (chapter 37) in action. The documents advise starting in `observe`, reading what the agent *would* have done, and only then switching to `auto`.

    ## The gate

    The first layer of Eval is the **gate**: deterministic code, no model. Whether the sender is on the allowed list, and whether the message **addresses** the agent (a mention or a direct message), picks a cell in the team's **admission table**:

    <!-- source dsh src/policy.ts -->
    ```ts
    function admissionCell(allowed: boolean, addressed: boolean): AdmissionCell | undefined {
      return allowed ? addressed ? undefined : 'listedUnaddressed' : addressed ? 'unlistedAddressed' : 'unlistedUnaddressed'
    }
    ```

    Reading notes: `a ? b : c` is TypeScript's form of Python's `b if a else c`, and here two are nested. `undefined` is TypeScript's `None`. The `undefined` case has no cell, because a listed person who addresses the agent is *always* explicit.

    ## Exercise 1: admission_cell

    Write `admission_cell(listed, addressed)`, mirroring that line: `None` for listed and addressed, `"listedUnaddressed"` for listed only, `"unlistedAddressed"` for addressed only, and `"unlistedUnaddressed"` for neither.
    """)
    return


@app.function
def admission_cell(listed, addressed):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("38", "admission_cell", admission_cell, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The team sets each cell to `explicit`, `inferred` or `ignore`. Then the gate itself, for a person's message. Here's the order, from `gate` in `src/policy.ts`. Reading notes: `&&` is `and`, `||` is `or`, `===` and `!==` are `==` and `!=`, `!x` is `not x`, and `…` marks lines left out.

    <!-- source dsh src/policy.ts -->
    ```ts
    if (allowed && trigger.addressed) return { admissible: true, strength: 'explicit', basis: { ...recorded, via: 'addressed', addressed: trigger.addressed } }
    …
    if (admission === 'explicit') return { admissible: true, strength: 'explicit', basis }
    …
    if (!(trigger.addressed === 'direct' && !allowed) && (allowed || mode !== 'off') && participating(thread, tickets)) {
      return { admissible: true, strength: allowed ? 'explicit' : 'inferred', basis: { ...recorded, via: 'followup' } }
    }
    if (admission === 'inferred' && mode !== 'off') return { admissible: true, strength: 'inferred', basis: { ...basis, observation: { mode } } }
    ```

    In words:

    1. A listed person who addresses the agent: **explicit**.
    2. The table cell says explicit (a team can open a support chat to anyone): **explicit**.
    3. A **follow-up**, a message on a Thread that already has a Ticket (`participating`): if `follow_up and (listed or mode != "off")`, then explicit for a listed person and inferred for anyone else.
    4. The cell says inferred and the mode isn't `off`: **inferred**.
    5. Anything else: not admitted, with a recorded reason.

    The order matters. Testing the explicit cell before the follow-up rule means a Thread's existing work can never *weaken* an explicit request. Bots have their own path: only bots on a list the team keeps can ever make an inferred candidate, never an explicit one.

    ## Exercise 2: gate

    Write `gate(listed, addressed, admission, follow_up, mode)`, giving `"explicit"`, `"inferred"` or `"not admitted"`. `admission` is the value of the sender's cell (`"explicit"`, `"inferred"` or `"ignore"`), and `follow_up` says whether the Thread already has a Ticket. Follow steps 1 to 5 above. We leave out the direct-message rule and the check that a scope receives every message.
    """)
    return


@app.function
def gate(listed, addressed, admission, follow_up, mode):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("38", "gate", gate, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The assessment

    A candidate that passes the gate goes to the Router as a compact **card**, a short summary with ids to read more (chapter 41). An explicit one is routed straight away, with the Router's routing tool, `orchestrator_route`. An inferred one is **assessed**, and the assessment isn't free text: it fills a structure, and code turns it into the outcome. The team writes **rules** for proactive work, for example "copy fixes" and "visual fixes" in the defaults, and the model must judge *every enabled rule exactly once*:

    <!-- source dsh src/policy.ts -->
    ```ts
    if (assessment.decision !== 'accept') return { kind: assessment.decision, reason: assessment.reason, audit }
    if (!directionAllowed(assessment.direction)) return { kind: 'ignore', audit, reason: { code: 'directionUnsettled' } }
    if (!matches.length) return { kind: 'ignore', audit, reason: { code: 'noRuleMatched' } }
    if (!assessment.intent_evidence) throw new Error('Acceptance requires intent_evidence')
    …
    if (strength === 'inferred' && settings.eval.mode !== 'auto') {
    ```

    Look at what that last test does. The model's "yes" is *not* trusted to start inferred work. Outside `auto`, code downgrades it to "would accept". An acceptance also needs a settled direction, at least one matched rule, and quoted evidence that someone actually wants this. You saw this idea in chapter 30 as "failures are data". Here it's a stronger one: **the model proposes, and code disposes.**

    ## Exercise 3: outcome

    Write `outcome(decision, matched, strength, mode)`, mirroring `assessmentOutcome`. `decision` is the model's word, and `matched` the list of rules it matched. (The code takes the strength too, though in practice only inferred candidates are assessed.)

    1. If `decision != "accept"`, give back `decision`
    2. If `matched` is empty, give back `"ignore"`
    3. If `strength == "inferred"` and `mode != "auto"`, give back `"would accept"`
    4. Otherwise `"accept"`

    (We skip the direction and evidence checks.)
    """)
    return


@app.function
def outcome(decision, matched, strength, mode):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("38", "outcome", outcome, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One Eval at a time, per Thread

    Messages arrive in bursts: "fix the button", "the blue one", "on the login page". Judging each alone would start three jobs. So **at most one Eval runs per Thread**, and Triggers that arrive meanwhile wait together for the next one, as a **batch**. The essay measured the payoff at the Router: as messages arrived faster, from about 1 to about 40 a minute, batches grew from 1.0 to 2.8 and the average time per decision fell from 6.4 to 1.6 seconds.

    An Eval also has a deadline, which counts from when its card *reached* the Router, so time spent queueing doesn't count. A late Eval isn't dropped at once:

    <!-- source dsh src/core/evals.ts -->
    ```ts
    private expire(thread: Thread): void {
      const expired = (thread.expired ?? 0) + 1
      if (expired > REQUEUES) { this.finish(thread, thread.evaluating!, { kind: 'unavailable', reason: [{ code: 'timeout' }] }); return }
    ```

    `REQUEUES` is 2. A busy Router gets two more chances, and the third expiry records `unavailable`, which the panel can show to a person. Nothing is silently lost.

    ## Exercise 4: on_deadline

    Write `on_deadline(expired)`, where `expired` is how many times this Thread's Eval has expired in a row so far. Add 1, and give back `"unavailable"` if that's more than 2, otherwise `"requeue"`.
    """)
    return


@app.function
def on_deadline(expired):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("38", "on_deadline", on_deadline, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The Mandate

    An accept creates a Ticket together with a **Mandate**: the version of the team's rules (the **Policy**) it was accepted under, the scope (what work it allows), the strength, the matched rules, the sender and how the Source received the message. It never changes. Wider scope means a new Ticket. And it's checked again before ended work resumes, so if, for example, the team has since switched off the rule an inferred Ticket was accepted under, the old Mandate no longer holds (`revalidate` in `src/policy.ts`; each kind of Mandate has its own test there).

    Only three things change a Mandate: an accept creates it, a person's stop withdraws it, and failed revalidation revokes it. A worker can't declare that someone asked it to stop. Only an Eval decides that.

    ## Why this is in code

    The prototype says it bluntly about its own prompts: editing them "does not change the gate, Mandates, tool checks or account wrappers, which remain enforced by code." Compare Pi, which leaves permission entirely to you and your container. Neither is wrong, but they're built for different settings: Pi for one expert at a keyboard, Super-agent for a team chat where strangers type.

    ## What you can do now

    - explain explicit and inferred strength, and the three accept modes, and why `observe` is the default
    - compute an admission cell and run the gate in order
    - explain why the explicit cell is tested before follow-ups
    - turn a model's assessment into an outcome, and say why code downgrades inferred accepts
    - explain per-Thread batching, delivered-time deadlines, and two requeues before unavailable
    - say what a Mandate holds, and the three ways it changes

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** `docs/002-authorization.md` covers hand-backs, resuming and revalidation in full, and `src/policy.ts` holds `gate`, `revalidate` and `assessmentOutcome`. `prompts/admission.md` is what the Router reads when it assesses a candidate.
    """)
    return


if __name__ == "__main__":
    app.run()
