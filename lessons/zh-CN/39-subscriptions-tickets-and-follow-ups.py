import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="39 · 订阅、工单与后续消息")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 39 · 订阅、工单与后续消息

    第 37 章的 `deliver` 是文章想法的最简形式：线程上后来的消息直接送到跟着它的工单。原型的真实规则更严格，每一条都堵住一个具体的漏洞。这一章讲 `src/core/intake.ts`、`src/core/tickets.ts` 和 `src/core/stage.ts`。

    ## 一条消息会送到谁

    工单可以是**活着的**（open）或**已结束的**（ended）。结束时，它保留自己的订阅。这是有意为之：一周后评审者在已合并的拉取请求上留的评论，仍应送到相关的工作。但不是什么都该唤醒已结束的工作。规则在 `Intake.receive` 里把触发分发给各个订阅工单的循环中（即扇出，fan-out）。`!!x` 把 `x` 变成真或假，`.includes(x)` 就是 Python 的 `x in`：

    <!-- source dsh src/core/intake.ts -->
    ```ts
    // Only a person's message on a non-context thread resumes an ended ticket; bots and CI do not (the Executor re-reads
    // threads when it resumes), and a human-stopped ticket receives nothing because its Mandate was withdrawn.
    const followed = !!thread.context?.includes(ticketId)
    const resumes = trigger.human && ticket?.ended?.by !== 'human' && !followed
    if (!ticket || (!liveTicket(ticket) && !resumes)) continue
    ```

    `resumes` 的每一部分都是一个决定：

    - **只有人。** 如果 CI 结果能唤醒已结束的工单，每晚的构建都会重启旧工作。人*真的*唤醒它时，Worker 反正会重读它的线程，所以能看到机器人说了什么。
    - **被人停止的不行。** 停止会撤回授权（第 38 章）。人停下的工作就保持停止。
    - **不能通过上下文线程。** Worker 可以只为*阅读*而跟一个相关的 Issue（`followForContext`）。这给它信息，不给它权限，所以不能从那里被唤醒。

    ## 练习 1：reaches

    `ticket` 是带 `"live"`（真或假）和 `"ended_by"`（`None`、`"human"`，或别的结束者）的字典。写 `reaches(ticket, human, context)`：`human` 表示消息是不是人写的，`context` 表示工单是否只为上下文而跟这个线程。下面是逻辑，即上面那段 TypeScript 的 Python 版（`not x` 就是 `!x`）。把它写成函数体：

    ```
    resumes = human and ticket["ended_by"] != "human" and not context
    return ticket["live"] or resumes
    ```
    """)
    return


@app.function
def reaches(ticket, human, context):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("39", "reaches", reaches, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 后续消息归它的 Worker

    现在是微妙的部分。以这种方式送到工单的人的消息，**根本不进入评估**，由它的 Worker 来决定。下面这段说的是：人在工单自己的线程上发的消息，归那个工单的执行器决定；主 Session 把消息附给工单时也一样；执行器拿着这条消息时，它不经评估、也没有决定；执行器要么在工单范围内处理，要么交回去重新路由。

    <!-- source dsh docs/002-authorization.md -->
    ```text
    A person's message on a Ticket's own Thread belongs to that Ticket's Executor to decide. The same applies when the main Session attaches a message to a Ticket. While an Executor owns the message, it takes no Eval and gets no Decision. Each Executor either handles the request within its Ticket's scope or hands it back for routing elsewhere.
    ```

    为什么？想想“顺便把页脚也修一下”。只有正在做这件活的 Worker 知道这合不合适。Router 要判断它，就得把整件活都加载进来。所以 Worker 在授权范围内处理它，或者附上理由**交回**（`handback`），只有那时它才成为 Router 的新候选。在此期间，Router 只能应人的要求*停止*那件工作。

    还有一道保险。工单结束时，Worker 从没读过的后续消息会回到评估，所以消息不会掉进“Worker 拿着”和“Worker 已做完”之间的缝里。

    ## 工单在等什么

    面板的规则又是注意力原则：只显示需要人处理的事。所以每个打开的工单恰好处于一个**阶段**（stage），由存下来的事实算出，阶段本身从不存储。表格在 `docs/003-execution.md` 里，代码逐行照它来写：

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

    这里的*送达*（delivery）指把消息交给 Worker 的会话，一直失败就需要人来看。**决定**（Decision）是评估的结果：accept、stop、ignore 或 unavailable。Worker 报告的状态是 `progress`、`waiting`、`ready`、`blocked`、`failed` 或 `dropped` 之一。报告为 `blocked` 或 `failed`、且之后没有人发过消息时，`waitsForPerson` 为真。一旦有人回话，Worker 就有了继续的依据，工单又回到工作中。给人的问题不会藏在长报告里：它放在单独的结构化字段 `human_action` 里，这样才能送到人那里（“answer”阶段）。

    ## 练习 2：ticket_stage

    `ticket` 是一个字典，有 `"ended"`、`"delivery_failed"`、`"question"`（`None` 或文字）、`"report"`（`None` 或一个状态）、`"person_since"`（那次报告之后是否有人回过话）和 `"assignment"`（`None`、`"pending"`、`"running"` 或 `"ended"`）。写 `ticket_stage(ticket)`，按下面的顺序逐行判断，返回第一个成立的：

    | 成立条件 | 返回 |
    |---|---|
    | `ended` | `"ended"` |
    | `delivery_failed` | `"person/delivery"` |
    | `question` | `"person/answer"` |
    | 报告是 `"ready"` | `"person/review"` |
    | 报告是 `"blocked"` 或 `"failed"`，且不是 `person_since` | `"person/"` 加上报告 |
    | 派工是 `None` 或 `"pending"` | `"working/queued"` |
    | 报告是 `"waiting"`，或派工是 `"ended"` | `"idle"` |
    | 其他情况 | `"working"` |
    """)
    return


@app.function
def ticket_stage(ticket):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("39", "ticket_stage", ticket_stage, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 拉取请求的活什么时候算完？

    有拉取请求时，Worker 说“ready”并不是终点：还得有人评审，它可能被合并、关闭或重新打开。所以工单会**跟**着拉取请求，由它们决定自己何时结束。工单自己开过拉取请求时，只算那些；否则它跟的所有拉取请求都得关闭。

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

    它以*最晚*的那次关闭为结局：merged 或 closed。而且它仍保留订阅，所以之后有人留评论，就会在同一个授权下恢复它。

    ## 练习 3：follows_done

    `follows` 的每一项是 `{"own": True 或 False, "closed": None 或 (时间, 结局)}`。时间是可以按顺序排序的文字。写 `follows_done(follows)`：

    1. `own` 是 `"own"` 为真的项；`own` 不为空时 `counted` 是 `own`，否则是全部 `follows`
    2. 如果 `counted` 为空，或者有一项的 `"closed"` 是 `None`，返回 `None`
    3. 否则返回它们的 `"closed"` 值 `sorted(...)` 之后的最后一个

    （元组按第一项排序，所以最晚的时间排在最后。）
    """)
    return


@app.function
def follows_done(follows):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("39", "follows_done", follows_done, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 和 Pi 对比

    在 Pi 里，人*就是*调度者。他们打字、转向、分支，工作都在一棵会话树里（第 34 章）。在 Super-agent 里，没人盯着每个 Worker，所以 harness 必须自己保存每件工作的状态，并说出哪些需要人。这就是它有这么多显式状态的原因：工单阶段、报告状态、派工状态、决定。每一种都是为了让“智能体为什么没回应？”“这个卡住了吗？”这类问题能从存下来的事实里得到回答，而不是靠模型的记忆。

    ## 现在你能做到

    - 说出哪些消息会送到活着的工单、哪些会唤醒已结束的工单，以及原因
    - 解释后续消息为什么交给它的 Worker 而不是评估，以及什么是交回
    - 解释工单结束时，未读的后续消息会怎样
    - 按表格顺序，从事实算出工单的阶段
    - 判断拉取请求类工单何时完成，以及以什么结局结束
    - 对比 Super-agent 的显式状态和 Pi 的单棵会话树

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** `docs/003-execution.md` 有阶段表，以及重试、重启和拉取请求的完整说明。在 `src/core/model.ts` 里读 `followupOwners` 和 `waitsForPerson`，好几条规则都用到它们。
    """)
    return


if __name__ == "__main__":
    app.run()
