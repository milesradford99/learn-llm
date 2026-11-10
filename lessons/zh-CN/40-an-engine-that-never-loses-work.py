import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="40 · 永不丢活的引擎")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 40 · 永不丢活的引擎

    Pi 在你的终端里运行，它停了你会发现。Super-agent 无人值守地连续运行好几天，有许多 Worker，还会因升级而重启。一条忘了发的消息，或者发了两遍的消息，都会耗费团队的注意力（第 37 章）。这一章讲让它可靠的机制。几乎全是普通的工程，只是做得很仔细，这本身就是要学的东西。

    ## 唯一的写入者，以及很小的核心

    领域里的一切，也就是触发、线程、决定、工单和派工，都恰好有**一个写入者**：`src/core/engine.ts` 里的 `Engine`。架构文档写得很明白：

    <!-- source dsh docs/007-architecture.md -->
    ```text
    All Engine methods are synchronous. Each command changes its aggregates in one transaction. `evaluateDue` uses a separate transaction for each Eval or Ticket it changes. Ports enqueue external work inside the transaction; adapters perform it after commit. Invalid requests throw `UserError`.
    ```

    意思是：Engine 的方法都是同步的（做完才返回）；每条命令在一个事务里改完数据；对外的工作在事务内只入队，提交之后才由适配器去做；无效请求抛出 `UserError`。**事务**（transaction）是对存储数据的一组修改，要么全部发生，要么都不发生。而 `src/core` 完全不做输入输出：没有网络、没有文件、不调用模型。它决定应该发生什么，并把要做的事*记录*下来，由叫做适配器（adapter）的其他部分稍后去做。同一份文档里还有两条规则：

    - 每个调用者只拿到 Engine 中属于自己角色的那一部分（`EngineForSources`、`EngineForMainSession`、`EngineForExecutor` 等），所以 GitHub 适配器*没法*停止工单。
    - 名字表明方向。请求用祈使语气，可能被拒绝，比如 `stopTicket`；适配器报告的事实用过去时，比如 `assignmentStarted`。

    对比 Pi：它的循环也不自己做输入输出，而是调用钩子（第 31 章）。Super-agent 走得更远，整个领域都能在没有模型、没有网络的情况下测试。

    ## 先存下，再去做

    核心想法是这样的。发一条群聊回复、启动一个 Worker、给它送一条消息，每一个都是一个**效果**（effect）。效果和导致它的决定在同一个事务里存下来，之后由一个持久化队列去执行（`src/app/effects.ts` 里的 `Effects`）。如果进程在“决定”和“执行”之间死掉，存下来的效果在下次启动时还在。这叫做**发件箱**（outbox）模式。对比第 34 章，Pi 也是先把每条记录写进日志，然后才让它起作用。

    这个队列还有一个窍门。每一个“种类 + 键”的组合只对应队列里的一行。再次放入同一种类、同一个键的效果，不会多出一行，而是替换内容并让它立即到期，同时保留它在队列里的位置。所以如果某个工单的状态在 Worker 忙碌时变了五次，Worker 只收到一次送达，而不是五次。

    ## 练习 1：enqueue

    `queue` 是一个行的列表，每行是 `{"kind", "key", "payload", "due"}`。写 `enqueue(queue, kind, key, payload, now)`，返回这个队列：

    1. 对每一行：如果它的 `"kind"` 和 `"key"` 都相同，把 `"payload"` 设为 `payload`，把 `"due"` 设为 `now`，然后返回队列
    2. 循环结束后，追加 `{"kind": kind, "key": key, "payload": payload, "due": now}` 并返回队列
    """)
    return


@app.function
def enqueue(queue, kind, key, payload, now):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("40", "enqueue", enqueue, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 重试，以及知道何时停手

    失败的效果会按**指数退避**（exponential backoff）重试，你在 Pi 的重试里见过（第 35 章）：

    <!-- source dsh src/app/effects.ts -->
    ```ts
    const delay = Math.min(handler.retry[1], handler.retry[0] * 2 ** Math.min(row.attempts, 16))
    ```

    `handler.retry[0]` 是首次延迟，`handler.retry[1]` 是上限，每种效果各不相同。给 Worker 送消息从 2 秒开始，到 60 秒停止增长；群聊回复从 5 秒开始，到 300 秒停止。`Math.min(row.attempts, 16)` 防止这个数字本身无限增长。某个效果连续失败三次后，它所属的种类才会在面板上显示为问题，这时才通知人。例外是给 Worker 送消息失败，它会立刻显示在对应的工单上。

    有些失败其实不算失败。没有空闲的 Worker 位置时，处理函数抛出 `Later`，这一行被放回去，不计入失败次数。

    ## 练习 2：retry_delay

    照上面那一行写 `retry_delay(attempts, first, cap)`。
    """)
    return


@app.function
def retry_delay(attempts, first, cap):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("40", "retry_delay", retry_delay, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 位置，以及人为什么优先

    运行 Worker 要花钱、占机器时间，所以团队会设一个上限。`src/executor/dsh.ts` 里的 `WorkerSlots` 负责分配位置。看这个条件：

    <!-- source dsh src/executor/dsh.ts -->
    ```ts
    reserve(ticketId: string, by: 'launch' | 'wake' | 'person'): void {
      if (this.starting.has(ticketId)) return
      if (this.used() >= this.max() + (by === 'person' ? 1 : 0)) throw new Later('No worker slot is free')
      this.starting.set(ticketId, by)
    }
    ```

    `launch` 是为新工单启动 Worker，`wake` 是为 CI 结果这类自动消息唤醒空闲的 Worker，`person` 是为人的消息而唤醒。为**人的消息**而唤醒时，可以超出上限**多用一个位置**，而且唤醒排在新启动之前。原因是：如果十件新活占满了所有位置，一个正在回复现有工作的人就得排在它们后面。新工作绝不能把已经参与进来的人挡在外面。这就是注意力原则，写成了一个 `+ 1`。

    ## 练习 3：reserve

    写 `reserve(used, max_workers, by)`，可以占用位置时返回 `True`。`by == "person"` 时额外加 1，否则加 0；`used < max_workers + extra` 时才有空位。
    """)
    return


@app.function
def reserve(used, max_workers, by):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("40", "reserve", reserve, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 不阻塞的循环

    在第 30 章的循环里，一个慢工具会阻塞一切，它返回之前什么都做不了。Worker 需要在构建和测试的*同时*听到人的后续消息，这样就不行。原型的*预设*（preset，DeepSeek Harness 给会话配的一组工具和设置）会把前台的 shell 命令在 **5 秒后转为后台任务**：命令继续运行，Worker 继续它的这一轮，任务结果之后作为新消息到达。Router 的**守卫**（拦截危险命令的检查）更严格，直接拒绝 `sleep` 和长时间等待，因为 Router 一阻塞，所有新请求都会被耽误。

    文章描述了更一般的形式：长命令和子任务变成异步的 `start()` 和 `stop()` 调用，立即返回，结果作为事件送回来。文章说这在 DeepSeek Harness 里已经实现，但**默认关闭**。所以你在原型里真正能找到的，是那条 5 秒规则。

    一轮结束时如果还有后台工作在跑，工单会保留它的位置。30 分钟后，剩下的任务被终止，工单报告为等待中。

    ## 重启

    重启时，正在运行的派工以“host restarted”结束，新派工带着一条**恢复说明**启动：在重复任何对外操作*之前*，先检查自己的目录、git 历史和 GitHub。已经报告过等待、就绪、受阻或失败的工作保持原样，除非那次报告之后又有消息送到，那样它也会重新开始。每个 Worker 实际把哪些消息读进了历史，都有存档记录，由它决定重新送哪些，所以消息既不会丢，也不会出现两次。

    ## 原型承认的局限

    这些文档对局限很坦诚，你也应该如此：

    <!-- source dsh docs/003-execution.md -->
    ```text
    These wrappers and directories establish defaults, not a security boundary. A worker can still execute a program by absolute path.
    ```

    Worker 运行命令**不需要批准**，而且以操作者自己的系统账户身份执行。禁用命令列表和命令包装器是在引导智能体，而不是关住它。这和第 33 章 Pi 的坦诚很接近。文章说得更远：如果它描述的云端循环被攻破，所有连接的机器都可能被控制，团队仍需要一种机制，让智能体能*使用*凭证却永远看不到凭证。它的长期设计，即循环放在无状态的云端容器里、每台机器上只跑一个小守护进程，还只是提议，并没有发布。

    ## 现在你能做到

    - 解释 Engine 的唯一写入者规则、它的事务，以及调用者拿到的角色切片
    - 解释先存后做的效果，以及它们为什么能扛过崩溃
    - 按种类和键合并效果
    - 算出有上限的退避时间，并说出失败何时才会通知人
    - 解释人多出来的那个 Worker 位置，以及它为什么存在
    - 解释 5 秒后台规则和 Router 的守卫，并把它们和文章的提议区分开
    - 说出原型里什么是安全边界、什么不是

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** `docs/007-architecture.md` 是地图：Effects 表、依赖规则和唯一写入者表。然后读 `src/executor/dsh.ts` 里的 `DshExecutor`，以及 `cordis.patch.yml` 里的两个智能体预设。
    """)
    return


if __name__ == "__main__":
    app.run()
