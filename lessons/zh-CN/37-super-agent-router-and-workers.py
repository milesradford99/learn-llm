import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="37 · Super-agent：一个 Router，许多 Worker")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 37 · Super-agent：一个 Router，许多 Worker

    Pi 是一个智能体，为一个盯着它的人工作。第二个项目问的是另一个问题：如果一个智能体为**整个团队**工作，听着大家的群聊和代码评审，同时推进许多件事，会怎样？

    2026 年 10 月 31 日，DeepSeek Harness 团队发布了一篇设计文章《Super-agent 设计探索》，以及一份 MIT 许可的参考实现 **`dsh-superagent-prototype`**。它作为插件包运行在 **DeepSeek Harness**（`dsh`）上，那是 DeepSeek 自己的开源智能体运行底座。下面三样东西要分清：

    | | 是什么 | 在哪里 |
    |---|---|---|
    | 文章 | 想法、示意图和实验 | DeepSeek 发表的文章 |
    | 原型 | 真实代码：`src/` 里约 15,000 行 TypeScript，测试代码也差不多这么多，外加 9 份设计文档 | `deepseek-ai/dsh-superagent-prototype` |
    | DeepSeek Harness | 原型所依附的运行底座：会话、工具、预设 | `deepseek-ai/deepseek-harness` |

    原型的 README 写得很清楚：它和内部版本、和文章都不一样，文章里的一些设计并没有实现。所以这条路线说*代码做了 X*，指的是原型；说*文章提出 X*，指的是文章里的想法，代码里不一定实现了。

    ## 问题：注意力

    文章的出发点根本不是模型，而是人。团队观察到，一个人通常只能跟住 3 到 5 个活跃的智能体会话。超过这个数，模型决定一件事做得多好，而人的注意力决定有多少件事能推进。

    原型的第一份文档把这变成了一条原则，每个设计选择都用它来衡量：

    <!-- source dsh docs/000-philosophy.md -->
    ```text
    **Take as little human attention as possible, and give as much value as possible.**

    Every part of the project is judged by the attention it costs. A message, a notification, a question, a setting, a panel element and a paragraph of documentation each cost someone's attention, and each has to give more value than that, or it is removed.
    ```

    记住这句话：尽量少占用人的注意力，尽量多给出价值。后面四章里的许多细节，比如群聊回复不超过 100 个词、面板只显示需要人处理的事，都直接来自它。

    ## 两层

    文章从两个极端出发。把所有事件都塞进一个会话，上下文完全连续，但一个循环一次只做一件事。每个事件开一个新会话，所有事都能同时跑，但工作支离破碎，好几个会话可能回复同一个人。

    所以工作被分到两层：

    - **Router**（路由者）是一个常驻会话（session）。它读所有进来的东西，判断哪些是新工作、哪些不是，并启动 Worker。它自己不做技术工作，所以可以用更轻的模型、更少的工具。在代码里它叫**主 Session**（main Session），提示词称它为 `orchestrator`。
    - **Worker**（工作者）各自负责一件工作，拥有全新的上下文、做工作用的工具（比 Router 多得多）和自己的目录。文章说它们通常用最强的模型；在原型里，每个角色用什么模型由团队配置。Worker 自己确认收到请求，也自己处理拉取请求上的评审讨论，但结果、答复和需要人回答的问题都经由 Router 转达。在代码里每个都是一个**执行器 Session**（Executor Session），名为 `worker-<工单 id>`。

    和 Pi 对比：Pi 的核心里根本没有子智能体（sub-agent，由另一个智能体像工具一样启动的智能体，第 36 章）。这也不是“团队负责人”把一个大任务拆成几部分。Router 只处理不同任务之间的关系，拆分任务的事留在各自的 Worker 内部。

    ## 代码使用的词

    原型有一套简短而严格的词汇，代码、提示词和面板都只用这些词。工作是这样流动的：

    <!-- source dsh docs/001-domain.md -->
    ```text
    Work moves from Trigger to Eval to execution. A Source produces Triggers. A Trigger fans out as context to the Tickets subscribed to its Threads. A person's message that reached a Ticket this way stays with that Ticket; every other Trigger enters Eval, which decides whether it asks for new work. When Eval accepts, it creates a Ticket with a Mandate. An Assignment hands the Ticket to an Executor, and results go back to the Source through a Channel.
    ```

    意思是：来源产生触发；触发作为上下文扇出（fan-out，分发）给订阅了其线程的工单；人的消息这样送到工单后就归那个工单，其他触发都进入评估，判断是否在请求新工作；评估接单时创建带授权的工单；派工把工单交给执行器；结果经渠道回到来源。

    | 词 | 含义 | 通俗地说 |
    |---|---|---|
    | 来源（Source） | 一串事件：GitHub，或飞书 | 消息从哪来 |
    | 触发（Trigger） | 一个永不改变的事件 | 一条消息、一条评论、一次 CI 结果 |
    | 线程（Thread） | 按时间排列的触发 | 一个群聊话题，或一个拉取请求的讨论 |
    | 评估（Eval） | 对一批触发的判断 | “这是在请求工作吗？” |
    | 工单（Ticket） | 一个已授权的工作单位 | 一件活 |
    | 授权（Mandate） | 接单时允许了什么、为什么 | 这件活的许可单 |
    | 订阅（Subscription） | 工单在听一个线程 | “这里说的话都转给我” |
    | 派工（Assignment） | 把工单交给一个执行器 | 一个 Worker 接手 |
    | 渠道（Channel） | 线程的写入端 | 回复发到哪里 |

    注意什么*不是*概念：人。人只是触发的一个字段（它的发送者）。设计文档把这类**被否决的概念**列出来，并给出每条的理由。这种做法很少见，值得学习。

    ## 第一站：收下一个触发

    下面是 `src/core/intake.ts` 里 `Intake.receive` 的开头。它是 TypeScript，第 31 章教过你怎么读。有两处新写法：`throw new Error(...)` 就是 Python 的 `raise`，`this.repo` 是交给 Engine 的存储。

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

    这里有两个想法。触发按 id 只存**一次**。GitHub 和飞书都可能把同一个事件送两遍，重启后来源也可能从原来的位置再读一遍。如果每份副本都产生新工作，智能体就会回答两次。像这样重复执行也安全，叫做**幂等**（idempotence）。另外，没人监听的触发也会**存下来**，并记录它为什么被丢弃。什么都不会悄悄丢失，所以面板总能回答“智能体为什么没回应？”。

    ## 练习 1：receive

    `store` 是从触发 id 到触发的字典，`trigger` 有 `"id"` 和一个 `"threads"` 列表。写 `receive(store, trigger, listened)`：

    1. 如果 `trigger["threads"]` 为空，`raise ValueError("A trigger must belong to at least one thread")`
    2. 如果它的 id 已经在 `store` 里，返回 `"duplicate"`
    3. 存下来：`store[trigger["id"]] = trigger`
    4. 如果没有 `listened`，返回 `"filtered"`，否则返回 `"stored"`
    """)
    return


@app.function
def receive(store, trigger, listened):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("37", "receive", receive, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 直接送达

    触发存下来之后发生的事，是文章的核心想法。关于某件工作的第一条消息经过 Router，由它启动一个 Worker。接单时，新工单就**订阅**了那条消息所在的线程，于是那里的后续消息**直接送到它的 Worker**，不必再问 Router。拉取请求上的一条后续评论，会送到开了这个拉取请求的 Worker。

    文章测量了这能省下多少。在每次约 300 条消息（其中约 20 条是真正的请求）的合成测试里，关掉直接送达后完成时间几乎不变，但 Router 要多做约 30% 的决定，从服务商缓存读取的词元（token）增加约 37%。（服务商可以从缓存里复用请求中重复的开头，并少收费；*缓存读取*数的就是这些词元。第 41 章会解释它为什么重要。）

    订阅是**多对多**的：两个 Worker 可以跟同一个拉取请求，两者都能听到那里的新评论，所以不用互相发消息，就能察觉彼此冲突。第 39 章讲真实的规则，那些更严格。这里先写简单版。

    ## 练习 2：deliver

    `subscriptions` 是从线程到工单 id 列表的字典。写 `deliver(trigger, subscriptions)`，按顺序、不重复地返回订阅了这个触发任一线程的工单；一个都没有就返回 `"router"`。

    1. `tickets = []`
    2. 对触发的每个线程，对 `subscriptions.get(thread, [])` 里的每个工单：如果还不在 `tickets` 里就追加
    3. `tickets` 为空时返回 `"router"`，否则返回 `tickets`
    """)
    return


@app.function
def deliver(trigger, subscriptions):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("37", "deliver", deliver, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 实验说明了什么，没说明什么

    | 设置 | 完成用时，典型 / 慢（分钟） | 每次运行的缓存读取 |
    |---|---|---|
    | Super-agent | 12.4 / 15.7 | 23.23M |
    | Super-agent，不直接送达 | 12.6 / 15.5 | 31.77M |
    | 单会话加子智能体工具，中性提示词 | 20.4 / 29.5 | 6.05M |
    | 单会话加子智能体工具，“优先委派”提示词 | 13.8 / 29.0 | 10.82M |

    （*典型*是中位数 p50，*慢*是第 95 百分位 p95：20 次里有 1 次更慢。M 表示百万词元。）

    像行家一样读这张表。Super-agent 更快，而且**稳定得多**，但它用的词元**更多**，不是更少。用中性提示词时，单会话根本没用过子智能体工具。文章很谨慎，称这些都是初步结果：负载是合成的，写代码被换成一个只会等待的工具，时间做了缩放，每种设置跑了 5 次。不要把这些数字当作代码质量的基准，它们并不衡量这个。

    ## 代码里的东西在哪

    | 文件夹 | 内容 |
    |---|---|
    | `src/core/` | 领域核心：`Engine` 和它的各部分 `Intake`、`Evals`、`Tickets`、`Assignments`、`Replies`。完全没有输入输出 |
    | `src/policy.ts`、`src/admission.ts` | 谁可以请求工作（第 38 章） |
    | `src/executor/` | `DshExecutor`：每个工单一个执行器 Session 和一个目录（第 40 章） |
    | `src/app/` | 运行时、存储、持久化的副作用、提示词、会话、管理接口 |
    | `src/sources/` | GitHub 和飞书，转换成触发 |
    | `prompts/`、`skills/` | 智能体读的内容（第 41 章） |
    | `docs/000` 到 `docs/008` | 设计文档，每份只建立在前面几份之上 |

    最后一行值得注意。这些文档按固定顺序写成，任何想法都不会在介绍之前被用到，这和本课程遵循的是同一条规则。

    ## 现在你能做到

    - 分清文章、原型和 DeepSeek Harness，说出一个说法来自哪一个
    - 说出注意力原则，以及它为什么是出发点
    - 解释 Router 和 Worker 两层、各自的模型和工具，以及它们在代码里的名字
    - 跟踪一个触发经过来源、线程、评估、工单、派工和渠道
    - 解释幂等的收取，以及被过滤的触发为什么也要存下来
    - 解释直接送达，并且不夸大地读懂实验表

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** 完整读一遍原型的 `docs/000-philosophy.md` 和 `docs/001-domain.md`：它们很短，那张被否决概念的表本身就是一课。仓库是 [deepseek-ai/dsh-superagent-prototype](https://github.com/deepseek-ai/dsh-superagent-prototype)，运行底座是 [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)。
    """)
    return


if __name__ == "__main__":
    app.run()
