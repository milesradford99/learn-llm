import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="41 · 提示词、上下文与考试")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 41 · 提示词、上下文与考试

    Super-agent 还剩最后一部分：它的智能体读什么。然后是我们一路铺垫的比较，以及一场考试。

    ## 智能体读什么

    原型把提示词写成 `prompts/` 下的文件，按固定顺序组装：

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

    也就是：共有说明和角色、三个团队部分、订阅与回复、派发（主 Session）或工作说明（执行器）、准入（仅主 Session）、交接、环境、策略（仅主 Session）。团队用大白话写三个部分：**沟通**、**工单范围**和**交付要求**，还可以覆盖任何内置部分。覆盖内容带版本；如果升级改变了内置默认值，面板会在团队文字旁边显示差异。

    为什么策略放在最后？它在**每次**调用模型时都会根据设置重新生成，有人改规则它就会变。如果请求的开头和服务商最近见过的一致，这部分的费用会便宜很多，这叫**前缀缓存**（prefix cache）。所以会变的东西应该尽量靠后，让前面的文字保持不变。Pi 在第 32 章从另一个方向想到了同一点：追加工具变化，而不是改写开头。

    ## 练习 1：prompt_order

    写 `prompt_order(role)`，`role` 是 `"orchestrator"`（Router）或 `"worker"`。按顺序返回各部分的名字：`"shared and role"`、`"team sections"`、`"subscriptions and replies"`，然后 orchestrator 是 `"dispatch"`、worker 是 `"work"`，接着只给 orchestrator 加 `"admission"`，然后是 `"handoff"` 和 `"environment"`，最后只给 orchestrator 加 `"policy"`。
    """)
    return


@app.function
def prompt_order(role):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "prompt_order", prompt_order, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 提示词改变不了什么

    同一份文档还给指令冲突时的优先级排了序。最高一级是：由工具和宿主强制执行的事实，如身份、权限、工具参数检查、回复用途、长度限制，提示词改变不了它们。最低一级是：外来文字，如消息、评论、文档和引文，只是数据，永远改变不了规则或授权。

    群聊（IM，即时消息）字数上限就是最高一级的好例子。提示词要求群聊消息默认一句话、“最多 100 个词”，而这个硬上限并不只交给提示词：代码会拒绝更长的确认回复，发件箱（Outbox）会拒绝更长的飞书回复。（GitHub 评论没有字数上限。）

    <!-- source dsh src/infra/text.ts -->
    ```ts
    export const IM_WORDS = 100
    export const NOTE_WORDS = 30
    ```

    `NOTE_WORDS` 限制 Router 给 Worker 的路由备注：30 个词，这样 Router 只转达事实，不会开始设计方案。计数用的是分词，所以中文也按词计算，而不是按字符。

    ## 把上下文当作缓存

    文章把上下文窗口描述成一个有两种操作的缓存。**内联**（Inline）把可能用到的信息直接放进上下文；**溢出**（Spill）把放不下的部分截掉，留一个指针供继续读取。在原型里，智能体看到的每样东西都以紧凑的**卡片**形式到达。触发是这样变成卡片的：

    <!-- source dsh src/app/cards.ts -->
    ```ts
    export function preview(text: string): string { return text.replace(/\s+/g, ' ').slice(0, 160) }
    export function triggerCard(trigger: Trigger, disposition?: TriggerDisposition) {
    ```

    这里只展示了第一个辅助函数和 `triggerCard` 的开头。TypeScript 里的 `/\s+/g` 表示全局匹配 `\s+`（`g` 即 global），`.slice(0, 160)` 保留前 160 个字符，相当于 `[:160]`，对文字和列表都适用。卡片里有前 160 个字符、完整长度（`chars`），以及一个读取其余内容的资源（`read: trigger/<id>`）。每个触发、线程和工单都有 id，每张卡片都写着相关事物的 id，所以模型可以一跳一跳地顺着找。文章称之为**回忆链**（Chain of Recall）：有人在另一个渠道提起一件旧工作时，Router 不需要把全部历史放进上下文，也能找回来。对比 Pi 的截断（第 33 章），它同样会说明切掉了什么、怎样拿到其余部分。

    ## 工具箱：正则表达式

    `import re` 提供**正则表达式**（regular expression），也就是匹配文字的模式。`\s+` 表示“一个或多个空格、制表符或换行”，`re.sub(pattern, replacement, text)` 替换所有匹配。字符串前的 `r` 让 `\` 保持原样。

    ## 练习 2：preview

    照那行 TypeScript 写 `preview(text)`：把每段连续的空白替换成一个空格，然后保留前 160 个字符。（函数开头写 `import re`。）
    """)
    return


@app.function
def preview(text):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "preview", preview, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pi 和 Super-agent 正面对比

    | 问题 | Pi | Super-agent |
    |---|---|---|
    | 为谁工作 | 一个坐在键盘前的人 | 一个团队，通过群聊和 GitHub |
    | 什么启动工作 | 人打字 | 通过准入和评估的触发 |
    | 工作单位 | 会话，JSONL 文件里的一棵树 | 工单，带授权和派工 |
    | 同时做许多事 | 交给扩展；核心里没有 | 设计的核心：一个 Router，许多 Worker |
    | 打断 | 转向和后续队列 | 订阅把后续消息送给对应的 Worker |
    | 权限 | 你的账户；隔离进程 | 允许名单、准入表、授权，都在代码里 |
    | 状态 | 只追加的日志及其投影 | 只有一个写入者的领域，存在 SQLite 里 |
    | 故障处理 | 重试、压缩、错误即消息 | 先存的效果、退避、重新排队、显式阶段 |
    | 与人沟通 | 那就是你的终端 | 只在有人必须行动或能用上结果时开口；群聊里最多 100 个词 |
    | 基础 | 自己的各个包 | DeepSeek Harness，作为插件包 |
    | 状态 | 产品，每天都在变 | 研究原型，这是它对自己的描述 |

    每个项目底下都有一个核心想法。**Pi 信任人**，所以它保持小巧，把每根操纵杆都交给人。**Super-agent 保护团队的注意力**，所以只在被允许时行动，只在必要时开口。再看看它们的共同点：失败也是数据，调用和结果必须配对，重要的东西不能只活在提示词里，输出必须有界，上下文是稀缺资源。

    ## 考试

    写 `exam()`，返回一个键为 `"q1"` 到 `"q12"`、值为小写字母的字典，比如 `return {"q1": "a", "q2": "b"}`，一直写到 `"q12"`。结果每题一行，只告诉你这一题对不对。括号里是章号。

    1. （31）一条回复有三个工具，第一个运行时来了一条转向消息。它何时到达模型？**a)** 立刻，另外两个被取消。**b)** 三个都运行完之后，下一次调用模型之前。**c)** 智能体完全结束之后。
    2. （31）为什么 Pi 让 `length` 停止的回复里所有工具调用失败？**a)** 服务商宕机了。**b)** 为了省词元。**c)** 修复过的参数可能通过校验却不完整。
    3. （32）Pi 的流函数怎样报告网络故障？**a)** 用一条 `stopReason` 为 `error` 的最终消息。**b)** 抛出异常。**c)** 返回 `None`。
    4. （33）为什么 `bash` 保留输出结尾，而 `read` 保留开头？**a)** 结尾更短。**b)** 错误通常在末尾，而文件用 `offset` 从头读。**c)** 碰巧如此。
    5. （34）Pi 会话管理器的 `branch(id)` 做什么？**a)** 复制文件。**b)** 删除之后的记录。**c)** 移动叶子指针。
    6. （35）压缩的切点为什么不能落在工具结果上？**a)** 保留部分会以调用已消失的结果开头，服务商会拒绝。**b)** 工具结果很小。**c)** 它们无法总结。
    7. （37）在 Super-agent 里，Router 做什么？**a)** 为每个任务写代码。**b)** 把一个大任务拆给一个团队。**c)** 读进来的东西，判断哪些是新工作并派发；工作由 Worker 来做。
    8. （38）在 `observe` 模式下，模型对一个推断候选说“accept”。会发生什么？**a)** 启动一个工单。**b)** 记为“会接单”，什么都不启动。**c)** 消息被删除。
    9. （39）有人在一个活着的工单的拉取请求上留言。它去哪里？**a)** 交给那个工单的 Worker，由它处理或交回；不经过评估。**b)** 交给 Router 做一次新的评估。**c)** 直到有人 @ 智能体之前，哪里都不去。
    10. （40）为什么为人的消息而唤醒时，可以超出上限多用一个位置？**a)** 人付钱更多。**b)** 这是个 bug。**c)** 让新工作永远挡不住正在继续现有工作的人。
    11. （41）为什么策略部分放在 Router 提示词的最后？**a)** 它最不重要。**b)** 它经常变，放在最后能让前面的文字保持不变，便于前缀缓存。**c)** 模型先读结尾。
    12. （40）在原型里，禁用命令列表和命令包装器是安全边界吗？**a)** 不是：它们只设定默认值，Worker 仍然可以用绝对路径运行程序。**b)** 是，完全是。**c)** 只在 Linux 上是。
    """)
    return


@app.function
def exam():
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "exam", exam, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 你算精通了吗？

    如果你做完了每个练习，你就懂了两套架构、它们为什么这样设计，也知道该去代码的哪里找。这是最难的部分，但还不是*经验*。想名副其实地说“精通”，请用真实模型完成下面这些。它们要花一两个周末，也正是面试官或维护者会问的内容：

    1. **Pi：** 安装它，做一个真实任务，用 `/tree` 换一种做法，用 `/compact` 观察摘要。
    2. **Pi：** 用 `pi --mode json` 运行一个任务，把每个事件对应到 `agent-loop.ts` 的某一行。
    3. **Pi：** 写一个阻止 `rm -rf` 并添加一个工具的扩展。然后读 `examples/extensions/plan-mode`。
    4. **Super-agent：** 克隆原型，运行 `pnpm check`，为第 38、39 章的每条规则在 `tests/` 里读一个测试。
    5. **Super-agent：** 安装 DeepSeek Harness，按 `SETUP.md` 在一个用完即弃的仓库上配置原型，接单模式保持 `observe`，读一周它记录下的决定。
    6. **Super-agent：** 改一个准入格子，预测哪些打开的工单会收到策略通知，然后验证。
    7. **两者：** 给它们同一个小 bug，比较词元、时间，以及它们占用了你多少注意力。

    ## 口试题

    大声回答，然后打开提纲。

    /// details | 讲讲一条请 Super-agent 修错别字的 GitHub 评论，从头到尾经历了什么。
    GitHub 来源把它变成该 Issue 线程上的一个触发，`Intake.receive` 只存一次。如果没有活着的工单跟着这个线程，它就排队等评估。准入关看到名单内的发送者 @ 了智能体，所以是明确请求。Router 收到卡片，用自己的路由工具 `orchestrator_route` 接单，在一个事务里创建工单、授权、派工、订阅和确认回复。效果队列在独立目录里启动一个 Worker，它确认一次、修改、开拉取请求并报告 `ready`；它开的拉取请求关闭后，工单结束。
    ///

    /// details | 为什么 Super-agent 比单会话加子智能体用的词元更多？为什么这可以接受？
    每个 Worker 读自己的上下文，Router 读所有进来的东西，所以缓存读取会累积：文章的测试里每次运行 23.23M，对比 6.05M 或 10.82M。换来的是更快、稳定得多的完成时间，也没有哪个会话被每条新消息打断。直接送达让成本不再继续上涨；没有它，缓存读取增加约 37%。
    ///

    /// details | 你要为主动工作加上“永远不要碰支付模块”，放在哪里？为什么？
    放在允许这类主动工作的那条规则的说明里，或者放在工单范围部分。匹配到的规则说明会送到那个工单的 Worker。这不是策略里允许名单的事，名单决定谁能启动工作，而不是工作能碰什么。而且 Worker 仍然什么都能运行，所以这只是引导，不是边界。
    ///

    /// details | 什么时候选 Pi，什么时候选 Super-agent 式的设计？
    Pi 适合想要掌控和可见性的单个专家，或作为嵌入产品的引擎。Super-agent 适合有源源不断的小型、彼此松散相关请求的团队，那里人的注意力是瓶颈。两者的风险不同。用 Pi，一切以你的权限、在你眼皮底下运行。用 Super-agent，智能体以团队账户公开行动，所以需要认真设置名单、规则和提示词，而且它还不是生产级软件。
    ///

    ## 你完成了这条路线

    你从一个 12 行的循环出发，现在已经重建了两个真实智能体的工作部件：一个让人能做得更多，一个尽量少向人索取。你能读懂任一代码库，解释它为什么这样写，并和它争论。

    ## 现在你能做到

    - 说出 Super-agent 提示词的组装顺序，并解释策略为什么放在最后
    - 区分提示词控制的东西和代码强制的东西
    - 解释内联与溢出、卡片和回忆链
    - 在每个主要决定上比较 Pi 和 Super-agent，并说出它们的共同点
    - 规划把理解变成经验的实践工作

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** 读原型的 `prompts/orchestrator.md` 和 `prompts/worker.md`，看上面这些原则如何写成智能体读的文字；读 `docs/005-panel.md` 了解面板。两个项目都变化很快：用 `python -m learn_llm.source_check` 对照新提交重新检查这些摘录。
    """)
    return


if __name__ == "__main__":
    app.run()
