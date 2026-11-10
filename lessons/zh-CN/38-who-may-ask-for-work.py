import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="38 · 谁可以请求工作")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 38 · 谁可以请求工作

    Pi 信任坐在键盘前的那一个人（第 36 章）。Super-agent 读的是团队的群聊和 GitHub，那里谁都能输入任何东西，包括一段写着“忽略你的指令”的网页。所以在任何工作开始之前，原型用代码而不是提示词回答两个问题：**谁可以在这里启动工作**，以及**这件工作可以走多远**。

    它的权限说明文档开篇就是其余一切遵循的规则：

    <!-- source dsh docs/002-authorization.md -->
    ```text
    The host enforces allowed lists, rules and Mandates. Messages, comments and quoted documents are inputs to a decision; they cannot change those permissions.
    ```

    也就是说：名单、规则和授权（Mandate）由宿主强制执行；消息、评论和引用的文档只是决策的输入，改变不了这些权限。

    ## 两种强度

    每个进来的请求都有一个**强度**（strength）：

    - **明确**（explicit）：团队允许的人直接请智能体做事
    - **推断**（inferred）：其他可能是请求的消息，比如名单内的人顺口提到错别字却没有 @ 智能体，或者名单外的人发的后续消息

    明确请求永远有效。推断请求取决于团队设定的**接单模式**（accept mode）：

    | 模式 | 推断候选 |
    |---|---|
    | `off` | 根本不考虑 |
    | `observe`（默认） | 会研判，但“是”只*记录*为“会接单” |
    | `auto` | “是”就开始工作 |

    这个默认值就是注意力原则（第 37 章）的体现。文档建议从 `observe` 开始，读一读智能体*本来会*做什么，再切换到 `auto`。

    ## 准入关

    评估的第一层是**准入**（gate）：确定性的代码，不用模型。发送者是否在允许名单里，以及消息是否**指向**智能体（@ 提及或私信），决定了团队**准入表**（admission table）里的一个格子：

    <!-- source dsh src/policy.ts -->
    ```ts
    function admissionCell(allowed: boolean, addressed: boolean): AdmissionCell | undefined {
      return allowed ? addressed ? undefined : 'listedUnaddressed' : addressed ? 'unlistedAddressed' : 'unlistedUnaddressed'
    }
    ```

    读法提示：`a ? b : c` 是 TypeScript 版的 Python `b if a else c`，这里嵌套了两层。`undefined` 就是 TypeScript 的 `None`。`undefined` 那种情况没有格子，因为名单内的人 @ 智能体*永远*是明确请求。

    ## 练习 1：admission_cell

    照那一行写 `admission_cell(listed, addressed)`：名单内且 @ 返回 `None`，只在名单内返回 `"listedUnaddressed"`，只 @ 了返回 `"unlistedAddressed"`，两者都不是返回 `"unlistedUnaddressed"`。
    """)
    return


@app.function
def admission_cell(listed, addressed):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("38", "admission_cell", admission_cell, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    团队把每个格子设为 `explicit`、`inferred` 或 `ignore`。接下来是准入本身，针对人发的消息。下面是 `src/policy.ts` 里 `gate` 的顺序。读法提示：`&&` 是 `and`，`||` 是 `or`，`===` 和 `!==` 是 `==` 和 `!=`，`!x` 是 `not x`，`…` 表示删去的行。

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

    用大白话说：

    1. 名单内的人 @ 智能体：**明确**。
    2. 表格格子写着明确（团队可以把一个支持群开放给所有人）：**明确**。
    3. **后续消息**，也就是已有工单的线程上的消息（`participating`）：如果 `follow_up and (listed or mode != "off")`，名单内的人算明确，其他人算推断。
    4. 格子写着推断，且模式不是 `off`：**推断**。
    5. 其他情况：不准入，并记录原因。

    顺序很重要。先判断明确格子、再判断后续消息，意味着线程上已有的工作永远不会*削弱*一个明确请求。机器人走自己的路：只有团队名单上的机器人才可能产生推断候选，而且永远产生不了明确请求。

    ## 练习 2：gate

    写 `gate(listed, addressed, admission, follow_up, mode)`，返回 `"explicit"`、`"inferred"` 或 `"not admitted"`。`admission` 是发送者所在格子的值（`"explicit"`、`"inferred"` 或 `"ignore"`），`follow_up` 表示这个线程是否已有工单。照上面第 1 到 5 步写。我们省略私信规则，以及“这个范围是否接收所有消息”的检查。
    """)
    return


@app.function
def gate(listed, addressed, admission, follow_up, mode):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("38", "gate", gate, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 研判

    通过准入的候选会以紧凑**卡片**（card，带 id 的简短摘要，第 41 章）的形式交给 Router。明确候选直接用 Router 的路由工具 `orchestrator_route` 派发；推断候选则要**研判**（assessment），而研判不是自由文字，而是填一个结构，再由代码转换成结果。团队为主动工作编写**规则**，默认的有“文案修正”和“视觉修正”，模型必须对*每条启用的规则恰好判断一次*：

    <!-- source dsh src/policy.ts -->
    ```ts
    if (assessment.decision !== 'accept') return { kind: assessment.decision, reason: assessment.reason, audit }
    if (!directionAllowed(assessment.direction)) return { kind: 'ignore', audit, reason: { code: 'directionUnsettled' } }
    if (!matches.length) return { kind: 'ignore', audit, reason: { code: 'noRuleMatched' } }
    if (!assessment.intent_evidence) throw new Error('Acceptance requires intent_evidence')
    …
    if (strength === 'inferred' && settings.eval.mode !== 'auto') {
    ```

    看看最后那个判断做了什么。模型的“是”*不足以*启动推断工作。在 `auto` 之外，代码把它降级为“会接单”。接单还需要方向已定、至少匹配一条规则，以及引用的证据，证明确实有人想要这件事。你在第 30 章见过“失败也是数据”，这里是更强的一条：**模型提议，代码定夺。**

    ## 练习 3：outcome

    照 `assessmentOutcome` 写 `outcome(decision, matched, strength, mode)`。`decision` 是模型给的结论，`matched` 是它匹配到的规则列表。（代码也接收强度，不过实际上只有推断候选才会被研判。）

    1. 如果 `decision != "accept"`，返回 `decision`
    2. 如果 `matched` 为空，返回 `"ignore"`
    3. 如果 `strength == "inferred"` 且 `mode != "auto"`，返回 `"would accept"`
    4. 否则返回 `"accept"`

    （这里省略方向和证据的检查。）
    """)
    return


@app.function
def outcome(decision, matched, strength, mode):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("38", "outcome", outcome, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 每个线程一次只做一个评估

    消息常常扎堆到来：“修一下按钮”“蓝色那个”“在登录页”。逐条判断会启动三件工作。所以**每个线程最多同时运行一个评估**，期间到达的触发一起等下一个，成为一**批**（batch）。文章测量了这对 Router 的好处：消息到达从每分钟约 1 条加快到约 40 条时，每批从 1.0 条增加到 2.8 条，每次决定的平均用时从 6.4 秒降到 1.6 秒。

    评估还有截止时间，从卡片*到达* Router 时开始计算，所以排队的时间不算。超时的评估不会立刻作废：

    <!-- source dsh src/core/evals.ts -->
    ```ts
    private expire(thread: Thread): void {
      const expired = (thread.expired ?? 0) + 1
      if (expired > REQUEUES) { this.finish(thread, thread.evaluating!, { kind: 'unavailable', reason: [{ code: 'timeout' }] }); return }
    ```

    `REQUEUES` 是 2。忙碌的 Router 还有两次机会，第三次超时记为 `unavailable`，面板可以把它显示给人看。什么都不会悄悄丢失。

    ## 练习 4：on_deadline

    写 `on_deadline(expired)`，`expired` 是这个线程的评估到目前为止连续超时的次数。加 1，如果大于 2 就返回 `"unavailable"`，否则返回 `"requeue"`。
    """)
    return


@app.function
def on_deadline(expired):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("38", "on_deadline", on_deadline, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 授权

    接单时会同时创建工单和**授权**（Mandate）：接单时所依据的团队规则（**策略**，Policy）的版本、范围（允许做哪些工作）、强度、匹配的规则、发送者，以及来源是怎么收到这条消息的。它永不改变。范围更大就意味着新工单。已结束的工作恢复之前，它会被重新校验：比如团队已经关掉了某个推断工单当初依据的规则，旧授权就不再有效（`src/policy.ts` 里的 `revalidate`，每种授权在那里都有自己的检查）。

    只有三件事能改变授权：接单创建它，人的停止撤回它，重新校验失败吊销它。Worker 不能声称有人让它停下，只有评估能决定这一点。

    ## 为什么放在代码里

    原型对自己的提示词说得很直白：修改提示词，不会改变准入关、授权、工具检查和账户包装，这些仍由代码强制执行。对比 Pi，它把权限完全交给你和你的容器。两者都没错，只是为不同场景而建：Pi 面向坐在键盘前的一位专家，Super-agent 面向有陌生人发言的团队群聊。

    ## 现在你能做到

    - 解释明确和推断两种强度、三种接单模式，以及 `observe` 为什么是默认值
    - 算出准入格子，并按顺序执行准入关
    - 解释为什么先判断明确格子、再判断后续消息
    - 把模型的研判转换成结果，并说明代码为什么要降级推断接单
    - 解释按线程分批、按送达时间计算的截止期限，以及两次重排后才记为 unavailable
    - 说出授权里有什么，以及它改变的三种方式

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** `docs/002-authorization.md` 完整讲了交回、恢复和重新校验，`src/policy.ts` 里有 `gate`、`revalidate` 和 `assessmentOutcome`。`prompts/admission.md` 是 Router 研判候选时读的内容。
    """)
    return


if __name__ == "__main__":
    app.run()
