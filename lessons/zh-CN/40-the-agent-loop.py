import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="40 · 智能体循环")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import FakeModel, ask, call, result, say, trace, user, watch

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 40 · 智能体循环

    从这一章开始，是主线课程之外的一条选修路线，主题是**智能体**（agent）。它只有一个目标：学完之后，你能把两个真实的开源智能体 **Pi** 和 **DeepSeek 的 Super-agent** 弄懂到源码层面，能讲清楚、能改动，也能为它们的设计辩护。

    你需要学完第 19 章的 Python（`while`、字典、类）。不需要 GPU，不需要 API 密钥，也不需要联网：模型由一个写好剧本的替身扮演，所以每个练习都在这个页面上运行，免费，而且每次结果都一样。

    每一章的做法相同：先读页面上引用的真实代码，再用 Python 亲手重建这个机制，由检查来打分。亲手重建，才能把“我读过”变成“我知道它为什么这样写”。

    ## 从模型到智能体

    你在主线课程里训练的所有模型都只做一件事：给一段文字，生成更多文字。聊天模型就是一个函数：

    ```
    reply = model(messages)
    ```

    它打不开文件，也运行不了代码。两次调用之间它什么也记不住：它看起来有记忆，只是因为我们每次都把整段对话重新发一遍。

    那模型是怎么“修 bug”的？窍门很小。我们告诉模型有哪些**工具**（tool），它的回复可以不是答案，而是一个请求：“请对 `main.py` 运行 `read`。”模型自己什么也不运行，是*我们的程序*运行工具，把结果加进对话，再调用一次模型。如此反复，直到模型不再请求工具、直接给出回答。

    这就是**智能体**：一个模型、一些工具、一个循环。模型外面的那层程序，也就是运行循环和它所需一切的部分，叫做 **harness**（运行底座）。Pi 自称“minimal, extensible agent harness”，而 DeepSeek 的 Super-agent 是搭在一个 harness（DeepSeek Harness）上的插件包。学到最后你会看到，它们是对同一个问题的两种截然不同的回答。

    ## 消息

    一段对话就是一个字典列表，这里用 Pi 的格式。每条消息都有 `role`，取值是 `"user"`、`"assistant"` 或 `"toolResult"`。下面是一段短对话。`trace` 是一个辅助函数，把每条消息显示成一行；`user`、`ask`、`call`、`result` 和 `say` 帮我们构造消息：
    """)
    return


@app.cell
def _():
    trace([
        user("What's 2 + 3?"),
        ask(call("add", {"x": 2, "y": 3}, id="call_1")),
        result("call_1", "add", "5"),
        say("It's 5."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是第二条消息的完整样子。助手消息里有一个**块**（block）列表，还有一个 `stopReason`，说明模型为什么停下：`"toolUse"` 表示“我在等工具”，`"stop"` 表示“我说完了”。
    """)
    return


@app.cell
def _():
    ask(call("add", {"x": 2, "y": 3}, id="call_1"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    注意 `id`。模型可以一次请求多个工具，每个结果通过 `toolCallId` 指明自己回答的是哪个调用。如果某个调用没有对应的结果，真实的模型服务商会拒绝这段对话，这在后面几章会给我们带来真正的麻烦。

    下面是模型替身。`FakeModel` 按顺序回放你给它的回复，并记住它看到过什么。
    """)
    return


@app.cell
def _():
    _model = FakeModel([ask(call("add", {"x": 2, "y": 3})), say("It's 5.")])
    _first = _model([user("What's 2 + 3?")])
    _first["stopReason"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：tool_calls

    写 `tool_calls(message)`，按顺序给出助手消息里所有 `toolCall` 块。这是一个末尾带 `if` 的列表推导式（第 12 章）：

    ```
    [block for block in message["content"] if block["type"] == "toolCall"]
    ```
    """)
    return


@app.function
def tool_calls(message):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("40", "tool_calls", tool_calls, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 工具箱：出错的时候

    工具可能失败：文件不存在、命令崩溃。Python 用**抛出异常**（exception）来表示失败，通常程序会停下并显示红色错误框。要接住异常，这样写：

    ```
    try:
        risky()
    except Exception as error:
        print(str(error))
    ```

    `try:` 下面的代码照常运行。如果其中一行抛出异常，Python 就跳到 `except`，`error` 里存着发生了什么，`str(error)` 是它的文字说明。要自己抛出异常，写 `raise ValueError("出了什么问题")`。

    ## 工具箱：嵌入值的文字

    在引号前加 `f`，`{ }` 里的内容就会被替换成它的值：当 `name = "read"` 时，`f"Tool {name} not found"` 得到 `Tool read not found`。在里面访问字典键时，要用另一种引号：`f"{call['name']}"`。

    ## 规则一：失败也是信息

    工具失败时，智能体**不能**崩溃。它要把失败当作普通结果交给模型，并标上 `isError`，因为模型往往能自己修好：读错误、改路径、再试一次。这是智能体代码和普通代码最大的不同：错误是*拿给模型看的*东西。

    不存在的工具也一样。模型有时会编造工具名。Pi 是这样做的：

    <!-- source pi packages/agent/src/agent-loop.ts -->
    ```ts
    const tool = tools.find((t) => t.name === toolCall.name);
    if (!tool) {
    …
        result: createErrorToolResult(`Tool ${toolCall.name} not found`),
        isError: true,
    ```

    这是 TypeScript，第 41 章会正式讲。现在你可以这样读：如果没有这个名字的工具，就生成一条错误结果。`…` 表示删去了几行。

    ## 练习 2：run_tool

    `tools` 是一个字典，从工具名映射到函数，每个函数接收参数字典，返回文字。写 `run_tool(tools, call)`，它永远返回一条结果消息：

    1. `name = call["name"]`。如果 `name` 不在 `tools` 里，返回 `result(call["id"], name, f"Tool {name} not found", True)`
    2. 否则在 `try:` 里写 `text = tools[name](call["arguments"])`。`tools[name]` 是一个函数，所以第二对括号就是调用它
    3. 如果抛出异常，返回一条装着 `str(error)` 的结果，末尾带 `True`
    4. 如果成功，返回 `result(call["id"], name, text)`
    """)
    return


@app.function
def run_tool(tools, call):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("40", "run_tool", run_tool, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 循环

    现在把整件事用大白话说一遍：

    1. 把对话给模型看，把它的回复加进对话。
    2. 如果回复里没有工具调用，停止。
    3. 运行每个工具调用，把每个结果加进对话。
    4. 回到第 1 步。

    模型可能永远循环下去，所以运行底座还需要一个上限。每绕一圈叫一**轮**（turn）。

    ## 练习 3：agent_loop

    写 `agent_loop(model, messages, tools, max_turns)`。它就地修改 `messages` 并把它返回。

    1. `turns = 0`，然后 `while turns < max_turns:`
    2. 循环里：`reply = model(messages)`，然后 `messages.append(reply)`，然后 `turns = turns + 1`
    3. `calls = tool_calls(reply)`；如果 `len(calls) == 0`，就 `return messages`
    4. `for call in calls:` 追加 `run_tool(tools, call)`
    5. 循环结束后，`return messages`
    """)
    return


@app.function
def agent_loop(model, messages, tools, max_turns):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("40", "agent_loop", agent_loop, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    看它工作。下面的 `read_file` 是一个只认识一个文件的工具，`watch` 运行你的循环并显示 trace。（练习 3 通过之前，这个单元格只会提示你。）模型先失败，读了错误，然后重试：
    """)
    return


@app.function
def read_file(arguments):
    if arguments["path"] != "notes.txt":
        raise ValueError(f"No such file: {arguments['path']}")
    return "buy milk"


@app.cell
def _():
    _model = FakeModel([
        ask(call("read", {"path": "note.txt"})),
        ask(call("read", {"path": "notes.txt"})),
        say("Your note says: buy milk."),
    ])
    watch(agent_loop, _model, [user("What's in my notes?")], {"read": read_file}, 10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    你刚刚造出了一个智能体，里面没有任何魔法。这条路线剩下的内容，都是认真对待这个循环之后才会遇到的事。

    ## 为什么模型的输入越来越大

    模型什么都不记得，所以每一轮都要把整个列表再发一遍。如果每轮增加的文字差不多，你发送的*总量*就随轮数的平方增长：第 1 轮发 1 份，第 2 轮发 2 份，第 50 轮发 50 份，跑 50 轮一共发 1,275 份。服务商全部都要收费，而模型一次能读的量也有限（这叫**上下文窗口**，context window）。
    """)
    return


@app.cell
def _():
    _turns = 50
    sum(range(1, _turns + 1))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 还剩哪些问题

    你这 12 行循环够做演示，真用起来会散架。下面列出这些问题，以及它们在哪里解决。Pi 是对一个循环的深入研究；Super-agent 研究的是许多循环如何为整个团队工作。

    | 问题 | 章 | 项目 |
    |---|---|---|
    | 用户想打断；回复被截断 | 31 | Pi |
    | 几十家模型服务商，格式各不相同；回复一点点到达 | 32 | Pi |
    | 工具返回一百万行；两个工具改同一个文件 | 33 | Pi |
    | 崩溃丢了对话；想换一条路再试 | 34 | Pi |
    | 对话超出上下文窗口 | 35 | Pi |
    | 每个人想要的行为都不一样 | 36 | Pi |
    | 一个人同时只能盯住几个智能体 | 37 | Super-agent |
    | 群里谁都能开口；谁有权启动工作？ | 38 | Super-agent |
    | 后来的消息要送到对应的工作，不必重新请求 | 39 | Super-agent |
    | 许多工作者、重启，什么都不能丢 | 40 | Super-agent |
    | 提示词、上下文和与人沟通 | 41 | Super-agent |

    ## 现在你能做到

    - 说清智能体、工具调用和 harness 分别是什么
    - 读懂消息格式，并解释 id 的作用
    - 解释为什么工具失败要交回给模型，而不是让程序停下
    - 凭记忆写出带轮数上限的智能体循环
    - 解释为什么长任务的成本比长度增长得更快

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** 这条路线引用的代码固定在 Pi 的提交 `6291b9b`（2026-11-01）和 DeepSeek `dsh-superagent-prototype` 的提交 `8c32a9d`（2026-10-31）。两个项目每天都在变，新版本可能不同：见 [badlogic/pi-mono](https://github.com/badlogic/pi-mono) 和 [deepseek-ai/dsh-superagent-prototype](https://github.com/deepseek-ai/dsh-superagent-prototype)。摘录按 MIT 许可引用，许可声明见 `learn_llm/data/AGENT-SOURCES.md`。
    """)
    return


if __name__ == "__main__":
    app.run()
