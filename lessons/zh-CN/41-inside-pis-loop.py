import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="41 · Pi 的循环内部")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import FakeModel, ask, call, result, say, user, watch

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 41 · Pi 的循环内部

    第 40 章的循环只有 12 行。Pi 的循环大约 170 行，在 `packages/agent/src/agent-loop.ts` 里，而核心就是你已经熟悉的形状。这一章你要学会读它，并弄清多出来的那些行是做什么的。

    先看地图。Pi 是一个 **monorepo**：一个代码库里有好几个包，彼此层层依赖。

    | 包 | 它是什么 |
    |---|---|
    | `ai` | 用同一种方式对接所有模型服务商（第 42 章） |
    | `agent` | 循环本身，一个不到 1,000 行的文件，对文件和终端一无所知 |
    | `coding-agent` | `pi` 命令：工具、会话、压缩、扩展（第 43 到 46 章） |
    | `tui` | 终端界面库 |
    | `mcp`、`codemode`、`durable`、`chord`、`schema`、`env`、`evals`、`telemetry` | 较新的积木（第 46 章） |

    这种划分是关于 Pi 要记住的第一件事：`agent` 是一个不带立场的小库，让它成为编程工具的一切都是从外面插进去的。

    ## 工具箱：读 TypeScript

    Pi 用 **TypeScript** 编写，也就是带类型的 JavaScript。你只需要读得懂。几乎所有写法都能直接对应到 Python：

    | TypeScript | Python | 含义 |
    |---|---|---|
    | `const x = 1`、`let y = 2` | `x = 1` | `const` 不再改变，`let` 可以改变 |
    | `x: string` | | 类型标注，不影响运行 |
    | `(a, b) => a + b` | `lambda a, b: a + b` | 一个小函数 |
    | `x?.y` | | `x.y`；如果 `x` 不存在，就什么也没有 |
    | `a ?? b` | | 取 `a`；如果它不存在，取 `b` |
    | `{ ...a, b: 1 }` | | 复制字典 `a` 并改一个字段 |
    | `[...a, b]` | | 复制列表 `a` 并多加一项 |
    | `a \|\| b` | `a or b` | 取 `a`；如果它是空的，取 `b` |
    | `list.push(x)` | `list.append(x)` | 加到末尾 |
    | `list.filter((c) => test)` | `[c for c in list if test]` | 保留一部分 |
    | `` `Tool ${name}` `` | `f"Tool {name}"` | 嵌入值的文字 |
    | `===` | `==` | 相等 |
    | `async`、`await` | | “这里可能要等；等的时候让别的工作继续” |

    最后一行很重要：一次模型调用要几秒钟，`await` 等它的时候不会冻住其他一切。读到 `await f()` 时，当成 `f()` 就好。

    表里的 `lambda` 是 Python 写无名小函数的方式：`lambda a, b: a + b` 相当于只有一行 `return` 的 `def`。另外，Python 的 `x or y` 表示两个条件有一个成立即可，相当于 TypeScript 的 `||`。

    ## 两层循环

    下面是 Pi 的 `runLoop` 的核心，细节已经删去（`…`）。慢慢读。你会认出第 40 章的循环就是*内层*的 `while`。

    <!-- source pi packages/agent/src/agent-loop.ts -->
    ```ts
    let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];
    …
    while (true) {
      let hasMoreToolCalls = true;
    …
      while (hasMoreToolCalls || pendingMessages.length > 0) {
    …
        const message = await streamAssistantResponse(currentContext, config, signal, emit, streamFunction);
        newMessages.push(message);
    …
        if (message.stopReason === "error" || message.stopReason === "aborted") {
    …
          await emit({ type: "agent_end", messages: newMessages });
          return;
        }
    …
        const toolCalls = message.content.filter((c) => c.type === "toolCall");
    …
        hasMoreToolCalls = false;
        if (toolCalls.length > 0) {
    …
        pendingMessages = (await config.getSteeringMessages?.()) || [];
    …
        }
    …
      const followUpMessages = (await config.getFollowUpMessages?.()) || [];
      if (followUpMessages.length > 0) {
    …
        pendingMessages = followUpMessages;
        continue;
      }
    …
      break;
    }
    ```

    读这段代码要注意三点。`(await config.getSteeringMessages?.()) || []` 的意思是“如果有这个钩子就调用它，什么都没得到就用空列表”。`hasMoreToolCalls` 在 `if` 里被重新设为真，那一行被我们删去了。`continue` 和 `break` 的作用和 Python 一样。

    为什么是两层？因为第 40 章忽略了一个问题：**智能体工作时，用户可以继续打字**，而用户的意思可能有两种。

    - **转向（steering）：** “停，文件错了！”这应该尽快生效，前提是安全。
    - **后续（follow-up）：** “做完之后跑一下测试。”这应该等智能体做完。

    所以内层循环在有工具调用*或*转向消息时运行，外层循环在内层跑空时取走后续消息。

    ## “尽快且安全”是什么时候？

    Pi 在**整条回复的工具都运行完之后**才取出转向消息，绝不在工具之间插入。如果一条回复请求了 `read` 和 `edit`，你在 `read` 时转向，`edit` 仍会运行。为什么不取消它？因为每个工具调用都必须有结果（第 40 章）。Pi 在这个钩子的注释里写得很直白：消息在下一次模型调用前加入，当前助手消息里的工具调用不会被跳过。

    队列默认一次只取一条。你打了三条消息，模型会一轮看到一条，这样它能逐条反应。下面的 `take_one` 做的就是这件事。

    ## 工具箱：list.pop

    `queue.pop(0)` 从列表里移除第一项并把它给你。`messages[-1]`（没有 pop）读取最后一项，但把它留在原处。

    ## 练习 1：take_one

    写 `take_one(queue)`：队列为空时返回 `[]`；否则返回只含最早一项的列表，并把这一项从队列移除。写成列表，循环就能用同一种方式处理“没有”和“有”。
    """)
    return


@app.function
def take_one(queue):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "take_one", take_one, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 退出规则

    源码里还有三个细节，每个都是真实故障留下的疤。

    **错误立即结束运行。** 如果回复的 `stopReason` 是 `"error"` 或 `"aborted"`，循环直接返回，不看队列。是否重试由上一层的 harness 决定（第 45 章）。

    **不能从助手消息继续。** Pi 可以用 `agentLoopContinue` 接着跑一次中断的运行，但最后一条消息必须是模型能回答的：

    <!-- source pi packages/agent/src/agent-loop.ts -->
    ```ts
    if (context.messages[context.messages.length - 1].role === "assistant") {
      throw new Error("Cannot continue from message role: assistant");
    }
    ```

    ## 练习 2：can_continue

    写 `can_continue(messages)`：空列表返回 `False`；最后一条消息的角色是 `"assistant"` 时返回 `False`；否则返回 `True`。
    """)
    return


@app.function
def can_continue(messages):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "can_continue", can_continue, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **被截断的回复可能说谎。** 回复可能写到一半就碰到输出上限，`stopReason` 为 `"length"`，其中的工具调用也许不完整。危险很隐蔽：Pi 会尽力修复流里残缺的 JSON，所以一个被截断的 `write` 调用可能解析和校验都完全通过，却只带着半个文件。运行它会悄悄毁掉数据。因此 Pi 不运行这条回复里的任何工具：

    <!-- source pi packages/agent/src/agent-loop.ts -->
    ```ts
    // A "length" stop means the output was cut off by the token limit, so
    // every tool call in the message may carry truncated arguments. Fail
    // them all instead of executing potentially borked calls.
    const executedToolBatch =
      message.stopReason === "length"
        ? await failToolCallsFromTruncatedMessage(toolCalls, emit)
        : await executeToolCalls(currentContext, message, config, signal, emit);
    ```

    ## 练习 3：truncated_result

    写 `truncated_result(call)`，为这样的调用生成错误结果。返回 `result(call["id"], call["name"], text, True)`，其中 `text` 如下（是 Pi 原文的简短版；`\"` 表示字符串里的引号）：

    ```
    f"Tool call \"{call['name']}\" was not executed: the response hit the output token limit."
    ```
    """)
    return


@app.function
def truncated_result(call):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "truncated_result", truncated_result, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：run_loop

    现在把它们合起来。写 `run_loop(model, messages, run_call, steering, follow_ups)`。`run_call(call)` 返回一条结果消息；`steering` 和 `follow_ups` 是当作队列用的列表；它修改 `messages` 并把它返回。

    1. `pending = take_one(steering)`，然后 `while True:`
    2. 里面：`more_calls = True`，然后 `while more_calls or len(pending) > 0:`
    3. 再里面：把 `pending` 里的消息全部追加到 `messages`，然后 `pending = []`
    4. `reply = model(messages)`，追加它
    5. 如果 `reply["stopReason"]` 是 `"error"` 或 `"aborted"`，`return messages`
    6. `calls = [block for block in reply["content"] if block["type"] == "toolCall"]`，和第 40 章一样；`more_calls = len(calls) > 0`
    7. 对每个调用：如果停止原因是 `"length"`，追加 `truncated_result(call)`，否则追加 `run_call(call)`
    8. `pending = take_one(steering)`（内层循环到此结束）
    9. 回到外层循环：`extra = take_one(follow_ups)`。如果为空，`return messages`；否则 `pending = extra`
    """)
    return


@app.function
def run_loop(model, messages, run_call, steering, follow_ups):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("41", "run_loop", run_loop, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面这次运行里，有一条后续消息在等着，还有一条转向消息在第一个工具运行时到达。`run_loop` 通过之后，你应该看到转向消息紧跟在工具结果之后，而后续消息要等模型回复了转向消息才出现：
    """)
    return


@app.function
def demo_steering():
    steering = []

    def run_call(call):
        steering.append(user("Actually, skip the tests."))
        return result(call["id"], call["name"], "42 files")

    model = FakeModel([ask(call("ls")), say("Skipping them."), say("Committed.")])
    return run_loop(model, [user("Run the tests")], run_call, steering, [user("Then commit")])


@app.cell
def _():
    watch(demo_steering)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 钩子：小循环如何保持小

    权限、上下文裁剪、成本上限怎么办？这些都不在循环里。`AgentLoopConfig` 提供了一些可以插入代码的位置：

    | 钩子 | 什么时候运行 | 用来做什么 |
    |---|---|---|
    | `transformContext` | 每次调用模型前 | 裁剪或添加上下文 |
    | `convertToLlm` | 每次调用模型前 | 把应用专用消息转成模型能懂的消息 |
    | `prepareRequest` | 每次调用模型前 | 为这次请求换模型或换上下文 |
    | `beforeToolCall` | 参数校验后、工具运行前 | 拦截调用 |
    | `afterToolCall` | 工具运行后 | 改写结果 |
    | `finishTurn`、`prepareNextTurn` | 一轮结束前后 | 提前结束；在两轮之间压缩 |
    | `getSteeringMessages`、`getFollowUpMessages` | 两个取出点 | 你刚用过的队列 |

    注意 `convertToLlm`。循环处理的是 `AgentMessage`，它比模型能懂的范围更宽：应用可以在记录里保存自己的消息类型（一条界面通知、一条用户执行的 shell 命令），到最后一刻才转成模型消息。这一行很小，影响很大：对话记录属于应用，而不属于服务商。

    循环运行时还会**发出事件**：`agent_start`、`turn_start`、`message_start`、`message_update`、`message_end`、`tool_execution_start`、`tool_execution_update`、`tool_execution_end`、`turn_end`、`agent_end`。它自己不画任何东西。终端界面、JSON 模式、RPC 模式都只是监听者，所以它们的行为完全一致。

    ## 现在你能做到

    - 说出 Pi 的各个包，以及每个包负责什么
    - 读懂 TypeScript，能跟上这个循环
    - 解释循环为什么有两层，以及转向和后续的区别
    - 准确说出转向消息何时生效，以及为什么不能更早
    - 解释为什么 `length` 停止会让回复里所有工具调用失败
    - 列出循环的钩子及其用途

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** 通读 `packages/agent/src/agent-loop.ts`，再读 `agent.ts`，它把循环包进一个带两个队列的有状态类。搜索 `declareToolChanges`（第 42 章）和 `executeToolCallsParallel`（第 43 章）。
    """)
    return


if __name__ == "__main__":
    app.run()
