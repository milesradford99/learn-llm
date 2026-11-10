import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="35 · 空间不够了")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import ask, attempt, call, result, say, user

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 35 · 空间不够了

    每个长时间运行的智能体最后都会遇到同一个问题：对话越来越长，直到放不进模型的上下文窗口。你不能简单丢掉旧消息，因为里面有智能体学到的东西和做过的决定。

    **压缩**（compaction）用模型写的摘要替换对话的旧部分。思路一句话就够，难点全在哪些消息不能分开。这一章研究 `packages/coding-agent/src/core/compaction/compaction.ts`，文件的第一条注释就点明了设计：“Pure functions for compaction logic. The session manager handles I/O”。

    ## 何时：廉价地数词元（token）

    每条消息都需要一个大小，而真正的分词器（第 17 章）每个模型都不同。所以 Pi 用一条经验法则：一个词元约等于 4 个字符：

    <!-- source pi packages/coding-agent/src/core/compaction/compaction.ts -->
    ```ts
      if (block.type === "text") {
        chars += block.text.length;
    …
      } else if (block.type === "toolCall") {
        chars += block.name.length + JSON.stringify(block.arguments).length;
      }
    }
    return Math.ceil(chars / 4);
    ```

    能用真实数字时，Pi 优先用真实数字：每条助手回复都会报告本次请求的 `usage`，所以到这条回复为止的上下文真实大小是已知的，只需要估算它之后的消息。（这是 `estimateContextTokens` 做的事。）

    ## 工具箱：向上取整

    `math.ceil(x)` 向上取整（需要 `import math`）：`math.ceil(1.25)` 是 2。`json.dumps(x, separators=(",", ":"))` 写出不带空格的 JSON，和 JavaScript 的 `JSON.stringify` 一致。

    ## 练习 1：estimate_tokens

    写 `estimate_tokens(message)`。把 `import json` 和 `import math` 写在函数内部的最前面。先数字符，再返回 `math.ceil(chars / 4)`。

    - 如果 `message["role"] == "assistant"`，遍历 `message["content"]` 并累加：`block["type"] == "text"` 时算 `len(block["text"])`，否则（工具调用）算 `len(block["name"]) + len(json.dumps(block["arguments"], separators=(",", ":")))`
    - 其他消息：`len(message["content"])`，这里用户消息和工具消息装的都是纯文字
    """)
    return


@app.function
def estimate_tokens(message):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("35", "estimate_tokens", estimate_tokens, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    判断本身只有一行。`reserveTokens` 为模型的下一条回复留出空间，默认 16,384，所以压缩在窗口完全写满之前就开始：

    <!-- source pi packages/coding-agent/src/core/compaction/compaction.ts -->
    ```ts
    export function shouldCompact(contextTokens: number, contextWindow: number, settings: CompactionSettings): boolean {
      if (!settings.enabled) return false;
      return contextTokens > contextWindow - settings.reserveTokens;
    }
    ```

    ## 练习 2：should_compact

    写 `should_compact(context_tokens, context_window, reserve)`。注意是严格的 `>`。
    """)
    return


@app.function
def should_compact(context_tokens, context_window, reserve):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("35", "should_compact", should_compact, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 在哪里切

    保留最近的部分（`keepRecentTokens`，默认 20,000），其余做成摘要。要找到分界线，就从最新的消息开始倒着走，累加大小，到预算时停下。

    但不能随便切。工具结果必须跟在工具调用后面，所以**切点绝不能落在工具结果上**：否则保留部分的开头就是一个问题已经消失的回答，服务商会拒绝（第 32 章）。

    <!-- source pi packages/coding-agent/src/core/compaction/compaction.ts -->
    ```ts
    // Walk backwards from newest, accumulating estimated message sizes
    let accumulatedTokens = 0;
    let cutIndex = cutPoints[0]; // Default: keep from first message (not header)

    for (let i = endIndex - 1; i >= startIndex; i--) {
    …
      accumulatedTokens += messageTokens;
    …
      if (accumulatedTokens >= keepRecentTokens) {
        // Prefer the closest valid cut point at or after this entry. If trailing
        // tool results exceed the budget by themselves, keep their preceding
        // assistant tool call instead of falling back to the first message.
        cutIndex = cutPoints.find((candidate) => candidate >= i) ?? cutPoints[cutPoints.length - 1];
        break;
      }
    }
    ```

    （`for (let i = endIndex - 1; i >= startIndex; i--)` 让 `i` 从最后一个下标倒数到第一个。）

    这条注释记录了一个修过的 bug。如果一个巨大的工具结果独自填满预算，它之后就没有合法切点了，而简单的兜底是第一条消息，也就等于根本没压缩。所以兜底改成了*最后一个*合法切点。

    ## 练习 3：find_cut_point

    写 `find_cut_point(messages, keep)`，给出第一条要保留的消息的下标。

    1. `cuts` = 所有 `messages[i]["role"]` 不是 `"toolResult"` 的下标 `i`
    2. `total = 0`，然后让 `i` 倒着遍历下标（`reversed(range(len(messages)))`），累加 `estimate_tokens(messages[i])`
    3. 如果 `total >= keep`（是 `>=`，不是 `>`）：`later` = 不小于 `i` 的切点。如果有，返回第一个；否则返回 `cuts[-1]`
    4. 循环结束后（始终没到预算），返回 `cuts[0]`
    """)
    return


@app.function
def find_cut_point(messages, keep):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("35", "find_cut_point", find_cut_point, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 能接着干活的摘要

    普通摘要不够用。下一个模型要接着工作，所以 Pi 要求固定格式，分为目标、约束、进度（已完成、进行中、受阻）、关键决定、下一步、关键上下文几部分：

    <!-- source pi packages/coding-agent/src/core/compaction/compaction.ts -->
    ```ts
    const SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.
    …
    Keep each section concise. Preserve exact file paths, function names, and error messages.`;
    ```

    同一个文件里还有这些选择：

    - **摘要层层累积。** 如果之前压缩过，Pi 不会从头总结摘要，而是把上一份发过去，要求更新：“PRESERVE all existing information”。
    - **文件列表由代码跟踪。** 读过和改过的文件从工具调用里提取，通过 `details` 跨压缩传递，并原样附上，不会在模型转述时丢失。
    - **在一轮中间切。** 一个用户请求可能产生一百次工具调用，所以切点可能落在一轮的中间。`isSplitTurn` 标记这种情况，Pi 会单独总结这一轮被切掉的前半部分。
    - **被截断的摘要算失败。** `getSummarizationFailure` 拒绝因长度停止的回复：半份检查点绝不能成为检查点。
    - **什么都不删。** 这就是第 34 章的日志。`compaction` 记录保存摘要和 `firstKeptEntryId`，投影从那里开始，原始消息仍在文件里。

    ## 失败的时候

    模型有两种不同的失败方式，harness 层分别处理（`agent-session.ts`）。这正是第 31 章循环干净退出的用意。

    **暂时性错误**（限流、服务器过载）会重试，默认最多 3 次，从 2 秒开始，每次等得更久，直到封顶：

    <!-- source pi packages/ai/src/utils/retry.ts -->
    ```ts
    const delay = policy.baseDelayMs * 2 ** Math.max(0, attempt - 1);
    ```

    **上下文溢出**原样重试还会失败，所以 `_isRetryableError` 把它排除在外，注释写着 “Context overflow is handled by compaction, not retry.” Pi 压缩后重试**一次**。如果再次溢出就停止，因为继续尝试可能永远循环：“Context overflow recovery failed after one compact-and-retry attempt.”

    失败的尝试留在日志里，但不进入模型看到的内容（`_omitRecoveryAttempt`），还是日志和投影分离的那一套。

    ## 练习 4：retry_delay

    写 `retry_delay(attempt, base, cap)`，用 `min`、`max` 和 `**`（第 00 章），照上面那行来写，结果不超过 `cap`。
    """)
    return


@app.function
def retry_delay(attempt, base, cap):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("35", "retry_delay", retry_delay, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：next_step

    写 `next_step(message, overflowed, compacted_already, retries, max_retries)`，给出 `"compact"`、`"retry"`、`"give up"` 或 `"done"` 之一：

    1. 如果 `overflowed`：`compacted_already` 时给 `"give up"`，否则给 `"compact"`
    2. 如果消息的 `stopReason` 是 `"error"`：`retries < max_retries` 时给 `"retry"`，否则给 `"give up"`
    3. 否则给 `"done"`
    """)
    return


@app.function
def next_step(message, overflowed, compacted_already, retries, max_retries):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("35", "next_step", next_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pi 为什么不用 `transformContext`

    你也许以为压缩放在循环的 `transformContext` 钩子里。并不是。压缩在会话层运行，发生在一次运行结束后或两轮之间，然后重新加载会话。所以压缩是日志里的一个持久事件，而不是每次请求临时制造的假象，磁盘上的文件和模型看到的内容始终一致。第 31 章的 `prepareNextTurn` 钩子部分就是为此而设：它的注释把压缩当作耗时准备工作的例子。

    记住这个设计：第 40 章里，Super-agent 同样先把要做的事存下来，再去做。

    ## 现在你能做到

    - 像 Pi 一样估算词元，并说出它何时改用真实数字
    - 解释预留空间和保留预算
    - 找到合法切点，并解释尾部结果那个 bug
    - 列出 Pi 的摘要好用的原因，以及文件列表为什么绕过模型
    - 解释溢出为什么只压缩一次而不是重试，以及压缩为什么不是每次请求的钩子

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** `packages/coding-agent/docs/compaction.md` 是很好的参考。然后读 `agent-session.ts` 里的 `_checkCompaction`，包括什么样的长度停止算可恢复。
    """)
    return


if __name__ == "__main__":
    app.run()
