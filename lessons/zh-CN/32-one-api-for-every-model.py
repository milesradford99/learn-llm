import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="32 · 一套接口对接所有模型")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import ask, attempt, call, result, say, user

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 32 · 一套接口对接所有模型

    在第 30、31 章里，模型是一个返回完整消息的函数。真实的模型既不完整，也不统一：回复是几个字几个字到达的，而且每家服务商都有自己的格式。Pi 的 `ai` 包把这两点都藏了起来，这也是 Pi 其余部分能保持小巧的原因之一。`packages/ai/src/api/` 里实现了 Anthropic、OpenAI、Google、Bedrock、Mistral 等家的接口。

    ## 为什么要流式？

    一条长回复可能要一分钟。如果等全部生成，屏幕一直空白，连接一断就什么都没了。所以服务商发送的是由小**事件**（event）组成的**流**（stream）。Pi 把每家服务商的流都转换成同一份事件列表：

    <!-- source pi packages/ai/src/types.ts -->
    ```ts
    export type AssistantMessageEvent =
      | { type: "start"; partial: AssistantMessage }
      | { type: "text_start"; contentIndex: number; partial: AssistantMessage }
      | { type: "text_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
      | { type: "text_end"; contentIndex: number; content: string; partial: AssistantMessage }
    …
      | { type: "toolcall_start"; contentIndex: number; partial: AssistantMessage }
      | { type: "toolcall_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
      | { type: "toolcall_end"; contentIndex: number; toolCall: ToolCall; partial: AssistantMessage }
    ```

    这个类型的读法是：一个事件是这些形状之一，靠 `type` 区分。**delta** 是一小段新内容，`contentIndex` 说明它属于哪个块，因为一条回复可以先有文字，再有工具调用，然后又有文字。思考块也是同样的做法。`partial` 是到目前为止的消息，监听者不必自己重建。

    在 Python 里，我们把事件写成字典，比如 `{"type": "text_delta", "contentIndex": 0, "delta": "Hel"}`。

    ## 练习 1：collect_text

    做监听者要做的事。写 `collect_text(events)`，给出一个字典，从 `contentIndex` 映射到目前收到的文字。从 `texts = {}` 开始。只看 `text_delta` 事件，令 `index = event["contentIndex"]`，像第 11 章一样用 `.get`：

    ```
    texts[index] = texts.get(index, '') + event["delta"]
    ```
    """)
    return


@app.function
def collect_text(events):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("32", "collect_text", collect_text, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 永不抛出异常的流

    流有两种结束方式：`{"type": "done", "message": ...}` 带着完整消息，或者 `{"type": "error", "error": ...}`，同样带着一条消息。第二点是关键的设计选择。这条规则就写在流函数类型的开头：

    <!-- source pi packages/agent/src/types.ts -->
    ```ts
    * Contract:
    * - Must not throw or return a rejected promise for request/model/runtime failures.
    * - Must return an AssistantMessageEventStream.
    * - Failures must be encoded in the returned stream via protocol events and a
    *   final AssistantMessage with stopReason "error" or "aborted" and errorMessage.
    ```

    （*promise* 是 TypeScript 里“稍后才到的结果”的说法。）现在你明白第 31 章的循环为什么只需检查 `stopReason` 了。网络故障不是一个穿透循环往外跳的异常，而是一条普通消息，可以保存、显示、重试。这是第 30 章规则一在模型身上的应用：**失败也是数据。**

    ## 练习 2：final_message

    写 `final_message(events)`，只看最后一个事件 `events[-1]`（列表不会为空）：`done` 返回它的 `"message"`，`error` 返回它的 `"error"`，否则（流被截断）返回 `None`。
    """)
    return


@app.function
def final_message(events):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("32", "final_message", final_message, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 同一条消息的三种写法

    这就是需要翻译层的原因。同一段交流，在 Pi 的存储格式和两家服务商格式里的样子：

    | | Pi | OpenAI chat | Anthropic |
    |---|---|---|---|
    | 助手文字 | `text` 块 | `content` 字段 | `text` 块 |
    | 工具调用 | `toolCall` 块 | `tool_calls` 列表，嵌套在 `function` 下，参数是 **JSON 文字** | `tool_use` 块，参数是对象 |
    | 工具结果 | 角色 `toolResult` | 角色 `tool` | **`user`** 消息里的 `tool_result` 块 |

    转换代码在每家服务商自己的文件里。在 `openai-completions.ts` 中，结果变成 `role: "tool"` 的消息；在 `anthropic-messages.ts` 中，变成 `tool_result` 块。Pi 存储自己的格式，到最后一刻才转换，所以一次会话可以中途换模型。

    ## 工具箱：JSON

    **JSON** 是结构化数据的标准文字格式，看起来很像 Python 字典。先 `import json`，然后 `json.dumps(value)` 生成文字，`json.loads(text)` 读回来。

    `''.join(pieces)` 把字符串列表粘在一起，`isinstance(x, list)` 判断 `x` 是不是列表。
    """)
    return


@app.cell
def _():
    import json as _json

    _text = _json.dumps({"x": 1})
    _text, _json.loads(_text)["x"]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：to_openai

    写 `to_openai(message)`，转换一条消息。（隐藏的准备代码没有导入 `json`，所以函数开头要写 `import json`。）

    1. `user`：`{"role": "user", "content": message["content"]}`
    2. `toolResult`：`{"role": "tool", "tool_call_id": message["toolCallId"], "content": message["content"]}`
    3. `assistant`：`{"role": "assistant", "content": ...}`，content 是所有文字块的 `block["text"]` 拼接的结果
    4. 如果有工具调用，再把 `out["tool_calls"]` 设为一个列表，每项是 `{"id": ..., "type": "function", "function": {"name": ..., "arguments": json.dumps(...)}}`
    """)
    return


@app.function
def to_openai(message):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("32", "to_openai", to_openai, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：to_anthropic

    Anthropic 更严格：一条回复的所有结果必须共用一条 `user` 消息。所以这个转换器接收整个列表。写 `to_anthropic(messages)`，构造列表 `out`：

    1. `user`：追加 `{"role": "user", "content": message["content"]}`
    2. `assistant`：追加 `{"role": "assistant", "content": blocks}`，其中文字块变成 `{"type": "text", "text": ...}`，工具调用变成 `{"type": "tool_use", "id": ..., "name": ..., "input": block["arguments"]}`
    3. `toolResult`：构造 `{"type": "tool_result", "tool_use_id": ..., "content": ..., "is_error": message["isError"]}`。如果 `out` 的最后一项是内容为列表的 `user` 消息，就把块追加到那个列表；否则追加一条装着 `[block]` 的新 `user` 消息。
    """)
    return


@app.function
def to_anthropic(messages):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("32", "to_anthropic", to_anthropic, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 修复对话

    存下来的对话可以回放给任何模型，所以 `packages/ai/src/api/transform-messages.ts` 在发送前会清理一遍。它移除以 `error` 或 `aborted` 结束的助手消息，因为它们可能只写了一半；它还为没有结果的工具调用补上结果，免得严格的服务商拒绝请求：

    <!-- source pi packages/ai/src/api/transform-messages.ts -->
    ```ts
    // Second pass: insert synthetic empty tool results for orphaned tool calls
    ```

    记住这一点：每个 harness 都得在某一层保证调用和结果配对。

    ## 把工具和提示词放进对话记录

    大多数框架把“系统提示词”和“工具”当作对话之外的设置。现在的 Pi 不这样。提示词和工具声明是**对话记录里的系统消息**，循环的注释写明了这个约定：“`context.tools` is what the runtime can execute; the transcript's system messages declare what the model may call.”

    扩展在会话中途添加工具时，Pi 不改写历史，而是追加一条系统消息，用 `toolsAdded` 和 `toolsRemoved` 说明变化。这样回放对话记录时，总能精确得到模型在每个时刻知道的内容，请求前面的文字也保持不变，服务商的缓存就仍然有用（第 41 章会解释）。

    差异是这样算出来的。注意声明发生变化的工具：它同时被列为移除和新增。

    <!-- source pi packages/ai/src/utils/transcript.ts -->
    ```ts
    toolsAdded: current
      .filter((tool) => {
        const previousTool = previousTools.get(tool.name);
        return previousTool === undefined || !declarationsEqual(previousTool, tool);
      })
    ```

    ## 练习 5：tool_changes

    写 `tool_changes(declared, current)`。两者都是从工具名到声明的字典。返回两个列表组成的元组 `(added, removed)`，各自保持字典里的顺序：

    - `added`：在 `current` 里、但不在 `declared` 里或声明不同的名字
    - `removed`：在 `declared` 里、但不在 `current` 里或声明不同的名字
    """)
    return


@app.function
def tool_changes(declared, current):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("32", "tool_changes", tool_changes, LOCALE)
    return


@app.cell
def _():
    attempt(tool_changes, {"read": "v1", "bash": "v1"}, {"read": "v2", "edit": "v1"})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    通过之后，这里显示 `(['read', 'edit'], ['read', 'bash'])`。`read` 变了，所以两个列表里都有它。

    ## 现在你能做到

    - 解释什么是流，读懂 Pi 的事件列表
    - 解释为什么 Pi 的流把失败变成消息，以及这对循环有什么影响
    - 把一段对话转换成两家服务商的格式，并解释 Anthropic 的分组规则
    - 说出 `transform-messages.ts` 修复什么、为什么修复
    - 解释 Pi 怎样把工具和提示词放进对话记录，又怎样宣告变化

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** 读 `packages/ai/src/types.ts`，再从头到尾读一家服务商的实现，`packages/ai/src/api/anthropic-messages.ts` 是个好选择。`packages/ai/src/utils/transcript.ts` 里是对话记录的规则。Pi 的 `packages/coding-agent/docs/custom-provider.md` 介绍了如何添加自己的服务商。
    """)
    return


if __name__ == "__main__":
    app.run()
