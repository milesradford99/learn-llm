import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="33 · 能在真实世界活下来的工具")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import attempt, call, result

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 33 · 能在真实世界活下来的工具

    Pi 默认的全部工具是 `read`、`bash`、`edit` 和 `write`，需要时还可以打开 `grep`、`find`、`ls` 和 `powershell`。它们都在 `packages/coding-agent/src/core/tools/` 里。（`bash` 运行一条 shell 命令，就是你在终端里输入的那种。）只给四个工具本身就是一种主张：`bash` 几乎无所不能，多数专用工具只会增加模型要读的东西。

    这些工具都很短。值得研究的是，每一个工具都对应着智能体的一种具体出错方式。

    ## 输出太多

    打印文件的 `cat` 命令，遇到大日志可能返回一百万行。这些输出会进入对话，并在*之后的每一轮*重新发送（第 30 章），直到模型读不下为止。所以任何输出都不会未经裁剪就交给模型：

    <!-- source pi packages/coding-agent/src/core/tools/truncate.ts -->
    ```ts
     * Truncation is based on two independent limits - whichever is hit first wins:
     * - Line limit (default: 2000 lines)
     * - Byte limit (default: 50KB)
     *
     * Never returns partial lines (except bash tail truncation edge case).
    …
    export const DEFAULT_MAX_LINES = 2000;
    export const DEFAULT_MAX_BYTES = 50 * 1024; // 50KB
    ```

    有三个细节很重要。上限既按行也按字节，因为压缩过的代码一行就可能有 10 MB。只保留整行，因为半行会误导人。方向也因工具而异：`read` 保留文件的**开头**，并告诉模型下一个 `offset` 该填多少；`bash` 保留**结尾**，因为命令输出的末尾才是错误所在，同时把全文存进临时文件，模型需要时可以去看。

    还必须告诉模型。悄悄截断看起来就像文件本来就短。工具的文字会说明切掉了什么、怎样拿到其余部分。

    回忆第 17 章：`text.encode("utf-8")` 给出字节，一个汉字占 3 字节。`"\n"` 是换行符，占 1 字节。`text.split("\n")` 把文字切成行，`"\n".join(lines)` 再拼回去。

    ## 练习 1：truncate_head

    写 `truncate_head(text, max_lines, max_bytes)`，返回一个元组：保留的文字，以及是否有内容被切掉。

    1. `kept = []` 和 `size = 0`
    2. 对 `text.split("\n")` 的每一行：`cost` 是它的字节数；如果 `kept` 非空，再加 1 算换行符
    3. 如果 `len(kept) == max_lines` 或 `size + cost > max_bytes`，返回 `"\n".join(kept), True`
    4. 否则把它加进 `kept`，并给 `size` 加上 `cost`
    5. 循环结束后，返回 `text, False`
    """)
    return


@app.function
def truncate_head(text, max_lines, max_bytes):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("33", "truncate_head", truncate_head, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 编辑而不破坏

    最直接的编辑方式是让模型发回整个新文件。这既慢又贵，还可能悄悄丢掉它懒得重复的行。行号也不好：前面的编辑一移动，行号就失效了。

    Pi 的 `edit` 接收成对的 `oldText` 和 `newText`，规则很简单：**旧文字必须恰好出现一次。** 零处匹配说明模型对文件的认识有误，两处匹配说明编辑有歧义。两者都变成错误结果，模型就会去读文件再试（规则一）。错误信息还会告诉它下一步怎么做：

    <!-- source pi packages/coding-agent/src/core/tools/edit-diff.ts -->
    ```ts
    `Found ${occurrences} occurrences of the text in ${path}. The text must be unique. Please provide more context to make it unique.`,
    ```

    模型还常常弄错看不见的细节：行尾空格、弯引号、特殊破折号。所以匹配带一点范围很窄的**模糊匹配**（fuzzy match）：先精确匹配，失败后再比较规范化之后的文字。`normalizeForFuzzyMatch` 的第一步是去掉每行行尾的空白：

    <!-- source pi packages/coding-agent/src/core/tools/edit-diff.ts -->
    ```ts
    export function normalizeForFuzzyMatch(text: string): string {
    …
          // Strip trailing whitespace per line
          .split("\n")
          .map((line) => line.trimEnd())
          .join("\n")
    ```

    这三行是接在 `text` 后面的连续调用：拆开，对每行运行一个函数（`.map` 是列表推导式的另一种写法），再拼回去。

    有一个细节能说明这里有多讲究。当成功的是模糊匹配时，替换只能发生在规范化后的文字上。如果直接这么做，整个文件的行尾空格都会消失，改一行也会产生巨大的 diff（改动行的列表）。所以 `applyReplacementsPreservingUnchangedLines` 只重写编辑碰到的行，其余每一行都从原文复制回来。下面的 `rewrite_touched_lines` 已经替你写好了这部分。

    ## 工具箱：抛出异常与文字工具

    `raise ValueError("message")` 由你自己抛出异常。第 30 章的 `run_tool` 把异常变成错误结果，所以工具直接抛出就行。

    `line.rstrip()` 去掉行尾的空格和制表符。`text.count(part)` 数 `part` 出现几次，`text.replace(a, b)` 做替换，`a in text` 判断 `a` 是否出现。

    ## 练习 2：fuzzy

    写 `fuzzy(text)`，去掉每行行尾的空白，其余（包括缩进）全部保留。
    """)
    return


@app.function
def fuzzy(text):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("33", "fuzzy", fuzzy, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是写好的 `rewrite_touched_lines`，你直接调用就行：
    """)
    return


@app.function
def rewrite_touched_lines(content, old, new):
    base = fuzzy(content)
    first = base[:base.index(fuzzy(old))].count("\n")
    last = first + fuzzy(old).count("\n")
    rewritten = base.replace(fuzzy(old), new).split("\n")
    lines = content.split("\n")
    middle = rewritten[first:len(rewritten) - (len(lines) - last - 1)]
    return "\n".join(lines[:first] + middle + lines[last + 1:])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：apply_edit

    写 `apply_edit(content, old, new)`，返回新的文件内容：

    1. 如果 `old == ''`：`raise ValueError("oldText must not be empty.")`
    2. `matches = fuzzy(content).count(fuzzy(old))`
    3. 如果为 0：`raise ValueError("Could not find the text. It must match exactly.")`
    4. 如果大于 1：`raise ValueError(f"Found {matches} occurrences of the text. It must be unique.")`
    5. 如果 `old in content`，返回 `content.replace(old, new)`
    6. 否则返回 `rewrite_touched_lines(content, old, new)`
    """)
    return


@app.function
def apply_edit(content, old, new):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("33", "apply_edit", apply_edit, LOCALE)
    return


@app.cell
def _():
    # The edit touches lines 1 and 2, and line 3 keeps its trailing spaces.
    attempt(apply_edit, "a  \nb\nc  \n", "a\nb", "A\nB")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    通过之后，这里显示 `'A\nB\nc  \n'`。

    ## 同时运行多个工具

    模型可能在一条回复里请求五次 `read`。一个个运行太浪费时间，所以 Pi 默认**并行**运行，分三个阶段：

    1. **逐个准备。** 找到工具、校验参数、运行 `beforeToolCall`。这里故意串行：钩子可能要问用户问题，或者碰到共享状态。
    2. **一起执行。**
    3. **按原顺序报告。** 不管完成先后，模型都必须按请求的顺序拿到结果。`Promise.all` 正好做到这一点：等全部完成，并保持列表顺序。（下面的代码里，每一项要么是已完成的结果，要么是待运行的函数，`typeof` 判断的就是这个。）

    <!-- source pi packages/agent/src/agent-loop.ts -->
    ```ts
    const orderedFinalizedCalls = await Promise.all(
      finalizedCalls.map((entry) => (typeof entry === "function" ? entry() : Promise.resolve(entry))),
    );
    const messages: ToolResultMessage[] = [];
    for (const finalized of orderedFinalizedCalls) {
      const toolResultMessage = createToolResultMessage(finalized);
      await emitToolResultMessage(toolResultMessage, emit);
      messages.push(toolResultMessage);
    }
    ```

    工具可以声明 `executionMode: "sequential"`，只要有一个这样的工具，整批都改为串行。

    ## 练习 4：in_source_order

    写 `in_source_order(calls, finished)`。`calls` 是模型请求的工具调用，`finished` 是按完成顺序排列的结果消息。按 `calls` 的顺序返回结果：先建一个从 `toolCallId` 到结果的字典，再对 `calls` 用列表推导式。
    """)
    return


@app.function
def in_source_order(calls, finished):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("33", "in_source_order", in_source_order, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 两次编辑，同一个文件

    并行带来新的危险。对同一文件的两次 `edit` 各自读取、修改、写回，后写的会抹掉先写的。Pi 的办法很小：每个文件一个队列，用真实路径作键，指向同一文件的符号链接（指向另一个文件的文件）也共用一个队列。下面的函数签名里，`<T>` 只是表示“任意类型”。

    <!-- source pi packages/coding-agent/src/core/tools/file-mutation-queue.ts -->
    ```ts
     * Serialize file mutation operations targeting the same file.
     * Operations for different files still run in parallel.
     */
    export async function withFileMutationQueue<T>(filePath: string, fn: () => Promise<T>): Promise<T> {
    ```

    每个新操作等待同一文件上的前一个操作，不同文件互不等待。这是正确性所需的最小方案，也很能概括 Pi 整体的风格。

    ## Pi 没做什么

    同样值得了解：`bash` 没有权限确认，也没有沙箱。Pi 的安全指南说得很直接：它拥有启动它的账户的全部权限，可以读取、修改、执行，而盯着对话记录并不是安全边界。想要隔离，就把整个 Pi 放进容器里运行。第 36 章还会回到这个话题，第 41 章会看 Super-agent 怎么说。

    ## 现在你能做到

    - 解释为什么必须裁剪输出、为什么还要按字节算，以及 `read` 和 `bash` 为什么切相反的两端
    - 解释编辑工具为什么用唯一文字，而不是行号或整个文件
    - 描述模糊匹配，以及它为什么只重写碰到的行
    - 描述并行工具调用的三个阶段，以及结果为什么保持顺序
    - 解释文件修改队列，以及为什么用真实路径作键

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** 把 `tools/edit.ts` 和 `tools/edit-diff.ts` 放在一起读，再读 `tools/output-accumulator.ts`，看 `bash` 的输出在命令运行时如何保存并溢出到文件。想亲自试用，可以从 [pi.dev](https://pi.dev) 安装 Pi，然后运行 `pi --help`。
    """)
    return


if __name__ == "__main__":
    app.run()
