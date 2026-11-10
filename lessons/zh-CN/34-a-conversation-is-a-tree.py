import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="34 · 对话是一棵树")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import attempt, say, trace, user

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 34 · 对话是一棵树

    到目前为止，对话只是内存里的一个列表。程序一崩溃，它就没了。如果智能体花二十分钟走错了方向，你也只能从头再来。

    Pi 用一个决定同时解决了这两个问题：会话是一份**只追加的日志**（append-only log），每条记录都写明自己的父记录。这样它就成了一棵**树**。这一章讲的是 `packages/coding-agent/src/core/session-manager.ts`。

    ## 格式

    会话是一个 **JSONL** 文件：每行一个 JSON 对象。正常使用时，记录只加在末尾，所以文件一旦存在，崩溃最多危及最后一行，之前的内容都还能读。（Pi 要等到第一条真实消息才写文件，版本迁移也可能重写它。）

    每条记录都有 `id` 和 `parentId`。下面的 TypeScript 里，`interface` 列出一个值有哪些字段，`extends` 表示“还包括那些字段”，`string | null` 表示文字或空值：

    <!-- source pi packages/coding-agent/src/core/session-manager.ts -->
    ```ts
    export interface SessionEntryBase {
      type: string;
      id: string;
      parentId: string | null;
      timestamp: string;
    }

    export interface SessionMessageEntry extends SessionEntryBase {
      type: "message";
      message: AgentMessage;
    }
    ```

    消息不是唯一的记录类型，还有 `model_change`、`thinking_level_change`、`compaction`、`branch_summary`、`custom` 等等。发生的一切都写进日志。

    会话还有一个指针，叫**叶子**（leaf）：你当前所在的记录。新记录的父记录就是叶子，然后它自己成为叶子。

    ## 练习 1：append

    我们用 `{"entries": [...], "leaf": ...}` 表示会话，叶子是一个 id；新会话里是 `None`（Python 里表示“没有值”）。写 `append(session, entry)`：

    1. `entry["id"] = f"e{len(session['entries'])}"`
    2. `entry["parentId"] = session["leaf"]`
    3. 把它加入 `session["entries"]`，把 `session["leaf"]` 设为它的 id，并返回这个 id
    """)
    return


@app.function
def append(session, entry):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("34", "append", append, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 回退而不删除

    精彩的部分来了。假设智能体在 `e1` 之后走错了。要重试时，Pi **什么也不删**，只是把叶子移回去：

    <!-- source pi packages/coding-agent/src/core/session-manager.ts -->
    ```ts
    branch(branchFromId: string): void {
      if (!this.byId.has(branchFromId)) {
        throw new Error(`Entry ${branchFromId} not found`);
      }
      this.leafId = branchFromId;
    }
    ```

    只是移动一个指针！下一条追加的记录以 `e1` 为父，于是 `e1` 有了两个子记录，旧尝试仍留在文件里。如果后来发现旧尝试更好，还可以回去。Pi 的 `/tree` 命令显示的就是这棵树。

    ```
    e0 ─ e1 ─ e2 ─ e3      （第一次尝试）
           └─ e4 ─ e5      （第二次尝试；叶子是 e5）
    ```

    任何记录到根都只有一条路径，所以只凭叶子就能确定一段完整对话。这叫**当前分支**（active branch），也就是模型看到的内容。要得到它，从叶子沿 `parentId` 走到根，再把列表反过来：

    <!-- source pi packages/coding-agent/src/core/session-manager.ts -->
    ```ts
    while (current) {
      path.push(current);
      current = current.parentId ? index.get(current.parentId) : undefined;
    }
    path.reverse();
    return path;
    ```

    ## 练习 2：active_path

    写 `active_path(session)`，给出从根到叶子的记录列表：

    1. 建一个字典 `by_id`，从 id 映射到记录
    2. `path = []` 和 `current = session["leaf"]`
    3. `while current is not None:` 取出记录，追加，并把 `current` 设为它的 `parentId`
    4. `path.reverse()` 就地反转列表，本身不返回任何东西，所以单独写一行，然后 `return path`

    `x is not None` 是判断“x 不是空值”的标准写法。
    """)
    return


@app.function
def active_path(session):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("34", "active_path", active_path, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：branch

    写 `branch(session, entry_id)`：如果没有这个 id，`raise ValueError(f"Entry {entry_id} not found")`；否则设置叶子。它不返回任何东西。
    """)
    return


@app.function
def branch(session, entry_id):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("34", "branch", branch, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：context

    会话里的内容比模型需要的多。写 `context(session)`，对当前路径上 `type` 为 `"message"` 的每条记录，给出 `entry["message"]` 组成的列表。
    """)
    return


@app.function
def context(session):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("34", "context", context, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    看一次重试。两次尝试都留在文件里（所以有 3 条记录），模型只看到其中一次。练习 1 到 4 通过之前，这个单元格只会提示你：
    """)
    return


@app.function
def demo_retry():
    session = {"entries": [], "leaf": None}
    first = append(session, {"type": "message", "message": user("Fix the bug")})
    append(session, {"type": "message", "message": say("I'll rewrite everything.")})
    branch(session, first)
    append(session, {"type": "message", "message": say("I'll change one line.")})
    return len(session["entries"]), trace(context(session))


@app.cell
def _():
    attempt(demo_retry)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 真实版本为什么步骤更多

    上面的 `context` 是简化版。Pi 的 `buildSessionProjection` 还要处理三件事：应用路径上最新的**压缩**（compaction），用摘要替换旧消息（下一章）；应用 `context_edit` 记录，改变模型看到的某条旧记录，而日志保留原文；从路径上读取设置，所以如果你分支回到某次换模型之前，旧模型也跟着回来。它们都遵循一个原则：**日志记录发生了什么，模型看到的内容由日志计算而来。** 名字里的 *projection*（投影）说的就是这个。在别的领域这叫**事件溯源**（event sourcing），正是它让这一切如此便宜。

    分支时，被放弃的路径上可能有有用的东西，所以 Pi 可以为它写一段摘要放到新分支上（`branchWithSummary`，代码在 `compaction/branch-summarization.ts`）。如果想要独立的文件，`createBranchedSession` 会把一条路径复制到新会话，文件头的 `parentSession` 记录它的来源。

    ## 练习 5：to_jsonl 和 from_jsonl

    保存和读取。写 `to_jsonl(session)`，每条记录一行 `json.dumps(entry)`，用 `"\n"` 连接。写 `from_jsonl(text)`：`entries` 是对 `text.split("\n")` 每行做 `json.loads(line)` 的结果，返回 `{"entries": entries, "leaf": entries[-1]["id"]}`，也就是说叶子是**最后一行**的 id。保存的会话至少有一行。Pi 也是这样：重新打开会话时从最新的记录继续。（每个函数开头写 `import json`。）
    """)
    return


@app.function
def to_jsonl(session):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("34", "to_jsonl", to_jsonl, LOCALE)
    return


@app.function
def from_jsonl(text):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("34", "from_jsonl", from_jsonl, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 现在你能做到

    - 解释 Pi 为什么用只追加的 JSONL，以及崩溃最多损失什么
    - 解释叶子是什么，以及为什么移动它就足以分支
    - 从叶子算出当前分支
    - 区分日志（发生了什么）和投影（模型看到什么）
    - 说出 `branchWithSummary` 和 `createBranchedSession` 的用途

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** `packages/coding-agent/docs/session-format.md` 记录了每种记录类型，`session-manager.ts` 开头的版本迁移代码展示了格式的演变。在真实会话里试试 `/tree`、`/fork` 和 `/resume`。
    """)
    return


if __name__ == "__main__":
    app.run()
