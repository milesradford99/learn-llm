import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="46 · 一个能按你心意改造的底座")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import attempt

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 46 · 一个能按你心意改造的底座

    Pi 的 README 说它 “ships with powerful defaults but skips features like sub-agents and plan mode”。这句话定义了整个项目。Pi 不想包含所有人想要的一切，而是想成为大家不用 fork 就能扩展的东西。这一章讲它是怎么组装起来的，也是 Pi 部分的收尾。

    ## 系统提示词

    默认提示词很短。第一句是：

    <!-- source pi packages/coding-agent/src/core/system-prompt.ts -->
    ```ts
    promptSections.preamble =
      "You are a coding assistant running inside pi, a coding agent harness. Help users inspect files, run commands, and modify code.";
    ```

    其余内容都是**命名的部分**（section）。之所以分开而不是一长串文字，原因在第 42 章的对话记录：某一部分变化时（比如技能重新加载），Pi 只发送这一部分的补丁，而不是整份新提示词（`diffSystemPromptSections`）。顺序是：开场白、`tools`、`rules`、`docs`，然后是可选的 `addendum`、`project_context`、`skills`，然后是 `cwd`，最后是自定义部分。

    其中两部分值得一看。`project_context` 装着你的 `AGENTS.md` 这类上下文文件。`docs` 列出 Pi 自己的文档位置，所以**模型可以读懂 Pi 的工作方式，替你扩展 Pi**：这就是 README 说的 “ask Pi to build what you want”。

    <!-- source pi packages/coding-agent/src/core/system-prompt.ts -->
    ```ts
    for (const [name, content] of Object.entries(promptSections)) {
      if (name !== "preamble") sections[name] = `<${name}>\n${content}\n</${name}>`;
    }
    ```

    ## 练习 1：build_prompt

    `sections` 是一个有序字典，从部分名映射到文字。写 `build_prompt(sections)`。跳过空文字。`preamble` 原样放入，其他部分变成 `f"<{name}>\n{text}\n</{name}>"`。全部用 `"\n\n"` 连接。（`for name, text in sections.items():` 见第 19 章，`continue` 跳到下一次循环。）
    """)
    return


@app.function
def build_prompt(sections):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("46", "build_prompt", build_prompt, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 技能：用多少付多少

    **技能**（skill）是一个带 `SKILL.md` 文件的文件夹，里面是针对某类任务的说明，比如怎样部署这个项目、怎样填写 PDF。把所有技能都放进提示词，每次请求都要付词元（token）。所以 Pi 采用**渐进式披露**（progressive disclosure）：提示词只列出每个技能的名字、一句话描述和路径。任务匹配时，模型用普通的 `read` 工具去读那个文件。

    妙处在于根本没有技能工具。技能只是模型自己决定要读的文件。设置了 `disable-model-invocation` 的技能不在列表里，只能由你用命令启动。

    这份列表以 XML 形式进入提示词，XML 是一种由 `<标签>` 构成的文字格式，所以描述必须转义。如果描述里有 `</skill>`，就可能冲出标签。这是对**提示词注入**（prompt injection，藏在模型所读文字里的指令）的一点小防御。

    <!-- source pi packages/coding-agent/src/core/skills.ts -->
    ```ts
    export function formatSkillsForPrompt(skills: Skill[], fileReadTool: "read" | "bash" | "indirect" = "read"): string {
      const visibleSkills = skills.filter((s) => !s.disableModelInvocation);
    …
        lines.push("  <skill>");
        lines.push(`    <name>${escapeXml(skill.name)}</name>`);
        lines.push(`    <description>${escapeXml(skill.description)}</description>`);
        lines.push(`    <location>${escapeXml(skill.filePath)}</location>`);
        lines.push("  </skill>");
    ```

    ## 练习 2：escape_xml

    写 `escape_xml(text)`，把 `&` 换成 `&amp;`，`<` 换成 `&lt;`，`>` 换成 `&gt;`。`.replace` 可以连着写，顺序很重要：如果先替换 `<`，它变成 `&lt;`，之后再替换 `&` 就成了 `&amp;lt;`。所以 `&` 要最先替换。（Pi 还会转义引号，这里略过。）
    """)
    return


@app.function
def escape_xml(text):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("46", "escape_xml", escape_xml, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：format_skills

    每个技能是一个字典，有 `name`、`description`、`filePath` 和 `disableModelInvocation`。写 `format_skills(skills)`：

    1. 保留 `disableModelInvocation` 为假的技能。如果一个也不剩，返回 `''`
    2. 否则返回下面这些行，用 `"\n"` 连接，外层标签之间每个技能一组，三个值都要转义：

    ```
    <available_skills>
      <skill>
        <name>pdf</name>
        <description>Read PDFs</description>
        <location>/s/pdf/SKILL.md</location>
      </skill>
    </available_skills>
    ```
    """)
    return


@app.function
def format_skills(skills):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("46", "format_skills", format_skills, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 扩展：运行在 harness 内部的代码

    **扩展**（extension）是 Pi 加载到自身进程里的 TypeScript 文件。Pi 自己能做的事，它几乎都能做：添加工具、命令、快捷键和模型服务商，替换页脚，监听事件。一个权限闸门大约十行：

    ```ts
    export default function (pi) {
      pi.on("tool_call", async (event, ctx) => {
        if (event.toolName === "bash" && event.input.command.includes("rm -rf")) {
          return { block: true, reason: "Blocked by policy" };
        }
      });
    }
    ```

    事件大约有四十种，包括 `context`（改写模型看到的内容）、`tool_call`、`tool_result`、`before_provider_headers` 和 `session_compact`。所以 Pi 故意不做的功能，几个文件就能做出来：子智能体、计划模式、权限闸门，甚至沙箱。

    `tool_call` 处理函数是这样运行的：第一个拦截的生效，后面的都不运行。（`for (const x of xs)` 就是 `for x in xs`，`{ handlers }` 从每一项里取出这个字段。）

    <!-- source pi packages/coding-agent/src/core/extensions/runner.ts -->
    ```ts
    for (const { handlers } of snapshotEventHandlers(this.extensions, "tool_call")) {
      for (const handler of handlers) {
        const handlerResult = await handler(event, ctx);

        if (handlerResult) {
          result = handlerResult as ToolCallEventResult;
          if (result.block) {
            return result;
          }
        }
      }
    }
    ```

    ## 练习 4：emit_tool_call

    写 `emit_tool_call(handlers, event)`。`handlers` 是函数列表，每个函数返回字典或 `None`。用 `outcome = None`；对每个处理函数，调用 `answer = handler(event)`，`answer` 不是 `None` 时存下来，如果 `answer.get("block")` 为真就立即返回。最后返回 `outcome`。
    """)
    return


@app.function
def emit_tool_call(handlers, event):
    # ← 把下面的 todo() 换成你的代码
    todo()

@app.cell(hide_code=True)
def _():
    check("46", "emit_tool_call", emit_tool_call, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一个核心，多种前端

    第 41 章说过，循环只发出事件。好处在于驱动它的方式有很多：

    | 模式 | 怎么用 | 用途 |
    |---|---|---|
    | 交互 | `pi` | 终端界面（`modes/interactive/`） |
    | 打印 | `pi -p "..."` | 一问一答，用于脚本 |
    | JSON | `--mode json` | 所有事件，以 JSON 行输出 |
    | RPC | `--mode rpc` | 长期运行的子进程，由另一个程序发送 JSON 行来驱动，任何语言都能用 |
    | SDK | `createAgentSession` | 嵌入你自己的 Node 程序。README 指向的真实案例是 OpenClaw |

    ## 信任与安全

    Pi 的安全立场很特别，你应该能准确说出来。**它不会在每次工具调用前询问。** 它的安全指南明确这样写，还补充说盯着对话记录、审查改动都不构成安全边界。安全来自限制进程能接触到什么，指南给出的最强实用方案是把整个 Pi 放进容器或虚拟机。

    **项目信任**（project trust）的作用更窄。`.pi/settings.json`、`.pi/mcp.json`、`.pi/extensions` 这类文件能执行代码，所以 Pi 从某个文件夹加载它们之前会询问。这防止克隆下来的仓库悄悄加载扩展。它不限制工具运行后能做什么，也挡不住文件内容里的提示词注入。

    ## 较新的包

    这个仓库变化很快，许多关于 Pi 的文章已经过时。在这个提交上：

    - **MCP 已内置。** MCP（Model Context Protocol）是连接外部服务器工具的标准。Pi 长期故意不做它，现在它作为内置扩展提供（`packages/coding-agent/src/extensions/mcp/`，协议客户端在 `packages/mcp`）。注意它落在哪里：是扩展，而不是核心。
    - **Codemode**（`packages/codemode`）让模型写一段调用工具的 JavaScript 脚本，在 WebAssembly 沙箱里运行，只有脚本的输出进入上下文。这是对第 43 章工具输出问题的另一种回答：十次工具调用不必在对话里留下十个结果。
    - **`durable`**（标记为实验性）是一个 harness，在显示之前先把每轮和每次工具调用提交到存储，进程死掉后还能恢复。可以和第 44 章的日志对比。
    - **`chord`** 用跨进程的有状态对象构建应用，支持热重载，`durable` 建立在它之上。

    ## 一段话讲完 Pi

    现在你应该能说出下面这段话，并为每一句辩护：*Pi 是一个分层的 TypeScript harness。与服务商无关的 `ai` 层把所有模型变成一条永不抛异常的事件流。很小的 `agent` 循环用两层处理转向和后续消息，所有策略都通过钩子注入。`coding-agent` 加上四把锋利的工具、树形只追加的会话（模型看到的只是投影），以及作为持久事件的压缩。其余一切都交给扩展、技能和包，并以你自己的权限运行。*

    ## 现在你能做到

    - 描述 Pi 的系统提示词如何构建，以及为什么分部分
    - 解释渐进式披露，以及为什么技能不需要专用工具
    - 用扩展写一个工具闸门，并说出处理函数的运行顺序
    - 列出驱动 Pi 的五种方式
    - 说清 Pi 的安全模型，以及项目信任管什么、不管什么
    - 说出最近的变化：MCP、codemode、durable

    ---
    <!-- go-deeper -->
    **继续深入（选读）：** `packages/coding-agent/docs/extensions.md` 和 `packages/coding-agent/examples/extensions/` 文件夹是最好的学习材料。`docs/security.md`、`docs/sdk.md`、`docs/rpc.md` 覆盖其余内容。真正检验理解的办法：写一个计划模式扩展。
    """)
    return


if __name__ == "__main__":
    app.run()
