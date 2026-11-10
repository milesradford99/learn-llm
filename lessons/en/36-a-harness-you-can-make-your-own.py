import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="36 · A harness you can make your own")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import attempt

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 36 · A harness you can make your own

    Pi's README says it "ships with powerful defaults but skips features like sub-agents and plan mode". That sentence defines the project. Pi doesn't try to contain everything people want; it tries to be something they can extend without forking it. This chapter covers how it's assembled, and closes out Pi.

    ## The system prompt

    The default prompt is short. Its first line is:

    <!-- source pi packages/coding-agent/src/core/system-prompt.ts -->
    ```ts
    promptSections.preamble =
      "You are a coding assistant running inside pi, a coding agent harness. Help users inspect files, run commands, and modify code.";
    ```

    Everything else is a **named section**. Sections are kept separate instead of being one long string for a reason: chapter 32's transcript. When one changes, for example when skills reload, Pi sends a patch for that section instead of a new prompt (`diffSystemPromptSections`). The order is: preamble, `tools`, `rules`, `docs`, then optional `addendum`, `project_context`, `skills`, then `cwd`, then any custom sections.

    Two of those are worth a look. `project_context` holds your `AGENTS.md`-style context files. And `docs` lists where Pi's own documentation lives, so **the model can read how Pi works and extend Pi for you**: that's what the README means by "ask Pi to build what you want".

    <!-- source pi packages/coding-agent/src/core/system-prompt.ts -->
    ```ts
    for (const [name, content] of Object.entries(promptSections)) {
      if (name !== "preamble") sections[name] = `<${name}>\n${content}\n</${name}>`;
    }
    ```

    ## Exercise 1: build_prompt

    `sections` is a dictionary in order, from section name to text. Write `build_prompt(sections)`. Skip empty text. `preamble` goes in as it is, and any other section becomes `f"<{name}>\n{text}\n</{name}>"`. Join them all with `"\n\n"`. (`for name, text in sections.items():` from chapter 17, and `continue` jumps to the next time round.)
    """)
    return


@app.function
def build_prompt(sections):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("36", "build_prompt", build_prompt, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Skills: pay for what you use

    A **skill** is a folder with a `SKILL.md` file: instructions for one kind of task, such as how to deploy this project or how to fill in a PDF. Putting every skill in the prompt would cost tokens on every request. So Pi uses **progressive disclosure**: the prompt lists only each skill's name, one-line description and path. When a task matches, the model uses its ordinary `read` tool to load the file.

    The elegance here is that there's no skill tool. A skill is just a file the model decides to read. A skill with `disable-model-invocation` is left out of the list, and you can only start it yourself with a command.

    The listing goes into the prompt as XML, a text format of `<tags>`, so descriptions must be escaped. If a description contained `</skill>`, it could break out of its tag. That's a small defence against **prompt injection**: instructions hidden in text that the model reads.

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

    ## Exercise 2: escape_xml

    Write `escape_xml(text)`, replacing `&` with `&amp;`, `<` with `&lt;` and `>` with `&gt;`. `.replace` calls can be chained, and the order matters: if you replace `<` first it becomes `&lt;`, and replacing `&` after that turns it into `&amp;lt;`. So `&` goes first. (Pi also escapes quotes. We'll skip that.)
    """)
    return


@app.function
def escape_xml(text):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("36", "escape_xml", escape_xml, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: format_skills

    Each skill is a dictionary with `name`, `description`, `filePath` and `disableModelInvocation`. Write `format_skills(skills)`:

    1. Keep the skills where `disableModelInvocation` is false. If none are left, give back `''`
    2. Otherwise give back these lines joined by `"\n"`, once per skill between the outer tags, with the three values escaped:

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
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("36", "format_skills", format_skills, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Extensions: code inside the harness

    An **extension** is a TypeScript file that Pi loads into its own process. It can do nearly everything Pi itself does: add tools, commands, keyboard shortcuts and model providers, replace the footer, and listen to events. A permission gate is about ten lines:

    ```ts
    export default function (pi) {
      pi.on("tool_call", async (event, ctx) => {
        if (event.toolName === "bash" && event.input.command.includes("rm -rf")) {
          return { block: true, reason: "Blocked by policy" };
        }
      });
    }
    ```

    There are around forty events, including `context` (rewrite what the model sees), `tool_call`, `tool_result`, `before_provider_headers` and `session_compact`. So the features Pi leaves out can be built in a few files: sub-agents, plan mode, permission gates, even a sandbox.

    This is how `tool_call` handlers run. The first one to block wins, and the rest don't run. (`for (const x of xs)` is `for x in xs`, and `{ handlers }` takes that field out of each item.)

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

    ## Exercise 4: emit_tool_call

    Write `emit_tool_call(handlers, event)`. `handlers` is a list of functions, each giving a dictionary or `None`. Use `outcome = None`; for each handler, call `answer = handler(event)`, and if `answer` isn't `None`, keep it, and give it back at once if `answer.get("block")` is true. At the end give back `outcome`.
    """)
    return


@app.function
def emit_tool_call(handlers, event):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("36", "emit_tool_call", emit_tool_call, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One core, many front ends

    Chapter 31 said the loop only emits events. The payoff is in how many ways you can drive it:

    | Mode | How | For |
    |---|---|---|
    | Interactive | `pi` | The terminal interface (`modes/interactive/`) |
    | Print | `pi -p "..."` | One prompt, one answer, for scripts |
    | JSON | `--mode json` | Every event, as JSON lines |
    | RPC | `--mode rpc` | A long-running child process that another program drives by sending JSON lines, from any language |
    | SDK | `createAgentSession` | Inside your own Node program. OpenClaw is the real-world example the README points to |

    ## Trust and safety

    Pi's safety position is unusual, and you should be able to state it exactly. **It doesn't ask before each tool call.** Its security guide says so, and adds that watching the transcript and reviewing changes don't create a security boundary. Safety comes from limiting what the process can reach, and the strongest practical option it names is running all of Pi inside a container or VM.

    **Project trust** does a narrower job. Files like `.pi/settings.json`, `.pi/mcp.json` and `.pi/extensions` can run code, so Pi asks before loading them from a folder. It stops a cloned repository from silently loading extensions. It doesn't limit what tools can do once running, and it doesn't stop prompt injection from a file's content.

    ## The newer packages

    The repository is moving fast, and many write-ups of Pi are already out of date. At this commit:

    - **MCP is built in.** MCP (Model Context Protocol) is a standard for connecting tools from outside servers. Pi long left it out on purpose; it now ships as a built-in extension (`packages/coding-agent/src/extensions/mcp/`, with the protocol client in `packages/mcp`). Note where it landed: as an extension, not in the core.
    - **Codemode** (`packages/codemode`) lets the model write a JavaScript script that calls tools, in a WebAssembly sandbox. Only the script's output reaches the context. That answers the tool-output problem from chapter 33 in a different way: ten tool calls don't need ten results in the conversation.
    - **`durable`** (marked experimental) is a harness that commits turns and tool calls to storage before showing them, so a dead process can resume. Compare it with chapter 34's log.
    - **`chord`** builds apps from stateful objects across processes, with hot reload, and `durable` is built on it.

    ## Pi in one paragraph

    You should now be able to say this and defend every clause: *Pi is a layered TypeScript harness. A provider-neutral `ai` layer turns every model into one event stream that never throws. A tiny `agent` loop has two levels for steering and follow-ups, and takes all of its policy through hooks. `coding-agent` adds four sharp tools, a tree-shaped append-only session whose model view is a projection, and compaction as a durable event. Everything else is left to extensions, skills and packages, and it runs with your own permissions.*

    ## What you can do now

    - describe how Pi's system prompt is built, and why it's in sections
    - explain progressive disclosure, and why a skill needs no special tool
    - write a tool gate as an extension, and say what order handlers run in
    - list Pi's five ways of being driven
    - state Pi's safety model, and what project trust does and doesn't cover
    - say what's changed recently: MCP, codemode, durable

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** `packages/coding-agent/docs/extensions.md` and the folder `packages/coding-agent/examples/extensions/` are the best way to learn this. `docs/security.md`, `docs/sdk.md` and `docs/rpc.md` cover the rest. A real test of understanding: write a plan-mode extension.
    """)
    return


if __name__ == "__main__":
    app.run()
