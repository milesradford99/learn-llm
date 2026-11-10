import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="44 · A conversation is a tree")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.agents import attempt, say, trace, user

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 44 · A conversation is a tree

    So far a conversation has been a list in memory. If the program crashes, it's gone. And if the agent spends twenty minutes going the wrong way, all you can do is start again.

    Pi fixes both with one decision: a session is an **append-only log**, and each entry records its parent. That makes it a **tree**. This chapter is about `packages/coding-agent/src/core/session-manager.ts`.

    ## The format

    A session is a file in **JSONL** format: one JSON object per line. In normal use, entries are only added at the end, so once the file exists, a crash risks at most the last line, and everything before it stays readable. (Pi waits for the first real message before writing the file, and migrations can rewrite it.)

    Each entry has an `id` and a `parentId`. In the TypeScript below, `interface` lists the fields a value has, `extends` means "those fields too", and `string | null` means text or nothing:

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

    Messages aren't the only kind of entry. There's also `model_change`, `thinking_level_change`, `compaction`, `branch_summary`, `custom`, and more. Everything that happens goes in the log.

    The session also has a pointer called the **leaf**: the entry you're currently at. A new entry's parent is the leaf, and then it becomes the leaf.

    ## Exercise 1: append

    We'll keep a session as `{"entries": [...], "leaf": ...}`, where the leaf is an id, or `None` (Python's value for "nothing") in a new session. Write `append(session, entry)`:

    1. `entry["id"] = f"e{len(session['entries'])}"`
    2. `entry["parentId"] = session["leaf"]`
    3. Add it to `session["entries"]`, set `session["leaf"]` to its id, and give back the id
    """)
    return


@app.function
def append(session, entry):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("44", "append", append, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Going back without deleting

    Here's the good part. Suppose the agent went wrong after `e1`. To try again, Pi **doesn't delete anything**. It moves the leaf back:

    <!-- source pi packages/coding-agent/src/core/session-manager.ts -->
    ```ts
    branch(branchFromId: string): void {
      if (!this.byId.has(branchFromId)) {
        throw new Error(`Entry ${branchFromId} not found`);
      }
      this.leafId = branchFromId;
    }
    ```

    Moving a pointer! The next entry appended gets `e1` as its parent, so `e1` now has two children, and the old attempt stays in the file. If it turns out the old attempt was better, you can go back to it. The `/tree` command in Pi shows this tree.

    ```
    e0 ─ e1 ─ e2 ─ e3      (the first try)
           └─ e4 ─ e5      (the second try; the leaf is e5)
    ```

    Any entry has exactly one path to the root, so the leaf alone defines a whole conversation. This is the **active branch**, and it's what the model sees. To find it, walk from the leaf to the root through `parentId`, then reverse the list:

    <!-- source pi packages/coding-agent/src/core/session-manager.ts -->
    ```ts
    while (current) {
      path.push(current);
      current = current.parentId ? index.get(current.parentId) : undefined;
    }
    path.reverse();
    return path;
    ```

    ## Exercise 2: active_path

    Write `active_path(session)`, the list of entries from the root to the leaf:

    1. Build a dictionary `by_id` from each entry's id to the entry
    2. `path = []` and `current = session["leaf"]`
    3. `while current is not None:` get the entry, append it, and set `current` to its `parentId`
    4. `path.reverse()` turns the list around in place and gives back nothing, so put it on its own line, then `return path`

    `x is not None` is the standard test for "x isn't nothing".
    """)
    return


@app.function
def active_path(session):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("44", "active_path", active_path, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: branch

    Write `branch(session, entry_id)`: if no entry has that id, `raise ValueError(f"Entry {entry_id} not found")`; otherwise set the leaf. It gives back nothing.
    """)
    return


@app.function
def branch(session, entry_id):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("44", "branch", branch, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: context

    The session holds more than the model needs. Write `context(session)`, the list of `entry["message"]` for each entry on the active path whose `type` is `"message"`.
    """)
    return


@app.function
def context(session):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("44", "context", context, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Watch a retry in action. Both attempts stay in the file (so 3 entries), and the model sees only one. Until exercises 1 to 4 pass, the cell just says so:
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
    ## Why the real version has more steps

    `context` above is a simplification. Pi's `buildSessionProjection` also deals with three things. It applies the newest **compaction** on the path, which replaces older messages with a summary (next chapter). It applies `context_edit` entries, which change what the model sees of an earlier entry while the log keeps the original. And it reads settings off the path, so if you branch back past a model change, the old model returns with it. All of them follow one principle: **the log records what happened; the model's view is computed from it.** The word *projection* in the name says the same thing. It's called **event sourcing** in other fields, and it's what makes all of this cheap.

    When you branch, the abandoned path could hold something useful, so Pi can summarise it and put the summary on the new branch (`branchWithSummary`, with the code in `compaction/branch-summarization.ts`). If you want a separate file instead, `createBranchedSession` copies one path out into a new session, and the header's `parentSession` records where it came from.

    ## Exercise 5: to_jsonl and from_jsonl

    Save and load. Write `to_jsonl(session)`, with one `json.dumps(entry)` per line, joined by `"\n"`. Write `from_jsonl(text)`: `entries` is `json.loads(line)` for each line of `text.split("\n")`, and it gives back `{"entries": entries, "leaf": entries[-1]["id"]}`, so the leaf is the **last line's** id. A saved session always has at least one line. Pi does the same: reopening a session continues from the most recent entry. (Start each function with `import json`.)
    """)
    return


@app.function
def to_jsonl(session):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("44", "to_jsonl", to_jsonl, LOCALE)
    return


@app.function
def from_jsonl(text):
    # ← replace todo() below with your code
    todo()

@app.cell(hide_code=True)
def _():
    check("44", "from_jsonl", from_jsonl, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you can do now

    - explain why Pi uses append-only JSONL, and what a crash can cost
    - explain what the leaf is, and why moving it is enough to branch
    - compute the active branch from a leaf
    - separate the log (what happened) from the projection (what the model sees)
    - say what `branchWithSummary` and `createBranchedSession` are for

    ---
    <!-- go-deeper -->
    **Go deeper (optional):** `packages/coding-agent/docs/session-format.md` documents every entry type, and the session version migrations at the top of `session-manager.ts` show how the format changed over time. In a real session, try `/tree`, `/fork` and `/resume`.
    """)
    return


if __name__ == "__main__":
    app.run()
