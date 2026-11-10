import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="19 · 分词")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.data import stories

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 19 · 分词

    第三部分按真实实验室的做法重建你的小 GPT。第一步和所有模型一样：把文本变成数字。

    到现在为止，我们用的是一份 2,000 个词的词表。它有三个问题。任何别的词都变成 `<unk>`，模型连照抄都做不到。词表不知道 `dog` 和 `dogs` 有关系。而且它只适用于词之间有空格的小写英文。

    反过来，如果改用单个字母，这些问题都能解决，但模型得一个字母一个字母地拼出每个词。所有现代模型的答案都在两者之间：**词元**（token）。它是一段文本，常常是一个完整的词，有时是词的一部分，并且是从数据里学出来的。这个方法叫**字节对编码**（byte-pair encoding，BPE）。它很简单，而且你会亲手写完全部代码。

    ## 从文本到字节

    计算机把一切都存成**字节**（byte），也就是 0 到 255 的整数。**UTF-8** 是用字节书写文本的标准方式：英文字母每个占 1 个字节，其他文字占得更多，汉字占 3 个。`text.encode("utf-8")` 给出字节，`list(...)` 把它们变成普通列表，`bytes([100, 111, 103])` 用数字列表造出字节，`.decode("utf-8")` 把字节变回文本。
    """)
    return


@app.cell
def _():
    list("dog".encode("utf-8")), list("你好".encode("utf-8")), bytes([100, 111, 103]).decode("utf-8")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    从字节出发，意味着任何语言的任何文本都能表示，而起始词表只有 256 项。

    ## 练习 1：to_bytes

    写 `to_bytes(text)`：

    ```
    return list(text.encode("utf-8"))
    ```
    """)
    return


@app.function
def to_bytes(text):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "to_bytes", to_bytes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 想法：合并最常见的一对

    BPE 就是一个动作，不断重复：找出出现次数最多的相邻编号对，给这一对一个新编号，然后到处替换。这里用字母写 `aaabdaaabac`，便于阅读：

    | 步骤 | 最常见的一对 | 新编号 | 文本 |
    |---|---|---|---|
    | 开始 | | | `a a a b d a a a b a c` |
    | 1 | `a a` | `Z` | `Z a b d Z a b a c` |
    | 2 | `Z a` | `Y` | `Y b d Y b a c` |
    | 3 | `Y b` | `X` | `X d X a c` |

    （第 2 步时，`Z a` 和 `a b` 都是 2 次，打成平手，我们取先出现的那个。在代码里，新编号是 256、257 和 258。）常见的字母对最先合并，然后小片段拼成大片段，常见的词最后就成了一个词元。

    ## 练习 2：count_pairs

    写 `count_pairs(ids)`，返回一个字典，从每个相邻的对映射到它的次数。这是第 13 章的配对，加上第 12 章的计数：

    1. `counts = {}`
    2. `for pair in zip(ids, ids[1:]):`，里面写 `counts[pair] = counts.get(pair, 0) + 1`
    3. 循环结束后，`return counts`
    """)
    return


@app.function
def count_pairs(ids):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "count_pairs", count_pairs, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：merge

    现在来替换。`for` 循环一次走一步，而这里有时要跳两步。Python 为此准备了 `while`：`while test:` 只要条件成立就重复里面的行，位置 `i` 由我们自己移动。

    **当心：** 如果 `i` 不增长，条件就一直成立，单元格永远结束不了。遇到这种情况，按单元格角落的方形停止按钮，然后检查改变 `i` 的那几行。

    写 `merge(ids, pair, new_id)`，返回一个新列表。`ids[i] == pair[0]` 和这一对的第一项比较。`and` 要求两个条件都成立，而且第一个不成立时 Python 会跳过其余的，所以 `i < len(ids) - 1` 要放在最前面：这样绝不会越过列表的末尾。`else:` 在 `if` 的条件不成立时运行。

    1. 两行：`out = []` 和 `i = 0`
    2. `while i < len(ids):`
    3. 里面写：`if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:`
    4. 再里面两行：`out.append(new_id)` 和 `i = i + 2`
    5. 和那个 `if` 对齐写 `else:`，里面两行：`out.append(ids[i])` 和 `i = i + 1`
    6. 循环结束后，`return out`
    """)
    return


@app.function
def merge(ids, pair, new_id):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "merge", merge, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：train_bpe

    训练就是把这个动作重复 `count` 次，并在字典里记下每次合并：从一对到新编号。新编号从 256 开始，排在字节之后。`max(counts, key=counts.get)` 给出次数最大的键，写法类似第 12 章的 `sorted`。

    写 `train_bpe(ids, count)`：

    1. `merges = {}`
    2. `for step in range(count):`
    3. 里面五行：`counts = count_pairs(ids)`、`pair = max(counts, key=counts.get)`、`new_id = 256 + step`、`merges[pair] = new_id`、`ids = merge(ids, pair, new_id)`
    4. 循环结束后，`return merges`

    Python 的字典会保持键加入的顺序。这一点很重要，因为之后要按同样的顺序重放合并。
    """)
    return


@app.function
def train_bpe(ids, count):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "train_bpe", train_bpe, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 编码和解码

    要**编码**新文本，先变成字节，再按顺序重放每一次合并。`d.items()` 逐个给出字典的键值对，所以 `for pair, new_id in merges.items():` 同时得到两者。

    要**解码**，需要知道每个编号背后的文本。造一张表：小于 256 的编号是单个字节，每个合并出的编号是它的两部分连起来。一对的两部分编号总是更小，所以按顺序走过合并时，它们总是已经准备好了。`bytes([i])` 造出一个字节，`+` 连接字节。`errors="replace"` 表示损坏的序列显示为 � 标记，而不是报错停下。

    ## 练习 5：encode

    写 `encode(text, merges)`：

    1. `ids = to_bytes(text)`
    2. `for pair, new_id in merges.items():`，里面写 `ids = merge(ids, pair, new_id)`
    3. 循环结束后，`return ids`

    ## 练习 6：decode

    写 `decode(ids, merges)`。这张表可以用列表，因为第 256 位正好接在第 255 位后面填入：

    1. `table = [bytes([i]) for i in range(256)]`
    2. `for pair, new_id in merges.items():`，里面写 `table.append(table[pair[0]] + table[pair[1]])`
    3. 循环结束后，`joined = b''.join([table[i] for i in ids])`
    4. `return joined.decode("utf-8", errors="replace")`

    （`b''.join(...)` 就是第 12 章的 join，用于字节，中间什么也不加。）
    """)
    return


@app.function
def encode(text, merges):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "encode", encode, LOCALE)
    return


@app.function
def decode(ids, merges):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "decode", decode, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 在故事上训练

    先完成全部六个练习。按钮在 300 篇故事上训练 500 次合并，最多大约一分钟。完成之前，需要它的单元格会显示 `Ancestor stopped`，这是正常的。如果之后改了练习，要再按一次按钮。单元格显示文本的字节数、变成的词元数和两者的比值。
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="训练分词器")
    train_button
    return (train_button,)


@app.cell
def _(train_button):
    mo.stop(not train_button.value, mo.md("*按下按钮开始训练。*"))
    train_stories, test_stories = stories()
    text = " ".join(train_stories[:300])
    raw = to_bytes(text)
    merges = train_bpe(raw, 500)
    tokens = encode(text, merges)
    len(raw), len(tokens), round(len(raw) / len(tokens), 2)
    return merges, test_stories


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 236,000 个字节变成大约 78,000 个词元：平均每个词元 3 个字节。下面是前 15 次和后 15 次合并，每个都解码出它的文本。空格是词元的一部分，所以一个词和它后面的空格常常合成一个词元：
    """)
    return


@app.cell
def _(merges):
    all_tokens = [decode([new_id], merges) for new_id in merges.values()]
    all_tokens[:15], all_tokens[-15:]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    早期的合并是字母片段，比如 `e ` 和 `th`。后期的很多是完整的词，比如 `when `（当）和 `lived `（住），有些甚至连起两个词，比如 `smiled and `（笑了，并且）。没有人告诉它什么是词。

    现在用没见过的文本测试。单元格把一个句子切成词元，再检查一篇没见过的故事解码后是否和原文完全相同。最后两项是一句中文。
    """)
    return


@app.cell
def _(merges, test_stories):
    def pieces(sentence):
        return [decode([token], merges) for token in encode(sentence, merges)]

    story = test_stories[0]
    (
        pieces("The dog chased a xylophone."),
        decode(encode(story, merges), merges) == story,
        pieces("小狗跑了"),
        decode(encode("小狗跑了", merges), merges),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - 常见的词是单个词元，连着它的空格。`chased`（追）在这些故事里不常见，所以被切成了几段。`xylophone`（木琴）从没见过，但什么也没坏：它被切成更小的片段，不需要 `<unk>`。
    - 整篇故事完全还原，包括大写和标点，这些都是我们的词表丢掉的。
    - 中文也还原了。我们的合并没有学过中文，所以它保持 12 个单独的字节，在列表里每个都显示为 �，因为汉字的一个字节没法单独显示。解码到一起，它们又完整了。如果用中文文本训练，它也会用同样的方式学出中文词元。

    ## 选择词表大小

    合并次数是一个超参数（第 13 章）。合并越多，同样的文本词元越少，能容纳 32 个词元的模型就看得越远，但表更大，要学的稀有词元也更多。英文模型通常用几万个。

    ## 让排行榜保持公平

    到现在为止，困惑度都是*按词*算的。用词元时，模型每个词要走更多步，每一步更容易，所以按词元算的困惑度没法和排行榜比较。因此，当我们用词元重新训练时，会把一篇故事的总意外程度加起来，再除以*词*数，和以前一样，让排行榜保持诚实。

    ## 接下来

    第 16 章的模型很小，有些选择也很简单。**第 20 章**把它升级成现代设计：更好的归一化，更好的位置处理方式，学习率的调整计划，还能保存一次训练并在以后继续。

    ## 你现在会做的事

    - 解释为什么词表和单个字母都不够好
    - 把任何语言的文本变成 UTF-8 字节并变回来
    - 写一个 BPE 训练器：统计相邻的对，合并最常见的，重复
    - 写按自己节奏前进的 `while` 循环
    - 按顺序重放合并，对新文本编码和解码
    - 说出词表大小要权衡什么

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 用于文本的 BPE 出自 Sennrich 等（2015），GPT-2 的分词器还会先用正则表达式切分文本，让合并不会跨过某些边界。真实的分词器把 `<|endoftext|>` 这样的*特殊词元*放在合并之外，并通过更新配对计数（而不是像我们这样每步都重新数）来加快训练。字典靠*哈希*（hashing）一下子找到键。
    """)
    return


if __name__ == "__main__":
    app.run()
