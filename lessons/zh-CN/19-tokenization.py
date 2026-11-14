import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="19 · 分词")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import hand, todo
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

    计算机把一切都存成**字节**（byte），也就是 0 到 255 的整数。**UTF-8** 是用字节书写文本的标准方式：英文字母每个占 1 个字节，其他文字占得更多，汉字占 3 个。

    /// admonition | 🧰 工具箱：文本和字节
    - `text.encode("utf-8")` 给出文本的字节。Python 显示字节时，像显示文本一样，只是前面多一个 `b`，比如 `b'dog'`。
    - `list(...)` 把字节变成普通的数字列表，`bytes([100, 111, 103])` 用这样的列表造出字节。
    - `.decode("utf-8")` 把字节变回文本。
    ///
    """)
    return


@app.cell
def _():
    as_bytes = "dog".encode("utf-8")
    as_bytes, list(as_bytes), list("你好".encode("utf-8")), bytes([100, 111, 103]).decode("utf-8")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    从字节出发，意味着任何语言的任何文本都能表示，而起始词表只有 256 项。

    ## 练习 1：to_bytes

    写 `to_bytes(text)`。它接收一个字符串，把它的 UTF-8 字节作为普通的数字列表返回：

    > `to_bytes("dog")` 得到 `[100, 111, 103]`
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
    hand.stuck([
        r"""用工具箱里的两个工具，一个套一个：先给文本编码，再把结果变成列表。""",
        r"""只要一行，开头是 `return list(`。""",
    ], LOCALE)
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

    有两个细节。在 `a a a` 里，`a a` 这一对可以从第一个 `a` 开始，也可以从第二个开始，但一个字母只能用一次：我们从左往右走，替换掉一对之后，从它的*后面*接着走。所以 `a a a` 变成 `Z a`。另外，第 2 步时 `Z a` 和 `a b` 都是 2 次，打成平手，我们取先出现的那个。在代码里，新编号是 256、257 和 258。常见的字母对最先合并，然后小片段拼成大片段，常见的词最后就成了一个词元。

    ## 动手算：两次合并

    轮到你了，这次是 12 个字母 `a b c a b c a b d a b c`。数一数最常见的一对出现了几次。把它合并成 `Z`，再数一数还剩多少个词元（合并后的一对算一个词元）。然后在新文本上再做一次，把新的一对叫作 `Y`。
    """)
    return


@app.cell(hide_code=True)
def _():
    merge_boxes = hand.boxes(['最常见的一对的次数', '第 1 次合并后的词元数', '现在最常见的一对的次数', '第 2 次合并后的词元数'])
    return (merge_boxes,)


@app.cell(hide_code=True)
def _(merge_boxes):
    hand.marked(merge_boxes, {'最常见的一对的次数': 4, '第 1 次合并后的词元数': 8, '现在最常见的一对的次数': 3, '第 2 次合并后的词元数': 5}, {
        '最常见的一对的次数': r"""`a b` 比 `b c` 出现得多，因为有一个 `a b` 后面跟的是 `d`""",
        '第 1 次合并后的词元数': r"""4 次合并，每次把两个词元变成一个，所以 12 要减去 4""",
        '现在最常见的一对的次数': r"""现在文本是 `Z c Z c Z d Z c`。`Z c` 出现了几次？""",
        '第 2 次合并后的词元数': r"""3 次合并，每次再少一个词元""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    两次合并之后，文本是 `Y Y Z d Y`：5 个词元，而不是 12 个，`Y` 代表 `a b c`。

    ## 练习 2：count_pairs

    写 `count_pairs(ids)`。`ids` 是一个编号列表。它返回一个字典：键是圆括号里的一对相邻编号，值是这一对出现的次数：

    > `count_pairs([1, 2, 1, 2, 3])` 得到 `{(1, 2): 2, (2, 1): 1, (2, 3): 1}`

    这是第 13 章把列表和自己配对的办法，加上第 12 章计数的办法。
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
    hand.stuck([
        r"""第 13 章：`zip(ids, ids[1:])` 给出每一对相邻的编号。第 12 章：`counts.get(key, 0) + 1` 给一个可能还不存在的次数加 1。`zip` 给出的每一对可以直接当键用。""",
        r"""一共 4 行，形状和第 12 章的 `count_words` 一样，循环是 `for pair in zip(ids, ids[1:]):`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 按自己的步子走的循环

    现在来替换。`for` 循环一次走一步，而合并有时要跳两步：替换掉一对之后，要把它的两半都跳过去。

    /// admonition | 🧰 工具箱：while、and、else
    - `while test:` 只要条件成立，就重复它下面缩进的那些行。没有任何东西会自己往前走：里面的行必须改变点什么，否则条件永远成立。
    - `a and b` 只有两个条件都成立时才成立。Python 先检查 `a`，如果它不成立，就根本不看 `b`。
    - 和 `if` 对齐的 `else:`，在 `if` 的条件*不*成立时运行它下面缩进的行。
    ///

    下面的单元格把三样都用上了。它从 20 数到 0：当这个数大于 5、而且走过的步数还不到 2 时，每次减 5；否则每次减 1。它显示一共走了多少步。
    """)
    return


@app.cell
def _():
    n_demo = 20
    steps_demo = 0
    while n_demo > 0:
        if n_demo > 5 and steps_demo < 2:
            n_demo = n_demo - 5
        else:
            n_demo = n_demo - 1
        steps_demo = steps_demo + 1
    steps_demo
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    先是 20、15、10，然后一步一步走 10 步到 0：一共十二步。

    **当心：** 如果 `while` 循环里面没有任何一行去改变条件所看的东西，条件就一直成立，单元格永远结束不了。遇到这种情况，按单元格角落的方形停止按钮，然后检查让你往前走的那几行。

    ## 练习 3：merge

    写 `merge(ids, pair, new_id)`。`ids` 是一个编号列表，`pair` 是圆括号里的两个编号，`new_id` 是用来替换它们的编号。它返回一个新列表：

    > `merge([1, 2, 3, 1, 2], (1, 2), 9)` 得到 `[9, 3, 9]`，`merge([1, 1, 1], (1, 1), 9)` 得到 `[9, 1]`

    用一个由你自己移动的位置 `i`，从 0 开始沿着 `ids` 走，边走边造新列表：

    - 如果这一对从位置 `i` 开始，也就是 `ids[i]` 是这一对的第一个编号，而且下一项是它的第二个编号，就把 `new_id` 加进新列表，并且**跳两步**；
    - 否则把 `ids[i]` 抄进新列表，走一步。

    走到最后一个位置时，后面已经没有下一项了，去取它会报错。所以要先确认*有*下一项，再去看它。
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
    hand.stuck([
        r"""先是一个空列表和 `i = 0`，然后是 `while i < len(ids):`。里面是一个用 `and` 连起三个条件的 `if`，再加一个 `else`。两个分支各加入一样东西，并给 `i` 加上一个数。`pair[0]` 和 `pair[1]` 是这一对的两个编号。""",
        r"""一共 10 行。三个条件里的第一个是 `i < len(ids) - 1`，意思是后面还有一项。它必须排在最前面，这样没有下一项时，Python 就不会去看 `ids[i + 1]`。如果单元格一直不结束，说明有一个分支没有给 `i` 加数。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 训练：重复这个动作

    训练就是把这个动作重复若干次，并在字典里记下每次合并：从一对到新编号。新编号从 256 开始，排在字节之后。

    /// admonition | 🧰 工具箱：值最大的那个键
    `max(d, key=d.get)` 给出字典里值最大的那个键，写法类似第 12 章的 `sorted`。对 `ages = {"ann": 30, "bo": 12, "cy": 30}`，它给出 `"ann"`：打成平手时，先存进去的键胜出。
    ///

    ## 练习 4：train_bpe

    写 `train_bpe(ids, count)`。`ids` 是一个编号列表，`count` 是要学的合并次数。它返回一个字典，从每个被合并的对映射到它的新编号。`count` 步里的每一步都这样做：

    1. 数出当前 `ids` 里的各个对；
    2. 取最常见的一对；
    3. 给它下一个新编号：第一步是 256，第二步是 257，依此类推；
    4. 把它记进字典；
    5. 在 `ids` 里替换这一对，这样下一步数的就是新文本里的对。

    上面的表格就是例子：`train_bpe(to_bytes("aaabdaaabac"), 3)` 得到 `{(97, 97): 256, (256, 97): 257, (257, 98): 258}`，因为 `a` 是字节 97，`b` 是字节 98。

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
    hand.stuck([
        r"""先是一个空字典，然后是 `for step in range(count):`，里面五行，每个编了号的步骤一行，用上你的 `count_pairs` 和 `merge`。步数 `step` 从 0 开始，所以新编号是 256 加上它。""",
        r"""一共 8 行。循环里的最后一行把 `merge` 的结果存回 `ids`。少了这一行，每一步找到的都是同一对。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 编码和解码

    要**编码**新文本，先变成字节，再按学到的先后顺序重放每一次合并。

    要**解码**，需要知道每个编号背后的字节。造一张表：小于 256 的编号是单个字节，每个合并出的编号是它的两部分连起来。一对的两部分编号总是更小，所以按顺序走过合并时，它们总是已经准备好了。

    /// admonition | 🧰 工具箱：字典里的键值对，以及连接字节
    - `d.items()` 逐个给出字典的键值对，所以 `for name, age in ages.items():` 同时得到两者。
    - `+` 连接字节：`bytes([104]) + bytes([105])` 是 `b'hi'`。
    - `b''.join(pieces)` 把一个由字节组成的列表连成一个，中间什么也不加。它就是第 12 章的 `join`，用在字节上。
    - `.decode("utf-8", errors="replace")` 把损坏的序列显示成 � 标记，而不是报错停下。
    ///

    ## 练习 5：encode

    写 `encode(text, merges)`。`text` 是一个字符串，`merges` 是 `train_bpe` 返回的东西。它返回一个编号列表：文本的字节，按顺序做完每一次合并之后的结果。

    > 用表格里的三次合并，`encode("aaabdaaabac", merges)` 得到 `[258, 100, 258, 97, 99]`

    ## 练习 6：decode

    写 `decode(ids, merges)`，也就是反过来走：它返回字符串形式的文本。

    1. 用列表造这张表：位置 $i$ 上放单个字节 $i$，$i$ 从 0 到 255。
    2. 按顺序走过每次合并。每次合并在表的末尾加一项：这一对的第一个编号对应的条目，连上第二个编号对应的条目。（第一次合并的编号是 256，正好是下一个空位，所以列表的位置和编号始终对得上。）
    3. 把所有 `ids` 对应的条目连起来，再把结果按 UTF-8 解码，损坏的序列用标记代替。

    > 用同样的合并，`decode([258, 100, 258, 97, 99], merges)` 得到 `"aaabdaaabac"`
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


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""从 `to_bytes(text)` 开始。然后对 `merges.items()` 做一个循环，它每次交给你一对和它的新编号，循环里用你的 `merge`。""",
        r"""一共 4 行。和 `train_bpe` 里一样，每次都要把 `merge` 的结果存回 `ids`。""",
    ], LOCALE)
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
    hand.stuck([
        r"""表一开始是 `[bytes([i]) for i in range(256)]`。在走过合并的循环里，加入 `table[pair[0]] + table[pair[1]]`。这里用不到 `new_id`：列表里的位置已经起了它的作用。""",
        r"""循环之后还有两行：用 `b''.join` 把 `[table[i] for i in ids]` 连起来，再返回结果的 `.decode("utf-8", errors="replace")`。""",
    ], LOCALE)
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

    第 17 章的模型很小，有些选择也很简单。**第 20 章**把它升级成现代设计：更好的归一化，更好的位置处理方式，还有带门的思考层。接着第 21 章来训练它：用上学习率的调整计划，还能保存一次训练并在以后继续。

    ## 你现在会做的事

    - 解释为什么词表和单个字母都不够好
    - 把任何语言的文本变成 UTF-8 字节并变回来
    - 亲手做 BPE 的合并，并写出训练器：统计相邻的对，合并最常见的，重复
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
