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

    到现在为止，我们用的是一份 2,000 个词的词表，它有三个局限：词表之外的词全都会变成 `<unk>`，模型连原样复述都做不到；在词表眼里，`dog` 和 `dogs` 是两个毫不相干的编号；而且它只适用于词与词之间用空格隔开的小写英文。

    反过来，如果改用单个字母，这三个问题都能解决，但模型就得一个字母一个字母地拼出每个词。所有现代模型采用的都是折中方案：**词元**（token）——它是一段文本，常常是一个完整的词，有时是词的一部分，完全从数据中自动学出。实现它的方法叫**字节对编码**（byte-pair encoding，BPE）。原理很简单，你会亲手写完它的全部代码。

    ## 从文本到字节

    计算机把所有内容都存成**字节**（byte），也就是 0 到 255 之间的整数。**UTF-8** 是目前通用的文本字节编码标准：英文字母每个占 1 个字节，其他文字占得更多，一个汉字通常占 3 个字节。

    /// admonition | 🧰 工具箱：文本和字节
    - `text.encode("utf-8")` 把字符串转成 UTF-8 字节串。Python 显示字节串时，会在引号前多一个 `b`，比如 `b'dog'`。
    - `list(...)` 把字节串拆成普通的整数列表；反过来，`bytes([100, 111, 103])` 把整数列表拼回字节串。
    - `.decode("utf-8")` 把字节串解码回可读的字符串。
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
    从字节出发，意味着任何语言的任何文本都能表示，而且起始词表只需要 256 项。

    ## 练习 1：to_bytes

    编写 `to_bytes(text)`：接收一个字符串，返回它对应的 UTF-8 字节整数列表：

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
        r"""把工具箱里的两个工具套起来用：先给文本编码，再把结果转成列表。""",
        r"""只需要一行，开头是 `return list(`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 核心思路：合并最常见的一对

    BPE 其实就是把同一个动作不断重复：找出出现次数最多的相邻编号对，给它分配一个新编号，再把文本里所有这一对都替换掉。为了方便阅读，下面先用字母演示 `aaabdaaabac`：

    | 步骤 | 最常见的一对 | 新编号 | 文本 |
    |---|---|---|---|
    | 开始 | | | `a a a b d a a a b a c` |
    | 1 | `a a` | `Z` | `Z a b d Z a b a c` |
    | 2 | `Z a` | `Y` | `Y b d Y b a c` |
    | 3 | `Y b` | `X` | `X d X a c` |

    有两个细节需要注意。第一，在 `a a a` 里，`a a` 既可以从第 1 个 `a` 开始，也可以从第 2 个开始，但每个字母只能参与一次合并：我们从左到右扫描，一旦把一对替换掉，就直接跳到这对的*后面*继续看，不再复用刚才合并过的字母，所以 `a a a` 会变成 `Z a`。第二，第 2 步里 `Z a` 和 `a b` 都出现了 2 次、打成平手，此时取最先出现的那一对。写进代码时，新编号会依次取 256、257、258……常见的字母对最先合并，小片段再拼成大片段，常见词最后就会自然合并成单个词元。

    ## 动手算：两次合并

    轮到你了，这次是 12 个字母 `a b c a b c a b d a b c`。数一数最常见的一对出现了几次，把它合并成 `Z`，再数一数还剩多少个词元（合并后的一对算一个词元）。然后在新的文本上再做一次，把新的一对记作 `Y`。
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
    两次合并之后，文本变成了 `Y Y Z d Y`：从 12 个缩减到 5 个词元，其中 `Y` 代表 `a b c`。

    ## 练习 2：count_pairs

    编写 `count_pairs(ids)`：`ids` 是一个编号列表，返回一个字典——键是圆括号括起来的一对相邻编号（比如 `(1, 2)`），值是这一对出现的次数：

    > `count_pairs([1, 2, 1, 2, 3])` 得到 `{(1, 2): 2, (2, 1): 1, (2, 3): 1}`

    小提示：`zip(xs, xs[1:])` 可以把列表里每个元素和它的下一个元素配成一对（见第 13 章）；字典的 `d.get(key, 0)` 能在键还不存在时返回默认值 0，方便累计次数（见第 12 章）。
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
        r"""第 13 章的 `zip(ids, ids[1:])` 会给出每一对相邻编号；第 12 章的 `counts.get(key, 0) + 1` 可以给可能还不存在的计数加 1。`zip` 产出的每一对都可以直接拿来当字典的键。""",
        r"""一共 4 行，写法和第 12 章的 `count_words` 一样，循环头写成 `for pair in zip(ids, ids[1:]):`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 自己控制步长的循环

    接下来写替换。`for` 循环每次只能老老实实前进 1 步，但合并时有时得一次跳 2 步：把一对合并掉之后，要把这对里的两个元素都跳过去。

    /// admonition | 🧰 工具箱：while、and、else
    - `while test:` 只要条件成立，就反复执行下面缩进的代码。循环本身不会自动推进，你必须在循环体里修改条件依赖的变量，否则条件会一直成立，变成死循环。
    - `a and b` 只有两个条件都成立时才为真。Python 先检查 `a`，只要 `a` 不成立，就根本不会去看 `b`。
    - 与 `if` 对齐的 `else:` 会在 `if` 的条件*不*成立时，执行它下面缩进的代码。
    ///

    下面的单元格同时用到了这三样。它从 20 倒数到 0：只要当前数字大于 5，且已经走的步数不到 2 步，就一次减 5；否则一次减 1。最后输出一共走了多少步。
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
    先是 20、15、10，接着一步一步减 10 次到 0，一共走了 12 步。

    **注意：** 如果 `while` 循环体内没有修改判断条件用到的变量，条件就会一直成立，单元格永远跑不完（死循环）。万一遇到这种情况，点一下单元格角落里的方形停止按钮，然后检查负责推进循环的那几行。

    ## 练习 3：merge

    编写 `merge(ids, pair, new_id)`：`ids` 是编号列表，`pair` 是圆括号括起来的两个编号，`new_id` 是用来替换它们的新编号。函数返回一个新列表：

    > `merge([1, 2, 3, 1, 2], (1, 2), 9)` 得到 `[9, 3, 9]`，`merge([1, 1, 1], (1, 1), 9)` 得到 `[9, 1]`

    用一个由你自己控制的位置指针 `i`，从 0 开始沿着 `ids` 向后扫描，边走边构造新列表：

    - 如果位置 `i` 正好是这一对的起点——也就是 `ids[i]` 等于 `pair` 的第一个编号，且下一项等于它的第二个编号——就把 `new_id` 追加进新列表，并让 `i` **跳两步**；
    - 否则把 `ids[i]` 原样放进新列表，让 `i` 前进一步。

    用最简单的例子 `ids = [1, 1, 1]`、合并 `(1, 1) -> 256` 走一遍指针 `i` 的变化（注意 `i` 的轨迹是 `0 → 2 → 3`，而不是 `0 → 1 → 2`）：

    | 轮次 | 当前 `i` | 条件检查 | 执行动作 | 当前结果 | 下一轮 `i` |
    |---|---|---|---|---|---|
    | 第 1 轮 | `0` | `i < 2` 且 `ids[0], ids[1]` 正好是 `(1, 1)` ✓ | 追加 `256`，跳两步 | `[256]` | `2` |
    | 第 2 轮 | `2` | `i < 2` 不成立（已是最后一项，没有下一项） | 追加 `ids[2]`（即 `1`），走一步 | `[256, 1]` | `3` |
    | 第 3 轮 | `3` | `i < len(ids)` 不成立 | 循环结束 | `[256, 1]` | — |

    走到最后一项时后面已经没有元素，如果直接去取 `ids[i + 1]` 就会越界报错。所以务必把 `i < len(ids) - 1` 写在 `and` 的**最前面**——只要它不成立，Python 就会立刻停下，根本不会去访问 `ids[i + 1]`。
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
        r"""先准备一个空列表并设 `i = 0`，然后写 `while i < len(ids):`。循环里放一个用 `and` 连起三个条件的 `if`，再配一个 `else`。两个分支各自往新列表追加一个元素，并推进 `i`。`pair[0]` 和 `pair[1]` 分别是这一对的两个编号。""",
        r"""一共 10 行。三个条件里的第一个必须是 `i < len(ids) - 1`（确认后面还有元素），放在最前面才能避免越界访问 `ids[i + 1]`。如果单元格一直跑不完，检查是不是漏了在某个分支里更新 `i`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 训练：不断重复这个动作

    训练就是把上面的动作重复若干次，并用一个字典记下每次合并（键是相邻编号对，值是新编号）。因为 0 到 255 已经分给了单个字节，新编号就顺着从 256 开始排。

    /// admonition | 🧰 工具箱：值最大的那个键
    `max(d, key=d.get)` 返回字典里值最大的那个键（写法类似第 12 章的 `sorted`）。例如对 `ages = {"ann": 30, "bo": 12, "cy": 30}`，它返回 `"ann"`：打成平手时，最先存进字典的键胜出。
    ///

    ## 练习 4：train_bpe

    编写 `train_bpe(ids, count)`：`ids` 是编号列表，`count` 是要学习的合并次数，返回一个记录每次合并（编号对 → 新编号）的字典。这 `count` 轮里，每一轮依次做五件事：

    1. 统计当前 `ids` 里每对相邻编号的出现次数；
    2. 找出出现次数最多的那一对；
    3. 给它分配下一个新编号：第 1 轮是 256，第 2 轮是 257，以此类推；
    4. 把这对编号和它的新编号记进字典；
    5. 用新编号替换 `ids` 里的这一对，让下一轮在更新后的序列上继续统计。

    前面表格里的例子正好对应：`train_bpe(to_bytes("aaabdaaabac"), 3)` 得到 `{(97, 97): 256, (256, 97): 257, (257, 98): 258}`，因为字母 `a` 的字节是 97，`b` 是 98。

    Python 字典会保留键插入的先后顺序。这一点很关键，因为后面编码时必须按完全相同的顺序重放这些合并。
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
        r"""先建一个空字典，然后写 `for step in range(count):`，循环体内正好五行，对应上面五个步骤，分别调用你的 `count_pairs` 和 `merge`。步数 `step` 从 0 开始，所以新编号就是 `256 + step`。""",
        r"""一共 8 行。循环体最后一行要把 `merge` 的结果重新存回 `ids`——少了这一行，每一轮找到的都会是同一对。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 编码和解码

    要**编码**新文本，先把文本转成字节列表，再严格按照训练时学到的先后顺序重放每一次合并。这个顺序不能乱：比如在前面的例子里，如果先去合并 `(256, 97)`，序列里当时根本还没有 `256`——必须先跑完 `(97, 97) -> 256`，`256` 才会出现。

    要**解码**，就得查出每个编号背后的原始字节。我们可以按顺序建一张查表列表 `table`：0 到 255 号位置放单个字节，之后每遇到一次合并，就把这对编号对应的字节拼起来放到表尾。因为组成一对的两个编号一定比新编号小，按顺序建表时它们肯定已经在表里了。举个 2 步的小例子（已知 `table[97] = b'a'`）：

    - 合并 `(97, 97) -> 256`：`table[256] = table[97] + table[97] = b'aa'`
    - 合并 `(256, 97) -> 257`：`table[257] = table[256] + table[97] = b'aaa'`

    另外有一个 UTF-8 的细节：一个汉字占 3 个字节，如果把这 3 个字节拆开、只拿其中 1 个字节单独解码，它拼不出完整的字符，就会显示成替换符 �。加上 `errors="replace"` 可以让程序遇到这种残缺字节时显示 �，而不是直接报错中断。

    /// admonition | 🧰 工具箱：字典里的键值对，以及连接字节
    - `d.items()` 依次给出字典里的每一组键和值，写成 `for name, age in ages.items():` 就能同时拿到两者。
    - `+` 可以直接拼接字节串：`bytes([104]) + bytes([105])` 得到 `b'hi'`。
    - `b''.join(pieces)` 把一个字节串列表无缝拼成一个大字节串，比如 `b''.join([b'hi', b'!'])` 得到 `b'hi!'`（就是第 12 章的 `join`，只是换成字节版）。
    - `.decode("utf-8", errors="replace")` 把字节串解码为字符串，遇到残缺的字节序列时显示为 �，不会报错停下。
    ///

    ## 练习 5：encode

    编写 `encode(text, merges)`：`text` 是字符串，`merges` 是 `train_bpe` 返回的合并字典。先把文本变成字节列表，再按顺序执行每一次合并，返回最终的编号列表：

    > 使用前面表格里的三次合并，`encode("aaabdaaabac", merges)` 得到 `[258, 100, 258, 97, 99]`

    ## 练习 6：decode

    编写 `decode(ids, merges)`，把编号列表还原回字符串：

    1. 建一张查表列表：位置 $i$ 上存放单个字节 $i$（$i$ 从 0 到 255）。
    2. 按顺序遍历每一次合并，把这一对里两个编号各自的字节片段拼在一起，追加到表的末尾。（第一次合并的新编号正好是 256，刚好落在列表下一个空位上，所以列表下标和编号始终一一对应。）
    3. 把 `ids` 里每个编号查到的字节片段全拼起来，按 UTF-8 解码，遇到残缺字节用替换符代替。

    > 使用同样的合并字典，`decode([258, 100, 258, 97, 99], merges)` 得到 `"aaabdaaabac"`
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
        r"""先用 `to_bytes(text)` 拿到字节列表，再遍历 `merges.items()`（每次拿到一对编号和它的新编号），在循环里调用你的 `merge`。""",
        r"""一共 4 行。和 `train_bpe` 一样，每次都要把 `merge` 的结果重新存回 `ids`。""",
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
        r"""查表一开始是 `[bytes([i]) for i in range(256)]`。在遍历合并的循环里，往表尾追加 `table[pair[0]] + table[pair[1]]`。这里用不到 `new_id`，列表下标本身就对应着编号。""",
        r"""循环之后还有两行：用 `b''.join` 把 `[table[i] for i in ids]` 拼起来，再返回结果的 `.decode("utf-8", errors="replace")`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 在故事上训练

    请先完成上面全部六个练习。按下按钮会在 300 篇故事上训练 500 次合并，大约需要一分钟以内。点击之前，后面依赖它的单元格会显示 `Ancestor stopped`，这是正常现象。如果之后修改了练习代码，记得再点一次按钮。这个单元格会输出原文的字节数、编码后的词元数，以及两者的压缩比。
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
    大约 236,000 个字节被压缩成了大约 78,000 个词元，平均每个词元包含 3 个字节。下面列出最先学到的 15 次合并和最后学到的 15 次合并，并把每个词元解码成文本。空格也算在词元里，所以一个词常常会和它后面的空格合成同一个词元：
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
    早期的合并大多是高频字母组合，比如 `e ` 和 `th`；后期很多已经是完整的单词，比如 `when `（当）和 `lived `（住），有些甚至连起了两个词，比如 `smiled and `（笑了，并且）。全程没有人告诉它什么是“词”。

    现在拿模型没见过的文本做个测试：把一个句子切成词元，再检查一篇未见过的故事经过“编码 → 解码”后是否与原文一字不差。最后两项是一句中文。
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
    - 常见词直接成了单个词元（还带着后面的空格）。`chased`（追）在这些故事里不常见，所以被拆成了几段。`xylophone`（木琴）在训练集里从未出现过，但分词器依然游刃有余：它被切分成更小的片段，完全不需要 `<unk>`。
    - 整篇未见过的故事一字不差地还原了回来，连大写字母和标点符号都完好无损——之前用固定词表时，这些信息全被丢掉了。
    - 中文也完好无损地还原了。因为我们的合并规则里没学过中文，4 个汉字保持为 12 个单独的字节；单个字节拆开看时无法独立显示，所以在片段列表里每个都显示成替换符 �，但把 12 个字节拼回一起解码，就又恢复成了完整的汉字。如果用中文语料训练，BPE 同样会学出中文词元。

    ## 选择词表大小

    合并次数是一个超参数（训练前由我们自己定好的设置，见第 13 章）。合并次数越多，同样一段文本拆出的词元越少，能容纳 32 个词元的模型就能看得更远；代价是词表更大，里面要学习的低频词元也会变多。英文大模型通常把词表设在几万个词元。

    ## 让排行榜保持公平

    到现在为止，我们算的都是*按词*计算的困惑度（困惑度 = $e^{\text{平均意外程度}}$，见第 13 章）。改用词元后，一个词往往要分好几步预测，每一步都更简单，所以“按词元算的困惑度”不能直接拿去和排行榜对比。因此，后面用词元重新训练模型时，我们依然把整篇故事的总意外程度加起来，再除以*词*数，保证排行榜上的成绩真正公平可比。

    ## 接下来

    第 17 章的小模型在不少地方都做了简化。**第 20 章**我们会把它升级为现代架构：换上更好的归一化、更合理的位置编码方式，以及带门控的前馈思考层。接着在第 21 章训练它，加上学习率调度，并支持保存和续训。

    ## 你现在会做的事

    - 解释为什么固定词表和单个字母各有什么短板
    - 把任意语言的文本转成 UTF-8 字节并还原
    - 手算 BPE 合并，并写出训练器：统计相邻对、合并最常见的一对、不断重复
    - 写出能自己控制步长的 `while` 循环
    - 按顺序重放合并规则，完成新文本的编码与解码
    - 说清词表大小背后的权衡

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 用于文本的 BPE 最早出自 Sennrich 等（2015）；GPT-2 的分词器还会先用正则表达式切分文本，避免合并跨过某些边界。真实的分词器会把 `<|endoftext|>` 这类*特殊词元*留在合并规则之外，并且在训练时只局部更新受影响的配对计数，而不是像我们这样每轮从头重数。字典之所以能瞬间查到键，背后靠的是*哈希*（hashing）。
    """)
    return


if __name__ == "__main__":
    app.run()
