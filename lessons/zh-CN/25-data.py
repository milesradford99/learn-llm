import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="25 · 数据")

with app.setup(hide_code=True):
    import zlib

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.diagrams import pipeline
    from learn_llm.rawdata import common_words, pile_tokens, raw_pile, seeded_hash
    from learn_llm.scaling import small_run, story_tokens
    from learn_llm.tokenizer import to_words

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 25 · 数据

    这门课里的每个模型，训练用的都是干净的文本：几千篇故事，每篇都不一样，每篇都是好英文。为真实的模型收集来的文本，完全不是这样。
    它是由程序去访问数十亿个网页，找到什么就留下什么收集来的。同一篇文章会出现在上百个网站上。每个页面都带着菜单和公告。相当大一部分根本不是语言。

    模型读什么，就学到什么。第 24 章测量了文本变多能换来什么。这一章讲的是文本本身值多少，以及怎样分三步，把一堆原始文本变成值得训练的数据：

    1. 去掉**完全相同的副本**；
    2. 去掉**近似重复**（near-copy），也就是只差几个词的文本；
    3. 去掉**垃圾文本**（junk）。

    ## 一堆原始文本

    我们用故事造了一小堆原始文本：2,000 篇，其中只有一半是我们想要的。
    """)
    return


@app.cell
def _():
    texts, kinds = raw_pile()
    [texts[kinds.index(kind)][:75] for kind in ["copy", "near-copy", "gibberish", "menu", "spam"]]
    return kinds, texts


@app.cell(hide_code=True)
def _(kinds, texts):
    kind_of = dict(zip(texts, kinds))

    def tally(some_texts):
        counts = {}
        for text in some_texts:
            counts[kind_of[text]] = counts.get(kind_of[text], 0) + 1
        return counts
    return kind_of, tally


@app.cell
def _(tally, texts):
    tally(texts)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第一个单元格显示每种不想要的文本的开头。第二个单元格数这堆文本：1,000 篇故事，250 篇是其中一些故事的副本，只是空格或大小写不同，250 篇是改了几个词的近似重复，还有 500 篇垃圾文本，分三种。

    `kinds` 里存着每篇文本的答案，`tally` 负责数。**真实的文本堆没有这样的标签。**我们是故意埋下这些问题，好让你数一数自己的清理抓到了多少。你的函数永远不会去看 `kinds`。

    ## 完全相同的副本，与哈希

    要找副本，最直接的办法是把每篇文本和其他每篇都比一遍。2,000 篇文本就是两百万次比较。对十亿个网页来说，这根本行不通。

    标准的工具是**哈希函数**（hash function）：一套固定的做法，能把任何文本变成一个数。同一篇文本总是得到同一个数。所以我们不去比较文本，而是为每篇文本算一次它的数，再找出现了两次的数。
    在集合里查一个数，不管集合多大，用的时间都一样短（第 20 章）。

    ## 动手算：一个玩具哈希

    这个哈希函数小到可以手算。给字母编号：$a = 1$，$b = 2$，$c = 3$，$d = 4$。把一个词的字母加起来，只留总数的最后一位。`cab`、`bad` 和 `abc` 的哈希各是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    hash_boxes = hand.boxes(['cab 的哈希', 'bad 的哈希', 'abc 的哈希'])
    return (hash_boxes,)


@app.cell(hide_code=True)
def _(hash_boxes):
    hand.marked(hash_boxes, {'cab 的哈希': 6, 'bad 的哈希': 7, 'abc 的哈希': 6}, {
        'cab 的哈希': r"""$3 + 1 + 2$""",
        'bad 的哈希': r"""$2 + 1 + 4$""",
        'abc 的哈希': r"""和 `cab` 是同样的字母，只是顺序不同""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `cab` 和 `abc` 是不同的词，哈希却相同。这叫**碰撞**（collision），是把一篇文本压成一个数的代价：文本比数多。
    我们的玩具有两个毛病。它只有 10 种可能的结果，而且不管字母的顺序。真正的哈希函数会把每个字节和它的位置彻底混合，改动一个字母，整个数都会变，而且它有几十亿种可能的结果。
    这样，两篇不同的文本大约每四十亿对才碰撞一次。对清理数据来说，这已经够好了：偶尔会有一篇好文本被误删。

    哈希之前还有一件事。对哈希函数来说，`The cat` 和 `the  cat` 是不同的文本。但对我们来说，它们是被另一个网站抄去的同一篇文本。所以我们先**规范化**（normalise）：把文本变成一种标准形式，全部小写、空格只留一个，再对它做哈希。

    /// admonition | 🧰 工具箱：切成词，和一个真正的哈希
    - `text.split()` 在每一段空格或换行处把文本切开，把各段作为列表给出：`"a  b\nc".split()` 是 `["a", "b", "c"]`。
    - `zlib.crc32(data)` 是一个真正的哈希函数，用来校验 zip 文件。它接收字节（第 19 章），给出一个从 0 到大约 40 亿的整数：`zlib.crc32(b"hi")` 是 3633523372。`zlib` 已经载入。
    ///

    ## 练习 1：normalise

    写 `normalise(text)`：返回全部小写的文本，每一段空格和换行都变成一个空格，开头和结尾都没有空格。

    `normalise("  The  CAT\nsat ")` 必须得到 `"the cat sat"`。
    """)
    return


@app.function
def normalise(text):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "normalise", normalise, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""把小写的文本切成词，再把这些词粘回去。第 12 章有粘合用的工具。""",
        r"""只要一行：`" ".join(...)`，括号里是 `.split()` 给出的列表。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：fingerprint

    写 `fingerprint(text)`：规范化之后的文本的 `zlib.crc32` 哈希。只有空格或大小写不同的两篇文本，必须得到同一个指纹（fingerprint）。

    `fingerprint(" THE CAT SAT ")` 必须得到 2690722008，和 `fingerprint("the cat sat")` 一样。
    """)
    return


@app.function
def fingerprint(text):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "fingerprint", fingerprint, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三步，每一步的结果交给下一步：规范化，变成字节，做哈希。第 19 章用 `.encode("utf-8")` 把文本变成字节。""",
        r"""只要一行：`zlib.crc32(...)`，括号里是规范化后再编码的文本。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：会增长的集合
    `seen = set()` 造出一个空集合，`seen.add(5)` 把 5 放进去。加入一个已经在里面的东西，什么也不会变。`5 in seen` 和 `5 not in seen` 问的是它在不在里面。
    ///

    ## 练习 3：drop_copies

    写 `drop_copies(texts)`。`texts` 是一个字符串列表。它返回一个新列表，每个指纹只留第一篇文本，顺序不变，指纹已经见过的后面那些文本都不要。
    留下的文本是原样的，保持原来写的样子，不是规范化之后的形式。

    `drop_copies(["A b", "c", "a  B", "c", "d"])` 必须得到 `["A b", "c", "d"]`。
    """)
    return


@app.function
def drop_copies(texts):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "drop_copies", drop_copies, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""两个容器，开始时都是空的：一个集合放见过的指纹，一个列表放留下的文本。把文本过一遍。每篇文本要么指纹是新的，这时两个容器都增长，要么不是，这时什么也不做。""",
        r"""大约 7 行。在循环里，把文本的指纹存进一个名字，然后 `if` 它 `not in` 集合：就把它加进集合，再把文本追加到列表。循环结束后返回列表。""",
    ], LOCALE)
    return


@app.cell
def _(tally, texts):
    kept = drop_copies(texts)
    len(kept), tally(kept)
    return (kept,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    250 个副本全都没了，还有几个碰巧一模一样的菜单也没了。把这堆文本过一遍，每篇文本只需记住一个小小的数。

    那 250 个近似重复全都还在。改掉一篇故事里的一个词，它的指纹就变成一个完全不同的数，这是设计使然。哈希能说“一样”，却永远说不出“差不多一样”。

    ## 近似重复：两篇文本有多像？

    我们需要一个数来表示两篇文本有多像。数共同的词不够，因为所有故事都有 `the` 和 `was` 这样的词。副本共有的是一*串串*的词。所以把每篇文本切成每一段连续的 $n$ 个词，叫它的**片段**（shingle），再把它们收进一个集合。
    当 $n = 2$ 时，`the cat sat down` 有三个片段：`(the, cat)`、`(cat, sat)` 和 `(sat, down)`。

    两个集合 $A$ 和 $B$ 有多像，用 **Jaccard 相似度**（Jaccard similarity）来衡量：

    > $$J(A, B) = \frac{\text{同时在两个集合里的项数}}{\text{至少在一个集合里的项数}}$$

    两个集合完全相同时它是 1，毫无共同之处时它是 0。

    ## 动手算：两个句子

    取 `the cat sat on the mat` 和 `the cat sat on the rug`，片段长 2 个词。每个句子有 5 个片段。同时在两个句子里的有几个？一共有多少个不同的片段？Jaccard 相似度是多少，保留两位小数？
    """)
    return


@app.cell(hide_code=True)
def _():
    jaccard_boxes = hand.boxes(['两个都有', '至少一个有', 'Jaccard'], step=0.01)
    return (jaccard_boxes,)


@app.cell(hide_code=True)
def _(jaccard_boxes):
    hand.marked(jaccard_boxes, {'两个都有': 4, '至少一个有': 6, 'Jaccard': 0.67}, {
        '两个都有': r"""两个句子到 `on the` 为止都一样""",
        '至少一个有': r"""共有的那些，再加上 `(the, mat)` 和 `(the, rug)`""",
        'Jaccard': r"""第一个答案除以第二个""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    六个词里改了一个，相似度仍有 0.67。文本更长时，改一个词的影响就更小了。

    /// admonition | 🧰 工具箱：关于集合的更多内容
    - 集合可以用花括号写：`{1, 2, 3}`。`set(a_list)` 用一个列表造出集合，重复的会被去掉。
    - `a & b` 是同时在两个集合里的项组成的集合，`a | b` 是至少在一个集合里的项组成的集合：`{1, 2, 3} & {2, 3, 4}` 是 `{2, 3}`，`{1, 2, 3} | {2, 3, 4}` 是 `{1, 2, 3, 4}`。`len` 数集合里的项数。
    - 列表不能放进集合，因为它以后还可能改变。`tuple(a_list)` 造出一个固定不变的副本，它就可以放：`tuple(["a", "b"])` 是 `("a", "b")`。
    ///

    ## 练习 4：shingles

    写 `shingles(words, n)`。`words` 是至少有 `n` 项的字符串列表。它返回由所有连续 `n` 个词的片段组成的集合，每个片段是一个元组。

    `shingles(["a", "b", "c", "d"], 2)` 必须得到 `{("a", "b"), ("b", "c"), ("c", "d")}`。
    """)
    return


@app.function
def shingles(words, n):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "shingles", shingles, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""从位置 `i` 开始的 `n` 个词是一个切片（第 09 章）。最后一个片段最晚能从哪里开始？在例子里，4 个词、片段长 2，起点是 0、1 和 2。""",
        r"""1 到 4 行。可以从空集合开始，在 `range(len(words) - n + 1)` 上循环，把每个 `tuple(words[i:i + n])` `add` 进去；也可以在方括号里造出元组的列表，再用 `set(...)` 包起来。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：jaccard

    写 `jaccard(a, b)`，参数是两个集合，都不是空的：就是上面的公式。

    对 `{1, 2, 3, 4, 5}` 和 `{1, 2, 3, 4, 6}`，它必须得到 $4 / 6$，约为 0.67。
    """)
    return


@app.function
def jaccard(a, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "jaccard", jaccard, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""工具箱里，分数的上面有一个工具，下面也有一个。""",
        r"""只要一行：一个 `len` 除以另一个 `len`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## MinHash：把一个集合变成几个数

    Jaccard 又把我们最初的问题带了回来。它比较的是两篇文本，而配对的数量太多。更糟的是，现在每次比较要处理两整个片段集合。我们想要指纹曾经给我们的东西：每篇文本只需记住一点点，比较又很快。

    这个想法是计算机科学里最巧妙的之一。对一个集合里的每个片段做哈希，只留下**最小**的那个哈希。把它叫作这个集合的最小值。现在问：两个集合 $A$ 和 $B$ 什么时候有相同的最小值？

    看所有至少在一个集合里的片段，再看其中哈希最小的那一个。好的哈希会把数打散得像随机的一样，所以这些片段里每一个成为那一个的可能性都相同。

    - 如果那个片段在**两个**集合里都有，它就是两边各自的最小值：两个最小值相等。
    - 如果它只在一个集合里，那个集合的最小值就比另一个的小：两者不同。

    所以两个最小值相等的概率，就是同时在两个集合里的片段所占的比例：**恰好就是 Jaccard 相似度**。每个集合一个数，回答的是“相等或不相等”，这个答案很粗糙。
    要更精细，就用许多不同的哈希函数，每个函数下各留一个最小值。这个列表就是集合的**签名**（signature），两个签名在多大比例的位置上一致，就是对 Jaccard 相似度的估计。这就是 **MinHash**。

    ## 动手算：三个哈希函数

    两个片段集合，用字母命名：$X = \{P, Q, R, S\}$ 和 $Y = \{Q, R, S, T\}$。它们 5 个片段里共有 3 个，所以 Jaccard 相似度是 0.6。下面是三个哈希函数的值：

    | 片段 | 哈希 1 | 哈希 2 | 哈希 3 |
    |---|---:|---:|---:|
    | $P$ | 7 | 1 | 6 |
    | $Q$ | 2 | 8 | 7 |
    | $R$ | 9 | 5 | 1 |
    | $S$ | 4 | 3 | 8 |
    | $T$ | 6 | 9 | 4 |

    写下每个集合的签名：它在哈希 1 下的最小值，然后是哈希 2、哈希 3。记住 $X$ 里没有 $T$，$Y$ 里没有 $P$。然后数一数两个签名在多少个位置上一致。
    """)
    return


@app.cell(hide_code=True)
def _():
    minhash_boxes = hand.boxes(['X，哈希 1', 'X，哈希 2', 'X，哈希 3', 'Y，哈希 1', 'Y，哈希 2', 'Y，哈希 3', '一致的位置'])
    return (minhash_boxes,)


@app.cell(hide_code=True)
def _(minhash_boxes):
    hand.marked(minhash_boxes, {'X，哈希 1': 2, 'X，哈希 2': 1, 'X，哈希 3': 1, 'Y，哈希 1': 2, 'Y，哈希 2': 3, 'Y，哈希 3': 1, '一致的位置': 2}, {
        'X，哈希 1': r"""7、2、9 和 4 里最小的""",
        'X，哈希 2': r"""1、8、5 和 3 里最小的""",
        'X，哈希 3': r"""6、7、1 和 8 里最小的""",
        'Y，哈希 1': r"""2、9、4 和 6 里最小的""",
        'Y，哈希 2': r"""8、5、3 和 9 里最小的：那个 1 属于 $P$，而 $Y$ 里没有它""",
        'Y，哈希 3': r"""7、1、8 和 4 里最小的""",
        '一致的位置': r"""把你的两个签名逐个位置比较""",
    }, LOCALE, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    三个位置里有两个一致，估计值是 0.67，真实值是 0.6。在哈希 2 下，所有值里最小的属于 $P$，而它只有 $X$ 有，所以两个最小值不同。三个哈希函数只能给出粗略的估计，真实的签名有几十到几百个数。

    我们需要很多哈希函数，`seeded_hash(seed, item)` 提供了它们：每个整数 `seed` 给出一个不同的哈希函数。对同一个片段 `s`，`seeded_hash(0, s)` 和 `seeded_hash(1, s)` 是互不相关的数。

    ## 练习 6：minhash

    写 `minhash(shingle_set, seeds)`。`shingle_set` 是一个非空的集合，`seeds` 是一个整数列表。它返回签名：一个列表，每个种子对应一个数，是该集合所有片段 `s` 上最小的 `seeded_hash(seed, s)`。

    对集合 `{"p"}` 和种子 `[5]`，它必须得到 `[seeded_hash(5, "p")]`：只有一个片段时，最小的哈希就是这个片段的哈希。
    """)
    return


@app.function
def minhash(shingle_set, seeds):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "minhash", minhash, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""两层循环，一个套着另一个：外层遍历种子，内层遍历片段。对每个种子，哈希列表的 `min`（第 12 章）给出签名的一个数。""",
        r"""1 到 4 行。对一个种子，这个数是 `min([seeded_hash(seed, s) for s in shingle_set])`。把每个种子对应的数收集到一个列表里。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 7：agreement

    写 `agreement(sig_a, sig_b)`，参数是两个长度相同的签名：返回它们在多大比例的位置上数相同。

    对 `[1, 2, 3, 4]` 和 `[1, 5, 3, 7]`，它必须得到 0.5。
    """)
    return


@app.function
def agreement(sig_a, sig_b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "agreement", agreement, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""用 `zip`（第 01 章）把两个列表一起走一遍，数出相等的配对。再除以长度。""",
        r"""1 到 5 行。在 `if a == b:` 里让计数器加 1，可以。对一个比较结果的列表用 `sum` 也可以，因为 `True` 算 1。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在来处理这堆文本。下面的单元格把剩下的每篇文本切成词（`to_words` 是第 12 章的小写词），取它的 3 词片段，再算出 32 个数的签名。要几秒钟：
    这是唯一慢的一步，而且是每篇文本做一次，不是每一对做一次。
    """)
    return


@app.cell
def _(kept):
    word_lists = [to_words(text) for text in kept]
    signatures = np.array([minhash(shingles(words, 3), range(32)) for words in word_lists])
    signatures.shape
    return signatures, word_lists


@app.cell(hide_code=True)
def _(kept, kind_of, signatures):
    near_i = [kind_of[text] for text in kept].index("near-copy")
    original_i = int(np.argmax(np.mean(signatures[:near_i] == signatures[near_i], axis=1)))
    return near_i, original_i


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    一张表，每篇文本占一行，每行 32 个数。我们在一对已知的文本上检验这个估计：`near_i` 是一篇近似重复的位置，`original_i` 是它所由来的那篇故事的位置。
    单元格显示它们真实的 Jaccard 相似度（用完整的集合算出），以及由两个签名得到的估计（`.tolist()` 把表的一行变回普通列表）：
    """)
    return


@app.cell
def _(near_i, original_i, signatures, word_lists):
    true_likeness = jaccard(shingles(word_lists[near_i], 3), shingles(word_lists[original_i], 3))
    round(true_likeness, 3), agreement(signatures[near_i].tolist(), signatures[original_i].tolist())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约是 0.75 和 0.91。这个估计很粗糙：32 个数只能以 $\frac{1}{32}$ 为一步来一致，而每一个都由运气决定，所以差 0.1 很平常。它不需要精确。
    两篇毫不相关的故事，相似度在 0.01 以下，再怎么走运，也不会把 0.01 错认成 0.75。

    下一个单元格按顺序走过这些文本，只有当一篇文本的签名和之前留下的每一篇都在不到一半的位置上一致时，才留下它。带 `np.mean` 的那一行就是你的 `agreement`，由 NumPy 一次对所有留下的签名做完：
    `signatures[unique] == signatures[i]` 把每个留下的行和第 `i` 行比较，每行上的均值就是一致的比例。
    """)
    return


@app.cell
def _(kept, signatures, tally):
    unique = [0]
    for i in range(1, len(kept)):
        agreements = np.mean(signatures[unique] == signatures[i], axis=1)
        if np.max(agreements) < 0.5:
            unique.append(i)
    deduped = [kept[i] for i in unique]
    len(deduped), tally(deduped)
    return (deduped,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每个近似重复都没了，每篇故事还都在。几乎所有的菜单也没了：它们是同一个模板，只改了几个数，彼此就是近似重复，其中一个被留下当作“原件”。
    这很典型。机器用模板生成的文本，是网上重复最多的文本。

    ## 垃圾文本：特征与分类器

    剩下的是故事、乱码和垃圾广告，它们每一篇都独一无二，所以找副本的办法帮不上忙。我们需要就一篇文本本身来判断它。两个简单的测量，叫**特征**（feature），就已经能说明很多：

    - 它的词里，常见英文词所占的比例（我们用故事里最常出现的 2,000 个词）；
    - 它的词里，互不相同的词所占的比例。

    ## 动手算：两个特征

    取文本 `buy buy cheap buy now`，假设常见词是 `buy` 和 `now`。它的 5 个词里，常见词占多大比例？其中有 3 个不同的词：3 占 5 的多大比例？
    """)
    return


@app.cell(hide_code=True)
def _():
    feature_boxes = hand.boxes(['常见词的比例', '不同词的比例'], step=0.1)
    return (feature_boxes,)


@app.cell(hide_code=True)
def _(feature_boxes):
    hand.marked(feature_boxes, {'常见词的比例': 0.8, '不同词的比例': 0.6}, {
        '常见词的比例': r"""`buy` 三次，`now` 一次：5 个里有 4 个""",
        '不同词的比例': r"""`buy`、`cheap` 和 `now`：5 个里有 3 个""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 8：known_share

    写 `known_share(words, known)`。`words` 是一个非空的字符串列表，`known` 是一个字符串集合。它返回 `words` 的项里在 `known` 中的比例。出现三次的词算三次。

    对 `["the", "cat", "zzz", "sat"]` 和集合 `{"the", "cat", "sat"}`，它必须得到 0.75。
    """)
    return


@app.function
def known_share(words, known):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "known_share", known_share, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""这又是 `agreement`，只是换了一个判断：数出 `w in known` 成立的词，再除以词的个数。""",
        r"""1 到 5 行：一个计数器加一个带 `if` 的循环，或者对 `w in known` 判断的列表用 `sum`。""",
    ], LOCALE)
    return


@app.cell
def _(kinds, texts):
    known = common_words()

    def features(text):
        words = to_words(text)
        return [known_share(words, known), len(set(words)) / len(words)]

    np.round([features(texts[kinds.index(kind)]) for kind in ["story", "gibberish", "spam"]], 2)
    return (features,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    依次是一篇故事、一段乱码和一条垃圾广告各一行。故事几乎全是常见词，大约一半的词互不相同。乱码几乎没有常见词。垃圾广告把少数几个词反复重复。

    现在我们可以凭眼睛定两个阈值。但真实的文本堆需要几十个特征，没有人能凭眼睛调几十个阈值。第 06 到 10 章解决的正是这个问题：让**分类器**从例子里学出规则。
    有人读几百篇文本，把每篇标成好文本或垃圾，用这些训练一个分类器，然后由它去判断那几百万篇永远没有人会读的文本。这叫**质量分类器**（quality classifier）。

    下面，前 200 篇文本扮演有标签的那些，标签取自我们埋下的答案：垃圾是 1，故事是 0。模型是第 10 章最小的那种：从 2 个特征到 2 个分数的一层。
    """)
    return


@app.cell
def _(deduped, features, kind_of):
    X = torch.tensor([features(text) for text in deduped])
    labels = torch.tensor([kind_of[text] != "story" for text in deduped]).long()
    torch.manual_seed(0)
    classifier = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.1)
    for step in range(300):
        optimizer.zero_grad()
        F.cross_entropy(classifier(X[:200]), labels[:200]).backward()
        optimizer.step()
    guesses = torch.argmax(classifier(X), dim=1)
    return guesses, labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    （`.long()` 把 `True` 和 `False` 变成 `F.cross_entropy` 要求作为标签的整数 1 和 0。）

    它在没训练过的文本上表现怎么样？第 18 章的三个计数能回答：抓到的垃圾、被误删的好故事，和漏过去的垃圾。
    """)
    return


@app.cell
def _(guesses, labels):
    new_guesses, new_labels = guesses[200:], labels[200:]
    caught = torch.sum((new_guesses == 1) & (new_labels == 1)).item()
    thrown_out = torch.sum((new_guesses == 1) & (new_labels == 0)).item()
    let_through = torch.sum((new_guesses == 0) & (new_labels == 1)).item()
    caught, thrown_out, let_through
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每一条垃圾都抓到了，没有误删任何故事：精确率和召回率都是 1。对这么干净的结果要保持怀疑。我们的垃圾是埋进去的，而埋进去的垃圾很容易认。
    真实的垃圾和真实的文本之间界线模糊：一半是广告的页面，被糟糕的机器翻译过的文本，一篇不知所云的故事。真实的流水线用的是同样的方法，只是分类器更大，特征更丰富。
    其中一个特征，是语言模型自己在这段文本上的困惑度，因为模型觉得非常意外的文本，往往不是语言。
    它们也接受低于 1 的精确率和召回率，并选择哪一种错误代价更大。

    把分类器判为好的文本留下来：
    """)
    return


@app.cell
def _(deduped, guesses, tally):
    clean = [deduped[i] for i in range(len(deduped)) if guesses[i] == 0]
    len(clean), tally(clean)
    return (clean,)


@app.cell(hide_code=True)
def _():
    pipeline([("原始文本堆", "2,000"), ("没有副本", "1,744"), ("没有近似重复", "1,351"), ("干净", "1,000")], ["指纹", "MinHash", "分类器"], "三步清理，以及每一步之后剩下多少篇文本")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这堆文本只剩一半大小，而且全是故事。

    ## 这有影响吗？

    老实的检验办法是两种都训练一遍。按钮会把第 24 章的小模型训练两次，每个词元用 64 个数，读 150,000 个词元：一次用原始文本堆，一次用你清理后的。
    两次的计算量相同。两次都在同样的测试故事上打分。大约要 20 秒。
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="在原始文本堆和干净文本上各训练一次（约 20 秒）")
    train_button
    return (train_button,)


@app.cell
def _(clean, texts, train_button):
    mo.stop(not train_button.value, mo.md("*按下按钮开始训练。*"))
    _train_tokens, test_tokens, vocab_size = story_tokens()
    raw_loss = small_run(64, 150000, pile_tokens(texts), test_tokens, vocab_size)
    clean_loss = small_run(64, 150000, pile_tokens(clean), test_tokens, vocab_size)
    round(raw_loss, 3), round(clean_loss, 3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    在我们的机器上：原始文本堆大约是 4.48，干净的是 4.39。计算量相同，干净的模型好 2%。这听起来不多，但用第 24 章的定律来估价就不一样了：
    文本每翻一倍，损失减少 9%，所以 2% 相当于多出大约 15% 的文本能换来的。这里它只花了几秒钟的清理。

    在真实的文本堆上，收益更大，而且不只体现在损失上。用副本训练的模型，会学着背诵它们。用菜单训练的模型，会写出菜单。而模型绝不该说出的内容，最好在它读到之前就清除。
    构建大模型的人，把很大一部分精力花在这一步上。

    ## 接下来

    模型现在读的是好文本，而它从文本里学到的，是把文本接着写下去。这和照你说的去做不是一回事。**第 26 章**讲训练的最后一步，教模型按我们希望的方式回应：
    从示例、从比较，也从奖励中学。

    ## 你现在会做的事

    - 解释什么是哈希函数、什么是碰撞，以及为什么要先规范化文本
    - 一遍扫过，从一堆文本里去掉完全相同的副本
    - 用片段和 Jaccard 相似度测量两篇文本有多像
    - 解释为什么两个集合的最小哈希一致的概率等于它们的相似度，并构建 MinHash 签名
    - 用特征描述一篇文本，并用少量有标签的例子训练一个质量分类器
    - 用原始数据和干净数据各训练一次，测出清理值多少

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** MinHash 出自 Broder（1997），他用它为 AltaVista 搜索引擎找重复的网页。把每个签名和每个留下的签名都比一遍，对数十亿个网页来说仍然太慢：
    *局部敏感哈希*（locality-sensitive hashing）把每个签名切成若干段，只比较至少有一整段相同的文本。大型数据流水线里的质量分类器，常常是 *fastText* 模型；Python 的 `hashlib` 里有比 `zlib.crc32` 更强的哈希函数。
    """)
    return


if __name__ == "__main__":
    app.run()
