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

    这门课前面训练的每个模型，吃的都是干净文本：几千篇故事，篇篇不同，而且都是流畅自然的英文。但真实大模型收集来的训练文本，完全是另一回事：爬虫程序扫过数十亿个网页，碰到什么就抓什么。同一篇文章可能被上百个网站转载；每个页面都夹杂着导航菜单和版权声明；还有相当大一部分根本算不上正常语言。

    模型读什么，就会学到什么。第 24 章测出了“数据变多”能换来多少提升；这一章我们关心的是“数据本身的质量”，以及怎样分三步，把一堆杂乱的原始网页变成值得训练的干净数据：

    1. 去掉**完全相同的副本**；
    2. 去掉**近似重复**（near-copy），也就是只改了几个词的文本；
    3. 去掉**垃圾文本**（junk）。

    ## 一堆原始文本

    我们拿故事集造了一小堆模拟的原始数据：一共 2,000 篇文本，其中只有一半是真正想要的故事。
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
    第一个单元格展示了各类问题文本的开头。第二个单元格统计了整堆数据的构成：1,000 篇好故事，250 篇只改了空格或大小写的完全副本，250 篇改了几个词的近似重复，还有 500 篇三种不同类型的垃圾文本。

    `kinds` 里记录着每篇文本的真实类别，`tally` 用来按类统计。**真实爬取的数据里可没有这种标签。**这里是我们特意埋下的“标准答案”，方便你检验清理效果；你自己写的清洗函数绝不会偷看 `kinds`。

    ## 完全相同的副本，与哈希

    要找副本，最直接的办法是把每篇文本和其他所有文本两两比一遍。2,000 篇文本就要比两百万次；放到十亿个网页上，根本行不通。

    标准做法是使用**哈希函数**（hash function）：一套固定的计算规则，能把任意一段文本算成一个整数（常叫**哈希值**）。只要文本内容完全一致，算出来的哈希值就一模一样。所以我们不用把文本两两比对，只需给每篇文本各算一次哈希值，再看看哪些值重复出现过就行了。
    在集合（`set`）里查一个值在不在，不管集合有多大，花的时间几乎一样短（第 20 章讲过，集合底层直接按哈希定位，不用逐个翻找）。

    ## 动手算：一个玩具哈希

    来看一个简单到能手算的哈希函数。先给字母编号：$a = 1$，$b = 2$，$c = 3$，$d = 4$。把单词里每个字母对应的数字加起来，只保留总和的个位数。`cab`、`bad` 和 `abc` 的哈希值各是多少？
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
        'abc 的哈希': r"""和 `cab` 字母相同，只是顺序不同""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `cab` 和 `abc` 是两个不同的词，算出的哈希值却相同。这叫**哈希碰撞**（collision），是把任意文本压缩成有限数字的必然代价：可能的文本无穷无尽，而数字的范围是有限的。
    我们这个玩具哈希有两个毛病：结果只有 10 种可能，而且不看字母顺序。真正的哈希函数会把每个字节和它所处的位置彻底搅匀——哪怕只改一个字母，算出的结果也会面目全非，而且可能的结果有几十亿种。
    这样一来，两篇不同文本撞上同一个哈希值的概率只有大约四十亿分之一。拿来清洗训练数据，这已经足够稳妥了：顶多极偶尔误删一篇好文章。

    做哈希之前还要处理一个小细节。在哈希函数眼里，`The cat` 和 `the  cat` 是完全不同的两段文字，但对我们来说，它们分明是同一句话被别的网站转载时改了格式。所以我们先对文本做**规范化**（normalise）：统一转成小写，把连续的空格或换行都合并成一个空格，再去算哈希。

    /// admonition | 🧰 工具箱：切词、拼接，和一个真正的哈希
    - `text.split()` 把文本按任意连续的空格或换行切开，返回词列表：`"a  b\nc".split()` 得到 `["a", "b", "c"]`。
    - `" ".join(words)` 用单个空格把字符串列表拼回一句话（第 12 章见过）：`" ".join(["hi", "there"])` 得到 `"hi there"`。
    - `zlib.crc32(data)` 是一个真正的哈希函数（zip 压缩包就用它校验文件）。它接收字节串（用 `.encode("utf-8")` 可以把字符串转成字节，第 19 章见过），返回一个 0 到约 40 亿之间的整数：`zlib.crc32(b"hi")` 是 3633523372。`zlib` 已经导入好了。
    ///

    ## 练习 1：normalise

    编写 `normalise(text)`：把文本全部转成小写，把每一段连续的空格和换行都压缩成单个空格，并去掉开头和结尾多余的空白。

    `normalise("  The  CAT\nsat ")` 必须得到 `"the cat sat"`。
    """)
    return


@app.function
def normalise(text):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "normalise", normalise, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""先把文本转成小写再切成词，然后用第 12 章的粘合工具（`" ".join(...)`）把词拼回去。""",
        r"""一行就能搞定：对转小写并 `.split()` 后的词列表，在外面套上 `" ".join(...)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：fingerprint

    编写 `fingerprint(text)`：计算文本规范化之后的 `zlib.crc32` 哈希值。只要两篇文本仅仅是空格或大小写不同，就必须算出同一个**指纹**（fingerprint）。

    `fingerprint(" THE CAT SAT ")` 必须得到 2690722008，和 `fingerprint("the cat sat")` 一样。
    """)
    return


@app.function
def fingerprint(text):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "fingerprint", fingerprint, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三步串联起来：先规范化，再转成字节，最后算哈希。第 19 章用 `.encode("utf-8")` 把文本转成字节。""",
        r"""一行就能搞定：把规范化并编码后的文本放进 `zlib.crc32(...)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：会增长的集合
    `seen = set()` 会创建一个空集合，`seen.add(5)` 把 5 放进去。如果 5 已经在集合里，再放一次也不会重复。用 `5 in seen` 和 `5 not in seen` 可以快速检查某个值在不在里面。
    ///

    ## 练习 3：drop_copies

    编写 `drop_copies(texts)`。`texts` 是一个字符串列表，函数返回一个新列表：对于每个指纹，只保留第一次出现的那篇文本，保持原有顺序，后面指纹重复的都跳过。
    注意保留的是**原始文本**本身，不是规范化后的版本。

    `drop_copies(["A b", "c", "a  B", "c", "d"])` 必须得到 `["A b", "c", "d"]`。
    """)
    return


@app.function
def drop_copies(texts):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "drop_copies", drop_copies, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""准备两个空容器：一个集合记录见过的指纹，一个列表存放留下的文本。把 `texts` 遍历一遍：如果当前文本的指纹是新的，就同时加入这两个容器；如果已经见过，就直接跳过。""",
        r"""大约 7 行。在循环里先算出当前文本的指纹并存到变量里，然后用 `if` 判断它是否 `not in` 集合：如果不在，就把它 `add` 进集合，再把原文本 `append` 到列表。循环结束后返回列表。""",
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
    250 篇完全副本全部清掉了，顺带还清掉了几篇一模一样的网页菜单。整堆数据只扫了一遍，每篇文本只需记下一个小小的整数。

    但那 250 篇近似重复全都留了下来。哪怕只改掉故事里的一个词，哈希指纹就会彻底变样——这本来就是哈希的设计目标。哈希只能回答“是否一模一样”，却说不出“是不是差不多”。

    ## 近似重复：两篇文本有多像？

    我们需要一个指标来衡量两篇文本有多像。光数共有多少个词还不够，因为几乎所有英文故事都会大量出现 `the`、`was` 这类常用词。真正能暴露抄袭或改写的，是一**连串**相同的词。所以我们把每篇文本切成所有长度为 $n$ 的连续词组，称为**片段**（shingle），再收进一个集合里。
    当 $n = 2$ 时，`the cat sat down` 会切出三个片段：`(the, cat)`、`(cat, sat)` 和 `(sat, down)`。

    两个集合 $A$ 和 $B$ 的相似程度，用 **Jaccard 相似度**（Jaccard similarity）来衡量：

    > $$J(A, B) = \frac{\text{同时在两个集合里的项数}}{\text{至少在一个集合里的项数}}$$

    两个集合完全相同时它是 1，毫无交集时它是 0。

    ## 动手算：两个句子

    拿 `the cat sat on the mat` 和 `the cat sat on the rug` 来试试，片段长度取 2 个词。每个句子都有 5 个片段。同时出现在两个句子里的有几个？合并起来一共有多少个不同的片段？Jaccard 相似度是多少（保留两位小数）？
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
        '至少一个有': r"""共有的那 4 个，再加上 `(the, mat)` 和 `(the, rug)`""",
        'Jaccard': r"""第一个答案除以第二个""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    六个词里改了一个，相似度仍有 0.67。文章更长时，改掉一两个词的影响就更微乎其微了。

    /// admonition | 🧰 工具箱：集合的更多用法
    - 集合可以用花括号直接写出：`{1, 2, 3}`；`set(a_list)` 可以把列表转成集合，顺便自动去掉重复项。
    - `a & b` 是交集（两边都有），`a | b` 是并集（至少一边有）：`{1, 2, 3} & {2, 3, 4}` 得到 `{2, 3}`，`{1, 2, 3} | {2, 3, 4}` 得到 `{1, 2, 3, 4}`。用 `len` 可以数集合里有几项。
    - 列表不能放进集合，因为列表的内容以后会变——内容一变，它的哈希值就变了，集合按哈希去找就再也找不到它。用 `tuple(a_list)` 可以做一个固定不变的元组副本，就能放进集合了：`tuple(["a", "b"])` 会得到 `("a", "b")`。
    ///

    ## 练习 4：shingles

    编写 `shingles(words, n)`。`words` 是至少有 `n` 项的字符串列表，函数返回由所有连续 `n` 个词的片段组成的集合，每个片段是一个元组。

    `shingles(["a", "b", "c", "d"], 2)` 必须得到 `{("a", "b"), ("b", "c"), ("c", "d")}`。
    """)
    return


@app.function
def shingles(words, n):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "shingles", shingles, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""从位置 `i` 开始取连续 `n` 个词，用的是切片 `words[i:i + n]`（第 09 章）。最后一个片段最晚能从哪里开始？在例子里，4 个词、片段长 2，起点是 0、1 和 2。""",
        r"""1 到 4 行。可以从空集合开始，在 `range(len(words) - n + 1)` 上循环，把每个 `tuple(words[i:i + n])` 用 `add` 放进去；也可以先用列表推导式生成元组列表，再用 `set(...)` 包起来。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：jaccard

    编写 `jaccard(a, b)`，传入两个非空集合，按上面的公式算出 Jaccard 相似度。

    对 `{1, 2, 3, 4, 5}` 和 `{1, 2, 3, 4, 6}`，它必须得到 $4 / 6$，约为 0.67。
    """)
    return


@app.function
def jaccard(a, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "jaccard", jaccard, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""工具箱里已经给了分子（交集）和分母（并集）对应的运算符。""",
        r"""一行就能搞定：一个 `len` 除以另一个 `len`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## MinHash：把一个集合浓缩成几个数

    Jaccard 相似度又把最初的难题带回来了：它必须把文本两两比较，配对数量实在太多；更糟的是，现在每次比较都要处理两大坨片段集合。我们还是想要“指纹”那种好处：每篇文本只需记住寥寥几个数，比起来又飞快。

    下面这个想法是计算机科学里最巧妙的技巧之一。把一个集合里的每个片段都算一遍哈希，但只留下**最小**的那个哈希值，叫作这个集合的最小值。现在想一想：两个集合 $A$ 和 $B$ 的最小值，什么时候会正好相等？

    拿个具体例子打比方：假设 $A$ 和 $B$ 合并在一起（并集）共有 5 个片段，其中 3 个是两家共有的，2 个是某一家独有的。好的哈希函数会把这 5 个片段随机打散成大小不一的数，所以取最小哈希值，就像**从这 5 个片段里公平抽 1 个中奖**：

    - 如果抽中了**共有的那 3 个**之一，两家都有这个片段，它自然同时成为两家各自的最小值——两边的最小值正好相等。
    - 如果抽中了**独有的那 2 个**之一，只有其中一家有它，这一家的最小值就会比另一家小——两边的最小值不同。

    因此，两个最小值相等的概率正好就是 $3 / 5$——也就是共有片段占全部片段的比例，**恰好等于 Jaccard 相似度**！

    不过，只用一个哈希函数，每篇文本只留 1 个数，只能回答“相等或不相等”，太粗糙了。想要估计得更准，就多准备几十个不同的哈希函数，在每个函数下各抽一次最小值。这一排最小值就是集合的**签名**（signature）；两个签名在多大比例的位置上一致，就是对 Jaccard 相似度的估计。这套方法就叫 **MinHash**。

    ## 动手算：三个哈希函数

    来看两个片段集合，用字母代表片段：$X = \{P, Q, R, S\}$，$Y = \{Q, R, S, T\}$。它们并集 5 个片段里共有 3 个，所以 Jaccard 相似度正好是 $3 / 5 = 0.6$。下面是三个哈希函数算出的值：

    | 片段 | 哈希 1 | 哈希 2 | 哈希 3 |
    |---|---:|---:|---:|
    | $P$ | 7 | 1 | 6 |
    | $Q$ | 2 | 8 | 7 |
    | $R$ | 9 | 5 | 1 |
    | $S$ | 4 | 3 | 8 |
    | $T$ | 6 | 9 | 4 |

    填出每个集合的签名：依次写出它在哈希 1、哈希 2、哈希 3 下的最小值。注意 $X$ 里没有 $T$，$Y$ 里没有 $P$。最后数一数两个签名有几个位置一致。
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
    三个位置里有两个一致，估计值是 0.67，已经很接近真实的 0.6 了。在哈希 2 下，全场最小的值属于 $P$，而 $P$ 只有 $X$ 独有，所以两边的最小值不同。只用三个哈希函数还比较粗糙，真实场景里一个签名通常会放几十到几百个数。

    我们需要很多彼此独立的哈希函数，这里直接用现成的 `seeded_hash(seed, item)`：每个整数 `seed`（种子）都对应一套不同的哈希函数。对同一个片段 `s`，`seeded_hash(0, s)` 和 `seeded_hash(1, s)` 是互不相关的两个数。

    ## 练习 6：minhash

    编写 `minhash(shingle_set, seeds)`。`shingle_set` 是一个非空集合，`seeds` 是一个整数列表。函数返回签名列表：每个种子对应一个数，即该集合所有片段 `s` 中最小的 `seeded_hash(seed, s)`。

    对集合 `{"p"}` 和种子 `[5]`，它必须得到 `[seeded_hash(5, "p")]`：只有一个片段时，最小的哈希就是它自己的哈希。
    """)
    return


@app.function
def minhash(shingle_set, seeds):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "minhash", minhash, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""两层循环：外层遍历每个种子，内层遍历集合里的每个片段。对每个种子，把所有片段算出的哈希值放进 `min(...)`（求最小值），就得到签名里的一个数。""",
        r"""1 到 4 行。对单个种子，这个最小值是 `min([seeded_hash(seed, s) for s in shingle_set])`。把每个种子算出的结果收进一个列表即可。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 7：agreement

    编写 `agreement(sig_a, sig_b)`，传入两个等长的签名列表，返回它们在多大比例的位置上数值相同。

    对 `[1, 2, 3, 4]` 和 `[1, 5, 3, 7]`，它必须得到 0.5。
    """)
    return


@app.function
def agreement(sig_a, sig_b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "agreement", agreement, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""用 `zip(sig_a, sig_b)`（第 01 章）同步遍历两个列表对应位置的元素，数出相等的对数，再除以签名长度。""",
        r"""1 到 5 行。可以用循环加 `if a == b:` 累加计数器，也可以对比较结果的列表直接求 `sum`（在 Python 里 `True` 算 1、`False` 算 0）。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在来实战处理这堆文本。下面的单元格把剩下的每篇文本切成小写词（`to_words` 是第 12 章写的分词函数），提取 3 词片段，再算出 32 个数的签名。运行要花几秒钟——这是唯一稍慢的一步，而且只需对**每篇文本**算一次，不用对**每一对文本**算一次。
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
    我们得到了一张签名表：每篇文本占一行，每行 32 个数。先拿一对已知的文本来检验估计效果：`near_i` 是一篇近似重复的位置，`original_i` 是它改自的那篇原故事的位置。
    下面的单元格同时算出它们真实的 Jaccard 相似度（用完整片段集合），以及用两个签名算出的估计值（`.tolist()` 把表格的一行转回普通列表）：
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
    大约是 0.75 和 0.91。这个估计确实有些粗糙：32 个数的一致比例只能是 $\frac{1}{32}$ 的整数倍，而且每个位置能不能撞上带有随机性，偏个 0.1 很正常。但它根本不需要精确。
    两篇毫不相干的故事，相似度连 0.01 都不到——再怎么碰运气，也不可能把 0.01 错认成 0.75。

    下一个单元格按顺序遍历剩余文本：只有当一篇文本的签名和之前**所有**已保留文本的一致比例都不到一半（$< 0.5$）时，才把它留下来。带 `np.mean` 的那一行干的就是你的 `agreement` 的活，只不过 NumPy 借助**广播**一次就把所有已保留的签名全比完了：

    - 假设目前保留了 $k$ 篇文本，`signatures[unique]` 的形状是 `(k, 32)`（每行 32 个数），当前文本的签名 `signatures[i]` 形状是 `(32,)`；
    - `signatures[unique] == signatures[i]` 会把 `(32,)` 自动对齐到这 $k$ 行的每一行，逐个位置比对，得到一张 `(k, 32)` 的 `True`/`False` 表；
    - 再沿着每一行（`axis=1`）求平均（`True` 算 1、`False` 算 0），就一次算出了当前文本与那 $k$ 篇保留文本各自的**一致比例**。
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
    250 篇近似重复全部清掉了，1,000 篇好故事一篇没少。几乎所有网页菜单也被清掉了：它们出自同一个模板，只改了几个数字，彼此之间就是近似重复，最后只剩一篇被当成“原件”留了下来。
    这在真实数据里非常常见——机器套模板批量生成的页面，正是网上重复最严重的内容。

    ## 垃圾文本：特征与分类器

    剩下的文本里有好故事、乱码和垃圾广告。后两种每篇都各不相同，去重手段派不上用场了，我们得单独判断每一篇文本本身的质量。只要两个简单的指标——机器学习里叫**特征**（feature）——就已经很说明问题：

    - 文章里常见英文词占多大比例（这里用故事集里最常出现的 2,000 个词作词表）；
    - 文章里互不相同的词占多大比例。

    ## 动手算：两个特征

    拿 `buy buy cheap buy now` 这句话来算，假设常见词表里只有 `buy` 和 `now`。它的 5 个词里，常见词占多大比例？里面共有 3 个不同的词，不同词又占 5 个词的多大比例？
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

    编写 `known_share(words, known)`。`words` 是一个非空的字符串列表，`known` 是一个字符串集合。函数返回 `words` 中落在 `known` 里的词所占的比例。同一个词出现三次就算三次。

    对 `["the", "cat", "zzz", "sat"]` 和集合 `{"the", "cat", "sat"}`，它必须得到 0.75。
    """)
    return


@app.function
def known_share(words, known):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "known_share", known_share, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""写法和 `agreement` 如出一辙，只是换了一个判断条件：数出满足 `w in known` 的词有多少个，再除以总词数。""",
        r"""1 到 5 行：用循环加 `if` 累加计数器，或者直接对 `w in known` 组成的布尔列表求 `sum`。""",
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
    这三行依次对应一篇故事、一段乱码和一条垃圾广告。好故事几乎全是常见词，大约一半的词互不相同；乱码几乎没有常见词；垃圾广告则来来回回就那么几个词。

    现在我们当然可以凭肉眼定两个阈值。但真实的数据清洗需要几十个特征，没人能靠肉眼手调几十个阈值。第 06 到 10 章解决的正是这个问题：让**分类器**自己从样本里学出判断规则。
    先由人工挑几百篇文本标好“好文本”或“垃圾”，拿去训练一个分类器，再让它去筛选那几百万篇人类永远读不完的网页。这就是**质量分类器**（quality classifier）。

    下面我们用前 200 篇文本充当人工标注的样本（标签来自我们预先埋下的答案：垃圾是 1，故事是 0）。模型用的是第 10 章最简单的线性分类器：只用一层 `nn.Linear(2, 2)`，把 2 个特征直接映射成 2 个分数。
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
    （`.long()` 把布尔值 `True` 和 `False` 转成 `F.cross_entropy` 要求的整数标签 1 和 0。）

    它在没训练过的文本上表现怎么样？用第 18 章的三个计数来看：抓到的垃圾、误删的好故事，以及漏过去的垃圾。
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
    每一条垃圾都抓到了，也没有误删任何一篇好故事：精确率（删掉的里真垃圾占多少）和召回率（所有真垃圾抓住了多少）都是 1。遇到这么完美的结果，反而要多个心眼——这里的垃圾是我们刻意造出来的，特征太明显了。
    真实世界里，垃圾文本和正常文本之间界线非常模糊：半页都是广告的文章、质量很差的机翻文本、前言不搭后语的故事……工业界的数据清洗流程用的是同样的思路，只是分类器更大、特征更丰富。
    其中一个常用特征，是语言模型自己读这段话时的困惑度——让模型大感意外的文本，往往不是正常人话。
    现实中精确率和召回率都不可能做到 1，团队必须权衡：误删好文章和漏放垃圾，哪一种代价更高。

    把分类器判为好文本的内容留下来：
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
    整堆数据缩小了一半，而且剩下的全是好故事。

    ## 这真的有用吗？

    最实在的检验办法，就是拿两份数据各训练一次。点击下面的按钮，就会把第 24 章的小型语言模型训练两遍（每个词元用 64 个数，各读 150,000 个词元）：一遍喂原始文本堆，一遍喂你清洗后的干净数据。
    两遍花的计算量完全相同，最后在同样的测试故事上打分。全程大约需要 20 秒。
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
    在我们的机器上，原始数据的测试损失大约是 4.48，清洗后的数据是 4.39。同样的算力下，干净数据训出的模型好了 2%。听起来不多，但按第 24 章的定律折算一下就不一样了：
    数据量每翻一倍，损失才降 9%，所以这 2% 的提升相当于白捡了大约 15% 的额外数据——而这里只花了几秒钟清洗。

    在真实的网页数据上，清洗带来的收益还要大得多，而且不只体现在损失数字上。喂了大量副本的模型，会学着一字不差地背诵原文；喂了导航菜单的模型，张口就会吐出菜单。至于那些绝不该让模型说出口的内容，最好的办法就是在它读到之前先清掉。
    造大模型的团队，往往把很大一部分精力都花在这一步。

    ## 接下来

    模型现在读的都是好文本了，但光靠读文本，它学会的只是“接着往下写”，这和“照你的要求办事”还不是一回事。**第 26 章**将讲到训练的最后一步，教模型按我们希望的方式回应：
    从示范中学、从对比中学，也从奖励中学。

    ## 你现在会做的事

    - 解释什么是哈希函数、什么是哈希碰撞，以及为什么要先规范化文本
    - 一遍扫过，从一堆文本里去掉完全相同的副本
    - 用片段和 Jaccard 相似度衡量两篇文本有多像
    - 解释为什么两个集合的最小哈希一致的概率等于它们的相似度，并构建 MinHash 签名
    - 用特征描述一篇文本，并用少量有标签的样本训练一个质量分类器
    - 用原始数据和干净数据各训练一次，测出清洗带来的收益

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** MinHash 由 Broder（1997）提出，当年被用来为 AltaVista 搜索引擎查重网页。不过面对数十亿个网页，把每个签名和之前所有保留签名逐一比对还是太慢：
    *局部敏感哈希*（locality-sensitive hashing）会把每个签名切成若干段，只比对至少有一整段完全相同的文本。大型数据流水线里的质量分类器常用 *fastText* 模型；另外，Python 标准库 `hashlib` 里也有比 `zlib.crc32` 更强的哈希函数。
    """)
    return


if __name__ == "__main__":
    app.run()
