import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="13 · 预测下一个词")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np

    from learn_llm import hand, todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.leaderboard import record, show

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 13 · 预测下一个词

    你用过的每一个聊天机器人，都在一遍又一遍地做同一件事：根据已有的词，预测下一个词。能做这件事的模型叫**语言模型**（language model）。这门课一路走来，为的就是它。

    这一章我们先搭出最简单的语言模型：不用神经网络，也不用算梯度，纯粹靠数数。这么做有两个原因：一是它真能写出像模像样的句子（虽然有点傻）；二是它能定下一个**基准分**，后面每一章的新模型都要想办法超越它。

    ## 把文本变成数字

    和第 12 章一样，我们把故事切成单词，留下最常见的 2,000 个。这次词表（vocabulary）里多了两项：

    - `<unk>`（unknown，未知）代表这 2,000 个词以外的任何词；
    - `<end>` 标记一个故事的结尾，让模型能学会什么时候停下。

    `to_words` 和 `count_words` 沿用第 12 章的写法；`story_ids` 是新加的，负责把一整批故事拼成一条长长的编号数组。这次我们用全部 12,000 篇故事来训练，留出 1,000 篇测试故事用来打分。
    """)
    return


@app.function
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.function
def count_words(words):
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts


@app.function
def story_ids(story_list, index):
    ids = []
    for story in story_list:
        for word in to_words(story):
            ids.append(index.get(word, index["<unk>"]))
        ids.append(index["<end>"])
    return np.array(ids)


@app.cell
def _():
    train_stories, test_stories = stories()
    counts = count_words(to_words(" ".join(train_stories)))
    vocab = sorted(counts, key=counts.get, reverse=True)[:2000] + ["<unk>", "<end>"]
    index = {}
    for i in range(len(vocab)):
        index[vocab[i]] = i
    train_ids = story_ids(train_stories, index)
    test_ids = story_ids(test_stories, index)
    len(vocab), len(train_ids), len(test_ids)
    return index, test_ids, train_ids, vocab


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    词表一共 2,002 项，训练集大约有 190 万个词，测试集大约 15.9 万个词。`index.get(word, index["<unk>"])` 会查出这个词的编号，词表里没有的生词就统一归到 `<unk>`。（这个单元格运行可能需要几秒钟，下一节填表的单元格也是如此。）

    ## 依赖前文的概率

    在 `once upon a`（从前）后面，下一个词几乎一定是 `time`。而在 `the` 后面，什么词都可能出现。所以我们想要的是：**已知**前面的内容时，每个词出现的概率。这叫**条件概率**（conditional probability），用一条竖线隔开：

    > $P(\text{next} \mid \text{previous})$ 读作“已知刚看到 `previous`，下一个是 `next` 的概率”

    估计它最简单的办法就是数数。看这段小文本，句号也算一个词：

    > the dog ran . the dog sat . the cat ran .

    `the` 出现了 3 次，其中两次后面是 `dog`，一次是 `cat`。所以 $P(\text{dog} \mid \text{the}) = 2 / 3$，$P(\text{cat} \mid \text{the}) = 1 / 3$，其余每个词都是 0。把这些出现次数整理成一张表——每个词占一行，每个可能的下一个词占一列——就得到了**二元模型**（bigram，专门看相邻的两个词）。

    ## 动手算：一张次数表

    下面是这段小文本的统计表，`the` 那一行已经填好。行表示排在前面的词，列表示紧跟在后面的词。

    | 前 ↓ 后 → | the | dog | cat | ran | sat | . |
    |---|---|---|---|---|---|---|
    | **the** | 0 | 2 | 1 | 0 | 0 | 0 |
    | **dog** | 0 | 0 | 0 | A | B | 0 |
    | **ran** | 0 | 0 | 0 | 0 | 0 | C |
    | **.** | D | 0 | 0 | 0 | 0 | 0 |

    在文本里数一数相邻词对，填出 A 到 D。再回答两个问题：这段文本一共有多少个相邻的词对？（它共有 12 个词。）$P(\text{ran} \mid \text{dog})$ 是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    count_boxes = hand.boxes(['A', 'B', 'C', 'D', '词对总数', 'P(ran | dog)'], step=0.01)
    return (count_boxes,)


@app.cell(hide_code=True)
def _(count_boxes):
    hand.marked(count_boxes, {'A': 1, 'B': 1, 'C': 2, 'D': 2, '词对总数': 11, 'P(ran | dog)': 0.5}, {
        'A': r"""`ran` 紧跟在 `dog` 后面出现了几次？""",
        'B': r"""`sat` 紧跟在 `dog` 后面出现了几次？""",
        'C': r"""`ran` 每次出现，后面都是句号""",
        'D': r"""最后一个句号后面什么都没有，所以只算前两个""",
        '词对总数': r"""除了最后一个词，每个词都是一个词对的开头""",
        'P(ran | dog)': r"""A 除以 `dog` 那一行的总数""",
    }, LOCALE, per_row=6)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：取相邻词对
    - `items[1:]` 是省略终点的切片：从下标 1 取到末尾，相当于去掉第一项。
    - `zip`（第 01 章）会把两个列表按位置两两配对，短的那个用完就停。把一个列表和它自己的 `[1:]` 配对，正好得到所有相邻的一对：例如 `letters = ["a", "b", "c"]` 时，`zip(letters, letters[1:])` 会依次给出 `("a", "b")` 和 `("b", "c")`。
    - `for first, second in zip(...)` 会在每次循环时，把这一对元素直接拆成 `first` 和 `second` 两个变量。
    ///

    ## 练习 1：bigram_counts

    写 `bigram_counts(ids, size)`。`ids` 是一个编号列表或数组，每个编号都在 0 到 `size - 1` 之间。函数返回形状为 `(size, size)` 的 NumPy 表格 `C`，其中 `C[a, b]` 记录 `b` 紧跟在 `a` 后面出现的次数：

    > `bigram_counts([0, 1, 0, 1, 2], 3)` 得到 `[[0, 2, 0], [1, 0, 1], [0, 0, 0]]`

    这里 `1` 跟在 `0` 后面两次，`0` 跟在 `1` 后面一次，`2` 跟在 `1` 后面一次；`2` 后面没跟过任何编号，所以它那一行全是 0。
    """)
    return


@app.function
def bigram_counts(ids, size):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "bigram_counts", bigram_counts, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""先用 `np.zeros` 建一张全零表（注意形状元组外面还要再套一层括号）。用工具箱里的 `zip` 逐个遍历相邻的编号对，把对应那一格加 1。循环结束后返回这张表。""",
        r"""一共 4 行代码。循环头是 `for a, b in zip(ids, ids[1:]):`，循环体里只有一行，用来更新 `C[a, b]`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 零概率带来的麻烦

    要把一行次数变成概率，只要除以这一行的总和。但这样一来，训练时没见过的词对，概率就会正好是 0。回想第 07 章的**意外程度**（surprise）$-\log p$：概率 $p$ 越小，意外程度越高；当 $p$ 趋近于 0 时，$-\log p$ 会一路飙升到无穷大（$\infty$）。只要测试故事里冒出一对没见过的词，整个测试分数就会直接变成无穷大，就好像模型断言这件事绝对不可能发生。

    常用的办法是：假装每个词对都额外多出现过一点点（$k$ 次）。这叫**平滑**（smoothing）：先给表里每个次数加上 $k$，再除以这一行的新总和。

    > $$P(b \mid a) = \frac{C[a, b] + k}{\text{第 } a \text{ 行每一项都加上 } k \text{ 之后的总和}}$$

    ## 动手算：平滑一行

    取上面表里 `the` 那一行 $[0, 2, 1, 0, 0, 0]$，用 $k = 1$ 做平滑。`dog` 的新次数、`sat` 的新次数、这一行的新总数各是多少？再算算比例：平滑之前，`dog` 占这一行的 $2 / 3$，现在是变大了还是变小了？把它的新概率保留两位小数。
    """)
    return


@app.cell(hide_code=True)
def _():
    smooth_boxes = hand.boxes(['dog 的新次数', 'sat 的新次数', '新总数', 'dog 的新概率'], step=0.01)
    return (smooth_boxes,)


@app.cell(hide_code=True)
def _(smooth_boxes):
    hand.marked(smooth_boxes, {'dog 的新次数': 3, 'sat 的新次数': 1, '新总数': 9, 'dog 的新概率': 0.33}, {
        'dog 的新次数': r"""$2 + k$""",
        'sat 的新次数': r"""$0 + k$：从没见过的词对，现在算见过一次""",
        '新总数': r"""原来的总数是 3，6 项每项又多了 1""",
        'dog 的新概率': r"""新次数除以新总数""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `dog` 的概率从 0.67 降到了 0.33，`sat` 则从 0 升到了 0.11。这一行原本只有 3 次真实计数，$k = 1$ 实在太大了——人为补上的次数比真实统计出来的还多。在真正的故事文本上，我们会把 $k$ 设得小得多。

    ## 练习 2：bigram_probs

    写 `bigram_probs(C, k)`。`C` 是形状为 `(size, size)` 的次数表，`k` 是一个数。它按上面的公式返回一张同形状的概率表，每一行加起来都是 1：

    > `bigram_probs(np.array([[1.0, 3.0], [0.0, 0.0]]), 0.5)` 得到 `[[0.3, 0.7], [0.5, 0.5]]`

    第一行变成 $[1.5, 3.5]$，总和是 5；第二行原本一次计数都没有，平滑后各项概率平分。
    要让每一行都除以**它自己**的行总和，给 `np.sum` 加上 `axis=1` 和 `keepdims=True`（和第 07 章的 `softmax` 一样）：保留成“竖着的一列”，才能像 `(3, 2) / (3, 1)` 这样按行相除；如果不加 `keepdims=True`，结果拍扁成一维，就会按列除错。
    """)
    return


@app.function
def bigram_probs(C, k):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "bigram_probs", bigram_probs, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""分两步：先给整张表加上 `k`（一个数加一个数组，会自动加到每一格里），再除以新表每一行的总和。""",
        r"""一共 2 行代码：第一行是 `smoothed = C + k`，第二行除以用 `axis=1` 和 `keepdims=True` 算出的行总和（形状为 `(size, 1)`）。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面看看模型认为 `the` 和 `once` 后面最可能跟哪五个词，以及对应的概率。这个单元格会调用你刚写的两个函数，所以在练习 1 和练习 2 通过之前它会报错——这是正常的，后面依赖它们的单元格也一样。

    /// admonition | 🧰 工具箱：四舍五入
    `round(x, 3)` 把小数保留到小数点后 3 位，例如 `round(0.12345, 3)` 得到 `0.123`，纯粹是为了显示更清爽。
    ///

    （`np.argsort(-row)` 会按从大到小排出下标，和第 12 章一样。`the` 后面概率最高的“词”是 `<unk>`，因为这一项打包了成千上万个生僻词。）
    """)
    return


@app.cell
def _(index, train_ids, vocab):
    C = bigram_counts(train_ids, len(vocab))
    P = bigram_probs(C, 0.01)
    likely = []
    for start_word in ["the", "once"]:
        row = P[index[start_word]]
        best = np.argsort(-row)[:5]
        likely.append([(vocab[j], round(row[j].item(), 3)) for j in best])
    likely
    return C, P


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 让模型写句子

    要生成文本，先选一个开头词，按它那一行的概率分布随机抽下一个词，接着再用新词继续抽，不断重复。这里必须**随机**抽，不能每次都死选概率最高的那个词——否则 `the` 后面永远跟同一个词，句子很快就会陷入死循环。按给定概率随机抽取，叫**采样**（sampling）。

    /// admonition | 🧰 工具箱：按概率抽签
    `rng.choice(n, p=chances)` 会从 $0$ 到 $n - 1$ 里抽一个整数。`chances` 是长度为 $n$、总和为 1 的概率列表，数字 $i$ 被抽中的概率就是 `chances[i]`。比如 `rng.choice(3, p=[0.2, 0.5, 0.3])` 大约有一半概率抽出 `1`。
    ///

    下面的单元格按这组概率抽 10 次试试看。
    """)
    return


@app.cell
def _():
    rng_demo = np.random.default_rng(1)
    [rng_demo.choice(3, p=[0.2, 0.5, 0.3]) for _draw in range(10)]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：sample_next

    写 `sample_next(p, rng)`。`p` 是一个概率向量，对应每个词编号的概率；`rng` 是随机数生成器。函数按这些概率随机抽出一个编号并返回。比如 `p = [0.4, 0.6]` 时，调用 1,000 次大约有 400 次得到 0。

    通过检查后，下面的单元格会从 `once` 开始，让模型连写 30 个词。
    """)
    return


@app.function
def sample_next(p, rng):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "sample_next", sample_next, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""只要一行：`return` 加上 `rng.choice`。工具箱的例子是在 3 个数里抽。这里可选的个数不是固定的：`p` 里有多少个概率，就有多少个。""",
        r"""`len(p)` 是可选的个数，`p` 本身写在 `p=` 后面。不要用 `np.argmax`：它总是选最可能的编号。""",
    ], LOCALE)
    return


@app.cell
def _(P, index, vocab):
    rng = np.random.default_rng(0)
    written = [index["once"]]
    for _step in range(30):
        written.append(sample_next(P[written[-1]], rng))
    " ".join([vocab[i] for i in written])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    拆开看，大多数相邻的两词搭配都还算通顺，连成整段话却前言不搭后语——这就是“记忆力只有一个词”的局限。（`written[-1]` 表示列表的最后一项。生成结果里可能会夹杂 `<end>`，因为我们这个小循环没有在结尾处停下，而是顺手接着写起了下一个故事。）

    ## 给模型打分：困惑度

    怎么衡量一个语言模型的好坏？我们需要一个统一的分数，而且必须在它**没见过**的测试文本上考它。做法很直接：逐词读过测试故事，每次都问一句——对于真正出现的下一个词，模型给它打了多大的概率？

    把这些概率全部乘起来，就是整段文本的概率。但 15.9 万个小于 1 的数连乘，结果会小到计算机根本存不下（直接变成 0）。

    /// admonition | 🧰 工具箱：对数把乘法变成加法
    $\log(a \times b) = \log a + \log b$。第 07 章你见过它反过来的样子：$e^{a + b} = e^a \times e^b$。所以不用把一大堆概率连乘，改成把它们的对数相加就行，数值始终稳稳当当。除法也一样：$\log(a / b) = \log a - \log b$。
    ///

    把每个词的概率取对数、求平均，再加个负号变成正数，就得到了第 07 章的老朋友：

    > 平均意外程度 = $-$（$\log p$ 的平均值），也就是第 07 章的交叉熵（cross-entropy）

    不过“平均意外程度是 3.99”听起来不够直观。用 $e^{\square}$ 把对数还原回去，就得到了**困惑度**（perplexity）：

    > $$\text{困惑度} = e^{\text{平均意外程度}}$$

    哪怕没学过高中对数，看下面这张速查表也能一眼明白 $e^{\square}$ 到底还原出了什么：

    | 概率 $p$ | 意外程度 $-\log p$ | $e^{-\log p} = 1 / p$ | 直观感受 |
    |---|---|---|---|
    | 1 | 0 | 1 | 百分百确定（1 选 1） |
    | 0.5 | 0.69 | 2 | 像在 2 个等概率词里猜 |
    | 0.1 | 2.30 | 10 | 像在 10 个等概率词里猜 |
    | 0.01 | 4.61 | 100 | 像在 100 个等概率词里猜 |

    规律非常干净：假设模型一无所知，只能给 $N$ 个词各分 $1 / N$ 的概率，那么 $\log(1 / N) = \log 1 - \log N = -\log N$，每个词的意外程度都是 $\log N$，平均值也是 $\log N$，取完 $e^{\log N}$ 刚好变回 $N$。因此，**困惑度为 $N$，就意味着模型平均每一步都在 $N$ 个等概率的词里拿不定主意。** 完美的模型得 1 分，分数越低越好。

    ## 动手算：算一次困惑度

    一个模型读了一段三个词的文本。对真正出现的下一个词，它给出的概率分别是 0.5、0.25 和 0.125。你需要查的数都在下面：

    | $p$ | 0.5 | 0.25 | 0.125 |
    |---|---|---|---|
    | 意外程度 $-\log p$ | 0.69 | 1.39 | 2.08 |

    | $x$ | 0.69 | 1.39 | 2.08 | 4.16 |
    |---|---|---|---|---|
    | $e^x$ | 2 | 4 | 8 | 64 |

    算出意外程度的总和、平均意外程度和困惑度。
    """)
    return


@app.cell(hide_code=True)
def _():
    perplexity_boxes = hand.boxes(['意外程度的总和', '平均意外程度', '困惑度'], step=0.01)
    return (perplexity_boxes,)


@app.cell(hide_code=True)
def _(perplexity_boxes):
    hand.marked(perplexity_boxes, {'意外程度的总和': 4.16, '平均意外程度': 1.39, '困惑度': 4}, {
        '意外程度的总和': r"""把第一张表里的三个意外程度加起来""",
        '平均意外程度': r"""总和除以 3 个词""",
        '困惑度': r"""$e$ 的“平均意外程度”次方：在第二张表里查这个平均值""",
    }, LOCALE, tolerance=0.011)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这个模型的犹豫程度，相当于平均在 4 个词里猜一个：最有把握时像 2 选 1，最没把握时像 8 选 1。注意困惑度是 $e$ 的**平均值**次方（得到 4），而不是 $e$ 的总和次方（64）。

    ## 练习 4：perplexity

    写 `perplexity(log_probs)`。`log_probs` 是一个 NumPy 数组，里面已经存好了每个真实下一个词的 $\log p$（都是负数或 0）。请按上面的公式返回一个数。提醒：`np.exp(x)` 就是 $e^x$。

    > `perplexity(np.log(np.array([0.5, 0.25, 0.125])))` 得到 `4.0`
    """)
    return


@app.function
def perplexity(log_probs):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "perplexity", perplexity, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三个工具，一层套一层：对这些对数用 `np.mean`，加一个负号把它变成平均意外程度，最外面再套上 `np.exp`。""",
        r"""只要一行，开头是 `return np.exp(`。如果结果小于 1，就是漏了负号。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下一个单元格会在测试故事上给三个模型打分，再顺便看看二元模型在自己训练过的文本上能得几分。

    /// admonition | 🧰 工具箱：填充与按位置批量取数
    - `np.full(n, x)` 会生成由 `n` 个 `x` 组成的数组，例如 `np.full(3, 0.5)` 得到 `[0.5, 0.5, 0.5]`。
    - `items[:-1]` 表示去掉最后一项的其余全部内容，正好和去掉第一项的 `items[1:]` 相对。
    - 可以同时用两个下标数组从二维表里批量取数：第一个数组指定行号，第二个数组指定列号，按位置两两配对取出对应格子。例如 `P[[0, 2], [1, 3]]` 会取出 `[P[0, 1], P[2, 3]]`。
    ///

    - **一无所知：** 每个词都平分 $1 / 2002$ 的概率。
    - **一元模型**（unigram）：只看每个词本身有多常见，完全不管前文。表格 `C` 第 `a` 行的总和，就是 `a` 后面跟着其他词的总次数（也就是 `a` 出现的次数），拿各行总和除以全表总和就能得到每个词的概率。
    - **二元模型：** `P[test_ids[:-1], test_ids[1:]]` 一次性查出测试文本里每对相邻词的概率。拿一小段 `test_ids = [3, 7, 2, 9]` 把两个切片上下对齐，一眼就能看清：

      ```
      test_ids[:-1]  =  [3, 7, 2]   # 前一个词（行号）
      test_ids[1:]   =  [7, 2, 9]   # 下一个词（列号）
      ```

      竖着一一配对，就正好取出 `[P[3, 7], P[7, 2], P[2, 9]]`。
    """)
    return


@app.cell
def _(C, P, test_ids, train_ids, vocab):
    know_nothing = perplexity(np.full(len(test_ids), np.log(1 / len(vocab))))
    unigram_p = np.sum(C, axis=1) / np.sum(C)
    unigram = perplexity(np.log(unigram_p[test_ids]))
    bigram = perplexity(np.log(P[test_ids[:-1], test_ids[1:]]))
    bigram_on_train = perplexity(np.log(P[train_ids[:-1], train_ids[1:]]))
    [round(score.item(), 1) for score in (know_nothing, unigram, bigram, bigram_on_train)]
    return bigram, know_nothing, unigram


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第一个分数正好是 2,002，和前面的推导严丝合缝。光是知道词有多常见，困惑度就掉到了大约 282；再往前多看一个词，又一路降到大约 54——模型现在平均相当于在 54 个词里猜一个。最后一个分数是二元模型在**训练集**上的成绩（大约 42）：和第 09 章一样，模型在背过的题目上总是考得更好，所以我们必须用测试故事来公平打分。

    **平滑该加多少？** `k` 是我们人为设定的，不是靠梯度下降学出来的参数。这种由人预先定好的设置叫**超参数**（hyperparameter）。下面看看 `k` 取四个不同值时，二元模型的测试集困惑度：
    """)
    return


@app.cell
def _(C, test_ids):
    smoothing_scores = []
    for k_try in [1, 0.1, 0.01, 0.001]:
        P_try = bigram_probs(C, k_try)
        smoothing_scores.append((k_try, round(perplexity(np.log(P_try[test_ids[:-1], test_ids[1:]])).item(), 1)))
    smoothing_scores
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    平滑太大，会把太多概率分给根本不成立的词对；平滑太小，模型又对训练集里见过的词对过于自信。`0.01` 差不多刚刚好。

    ## 往回看两个词

    只记一个词还是太短了。**三元模型**（trigram）会往回看两个词：$P(\text{next} \mid a, b)$。如果还用表格存，就需要 $2{,}002 \times 2{,}002 \times 2{,}002$ 个格子，足足约 80 亿格，而且绝大部分都是 0。所以我们改用字典，只记录真正出现过的组合：把前两个词 `(a, b)` 当作键，对应的值是一个小字典，记录后面跟过哪些词、各出现了几次。

    /// admonition | 🧰 工具箱：字典的实用技巧
    - 元组（圆括号里的一对值）可以直接当字典的键：`seen[("the", "dog")] = 2`。
    - `key not in seen` 是 `in` 的反面，字典里没有这个键时返回 `True`。
    - `seen.values()` 会取出字典里所有的值，`sum(...)` 可以把它们加起来：例如 `ages = {"ann": 30, "bo": 12}` 时，`sum(ages.values())` 等于 42。
    - `array.tolist()` 把 NumPy 数组转成普通的 Python 列表，循环遍历和拿来做字典键都更快。
    ///

    在下面的函数里，`followers` 不是复制出来的新字典，它指向的就是 `table` 里面的那个小字典，所以修改 `followers` 就是在直接修改 `table`。另外，`zip` 把三个列表配对起来和配两个一样方便。
    """)
    return


@app.function
def trigram_counts(ids):
    table = {}
    for a, b, c in zip(ids, ids[1:], ids[2:]):
        if (a, b) not in table:
            table[(a, b)] = {}
        followers = table[(a, b)]
        followers[c] = followers.get(c, 0) + 1
    return table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    以编号 `[0, 1, 2, 0, 1, 3]` 为例，词对 `(0, 1)` 出现了两次，后面一次跟 `2`，一次跟 `3`。运行下面单元格看看生成的整张表：
    """)
    return


@app.cell
def _():
    trigram_counts([0, 1, 2, 0, 1, 3])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：trigram_chance

    写 `trigram_chance(table, a, b, c)`。`table` 是 `trigram_counts` 返回的字典，`a`、`b`、`c` 是三个词编号。函数返回一个数：在词对 `(a, b)` 出现的所有次数里，后面紧跟 `c` 的比例。如果 `(a, b)` 从没出现过，返回 `0.0`；如果 `c` 从没跟在 `(a, b)` 后面，也返回 `0.0`。例如 `table = {(0, 1): {2: 3, 3: 1}}` 时：

    > `trigram_chance(table, 0, 1, 2)` 得到 `0.75`，`trigram_chance(table, 5, 5, 2)` 得到 `0.0`

    词对 `(0, 1)` 一共出现过 $3 + 1 = 4$ 次，其中 3 次后面跟的是 2。
    """)
    return


@app.function
def trigram_chance(table, a, b, c):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "trigram_chance", trigram_chance, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""先处理没见过的词对：一个带 `not in` 的 `if`，里面写 `return`。过了这一步，词对一定是见过的，就从 `table` 里取出它的小字典。答案是小字典里的一个次数，除以它所有次数的总和。""",
        r"""一共 4 行。`c` 的次数用 `.get` 取，取不到就用 0，因为 `c` 不一定是键。总和是对小字典的 `.values()` 用 `sum`。出现 `KeyError`，就是漏了第一步的判断。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    三元模型的零概率问题更严重：测试故事里有很多两个词的组合在训练时根本没见过。所以我们把两个模型**混合**起来：一半概率来自三元模型，一半来自平滑后的二元模型。就算三元模型没见过某个组合，二元模型也能兜底。下面的循环运行需要几秒钟：
    """)
    return


@app.cell
def _(P, test_ids, train_ids):
    table = trigram_counts(train_ids.tolist())
    mixed = []
    for a, b, c in zip(test_ids.tolist(), test_ids.tolist()[1:], test_ids.tolist()[2:]):
        mixed.append(0.5 * trigram_chance(table, a, b, c) + 0.5 * P[b, c])
    trigram = perplexity(np.log(np.array(mixed)))
    round(trigram.item(), 1)
    return (trigram,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    降到了大约 38：又明显前进了一大步。

    ## 排行榜

    从这一章起，后面所有语言模型都会用同一把尺子来考：在同一批 1,000 篇测试故事上，按词计算困惑度。你的成绩会自动保存下来，排行榜会一章一章变长，最强的模型排在最上面：
    """)
    return


@app.cell(hide_code=True)
def _(bigram, know_nothing, trigram, unigram):
    record("uniform", know_nothing)
    record("unigram", unigram)
    record("bigram", bigram)
    record("trigram", trigram)
    show(LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 只靠数数做不到的事

    为什么不继续往回多看几个词，比如看四个词、十个词？因为表里很快就没数据可数了：绝大多数十个词连在一起的句子，在训练集里连一次都没出现过。而且还有更根本的硬伤：对只会数数的模型来说，`the dog ran`（狗跑了）和 `the cat ran`（猫跑了）毫无关联，每个词都只是一个孤立的行号。而第 12 章已经给了我们词向量，`dog` 和 `cat` 在向量空间里离得很近——基于向量的模型就能举一反三，把学到的规律在相似的词之间共享。

    ## 接下来

    在**第 14 章**，你会搭一个逐词阅读文本的神经网络，它一边读一边更新自己的记忆。这叫**循环神经网络**（recurrent neural network），它会一举登上这张排行榜的榜首。

    ## 你现在会做的事

    - 说清楚语言模型在做什么，并读懂 $P(\text{next} \mid \text{previous})$
    - 统计二元和三元词组，把出现次数换算成概率
    - 解释零概率为什么会出问题，并用平滑或混合来解决
    - 用采样生成文本
    - 用手算和代码计算困惑度，并说清它的直观含义
    - 知道什么是超参数，并解释为什么不该用测试集来挑超参数
    - 说明“只靠数数”为什么走不通，以及词向量能带来什么

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 往回看固定个数词的模型统称为 *n-gram* 模型。我们把三元和二元按比例相加的做法叫*线性插值*（linear interpolation）；专门用来挑选超参数的第三份数据叫*验证集*（validation set）——像我们刚才那样直接用测试故事挑 `k`，会漏进一点点测试集的信息，不过这里只挑一个数字，影响微乎其微。另外，困惑度是 $e$ 的*交叉熵*次方（如果交叉熵用比特计量，则是 $2$ 的交叉熵次方）。
    """)
    return


if __name__ == "__main__":
    app.run()
