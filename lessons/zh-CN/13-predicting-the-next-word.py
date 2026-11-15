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

    这一章你会造出最简单的语言模型，没有网络，也没有梯度，只靠数数。这样做有两个理由：它们能生成真正的文本（虽然有点傻），还给了我们一个**分数**，之后的每个模型都要打败它。

    ## 把文本变成数字

    和第 12 章一样，我们把故事切成单词，留下最常见的 2,000 个。这次词表（vocabulary）里多了两项：

    - `<unk>`（unknown，未知）代表这 2,000 个词以外的任何词；
    - `<end>` 标记一个故事的结尾，让模型能学会什么时候停下。

    `to_words` 和 `count_words` 是第 12 章的。`story_ids` 是新的：它把故事列表变成一个长长的编号数组。这次我们用全部 12,000 篇故事训练，用 1,000 篇测试故事打分。
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
    词表有 2,002 项，训练用的词大约 190 万个，留出的大约 159,000 个。`index.get(word, index["<unk>"])` 给出这个词的编号，不认识的词就给 `<unk>` 的编号。（这个单元格可能要几秒钟。下一节填表的单元格也一样。）

    ## 依赖于前文的概率

    在 `once upon a`（从前）后面，下一个词几乎一定是 `time`。在 `the` 后面，什么词都可能出现。所以我们想要的是：**已知**前面的内容时，每个词出现的概率。这叫**条件概率**（conditional probability），用一条竖线来写：

    > $P(\text{next} \mid \text{previous})$ 读作“已知刚看到 `previous`，下一个是 `next` 的概率”

    估计它最简单的办法是数数。看这段小文本，句号也算一个词：

    > the dog ran . the dog sat . the cat ran .

    `the` 出现了 3 次，其中两次后面是 `dog`，一次是 `cat`。所以 $P(\text{dog} \mid \text{the}) = 2 / 3$，$P(\text{cat} \mid \text{the}) = 1 / 3$，其余每个词都是 0。把这些次数列成一张表，每个词一行，每个可能的下一个词一列，这样的模型叫**二元模型**（bigram）：它看的是相邻的两个词。

    ## 动手算：一张次数表

    下面是这段小文本的表，`the` 那一行已经填好。行是排在前面的词，列是紧跟在后面的词。

    | 前 ↓ 后 → | the | dog | cat | ran | sat | . |
    |---|---|---|---|---|---|---|
    | **the** | 0 | 2 | 1 | 0 | 0 | 0 |
    | **dog** | 0 | 0 | 0 | A | B | 0 |
    | **ran** | 0 | 0 | 0 | 0 | 0 | C |
    | **.** | D | 0 | 0 | 0 | 0 | 0 |

    在文本里数一数词对，填出 A 到 D。再回答两个问题：这段文本一共有多少个相邻的词对？（它有 12 个词。）$P(\text{ran} \mid \text{dog})$ 是多少？
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
    /// admonition | 🧰 工具箱：相邻的一对
    - `items[1:]` 是一个没写终点的切片：从位置 1 开始的全部内容，也就是去掉第一项的列表。
    - `zip`（第 01 章）把两个列表逐项配对，短的那个用完就停。把一个列表和它自己的 `[1:]` 配对，就得到所有相邻的一对：对 `letters = ["a", "b", "c"]`，`zip(letters, letters[1:])` 先给出 `("a", "b")`，再给出 `("b", "c")`。
    - `for first, second in zip(...)` 一边走，一边给每一对的两半各起一个名字。
    ///

    ## 练习 1：bigram_counts

    写 `bigram_counts(ids, size)`。`ids` 是一个编号列表或数组，每个编号在 0 到 `size - 1` 之间。它返回形状为 `(size, size)` 的 NumPy 表 `C`，其中 `C[a, b]` 是 `b` 紧跟在 `a` 后面的次数：

    > `bigram_counts([0, 1, 0, 1, 2], 3)` 得到 `[[0, 2, 0], [1, 0, 1], [0, 0, 0]]`

    这里 1 跟在 0 后面两次，0 跟在 1 后面一次，2 跟在 1 后面一次。2 后面从没跟过别的编号，所以它那一行全是 0。
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
        r"""从一张全是零的表开始（`np.zeros`，形状要单独再加一层括号）。用工具箱里的 `zip` 逐个访问相邻的一对，把这一对在表里的那一格加 1。循环结束后返回这张表。""",
        r"""一共 4 行。循环是 `for a, b in zip(ids, ids[1:]):`，里面只有一行，改的是 `C[a, b]`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 零带来的麻烦

    把一行次数变成概率，只要除以这一行的总数。但这样一来，训练时没见过的词对概率就正好是 0，而第 07 章的意外程度（surprise）$-\log 0$ 是无穷大。测试故事里只要有一对没见过的词，整个分数就成了无穷大，就好像模型断言这件事绝不可能发生。

    常用的办法是：假装每个词对都多出现过一点点，也就是 $k$ 次。这叫**平滑**（smoothing）：先给每个次数加上 $k$，再除以这一行的新总数。

    > $$P(b \mid a) = \frac{C[a, b] + k}{\text{第 } a \text{ 行每一项都加上 } k \text{ 之后的总数}}$$

    ## 动手算：平滑一行

    取上面表里 `the` 那一行 $[0, 2, 1, 0, 0, 0]$，用 $k = 1$ 做平滑。`dog` 的新次数、`sat` 的新次数、这一行的新总数各是多少？再看它们占总数的比例：平滑之前，`dog` 占这一行的 $2 / 3$。现在是多了还是少了？把它的新概率写到两位小数。
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
    `dog` 的概率从 0.67 降到 0.33，`sat` 从 0 升到 0.11。这一行只有 3 次真实的计数，$k = 1$ 实在太大了：编出来的次数比真实的还多。在真正的文本上，我们会用小得多的 $k$。

    ## 练习 2：bigram_probs

    写 `bigram_probs(C, k)`。`C` 是形状为 `(size, size)` 的次数表，`k` 是一个数。它按上面的公式返回一张形状相同的表，所以每一行加起来都是 1：

    > `bigram_probs(np.array([[1.0, 3.0], [0.0, 0.0]]), 0.5)` 得到 `[[0.3, 0.7], [0.5, 0.5]]`

    第一行变成 $[1.5, 3.5]$，总数是 5。第二行原来一次计数都没有，平滑让它的各项概率相等。你需要每行一个总数，而且总数的形状要能让每一行各除各的：第 07 章的 `softmax` 就是这么做的。
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
        r"""分两步：先给整张表一次加上 `k`（一个数加一个数组，就是给每一项都加上它），再除以新表每一行的总数。`np.sum` 加上 `axis=1`，每行给出一个总数。""",
        r"""一共 2 行。第一行是 `smoothed = C + k`。第二行的 `np.sum` 还要加 `keepdims=True`，这样总数的形状是 `(size, 1)`，每一行除以的才是自己的总数。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面看看模型认为 `the` 和 `once` 后面可能跟什么：最可能的五个词，以及它们的概率。这个单元格用到你写的两个函数，所以练习 1 和练习 2 通过之前它会显示错误。这是正常的，后面用到它们的单元格也一样。

    /// admonition | 🧰 工具箱：四舍五入
    `round(x, 3)` 保留小数点后 3 位：`round(0.12345, 3)` 是 `0.123`。它只是为了让数字好看。
    ///

    （`np.argsort(-row)` 按从大到小列出位置，和第 12 章一样。`the` 后面最可能的“词”是 `<unk>`：这一项代表成千上万个少见的词。）
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
    ## 写文本

    要生成文本，先从一个词开始，按它那一行的概率随机抽下一个词，然后重复。要**随机**抽取，而不是总选最可能的词，这一点很重要：`the` 后面最可能的词总是同一个，文本很快就会原地打转。按一组概率来抽取，叫**采样**（sampling）。

    /// admonition | 🧰 工具箱：按概率抽取
    `rng.choice(n, p=chances)` 从 $0$ 到 $n - 1$ 中抽一个整数。`chances` 里有 $n$ 个数，加起来是 1，数字 $i$ 被抽中的概率是 `chances[i]`。所以 `rng.choice(3, p=[0.2, 0.5, 0.3])` 大约有一半的时候给出 1。
    ///

    下面的单元格按这组概率抽 10 次。
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

    写 `sample_next(p, rng)`。`p` 是一个概率向量，每个编号一个概率；`rng` 是一个随机数生成器。它按这些概率抽出一个编号并返回。取 `p = [0.4, 0.6]` 时，调用 1,000 次，应该有大约 400 次得到 0。

    检查之后的单元格从 `once` 开始，写 30 个词。
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
    大多数相邻的词对都说得通，整段话却不知所云。这就是只记得一个词的模型。（`written[-1]` 是列表的最后一项。文本里可能出现 `<end>`：我们这个小循环不会在那里停下，而是接着写下一个故事。）

    ## 一个分数：困惑度

    语言模型好不好？我们需要一个数字，而且要在它没见过的文本上衡量。逐词走过测试故事，问一个问题：模型给真正出现的下一个词多大的概率？

    把这些概率全部乘起来，就是整段文本的概率，但 159,000 个小数相乘，结果小到计算机存不下。

    /// admonition | 🧰 工具箱：对数把乘法变成加法
    $\log(a \times b) = \log a + \log b$。第 07 章见过它反过来的样子：$e^{a + b} = e^a \times e^b$。所以不用把很多概率相乘，把它们的对数相加就行，这样不会有哪个数小到存不下。除法也一样：$\log(a / b) = \log a - \log b$。
    ///

    把这些概率的对数加起来，除以词数，再变个号，就得到你已经认识的量：

    > 平均意外程度 = $-$($\log p$ 的均值)，也就是第 07 章的交叉熵（cross-entropy）

    为了更好读，我们用 $e^{\square}$ 把对数还原回去，就得到**困惑度**（perplexity）：

    > $$\text{困惑度} = e^{\text{平均意外程度}}$$

    这个数字的意思是：假设模型一无所知，给 $N$ 个词每个 $1 / N$ 的概率。因为 $\log(1 / N) = \log 1 - \log N = -\log N$，所以每次的意外程度都是 $\log N$，均值也是 $\log N$，困惑度就是 $e^{\log N} = N$。所以**困惑度为 N，意味着模型的犹豫程度，相当于在 N 个机会均等的词里选一个。** 完美的模型得 1 分。越低越好。

    ## 动手算：一个困惑度

    一个模型读了一段三个词的文本。对真正出现的下一个词，它给的概率分别是 0.5、0.25 和 0.125。你需要的数都在这里：

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
    模型的犹豫程度，相当于在 4 个词里选一个：最有把握时像 2 选 1，最没把握时像 8 选 1。注意答案是 $e$ 的*平均值*次方，也就是 4，不是 $e$ 的总和次方 64。

    ## 练习 4：perplexity

    写 `perplexity(log_probs)`。`log_probs` 是一个 NumPy 数组，存着每个真实的下一个词的 $\log p$：对数已经取好了，它们是负数或 0。它按上面两个公式返回一个数。`np.exp(x)` 就是 $e^x$。

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
    下一个单元格在测试故事上给三个模型打分，再给二元模型在训练文本上打一次分。

    /// admonition | 🧰 工具箱：填充和挑选
    - `np.full(n, x)` 是 `n` 个 `x` 组成的数组：`np.full(3, 0.5)` 是 `[0.5, 0.5, 0.5]`。
    - `items[:-1]` 是去掉最后一项的全部内容，正好和 `items[1:]` 相对。
    - 一张表可以同时用两个位置数组来取数，一个管行，一个管列。它们为每一对各取一项：`P[[0, 2], [1, 3]]` 给出 `P[0, 1]` 和 `P[2, 3]`。
    ///

    - **一无所知：** 每个词都是 $1 / 2002$。
    - **一元模型**（unigram）：只看每个词有多常见，不管前面是什么。`C` 第 `a` 行的总数，是 `a` 后面跟着别的词的次数，也就是 `a` 出现的次数，所以各行的总数正是我们要的。
    - **二元模型：** `P[test_ids[:-1], test_ids[1:]]` 为测试文本里每个相邻的词对查出一个概率。
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
    第一个数是 2,002，和上面算出来的结果一致。知道词有多常见，就降到大约 282。知道前一个词，就降到大约 54：模型的犹豫程度，相当于在 54 个词里选一个。最后一个数是二元模型在它自己的训练文本上的分数：大约 42。和第 09 章一样，模型在见过的东西上通常表现更好，所以我们要在测试故事上衡量。

    **平滑多少合适？** `k` 是我们选的，不是靠梯度下降训练出来的旋钮。这样的选择叫**超参数**（hyperparameter）。下面是四个取值下二元模型的测试分数：
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
    平滑太多，会把概率分给胡说八道的词对；太少，模型又对见过的东西过于自信。0.01 差不多正好。

    ## 往回看两个词

    只记一个词实在不多。**三元模型**（trigram）往回看两个词：$P(\text{next} \mid a, b)$。这样一张表需要 $2{,}002 \times 2{,}002 \times 2{,}002$ 项，大约 80 亿，但几乎全是 0。所以我们改用字典（dictionary），只存真正见过的：每一对 `(a, b)` 是一个键，它的值是一个小字典，记录后面跟过什么。

    /// admonition | 🧰 工具箱：字典的另外几个工具
    - 键可以是圆括号里的一对：`seen[("the", "dog")] = 2`。
    - `key not in seen` 是 `in` 的反面：字典里没有这个键时为真。
    - `seen.values()` 逐个给出字典的值，`sum(...)` 把一串数加起来：对 `ages = {"ann": 30, "bo": 12}`，`sum(ages.values())` 是 42。
    - `array.tolist()` 把 NumPy 数组变成普通列表，用来循环、当字典的键都更快。
    ///

    在下面的函数里，`followers` 不是副本，它就是 `table` 里的那个小字典，所以改它就是改表。`zip` 配三个列表和配两个一样容易。
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
    对编号 `[0, 1, 2, 0, 1, 3]`，词对 `(0, 1)` 出现了两次，后面一次是 2，一次是 3。这个单元格显示整张表：
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

    写 `trigram_chance(table, a, b, c)`。`table` 是 `trigram_counts` 返回的东西，`a`、`b`、`c` 是三个编号。它返回一个数：在词对 `(a, b)` 出现的所有次数里，后面跟着 `c` 的比例。从没见过的词对得到 `0.0`，从没跟在这个词对后面的 `c` 也得到 `0.0`。取 `table = {(0, 1): {2: 3, 3: 1}}`：

    > `trigram_chance(table, 0, 1, 2)` 得到 `0.75`，`trigram_chance(table, 5, 5, 2)` 得到 `0.0`

    词对 `(0, 1)` 出现了 $3 + 1 = 4$ 次，其中 3 次后面是 2。
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
    三元模型的零问题更严重：测试故事里的很多词对从没见过。所以我们把模型**混合**起来：一半概率来自三元模型，一半来自平滑后的二元模型。三元模型什么也不知道时，二元模型还有答案。下面的循环要几秒钟：
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
    大约 38：又明显往下走了一步。

    ## 排行榜

    从现在起，这门课的每个语言模型都用同样的方式打分：在同样的 1,000 篇测试故事上，按词计算困惑度。你的分数会保存下来，所以这张表会一章一章地变长，最好的模型排在最上面：
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
    ## 数数做不到的事

    为什么不继续往回数，数四个词、十个词？因为表里的证据不够了：大多数十个词的序列连一次都没出现过。还有一个更深的弱点。对只会数数的模型来说，`the dog ran`（狗跑了）对 `the cat ran`（猫跑了）毫无帮助，每个词都只是一个行号。但第 12 章给了我们向量，`dog` 和 `cat` 在里面离得很近，所以基于向量的模型可以在相似的词之间共享学到的东西。

    ## 接下来

    在**第 14 章**，你会造一个逐词读文本的网络，它一边读一边更新记忆。这叫**循环神经网络**（recurrent neural network），它会登上这张排行榜的榜首。

    ## 你现在会做的事

    - 解释语言模型做什么，并读懂 $P(\text{next} \mid \text{previous})$
    - 统计二元和三元的词对，把次数变成概率
    - 解释为什么概率为零是个问题，并用平滑或混合来解决
    - 用采样生成文本
    - 用手算和代码计算困惑度，并说出它的含义
    - 说出什么是超参数，并解释为什么不该用测试数据来选它
    - 解释数数为什么行不通，以及向量带来了什么

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 往回看固定个数词的模型叫 *n-gram* 模型。我们的混合叫*线性插值*（linear interpolation），用来选超参数的第三份数据叫*验证集*（validation set）：像我们这样用测试故事选 `k`，会让一点测试信息漏进去，不过只选一个小数字，影响可以忽略。困惑度是 `e` 的*交叉熵*次方，如果用比特来计量，就是 2 的交叉熵次方。
    """)
    return


if __name__ == "__main__":
    app.run()
