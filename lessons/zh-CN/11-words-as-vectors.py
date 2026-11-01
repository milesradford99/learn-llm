import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="11 · 把词变成向量")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 11 · 把词变成向量

    课程的第二部分讲语言。从这里到最后一章，我们都用同一份文本：一批简短的小故事，叫 **TinyStories**。这样每个新模型都能和之前的模型比较。它已经打包在课程里，而且所有语言版本用的都是这份英文文本。下面是 12,000 篇故事中第一篇的开头：是英文原文，不用读懂，只要知道这是一段文本就行。（另外 1,000 篇 `test_stories` 留到第 12 章用。）
    """)
    return


@app.cell
def _():
    train_stories, test_stories = stories()
    train_stories[0][:300]
    return (train_stories,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    网络处理的是数字，所以第一个问题是怎样把词变成数字。最直接的办法是第 06 章的 one-hot 行：如果认识 2,000 个词，每个词就是一行 2,000 个数字，其中只有一个 1。但这对网络毫无帮助。任意两个不同的 one-hot 行，点积都是 0，所以 `dog` 和 `cat` 的差别，跟 `dog` 和 `yesterday` 的差别一样大。

    我们想要更好的东西：一个很短的**词向量**（word vector），比如 32 个数字，并且让意思相近的词得到相近的向量。这些数字没人写得出来。但有个办法只靠文本就能把它们学出来，它基于一个观察：**看一个词的邻居，就能了解这个词。** 遮住下面句子里的一个词，你依然能猜出它是哪一类词：

    > The ▢ barked at the mailman and wagged its tail.（这只 ▢ 冲着邮递员叫，还摇着尾巴。）

    出现在相同邻居中间的词，意思往往相近。所以我们给网络一个任务：预测一个词的邻居，然后留下它在这个过程中发明的向量。这就是 **word2vec** 方法。

    ## 从文本到单词

    先把文本切成单词。下面这个辅助函数把所有字母变成小写，保留连续的字母和撇号，其余的都丢掉。（`re` 是 Python 的文本模式工具，不用专门学；`[a-z']+` 的意思就是“一个或多个字母或撇号”。）
    """)
    return


@app.function
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.cell
def _(train_stories):
    words = to_words(" ".join(train_stories[:6000]))
    len(words), words[:12]
    return (words,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    （`" ".join(list)` 把字符串列表粘成一个字符串，中间加空格。6,000 篇故事大约有 965,000 个词，对这一章足够了。用 12,000 篇只不过多花一倍时间。）

    ## 用字典计数

    我们需要知道每个词出现了多少次。Python 为此准备了**字典**（dictionary）：它存的是一对对的**键**（key）和**值**（value），能一下子找到某个键。列表按位置查找，字典按键查找。

    - `{}` 造出一个空字典；`counts["dog"] = 3` 把 3 存在键 `"dog"` 下面。
    - `counts["dog"]` 把它读出来。
    - `counts.get("dog", 0)` 也是读取，但键还不存在时给出 0，而不是报错停下。
    """)
    return


@app.cell
def _():
    pets = {}
    pets["dog"] = 3
    pets["cat"] = pets.get("cat", 0) + 1
    pets, pets["dog"], pets.get("fish", 0)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：count_words

    写 `count_words(words)`，返回一个字典，从每个词映射到它出现的次数：

    1. `counts = {}`
    2. `for word in words:`，里面写 `counts[word] = counts.get(word, 0) + 1`
    3. 循环结束后，`return counts`
    """)
    return


@app.function
def count_words(words):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "count_words", count_words, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在我们留下最常见的 2,000 个词作为**词表**（vocabulary），并给每个词一个数字，也就是它的**编号**（id）：它在列表里的位置。

    - `sorted(counts, key=counts.get, reverse=True)` 列出所有键，按次数从大到小排序。
    - `index` 是反过来的字典：从词到编号。
    - `ids` 是用编号表示的全部文本。造列表的表达式末尾的 `if word in index` 跳过词表以外的词。
    """)
    return


@app.cell
def _(words):
    counts = count_words(words)
    vocab = sorted(counts, key=counts.get, reverse=True)[:2000]
    index = {}
    for i in range(len(vocab)):
        index[vocab[i]] = i
    ids = [index[word] for word in words if word in index]
    len(counts), vocab[:8], len(ids)
    return counts, ids, index, vocab


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    一共出现了大约 6,800 个不同的词，这 2,000 个就覆盖了文本的 97%。小故事的词汇量小，所以很适合在笔记本上跑。

    ## 两个向量有多像？

    第 05 章的点积是逐个位置相乘再相加。两个向量方向一致时，点积就大。但它也会随长度增长，而衡量相似只关心方向。所以我们除以两个向量的长度。用 `·` 表示点积（代码里是 `@`），向量的**长度**是 `√(a · a)`，也就是各项平方和的平方根。这个度量叫**余弦相似度**（cosine similarity）：

    > similarity(a, b) = (a · b) / (a 的长度 × b 的长度)

    方向相同时是 1，互相垂直（毫无共同点）时是 0，方向相反时是 −1。取 `a = [3, 4]`，`b = [6, 8]`，点积是 3 × 6 + 4 × 8 = 50。长度是 √(9 + 16) = 5 和 √(36 + 64) = 10。所以相似度是 50 / (5 × 10) = 1：`b` 是 `a` 拉长一倍，方向相同。

    ## 练习 2：similarity

    写 `similarity(a, b)`，输入是两个 NumPy 向量：

    1. `length_a = np.sqrt(a @ a)`
    2. `length_b = np.sqrt(b @ b)`
    3. `return (a @ b) / (length_a * length_b)`
    """)
    return


@app.function
def similarity(a, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "similarity", similarity, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 任务：猜邻居

    训练样本是一对对编号。沿着文本滑动；每个词轮流当**中心词**（centre word），两边 `window` 个位置以内的每个词都是它的**上下文词**（context word）。对文本 `the dog ran fast`，`window = 1`：

    | 中心词 | 上下文词 | 词对 |
    |---|---|---|
    | `the` | `dog` | (the, dog) |
    | `dog` | `the`、`ran` | (dog, the)、(dog, ran) |
    | `ran` | `dog`、`fast` | (ran, dog)、(ran, fast) |
    | `fast` | `ran` | (fast, ran) |

    从中心词预测上下文词，这个任务叫 **skip-gram**。

    ## 练习 3：skipgram_pairs

    写 `skipgram_pairs(ids, window)`，返回两个长度相同的列表：中心词和上下文词。对每个位置 `i`，邻居从 `i - window` 到 `i + window`，但必须留在列表范围内，也不能是 `i` 自己。`max(a, b)` 取较大的数，`min(a, b)` 取较小的数，正好用来处理边界。`!=` 表示“不等于”。`range` 在最后一个数之前就停下，所以要 `+ 1` 才能走到 `i + window`。

    1. 两行：`centres = []` 和 `contexts = []`
    2. `for i in range(len(ids)):`
    3. 里面写：`for j in range(max(0, i - window), min(len(ids), i + window + 1)):`
    4. 再里面：`if j != i:`
    5. 再里面两行：`centres.append(ids[i])` 和 `contexts.append(ids[j])`
    6. 循环结束后，`return centres, contexts`
    """)
    return


@app.function
def skipgram_pairs(ids, window):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "skipgram_pairs", skipgram_pairs, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 真的还是编的？换个便宜的问题

    每个词有两个向量，当中心词时用一个（`u`），当上下文词时用另一个（`v`）。词义最后会落在 `u` 里。我们希望当 `v` 确实是邻居时，`u · v` 很高。

    最直接的模型是第 06 章那一套：给全部 2,000 个词打分，再做 softmax。但这意味着每一对都要算 2,000 个点积，而词对有几百万个。我们改问一个便宜得多的问题：*这一对是真的吗？* 对每个真实的词对，再随机抽几个**噪声词**（noise word），训练网络把真的上下文词和假的区分开。这叫**负采样**（negative sampling）。

    回答是或否，需要从一个分数 `s` 得到一个概率。对两个分数做 softmax：“真”的分数是 `s`，“假”的分数是 0。（为什么是 0？第 06 章的平移说明，softmax 只关心分数之间的差，所以其中一个不妨取 0。）任何数的 0 次方都是 1，所以 e^0 = 1，“真”的份额是

    > σ(s) = e^s / (e^s + e^0) = e^s / (e^s + 1)

    这叫 **sigmoid**。`s` 很大时它接近 1，`s = 0` 时正好是 0.5，`s` 很负时接近 0。还有一个好用的事实：“假”的份额是 1 − σ(s)，它等于 σ(−s)。（验证一下：σ(−s) = e^(−s) / (e^(−s) + 1)。分子分母同乘 e^s，注意 e^(−s) × e^s = e^0 = 1，得到 1 / (1 + e^s)，正好等于 1 − e^s / (e^s + 1)。）

    损失就是第 06 章的意外程度，即正确答案所得份额的 `−log`，把一个真实词和 `k` 个噪声词加起来。对真实词，正确答案是“真”，份额是 σ(u · v)。对噪声词 `n`，正确答案是“假”，份额是 σ(−u · n)：

    > 一对的损失 = −log σ(u · v) − log σ(−u · n₁) − … − log σ(−u · nₖ)

    第一项把 `u` 拉向真实的邻居，其余各项把它从假词那里推开。在 PyTorch 里，`F.logsigmoid(s)` 安全地计算 `log σ(s)`，道理和第 06 章 softmax 的平移相似。

    ## 练习 4：pair_loss

    写 `pair_loss(u, v, n)`，一次处理一整批。`u` 和 `v` 的形状是 `(b, d)`，表示 `b` 对、每个向量 `d` 个数字。`n` 的形状是 `(b, k, d)`，每对 `k` 个噪声向量。（这里的 `n` 指噪声，不是前面章节里的样本数。）

    1. `real = torch.sum(u * v, dim=1)`，每对一个点积，形状是 `(b,)`
    2. `fake = torch.sum(n * u.unsqueeze(1), dim=2)`。`u.unsqueeze(1)` 的形状是 `(b, 1, d)`，大小为 1 的维度会被广播，所以它的作用相当于 `(b, k, d)`。这样每对得到 `k` 个点积，形状是 `(b, k)`
    3. `surprise = -F.logsigmoid(real) - torch.sum(F.logsigmoid(-fake), dim=1)`
    4. `return torch.mean(surprise)`
    """)
    return


@app.function
def pair_loss(u, v, n):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "pair_loss", pair_loss, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 训练

    最后一个 PyTorch 工具：`nn.Embedding(2000, 32)` 是一张 2,000 行、每行 32 个旋钮的表。给它任意形状的编号张量，它就为每个编号查出一行，所以 `(b,)` 的编号给出 `(b, 32)` 的向量，`(b, k)` 的编号给出 `(b, k, 32)`。就这么简单：一次查表，而表里的数字会学习。

    噪声词按概率表 `noise` 抽取：每个词的次数取 0.75 次方，再缩放到总和为 1。小于 1 的次方对大数压得多、对小数压得少（`100 ** 0.75` 大约是 32，`10000 ** 0.75` 是 1,000），所以常见词没那么占优势了，少见的词也有机会当噪声词。`torch.multinomial(noise, m, replacement=True)` 按这些概率抽出 `m` 个编号。

    `train_vectors` 就是第 09 章的 `train`，把交叉熵换成了 `pair_loss`。它做的事是：建两张表，交给 Adam 训练；每一批先查出 `u`、`v` 和 `n`，然后清空、反向、走一步。它返回 NumPy 数组形式的中心词表，每个词一行；上下文词表则丢掉。`nn.init.normal_` 给表一个小的随机起点。
    """)
    return


@app.function
def train_vectors(centres, contexts, noise, size, epochs, batch_size, rate):
    centre_table = nn.Embedding(len(noise), size)
    context_table = nn.Embedding(len(noise), size)
    nn.init.normal_(centre_table.weight, 0, 0.1)
    nn.init.normal_(context_table.weight, 0, 0.1)
    optimizer = torch.optim.Adam([centre_table.weight, context_table.weight], lr=rate)
    for epoch in range(epochs):
        order = torch.randperm(len(centres))
        for start in range(0, len(centres), batch_size):
            batch = order[start:start + batch_size]
            noise_ids = torch.multinomial(noise, 5 * len(batch), replacement=True)
            n = context_table(noise_ids.reshape(len(batch), 5))
            optimizer.zero_grad()
            loss = pair_loss(centre_table(centres[batch]), context_table(contexts[batch]), n)
            loss.backward()
            optimizer.step()
    return centre_table.weight.detach().numpy()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    按钮先造出词对（`window = 2` 时大约 370 万对），再训练 2 轮：在笔记本上大约半分钟。完成之前，需要它的结果的单元格会显示 `Ancestor stopped`，这是正常的。完成后你会看到 `(2000, 32)`：2,000 个词，每个 32 个数字。
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="训练词向量")
    train_button
    return (train_button,)


@app.cell
def _(counts, ids, train_button, vocab):
    mo.stop(not train_button.value, mo.md("*按下按钮开始训练。*"))
    centres, contexts = skipgram_pairs(ids, 2)
    freq = torch.tensor([counts[word] for word in vocab], dtype=torch.float32)
    noise = freq ** 0.75 / torch.sum(freq ** 0.75)
    torch.manual_seed(0)
    E = train_vectors(torch.tensor(centres), torch.tensor(contexts), noise, 32, 2, 1024, 0.003)
    E.shape
    return (E,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 它学到了什么？

    没有人告诉网络任何词的意思，它也没见过词典。它只看到一对对邻居。现在来找找，哪些词的向量和某个词最相似。

    `nearest` 把每一行都缩放到长度 1。两个长度都是 1，相似度公式里的除法就没有了，所以光用 `@` 就能一次算出和所有词的相似度。`np.argsort` 给出从小到大排序的位置，所以我们对 `-sims` 排序，让最大的排在最前面。排第一的总是词本身，所以跳过它。在框里换一个词，按回车：
    """)
    return


@app.function
def nearest(query, E, vocab, count):
    unit = E / np.sqrt(np.sum(E * E, axis=1, keepdims=True))
    sims = unit @ (query / np.sqrt(query @ query))
    return [vocab[i] for i in np.argsort(-sims)[:count]]


@app.cell(hide_code=True)
def _():
    word_box = mo.ui.text(value="dog", label="词：")
    word_box
    return (word_box,)


@app.cell
def _(E, index, vocab, word_box):
    asked = word_box.value.strip().lower()
    nearest(E[index[asked]], E, vocab, 6)[1:] if asked in index else "这个词不在词表里。"
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    再看几个词的表格：
    """)
    return


@app.cell(hide_code=True)
def _(E, index, vocab):
    table = ["| 词 | 最近的词 |", "|---|---|"]
    for word in ["dog", "happy", "mom", "red", "ran"]:
        table.append(f"| {word} | " + ", ".join(nearest(E[index[word]], E, vocab, 6)[1:]) + " |")
    mo.md("\n".join(table))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `dog`（狗）周围是动物，`happy`（开心）周围是感受和状态，`mom`（妈妈）周围是家人，`red`（红色）周围是颜色，`ran`（跑）周围是各种移动方式。（你的列表每次运行可能略有不同。）这一切都只来自谁站在谁旁边。

    ## 意义的算术

    向量之间的差也能表示关系。如果从 `boy`（男孩）到 `girl`（女孩）的这一步大致是“变成女性”的方向，那么从 `he`（他）走同样一步，应该落在 `she`（她）附近。下面的单元格计算 `b − a + c`，并找出不属于这三个词的最近的词。（`not in` 是 `in` 的反义。）
    """)
    return


@app.function
def analogy(a, b, c, E, index, vocab):
    query = E[index[b]] - E[index[a]] + E[index[c]]
    return [word for word in nearest(query, E, vocab, 6) if word not in (a, b, c)][:3]


@app.cell(hide_code=True)
def _(E, index, vocab):
    rows = ["| a 之于 b | 如同 c 之于 | 最佳猜测 |", "|---|---|---|"]
    for a, b, c in [("boy", "girl", "he"), ("he", "she", "his"), ("boy", "girl", "brother"), ("go", "went", "see")]:
        rows.append(f"| {a} : {b} | {c} | " + ", ".join(analogy(a, b, c, E, index, vocab)) + " |")
    mo.md("\n".join(rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    表里的英文词：`his` 他的，`her` 她的，`brother` 兄弟，`sister` 姐妹，`go`/`went` 去/去了，`see`/`saw` 看见/看见了。前三行排第一的猜测是 `she`、`her` 和 `sister`，最后一行的过去式 `saw` 排第二。（`lily's` 是“莉莉的”，算一个词，因为我们的切词保留撇号。）向量学到了男性和女性，甚至一点语法，而没有人告诉它们这些是什么。

    别指望它完美。它只有每个词 32 个数字，训练文本也只有约一百万个词，所以有些结果很奇怪，重新运行时也会有些不同。它也只学得到文本里有的东西。用儿童故事训练出的向量懂狗和妈妈，不懂化学。文本的各种习惯，包括关于谁该做什么的偏见，也会进到向量里。

    ## 接下来

    词向量抓住了词的意思，但到现在为止，还没有任何东西用到词的*顺序*。在**第 12 章**，你会造出第一个真正的语言模型：它预测下一个词，并能生成文本。为了衡量它，你会认识**困惑度**（perplexity），并开始建立**排行榜**（leaderboard）：在同一批留出的故事上算出的一个分数，从那里到课程结束，每个模型都要尝试打破它。

    ## 你现在会做的事

    - 解释为什么 one-hot 行不含词义，词向量又增加了什么
    - 用字典计数，并把文本变成编号
    - 用余弦相似度衡量两个向量有多像
    - 从文本造出 skip-gram 词对
    - 从 softmax 推出 sigmoid，并解释负采样
    - 把 `nn.Embedding` 用作一张学出来的向量表
    - 探索最近的词和类比，并说出它们的局限

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 这个方法是 *word2vec*（Mikolov 等，2013）。实际运行时还会丢掉一部分最常见的词，比如 `the`；*GloVe* 则直接统计词对，得到类似的向量。“词义体现在邻居里”这个想法叫*分布假说*（distributional hypothesis）。故事来自 *TinyStories*（Eldan 和 Li，2023）。
    """)
    return


if __name__ == "__main__":
    app.run()
