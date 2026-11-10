import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="15 · Transformer")

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
    from learn_llm.images import maps
    from learn_llm.leaderboard import record, show

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 15 · Transformer

    第 14 章最后留了个问题：既然真正起作用的是注意力，为什么还要留着 LSTM？LSTM 只能逐词地读，每一步都要等前一步，责任也要走很长的路。

    **Transformer** 彻底去掉了循环。每个词直接看它前面的词，而且同时进行。这个设计发表于 2017 年，今天所有的大语言模型都建立在它之上。学完这一章，你会造出一个 Transformer：一个小小的 **GPT**，大约一百万个旋钮，它会登上你排行榜的第一名。

    里面的每个想法你都见过：注意力（第 14 章）、`x + f(x)`（第 10 章）、层和 Adam（第 09 章）。这一章只增加四个小部件，然后把它们组装起来。

    ## 自注意力

    在第 14 章，解码器的状态向编码器的状态提问。在**自注意力**（self-attention）里，同一段文本的词互相提问。每个词的向量经过三个不同的 `nn.Linear` 层，得到三个角色：

    - **查询**（query）：我在找什么？
    - **键**（key）：我能提供什么？
    - **值**（value）：被选中时我传出什么？

    接下来的注意力和以前一样：比较、分配、混合。所以每个词的新向量，是它认为相关的那些词的混合。

    ## 部件 1：不许偷看

    语言模型要预测下一个词，所以一个词不能看到它后面的词，否则模型直接抄答案就行了。我们需要一张表，标出要挡住的位置。4 个词的情况如下，`True` 表示“挡住”：

    ```
    False  True  True  True      第 0 个词只看到自己
    False False  True  True      第 1 个词看到第 0、1 个词
    False False False  True
    False False False False      第 3 个词看到它前面的全部
    ```

    `torch.ones(n, n, dtype=torch.bool)` 是一张全是 `True` 的表，`torch.triu(table, diagonal=1)` 保留主对角线以上的部分，其余设为 `False`（“triu”是 upper triangle，上三角的缩写）。

    ## 练习 1：future_mask

    写 `future_mask(n)`：

    ```
    return torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    ```
    """)
    return


@app.function
def future_mask(n):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "future_mask", future_mask, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 部件 2：保持尺度的注意力

    使用掩码时，我们在 softmax 之前把被挡住的分数设为负无穷。因为 $e^{-\infty}$ 是 0，被挡住的词份额正好是 0，其余份额加起来仍然是 1。`scores.masked_fill(mask, -torch.inf)` 在 `mask` 为 `True` 的地方存入这个数。

    和第 14 章相比还有一处改动。两个各有 $d$ 个随机数的向量做点积，是把 $d$ 项加起来。根据第 08 章的规则，方差相加，所以对于普通大小的数，它的典型大小随 $\sqrt{d}$ 增长。分数太大时，softmax 几乎把全部份额都给一个词，其余的词就几乎收不到责任。所以我们把分数除以 $\sqrt{d}$，让它保持在合适的尺度上。

    ## 练习 2：attend

    写 `attend(Q, K, V, mask)`。形状和第 14 章一样，`d` 是 `K` 的最后一维大小。用 `-1` 表示“最后一维”，`-2` 表示它前面一维，这样函数对任意维数都能用，我们很快就会需要。它返回混合结果和份额。

    1. `scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])`
    2. `scores = scores.masked_fill(mask, -torch.inf)`
    3. `A = torch.softmax(scores, dim=-1)`
    4. `return A @ V, A`
    """)
    return


@app.function
def attend(Q, K, V, mask):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "attend", attend, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 部件 3：多个头

    一轮注意力给每个词一个混合结果。但一个词可能同时想知道几件事：主语是谁？现在是什么时态？所以我们并排跑几轮，叫**头**（head），每个头有自己的查询、键和值。

    这不会增加大小。当 `d = 128`、有 4 个头时，每个向量切成 4 段，每段 32 个数字，每个头处理自己的一段，最后再把 4 个结果拼起来。形状的变化是：

    > `(batch, words, 128)` → `(batch, words, 4, 32)` → `(batch, 4, words, 32)`

    最后一次交换让 `heads` 排到和批相同的位置上，这样 `@` 和同一个 `attend` 就会把每个头当成一段独立的文本。

    ## 练习 3：split_heads

    写 `split_heads(x, heads)`，`x` 的形状是 `(batch, words, d)`：

    1. `batch, words, d = x.shape`
    2. `return x.reshape(batch, words, heads, d // heads).transpose(1, 2)`
    """)
    return


@app.function
def split_heads(x, heads):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "split_heads", split_heads, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是完整的层。末尾的 `reshape` 撤销 `split_heads`，最后一个 `nn.Linear` 把各个头的发现混合起来。`self.shares` 保存最新的份额，方便以后查看。
    """)
    return


@app.cell
def _():
    class SelfAttention(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.heads = heads
            self.query = nn.Linear(d, d, bias=False)
            self.key = nn.Linear(d, d, bias=False)
            self.value = nn.Linear(d, d, bias=False)
            self.mix = nn.Linear(d, d)

        def forward(self, x):
            batch, words, d = x.shape
            Q = split_heads(self.query(x), self.heads)
            K = split_heads(self.key(x), self.heads)
            V = split_heads(self.value(x), self.heads)
            blended, self.shares = attend(Q, K, V, future_mask(words))
            return self.mix(blended.transpose(1, 2).reshape(batch, words, d))
    return (SelfAttention,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 注意力不识数

    做个试验。取三个词向量，去掉掩码运行注意力。然后把词的顺序反过来，再运行一次注意力，然后把结果的顺序也反过来。每个词向同样的词问了同样的问题，所以应该得到同样的混合结果。单元格打印两次结果的最大差别：
    """)
    return


@app.cell
def _(SelfAttention):
    torch.manual_seed(0)
    layer_demo = SelfAttention(8, 2)
    x_demo = torch.rand(1, 3, 8)
    Q_demo = split_heads(layer_demo.query(x_demo), 2)
    K_demo = split_heads(layer_demo.key(x_demo), 2)
    V_demo = split_heads(layer_demo.value(x_demo), 2)
    no_mask = torch.zeros(3, 3, dtype=torch.bool)
    forwards = attend(Q_demo, K_demo, V_demo, no_mask)[0]
    backwards = attend(torch.flip(Q_demo, dims=[2]), torch.flip(K_demo, dims=[2]), torch.flip(V_demo, dims=[2]), no_mask)[0]
    torch.max(torch.abs(forwards - torch.flip(backwards, dims=[2]))).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    结果大约是 `6e-08`，也就是 0.00000006：只是舍入误差，所以两者相同。对注意力来说，`the dog bit the man`（狗咬了人）和 `the man bit the dog`（人咬了狗）是同一个词袋（bag of words）。LSTM 按顺序读，顺序是白送的。这里我们得自己提供。

    ## 部件 4：位置和稳定的尺度

    **位置。** 最简单的办法：再加一个 `nn.Embedding`，每个位置一个学出来的向量。位置 0 有一个向量，位置 1 有另一个，依此类推。把它加到词的向量上，词就知道自己站在哪里了。（第 18 章有更聪明的办法。）这也带来一个限制：位置表有 32 项，模型就只能看到最近的 32 个词。

    **稳定的尺度。** 堆很多层时，数值可能衰减或爆炸，这就是第 08 章深度表格里的麻烦。**层归一化**（layer normalisation）用第 08 章的方差，把每个词的向量重新缩放到均值 0、典型大小 1。真实的层还会多加两个学习旋钮，`nn.LayerNorm` 替我们做了。

    ## 练习 4：layer_norm

    写 `layer_norm(x)`，沿最后一维计算。`keepdim=True` 就是 PyTorch 版的 `keepdims=True`，`unbiased=False` 要的是第 08 章那种普通方差。`1e-5` 防止除以 0。

    1. `mean = torch.mean(x, dim=-1, keepdim=True)`
    2. `variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)`
    3. `return (x - mean) / torch.sqrt(variance + 1e-5)`
    """)
    return


@app.function
def layer_norm(x):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "layer_norm", layer_norm, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 块

    注意力让词互相交流。然后每个词需要时间想想听到了什么，所以一个**块**（block）还有第二部分：第 07 章的小网络，分别作用在每个词上，隐藏层的宽度是四倍。

    两部分都走第 10 章那条责任的通道 `x + f(x)`，归一化放在 `f` 之前：

    > `x = x + attention(norm1(x))`

    > `x = x + think(norm2(x))`

    ## 练习 5：block_step

    写 `block_step(x, norm1, attention, norm2, think)`：

    1. `x = x + attention(norm1(x))`
    2. `x = x + think(norm2(x))`
    3. `return x`
    """)
    return


@app.function
def block_step(x, norm1, attention, norm2, think):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "block_step", block_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 完整的模型

    词表加位置表，一叠块，最后再归一化一次，然后用 `nn.Linear` 给每个词打分。`torch.arange(n)` 是张量 $0, 1, \ldots, n - 1$。数据是第 12 章的：同样的词表，同样的测试故事，所以分数能上同一张排行榜。这些函数之后的单元格显示词表大小、训练词数和测试词数。
    """)
    return


@app.cell
def _(SelfAttention):
    class Block(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.attention = SelfAttention(d, heads)
            self.norm2 = nn.LayerNorm(d)
            self.think = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

        def forward(self, x):
            return block_step(x, self.norm1, self.attention, self.norm2, self.think)

    class TinyGPT(nn.Module):
        def __init__(self, vocab_size, longest, d, heads, layers):
            super().__init__()
            self.words = nn.Embedding(vocab_size, d)
            self.positions = nn.Embedding(longest, d)
            self.blocks = nn.Sequential(*[Block(d, heads) for i in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, vocab_size)

        def forward(self, ids):
            x = self.words(ids) + self.positions(torch.arange(ids.shape[1]))
            return self.out(self.norm(self.blocks(x)))
    return (TinyGPT,)


@app.function
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.function
def story_ids(story_list, index):
    ids = []
    for story in story_list:
        for word in to_words(story):
            ids.append(index.get(word, index["<unk>"]))
        ids.append(index["<end>"])
    return np.array(ids)


@app.function
def windows(ids, length):
    n = (len(ids) - 1) // length
    return ids[:n * length].reshape(n, length), ids[1:n * length + 1].reshape(n, length)


@app.function
def perplexity(log_probs):
    return np.exp(-np.mean(log_probs))


@app.function
def train_gpt(model, X, Y, epochs, batch_size, rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    size = model.out.out_features
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X[batch]).reshape(-1, size), Y[batch].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return loss.item()


@app.function
def score_gpt(model, X, Y):
    size = model.out.out_features
    log_probs = []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            S = model(X[start:start + 64])
            log_probs.append(-F.cross_entropy(S.reshape(-1, size), Y[start:start + 64].reshape(-1), reduction="none").numpy())
    return perplexity(np.concatenate(log_probs))


@app.cell
def _():
    train_stories, test_stories = stories()
    counts = {}
    for word in to_words(" ".join(train_stories)):
        counts[word] = counts.get(word, 0) + 1
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
    按下按钮，会建一个模型：每个词用 128 个数字表示，有 4 个头、3 个块，并用全部 190 万个词训练两遍。在笔记本上大约 3 到 7 分钟。完成之前，需要它的结果的单元格会显示 `Ancestor stopped`，这是正常的。如果重新运行训练单元格，要再按一次按钮。单元格显示三个数字：旋钮数、最后的训练损失、测试困惑度。
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="训练 Transformer（3 到 7 分钟）")
    train_button
    return (train_button,)


@app.cell
def _(TinyGPT, test_ids, train_button, train_ids, vocab):
    mo.stop(not train_button.value, mo.md("*按下按钮开始训练。*"))
    X_np, Y_np = windows(train_ids, 32)
    X_test_np, Y_test_np = windows(test_ids, 32)
    torch.manual_seed(0)
    model = TinyGPT(len(vocab), 32, 128, 4, 3)
    knobs = sum([p.numel() for p in model.parameters()])
    last_loss = train_gpt(model, torch.tensor(X_np), torch.tensor(Y_np), 2, 64, 0.002)
    gpt_score = score_gpt(model, torch.tensor(X_test_np), torch.tensor(Y_test_np))
    knobs, last_loss, gpt_score.item()
    return gpt_score, model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 110 万个旋钮，测试困惑度大约 23。下面是排行榜：
    """)
    return


@app.cell(hide_code=True)
def _(gpt_score):
    record("transformer", gpt_score)
    show(LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    如果你跑过第 13 章，这里就是第一名：23 对 LSTM 的 31，训练时间差不多。第 12 章数数的模型远远落在后面。

    ## 写作，并看它往哪里看

    `write` 是第 13 章的循环，只改了一处：不再带着记忆走，而是每一步都把最近的 32 个词重新交给模型（`words[-32:]`）。你会看到流畅但东拉西扯的句子：困惑度 23 也就是这个水平。
    """)
    return


@app.function
def write(model, start, count, temperature, index, vocab):
    words = [index[start]]
    with torch.no_grad():
        for step in range(count):
            S = model(torch.tensor([words[-32:]]))
            p = torch.softmax(S[0, -1] / temperature, dim=0)
            words.append(torch.multinomial(p, 1).item())
    return " ".join([vocab[i] for i in words])


@app.cell
def _(index, model, vocab):
    torch.manual_seed(1)
    write(model, "once", 50, 0.8, index, vocab)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是最后一个块里一个头的份额，和第 14 章一样：每个词一行，每一列是它看的一个前面的词。右上角是空的，因为那是未来。
    """)
    return


@app.cell
def _(index, model, vocab):
    sentence = "once upon a time there was a little girl named lily she loved her dog".split()
    with torch.no_grad():
        model(torch.tensor([[index[word] for word in sentence]]))
    looks = model.blocks[-1].attention.shares[0, 0].numpy()
    mo.vstack([maps([looks], ["词（纵向）看向词（横向）"], 320), " ".join(sentence)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    （这是 4 个头里的第一个。）每一行加起来是 1。别指望每个头都有清晰的含义：这样的图案常常很难读懂，这是正常的。

    ## 接下来

    你已经造出了现代语言模型的核心，而且它小到可以完全弄懂。它和那些著名模型的差别，主要在规模和数据：大模型有几十亿个旋钮，用的文本也多得多。

    有件事你见过，但还没有质疑过：除了一个分数，我们怎么知道模型真的擅长某个任务？**第 16 章**讲怎样把模型用到新任务上，以及评估时自己骗自己的办法有多少种。

    ## 你现在会做的事

    - 解释自注意力里的查询、键和值
    - 用掩码挡住未来，并把分数按 $\sqrt{d}$ 缩放
    - 把向量切成多个头，并说出为什么这不增加大小
    - 解释为什么只有注意力时会忽略词序，以及位置怎样解决这个问题
    - 做层归一化，并用两步 `x + f(x)` 搭出一个块
    - 训练一个小 GPT，把它放上排行榜

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** Transformer 出自 *Attention Is All You Need*（Vaswani 等，2017），GPT 出自 OpenAI 2018 年起的论文。原始设计用固定的正弦波表示位置。“先归一化再注意力”的顺序叫 *pre-norm*。真实模型还会用 *dropout*。
    """)
    return


if __name__ == "__main__":
    app.run()
