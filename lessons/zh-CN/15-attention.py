import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="15 · 注意力")

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

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 15 · 注意力

    第 14 章的 LSTM 把读到的一切都塞进一个 128 个数字的向量。预测故事的下一个词，这已经相当管用。现在换一件更难的工作：翻译。模型要读完一整句话，再用另一种语言写出来。很自然的做法是用两个 LSTM：**编码器**（encoder）把句子读进记忆，**解码器**（decoder）从这个记忆出发往外写。

    麻烦出在这唯一的记忆上：一个长句子的全部内容都要穿过一个小小的向量。想想你自己怎么翻译长句子。你不会把它背下来然后转过身去，而是需要什么就回头看什么。

    这种回头看就叫**注意力**（attention），它是 ChatGPT 里那个 T 背后的想法，T 代表 Transformer（第 16 章）。

    ## 一个能检查答案的任务

    课程里没有翻译好的文本，所以我们用一个形式相同的任务，每个答案一眼就能核对：**把文本倒着写出来**。

    > 输入：`the dog ran home` → 输出：`home ran dog the`

    这是个好测试，因为第一个要写的词是最后读到的，最后一个要写的词是最早读到的。这里不需要理解，只需要把整个序列一字不差地记住。下面的单元格准备好 500 个常用词，外加一个后面会解释的新词 `<start>`，以及一个函数，用来随机抽取指定长度的文本片段和它的倒序。`torch.flip` 沿某一维反转，`rng.integers` 可以一次抽出一批起点，`np.stack` 把一个由行组成的列表叠成一张表。（`to_words` 是第 12 章的切词函数。）
    """)
    return


@app.function
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.function
def draw_batch(ids, length, size, rng):
    starts = rng.integers(0, len(ids) - length, size)
    X = torch.tensor(np.stack([ids[s:s + length] for s in starts]))
    return X, torch.flip(X, dims=[1])


@app.cell
def _():
    train_stories = stories()[0]
    words = to_words(" ".join(train_stories[:3000]))
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    vocab = ["<start>"] + sorted(counts, key=counts.get, reverse=True)[:500]
    index = {}
    for i in range(len(vocab)):
        index[vocab[i]] = i
    ids = np.array([index[word] for word in words if word in index])
    X_demo, Y_demo = draw_batch(ids, 6, 1, np.random.default_rng(0))
    [vocab[i] for i in X_demo[0].tolist()], [vocab[i] for i in Y_demo[0].tolist()]
    return Y_demo, ids, vocab


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 解码器怎样学习

    解码器一次写一个词，写出的每个词都是下一步的输入。训练时，我们不喂给它自己的猜测，因为起初大多是错的，会带歪后面的每一步。我们在每一步都给它**正确的**上一个词。这叫**教师强制**（teacher forcing），它让所有步骤可以同时训练。所以解码器的输入就是答案向右挪一位，最前面放一个 `<start>` 词（编号 0）：

    ```
    答案 Y：      home ran  dog  the
    解码器输入：  <start> home ran dog
    ```

    ## 练习 1：shift_right

    写 `shift_right(Y)`，`Y` 是形状为 `(batch, length)` 的表。`Y[:, :-1]` 去掉最后一列。`Y[:, :1]` 是第一列，`torch.zeros_like` 给出形状和类型都相同的零。第 14 章的 `torch.cat` 把两者拼起来：

    ```
    return torch.cat([torch.zeros_like(Y[:, :1]), Y[:, :-1]], dim=1)
    ```
    """)
    return


@app.function
def shift_right(Y):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "shift_right", shift_right, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是模型，它是把第 14 章的 `Reader` 分成了两半。编码器给出它的状态 `K`（每读一个词一个向量）和最后的记忆。解码器从这个记忆出发，给出自己的状态 `Q`（每写一个词一个向量）。

    `use` 是我们传进去的函数，就像第 11 章 `Block` 里的 `step`：它决定怎么处理 `Q` 和 `K`。它返回两样东西：要打分的向量和份额（马上会讲到），`[0]` 取第一个。`no_attention` 原样返回 `Q`，没有份额，完全不管 `K`，所以从编码器传到解码器的只有最后的记忆。`size` 是记忆大小，`width` 是最后一层接收的数字个数。
    """)
    return


@app.function
def no_attention(Q, K):
    return Q, None


@app.cell
def _():
    class Translator(nn.Module):
        def __init__(self, vocab_size, size, use, width):
            super().__init__()
            self.table = nn.Embedding(vocab_size, 64)
            self.encoder = nn.LSTM(64, size, batch_first=True)
            self.decoder = nn.LSTM(64, size, batch_first=True)
            self.use = use
            self.out = nn.Linear(width, vocab_size)

        def read(self, X):
            return self.encoder(self.table(X))

        def forward(self, X, decoder_in):
            K, memory = self.read(X)
            Q, _memory = self.decoder(self.table(decoder_in), memory)
            return self.out(self.use(Q, K)[0])
    return (Translator,)


@app.function
def train_translator(model, ids, steps, rng):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for step in range(steps):
        length = rng.integers(3, 21).item()
        X, Y = draw_batch(ids, length, 64, rng)
        S = model(X, shift_right(Y))
        optimizer.zero_grad()
        F.cross_entropy(S.reshape(-1, S.shape[2]), Y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@app.function
def translate(model, X):
    K, memory = model.read(X)
    word = torch.zeros(len(X), 1, dtype=torch.long)
    written = []
    shares = []
    for t in range(X.shape[1]):
        Q, memory = model.decoder(model.table(word), memory)
        mixed, A = model.use(Q, K)
        word = torch.argmax(model.out(mixed), dim=2)
        written.append(word)
        shares.append(A)
    return torch.cat(written, dim=1), shares


@app.function
def share_correct(model, ids, length, rng):
    X, Y = draw_batch(ids, length, 500, rng)
    with torch.no_grad():
        return torch.mean((translate(model, X)[0] == Y).float()).item()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `train_translator` 用 3 到 20 个词的片段训练。`translate` 则是真正地写，没有老师：每个词都来自解码器自己的上一个猜测，取最可能的词（`torch.argmax`）。`share_correct` 给某个长度的 500 个样本打分：位置上的词写对的比例。

    按下按钮会训练 1,500 批，大约半分钟，然后显示（长度，正确比例）这样的数对，长度依次是 5、10、15、20：0.988 表示 98.8%。
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="不用注意力训练")
    train_button
    return (train_button,)


@app.cell
def _(Translator, ids, train_button, vocab):
    mo.stop(not train_button.value, mo.md("*按下按钮开始训练。*"))
    torch.manual_seed(0)
    rng_plain = np.random.default_rng(0)
    plain_model = Translator(len(vocab), 128, no_attention, 128)
    train_translator(plain_model, ids, 1500, rng_plain)
    plain_scores = []
    for plain_length in [5, 10, 15, 20]:
        plain_scores.append((plain_length, round(share_correct(plain_model, ids, plain_length, rng_plain), 3)))
    plain_scores
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    你应该看到：长度 5 时大约 99%，10 时大约 78%，15 时大约 42%，20 时大约 23%。句子越长越糟，因为 128 个数字装不下。这就是**瓶颈**（bottleneck）。

    ## 加权平均

    解决办法从一个小想法开始。假设有三个向量，我们想要它们的混合：第一个占 70%，第二个占 20%，第三个占 10%。这就是**加权平均**（weighted average）：每个向量乘以它的份额，再加起来。如果这些向量是矩阵 `V` 的行，份额是向量 `a`，那就是 `a @ V`，第 05 章的向量乘矩阵。

    份额是 `[1, 0, 0]` 时，得到的正好是第一个向量。所以加权平均是一种柔和的查找：可以取出一行，也可以混合几行。

    ## 练习 2：weighted_average

    写 `weighted_average(a, V)`，份额 `a` 的形状是 `(n,)`，行 `V` 的形状是 `(n, d)`：

    ```
    return a @ V
    ```
    """)
    return


@app.function
def weighted_average(a, V):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "weighted_average", weighted_average, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 往哪里看：查询、键、值

    现在解码器可以回头看了。写的每一步，它都有一个状态，而编码器为每个读到的词留下一个状态。注意力分三步，每一步你都学过：

    1. **比较。** 把解码器的状态和每个编码器状态做点积。点积大表示“这个有关”（第 12 章：点积衡量向量有多像）。
    2. **分配。** softmax 把这些分数变成加起来为 1 的份额（第 06 章）。
    3. **混合。** 用这些份额对编码器状态做加权平均。

    结果是一份全新的输入摘要，只为写当前这个词而做。这些名称来自查找东西的过程。**查询**（query）`q` 是你要找的，**键**（key）`K` 是拿来比较的标签，**值**（value）`V` 是你取回的内容。这里键和值都是编码器状态，但它们不必相同，下一章就不一样了。

    解码器状态和编码器状态的点积凭什么会大？一开始并不大。是训练让有用的那些变大，下面就会看到。

    ## 练习 3：attention_shares

    写 `attention_shares(q, K)`，一个查询 `q` 的形状是 `(d,)`，键 `K` 的形状是 `(n, d)`。`K @ q` 给出每个键一个点积：

    ```
    return torch.softmax(K @ q, dim=0)
    ```
    """)
    return


@app.function
def attention_shares(q, K):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "attention_shares", attention_shares, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：attend

    现在全部一起来：一批文本，每段有很多查询。`Q` 的形状是 `(batch, writes, d)`，`K` 的形状是 `(batch, reads, d)`，`V` 的形状是 `(batch, reads, e)`：值可以有自己的大小。对三维张量，`@` 按文本逐段作用在最后两维上。`K.transpose(1, 2)` 交换第 1、2 维，相当于对每段文本做 `.T`。

    写 `attend(Q, K, V)`，返回混合后的向量和份额：

    1. `scores = Q @ K.transpose(1, 2)`，形状是 `(batch, writes, reads)`：每个查询对每个键
    2. `A = torch.softmax(scores, dim=2)`，这样每个查询的份额加起来是 1
    3. `return A @ V, A`，加权平均的形状是 `(batch, writes, e)`，再加上份额
    """)
    return


@app.function
def attend(Q, K, V):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "attend", attend, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    接入模型时，`with_attention` 把每个解码器状态和它的混合摘要并排拼起来，所以最后一层的宽度是记忆大小的两倍。其余都不变：同样的模型，同样的数据，同样的 1,500 批，同样的记忆大小。在你按下按钮之前，后面的单元格会显示 `Ancestor stopped`，这是正常的。
    """)
    return


@app.function
def with_attention(Q, K):
    blended, A = attend(Q, K, K)
    return torch.cat([Q, blended], dim=2), A


@app.cell(hide_code=True)
def _():
    train_button_2 = mo.ui.run_button(label="用注意力训练")
    train_button_2
    return (train_button_2,)


@app.cell
def _(Translator, ids, train_button_2, vocab):
    mo.stop(not train_button_2.value, mo.md("*按下按钮开始训练。*"))
    torch.manual_seed(0)
    rng_attention = np.random.default_rng(0)
    attention_model = Translator(len(vocab), 128, with_attention, 256)
    train_translator(attention_model, ids, 1500, rng_attention)
    attention_scores = []
    for attention_length in [5, 10, 15, 20]:
        attention_scores.append((attention_length, round(share_correct(attention_model, ids, attention_length, rng_attention), 3)))
    attention_scores
    return attention_model, rng_attention


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    你应该看到：长度 15 以内接近 100%，长度 20 时大约 95%，而之前只有 23%。记忆并没有变大，只是不用再背着所有东西了：解码器需要什么，就去取什么。

    ## 看它往哪里看

    份额不仅有用，还能读懂。下面是一段 12 个词的文本的份额：每写一个词一行，每读一个词一列。蓝色越深，份额越大。图下面是读到的文本和写出的文本。写第一个词时，好模型应该看最后读到的词；写最后一个词时，看最早读到的词。
    """)
    return


@app.cell
def _(attention_model, ids, rng_attention, vocab):
    X_look, Y_look = draw_batch(ids, 12, 1, rng_attention)
    with torch.no_grad():
        written_look, shares_look = translate(attention_model, X_look)
    A_look = torch.cat(shares_look, dim=1)[0].numpy()
    mo.vstack([
        maps([A_look], ["写出（纵向）与读到（横向）"], 280),
        " ".join([vocab[i] for i in X_look[0].tolist()]),
        " ".join([vocab[i] for i in written_look[0].tolist()]),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    它看起来应该是一条从右上到左下的蓝线。没有人告诉模型该往哪里看：每个写错的词带来的责任，让解码器状态和编码器状态学会了互相对齐。

    注意力对责任也有帮助。现在从每个写出的词到每个读到的词，都有一条捷径，这和第 11 章的跳跃连接、第 14 章的细胞是同一个技巧。

    ## 接下来

    有个问题。LSTM 还在那里，逐词地读，而且很慢，因为每一步都要等前一步。既然真正起作用的是注意力，我们还需要 LSTM 吗？如果每个词直接同时关注其他词呢？

    这就是 **Transformer**。在**第 16 章**，你会只用注意力造出一个 Transformer，把它放上排行榜，看着它拿下第一名。

    ## 你现在会做的事

    - 解释编码器-解码器的设计，以及一个固定记忆为什么是瓶颈
    - 用教师强制训练解码器
    - 用 `@` 计算加权平均
    - 解释查询、键和值
    - 为一整批数据写出注意力的三步：比较、分配、混合
    - 读懂注意力图

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 用于翻译的注意力出自 Bahdanau、Cho 和 Bengio（2014）；这种在解码器步骤之后加入的点积形式，接近 Luong 等人（2015）的做法。真实的翻译系统需要*束搜索*（beam search），而不是总选最可能的词，并用 *BLEU* 这样的分数衡量质量。
    """)
    return


if __name__ == "__main__":
    app.run()
