import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="17 · Transformer")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.diagrams import block_road
    from learn_llm.images import maps
    from learn_llm.leaderboard import record, show

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 17 · Transformer

    在第 16 章，你造出了自注意力层：每个词同时用好几个头向它前面的词提问。那一章留下一个缺口：对注意力来说，`the dog bit the man`（狗咬了人）和 `the man bit the dog`（人咬了狗）是同一个词袋。

    这一章先补上这个缺口，再把模型搭起来。学完之后，你会有一个小小的 **GPT**，大约一百万个旋钮，它会登上你排行榜的第一名。还差三个小部件：位置、稳定的尺度，以及块。其余的你都见过：`x + f(x)`（第 11 章）、层和 Adam（第 10 章），还有第 14 章的训练循环。

    ## 来自第 16 章

    下面是你在第 16 章写的五个函数（已经写好），以及用它们搭成的层。
    """)
    return


@app.function
def future_mask(n):
    return torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)


@app.function
def attend(Q, K, V, mask):
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    scores = scores.masked_fill(mask, -torch.inf)
    A = torch.softmax(scores, dim=-1)
    return A @ V, A


@app.function
def split_heads(x, heads):
    batch, words, d = x.shape
    return x.reshape(batch, words, heads, d // heads).transpose(1, 2)


@app.function
def join_heads(x):
    batch, heads, words, piece = x.shape
    return x.transpose(1, 2).reshape(batch, words, heads * piece)


@app.function
def self_attention_step(x, to_query, to_key, to_value, mix, heads):
    Q = split_heads(to_query(x), heads)
    K = split_heads(to_key(x), heads)
    V = split_heads(to_value(x), heads)
    blended, A = attend(Q, K, V, future_mask(x.shape[1]))
    return mix(join_heads(blended)), A


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
            out, self.shares = self_attention_step(x, self.query, self.key, self.value, self.mix, self.heads)
            return out
    return (SelfAttention,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 部件 1：位置

    要让注意力知道顺序，就得告诉每个词它站在哪里。办法再简单不过：再加一个 `nn.Embedding`，每个**位置**（position）一个学出来的向量。位置 0 有一个向量，位置 1 有另一个，依此类推。把它加到词的向量上，同一个词站在不同的地方，向量就不再相同。

    这也带来一个限制。位置表只有 32 项，对位置 33 就无话可说，所以我们的模型只能看到最近的 32 个词。（第 20 章有更聪明的办法。）

    ## 练习 1：embed

    /// admonition | 🧰 工具箱：一排位置
    `torch.arange(4)` 是由整数 0、1、2、3 组成的张量。把一个编号张量交给 `nn.Embedding`，每个编号得到一个向量（第 12 章），所以 `table(torch.arange(4))` 给出位置 0 到 3 的向量，形状是 `(4, d)`。

    形状 `(batch, 4, d)` 和 `(4, d)` 相加时会广播（第 05 章）：同样的四个位置向量加到这一批的每一段文本上。
    ///

    写 `embed(ids, words, positions)`。`ids` 的形状是 `(batch, n)`，`words` 和 `positions` 是两张 `nn.Embedding` 表。为每个词返回它的词向量加上它所在位置的向量，形状是 `(batch, n, d)`。词数 `n` 是 `ids.shape[1]`。
    """)
    return


@app.function
def embed(ids, words, positions):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("17", "embed", embed, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""查两次表，再相加：词表拿到的是编号，位置表拿到的是位置 0 到 `n - 1`。""",
        r"""只要一行。位置和站在那里的是哪个词无关：它们就是 `torch.arange(ids.shape[1])`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    管用吗？下面是两段各有三个词编号的文本（编号是随手编的）：`4 7 1` 和 `7 4 1`，同样的词，前两个换了位置。最后一个词在两段里相同，而且在两段里都能看到全部三个词。单元格用一个新建的注意力层分别处理两段文本，比较最后一个词得到的结果：先只用词向量，再用你的 `embed`。它显示两种情况下的最大差别。（它用到你的 `embed`，所以练习 1 通过之前会显示错误。这是正常的，后面用到你的函数的单元格也一样。）
    """)
    return


@app.cell
def _(SelfAttention):
    torch.manual_seed(0)
    words_demo = nn.Embedding(10, 8)
    positions_demo = nn.Embedding(3, 8)
    layer_demo = SelfAttention(8, 2)
    one_order = torch.tensor([[4, 7, 1]])
    other_order = torch.tensor([[7, 4, 1]])
    with torch.no_grad():
        bare = layer_demo(words_demo(one_order))[0, -1] - layer_demo(words_demo(other_order))[0, -1]
        placed = layer_demo(embed(one_order, words_demo, positions_demo))[0, -1] - layer_demo(embed(other_order, words_demo, positions_demo))[0, -1]
    torch.max(torch.abs(bare)).item(), torch.max(torch.abs(placed)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第一个数是 0，或者是 `3e-08` 这样的舍入误差：只用词向量时，不管别的词按什么顺序来，最后一个词得到的混合结果完全一样。第二个数明显不是 0，大约 0.25。加上位置之后，这个层能把两段文本分开了。

    ## 部件 2：稳定的尺度

    堆很多层时，数值可能衰减或爆炸，这就是第 09 章深度表格里的麻烦。He 初始化解决的是一叠普通的层在训练开始时的问题。这里我们用一个自始至终都起作用的工具：**层归一化**（layer normalisation）。它把每个词的向量单独重新缩放到均值 0、典型大小 1：先减去向量的均值，再除以它的典型大小，也就是第 09 章方差的平方根。

    拿向量 `[10, 10, 30, 30]` 来说。它的均值是 20。减去均值得到 `[-10, -10, 10, 10]`，方差是这些数平方的均值，100，典型大小是 $\sqrt{100} = 10$。除一下得到 `[-1, -1, 1, 1]`。

    再试试 `[1000, 1000, 3000, 3000]`：结果还是 `[-1, -1, 1, 1]`。不管进来的数有多大或多小，传给下一层的都在合适的尺度上。

    ## 动手算：归一化一个向量

    轮到你了，向量是 `[3, 9, 11, 17]`。算出它的均值和方差，再算出归一化之后的第一个数和最后一个数。
    """)
    return


@app.cell(hide_code=True)
def _():
    norm_boxes = hand.boxes(['均值', '方差', '第一个数', '最后一个数'], step=0.1)
    return (norm_boxes,)


@app.cell(hide_code=True)
def _(norm_boxes):
    hand.marked(norm_boxes, {'均值': 10, '方差': 25, '第一个数': -1.4, '最后一个数': 1.4}, {
        '均值': r"""四个数加起来，除以 4""",
        '方差': r"""每个数减去均值，平方，再求均值：$(49 + 1 + 1 + 49) / 4$""",
        '第一个数': r"""$(3 - \text{均值}) / \sqrt{\text{方差}}$，是个负数""",
        '最后一个数': r"""$(17 - \text{均值}) / \sqrt{\text{方差}}$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：layer_norm

    /// admonition | 🧰 工具箱：沿最后一维求均值和方差
    对任意形状的张量 `T`，`torch.mean(T, dim=-1, keepdim=True)` 是沿最后一维的均值。`keepdim=True` 就是 PyTorch 版的 `keepdims=True`（第 07 章）：结果保留大小为 1 的最后一维，所以能逐行从 `T` 里减掉。

    `torch.var(T, dim=-1, keepdim=True, unbiased=False)` 用同样的方式求方差。`unbiased=False` 要的是第 09 章那种普通方差。（不写的话，PyTorch 会除以“个数减 1”，那是统计学里的一个修正，我们用不着。）
    ///

    写 `layer_norm(x)`，沿最后一维归一化，`x` 可以是任意形状：

    > $$\frac{x - \text{均值}}{\sqrt{\text{方差} + 0.00001}}$$

    那个很小的数是为了防止向量里所有数都相等时除以 0。在 Python 里可以写成 `1e-5`。`torch.sqrt` 对张量开平方。

    有一项检查会把你的函数和 PyTorch 自带的 `nn.LayerNorm` 比较。真实的层还给每个数字多加两个学习旋钮：一个缩放，一个平移，这样训练可以在有用的地方把归一化抵消掉。
    """)
    return


@app.function
def layer_norm(x):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("17", "layer_norm", layer_norm, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三行：均值、方差，然后是公式。均值和方差都要 `dim=-1` 和 `keepdim=True`。""",
        r"""`1e-5` 放在平方根里面，加在方差上。如果和 `nn.LayerNorm` 的比较只差一点点，看看有没有写 `unbiased=False`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 部件 3：块

    注意力让词互相交流。然后每个词需要时间想想听到了什么。所以一个**块**（block）有两部分：一个自注意力层，后面跟着第 08 章的小网络。小网络分别作用在每个词上，隐藏层的宽度是四倍。

    两部分都走第 11 章那条责任的通道 `x + f(x)`，归一化放在每一部分之前：

    > $$x \leftarrow x + \text{attention}(\text{norm}_1(x))$$

    > $$x \leftarrow x + \text{think}(\text{norm}_2(x))$$

    箭头的意思是“`x` 变成”。下面把它画成图。蓝色的通道让 `x` 原封不动地绕过每一部分，每一部分只是把自己的贡献加上去。
    """)
    return


@app.cell(hide_code=True)
def _():
    block_road('x', '归一化', '注意力', '思考',
               'x 经过归一化和注意力，再加回自己；然后经过归一化和思考，再加回一次')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：走一遍块

    用几个好算的假层，跟着一个数走一遍。`x` 一开始是 3。两个归一化都是减 1，`attention` 是乘 2，`think` 是乘 3。第一行之后 `x` 是多少？第二行之后呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    road_boxes = hand.boxes(['注意力之后', '思考之后'])
    return (road_boxes,)


@app.cell(hide_code=True)
def _(road_boxes):
    hand.marked(road_boxes, {'注意力之后': 7, '思考之后': 25}, {
        '注意力之后': r"""归一化得到 $3 - 1 = 2$，注意力把它翻倍成 4，再加到一开始的 3 上""",
        '思考之后': r"""从你的第一个答案出发：归一化减去 1，思考乘以 3，结果再加到第一个答案上""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：block_step

    写 `block_step(x, norm1, attention, norm2, think)`，完成上面那两行。四个层都像函数一样调用。
    """)
    return


@app.function
def block_step(x, norm1, attention, norm2, think):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("17", "block_step", block_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""每一行都是往 `x` 上加，而不是把它换掉。""",
        r"""归一化发生在里面，在进入每一部分的路上：`attention` 拿到的是 `norm1(x)`，但被加上去的是原来的 `x`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 完整的模型

    现在把部件装到一起。跟着形状走一遍：一批文本，每段 `n` 个词，每个词 `d` 个数字。

    | 步骤 | 形状 |
    |---|---|
    | 词的编号 | `(batch, n)` |
    | `embed`：词向量加位置向量 | `(batch, n, d)` |
    | 一个接一个的块 | `(batch, n, d)` |
    | 最后再归一化一次 | `(batch, n, d)` |
    | `out`：一个 `nn.Linear`，给词表里的每个词一个分数 | `(batch, n, vocab)` |

    每个位置上的分数，是模型对**下一个**词的预测。有了掩码，位置 5 只见过第 0 到 5 个词，所以每段文本的每个位置同时都是一个训练样本。

    ## 练习 4：gpt_forward

    照着这张表写 `gpt_forward(ids, words, positions, blocks, norm, out)`。`blocks` 是一个已经装好整叠块的层（`nn.Sequential`，第 10 章），所以只调用一次。用上你的 `embed`。
    """)
    return


@app.function
def gpt_forward(ids, words, positions, blocks, norm, out):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("17", "gpt_forward", gpt_forward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""四次调用，每次都作用在上一次的结果上：`embed`、`blocks`、`norm`、`out`。""",
        r"""顺序很重要：块之后才归一化，最后才打分。`embed` 需要编号和两张表。""",
    ], LOCALE)
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
            return gpt_forward(ids, self.words, self.positions, self.blocks, self.norm, self.out)
    return (TinyGPT,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是写成 PyTorch 模块的块和模型。它们各自保存自己的层，`forward` 分别就是你写的函数。`[Block(d, heads) for i in range(layers)]` 造出这些块，前面的 `*` 把它们一个个交给 `nn.Sequential`（第 09 章）。

    ## 动手算：旋钮都在哪里

    训练大模型之前，先数一个小的：`TinyGPT(10, 4, 8, 2, 1)` 的词表有 10 个词，4 个位置，每个词 8 个数字，2 个头，1 个块。一个有 `r` 行、每行 `d` 个数字的 `nn.Embedding` 有 $r \times d$ 个旋钮。`nn.Linear(a, b)` 有 $a \times b$ 个权重和 `b` 个偏置。读上面两个类，数出其中四个部分。
    """)
    return


@app.cell(hide_code=True)
def _():
    count_boxes = hand.boxes(['词表', '位置表', '思考', 'out'])
    return (count_boxes,)


@app.cell(hide_code=True)
def _(count_boxes):
    hand.marked(count_boxes, {'词表': 80, '位置表': 32, '思考': 552, 'out': 90}, {
        '词表': r"""10 个词，每个 8 个数字""",
        '位置表': r"""4 个位置，每个 8 个数字""",
        '思考': r"""两层：`nn.Linear(8, 32)` 有 $8 \times 32 + 32$ 个旋钮，`nn.Linear(32, 8)` 有 $32 \times 8 + 8$ 个""",
        'out': r"""`nn.Linear(8, 10)`：$8 \times 10$ 个权重和 10 个偏置""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是 PyTorch 给整个小模型数出的旋钮数：
    """)
    return


@app.cell
def _(TinyGPT):
    small = TinyGPT(10, 4, 8, 2, 1)
    sum([p.numel() for p in small.parameters()])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    1,066 个。你数的四部分占了其中 754 个。其余的是注意力层（三个 $8 \times 8$ 的矩阵，加上 72 个旋钮的 `mix`）和三个归一化，每个 16 个旋钮。

    留意那两张“词表里每个词一行”的表：词表和 `out`。在小模型里，它们占 1,066 个旋钮里的 170 个。词表变成真实大小之后，它们会成为模型里最大的部分，马上就能看到。

    ## 数据和训练循环

    数据是第 13 章的：同样的词表，同样的测试故事，所以分数能上同一张排行榜。下面的函数来自第 12 到 14 章：把文本切成词、把故事变成编号、把编号切成窗口、困惑度、带温度地抽一个词、带裁剪的训练循环，以及打分。这些函数之后的单元格显示词表大小、训练词数和测试词数。
    """)
    return


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
def sample_word(scores, temperature):
    p = torch.softmax(scores / temperature, dim=0)
    return torch.multinomial(p, 1).item()


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
    ## 动手算：送进损失的是什么

    读一读 `train_gpt`。我们会用 32 个词的窗口训练，每批 64 个窗口，词表有 2,002 个词。`F.cross_entropy` 要的是一张表：每个预测一行，词表里每个词一列，所以 `reshape(-1, size)` 把批和位置压成一维。对一批数据来说，这张表有多少行、多少列？
    """)
    return


@app.cell(hide_code=True)
def _():
    loss_boxes = hand.boxes(['行数', '列数'])
    return (loss_boxes,)


@app.cell(hide_code=True)
def _(loss_boxes):
    hand.marked(loss_boxes, {'行数': 2048, '列数': 2002}, {
        '行数': r"""每个窗口的每个位置都是一个预测：64 个窗口，每个 32 个词""",
        '列数': r"""词表里的每个词一个分数""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    训练一步就同时从 2,048 个预测里学习。LSTM 给出的预测一样多，但它得把 32 个词一个接一个地走完。这里所有位置是一起算的。

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
    大约 110 万个旋钮，测试困惑度大约 23。这些旋钮里，词表和 `out` 各占大约 25.7 万个，合起来将近模型的一半，三个块一共大约 59.4 万个。下面是排行榜：
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
    如果你跑过第 14 章，这里就是第一名：23 对 LSTM 的 31，训练时间差不多。第 13 章数数的模型远远落在后面。

    ## 练习 5：write_ids

    第 14 章的 `write` 带着记忆一个词一个词往前走。Transformer 没有记忆。每一步它都把到目前为止的文本重新读一遍，再预测下一个词。但它的位置表行数有限，所以只能把最近的词交给它。

    /// admonition | 🧰 工具箱：列表的末尾
    `items[-3:]` 是列表的最后 3 项。如果列表不到 3 项，就是整个列表。
    ///

    写 `write_ids(model, first, count, temperature, longest)`。它从词编号 `first` 开始，一次添一个，再添 `count` 个编号，最后返回整个编号列表。每添一个新编号：

    1. 取最近的 `longest` 个编号，做成形状为 `(1, n)` 的张量：只装一段文本的一批。`torch.tensor([some_list])` 就能做到。
    2. 运行模型。分数的形状是 `(1, n, vocab)`。
    3. 对下一个词的预测在这唯一一段文本的最后一个位置：`S[0, -1]`。用第 14 章的 `sample_word(scores, temperature)` 从这些分数里抽一个编号，添到列表末尾。

    这里不训练任何东西，所以把循环放在 `torch.no_grad()` 里（第 10 章）。
    """)
    return


@app.function
def write_ids(model, first, count, temperature, longest):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("17", "write_ids", write_ids, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""框架和第 14 章一样：一个从单个编号开始的列表，一个跑 `count` 次的 `for` 循环，每轮最后 `append`。""",
        r"""做张量之前先用 `-longest:` 切列表。交给模型的编号绝不能超过 `longest` 个，否则位置表就不够用了。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在让它从 `once` 开始写 50 个词，温度 0.8。你会看到流畅但东拉西扯的句子：困惑度 23 也就是这个水平。
    """)
    return


@app.cell
def _(index, model, vocab):
    torch.manual_seed(1)
    " ".join([vocab[i] for i in write_ids(model, index["once"], 50, 0.8, 32)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是最后一个块里一个头的份额，和第 15 章一样：每个词一行，每一列是它看的一个前面的词。右上角是空的，因为那是未来。
    """)
    return


@app.cell
def _(index, model):
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

    有件事你见过，但还没有质疑过：除了一个分数，我们怎么知道模型真的擅长某个任务？**第 18 章**讲怎样把模型用到新任务上，以及评估时自己骗自己的办法有多少种。

    ## 你现在会做的事

    - 用位置表让注意力知道顺序
    - 动手和用代码做层归一化，并说出它的用处
    - 用两步 `x + f(x)` 搭出一个块
    - 跟着形状走完一个 GPT，从编号到分数
    - 数出一个模型的旋钮，并说出大部分在哪里
    - 训练一个小 GPT，把它放上排行榜，再让它写作

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** Transformer 出自 *Attention Is All You Need*（Vaswani 等，2017），GPT 出自 OpenAI 2018 年起的论文。原始设计用固定的正弦波表示位置。“先归一化再注意力”的顺序叫 *pre-norm*。真实模型还会用 *dropout*，而且很多模型让词向量和 `out` 共用一张表，这叫 *weight tying*。
    """)
    return


if __name__ == "__main__":
    app.run()
