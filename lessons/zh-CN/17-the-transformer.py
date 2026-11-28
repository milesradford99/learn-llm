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

    在第 16 章，你造出了自注意力层：每个词同时用好几个头向它前面的词提问。那一章留下一个缺口：对注意力来说，`the dog bit the man`（狗咬了人）和 `the man bit the dog`（人咬了狗）只是装了相同词的一袋词，分不清先后顺序。

    这一章先补上这个缺口，再把完整的模型搭起来。学完之后，你会亲手建成一个小型 **GPT**，拥有大约一百万个旋钮，并登上排行榜第一名。目前还差最后三个部件：位置、稳定的数值尺度，以及把它们装在一起的块。其余部分你都已经掌握了：残差连接 `x + f(x)`（第 11 章）、PyTorch 层和 Adam 优化器（第 10 章），以及第 14 章的训练循环。

    ## 来自第 16 章

    下面已经放好了你在第 16 章完成的五个函数，以及用它们组装成的 `SelfAttention` 层。
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

    要让注意力分清词序，就得告诉每个词它处在句子的第几个位置。最直接的办法是再加一张 `nn.Embedding` 表，给每个**位置**（position）配一个可学习的向量：位置 0 有自己的向量，位置 1 有另一个，依此类推。把位置向量加到词向量上，同一个词出现在不同位置时，得到的向量就不再相同。

    这也带来一个限制。如果位置表只有 32 行，从 0 数起就是位置 0 到 31，到了第 33 个词（也就是下标 32）就会越界，所以模型一次最多只能看到最近的 32 个词。（第 20 章会介绍更聪明的办法。）

    ## 练习 1：embed

    /// admonition | 🧰 工具箱：一排位置
    `torch.arange(4)` 会生成由整数 0、1、2、3 组成的张量。把编号张量传给 `nn.Embedding`，每个编号都会查出一个向量（第 12 章），所以 `table(torch.arange(4))` 会取出位置 0 到 3 的向量，形状是 `(4, d)`。

    形状 `(batch, 4, d)` 和 `(4, d)` 相加时会自动广播（第 05 章）：这四个位置向量会同时加到这一批里的每一段文本上。
    ///

    编写 `embed(ids, words, positions)`。其中 `ids` 的形状是 `(batch, n)`，`words` 和 `positions` 是两张 `nn.Embedding` 表。返回每个词的词向量与所在位置向量之和，形状为 `(batch, n, d)`。每段文本的词数 `n` 可以用 `ids.shape[1]` 拿到。
    """)
    return


@app.function
def embed(ids, words, positions):
    # ← 把下面的 todo() 换成你的代码
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
    管用吗？下面有两段各含 3 个词编号的示例文本：`4 7 1` 和 `7 4 1`——词完全一样，只是前两个换了顺序。两段文本的最后一个词都是 `1`，而且它都能看到全部三个词。下面的单元格用一个新建的注意力层分别处理两段文本，比较最后一个词的输出：先只用词向量，再换用你的 `embed`，并打印两种情况下的最大差值。（这里用到了你的 `embed`，所以在练习 1 通过之前会报错，这是正常的；后面用到你写的函数的单元格也一样。）
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
    第一个数是 0（或 `3e-08` 这样的浮点舍入误差）：只用词向量时，不管前面的词怎么调换顺序，最后一个词算出的混合结果完全一样。第二个数明显不为 0，大约是 0.25：加上位置向量后，注意力层就能区分这两段文本了。

    ## 部件 2：稳定的尺度

    网络堆得很深时，数值在层层传递中容易越来越小（衰减）或越来越大（爆炸），这就是第 09 章见过的难题。He 初始化只能稳住训练刚开始时的数值尺度，这里我们用一个在整个训练过程中持续起作用的工具：**层归一化**（layer normalisation）。它把每个词的向量单独重新缩放到均值 0、典型大小 1：先减去这个向量的均值，再除以它的典型大小（即方差的平方根，也就是标准差）。

    拿向量 `[10, 10, 30, 30]` 来说：它的均值是 20；减去均值得到 `[-10, -10, 10, 10]`；方差是这些数平方的均值，等于 100；典型大小是 $\sqrt{100} = 10$；最后除以 10，得到 `[-1, -1, 1, 1]`。

    再试试 `[1000, 1000, 3000, 3000]`：结果依然是 `[-1, -1, 1, 1]`。无论传进来的数值原本有多大或多小，归一化后传给下一层的都处在稳定的尺度上。

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
    对任意形状的张量 `T`，`torch.mean(T, dim=-1, keepdim=True)` 计算沿最后一维的均值。`keepdim=True` 就是 PyTorch 版的 `keepdims=True`（第 07 章）：结果保留大小为 1 的最后一维，方便通过广播逐行从 `T` 里减去。

    `torch.var(T, dim=-1, keepdim=True, unbiased=False)` 用同样的方式求方差。`unbiased=False` 表示按第 09 章的普通公式直接除以元素个数（不写的话，PyTorch 默认除以“个数减 1”，那是统计学里的样本方差修正，这里不需要）。
    ///

    编写 `layer_norm(x)`，对任意形状的 `x` 沿最后一维做归一化：

    > $$\frac{x - \text{均值}}{\sqrt{\text{方差} + 0.00001}}$$

    加上这个很小的数是为了防止向量里所有数都相等（方差为 0）时出现除以 0。在 Python 里它可以写成 `1e-5`，对张量开平方用 `torch.sqrt`。

    有一项检查会把你的函数和 PyTorch 自带的 `nn.LayerNorm` 对比。真实的 `nn.LayerNorm` 还会给向量的每一维额外配两个可学习的旋钮——一个缩放、一个平移——让模型在需要时能够微调甚至抵消归一化。
    """)
    return


@app.function
def layer_norm(x):
    # ← 把下面的 todo() 换成你的代码
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

    自注意力层让词与词之间互相交流信息；紧接着，每个词还需要经过一个独立的小网络，把刚收集到的信息进一步消化和变换。所以一个 Transformer **块**（block）由两部分组成：先是一个自注意力层，后面跟着第 08 章那种两层小网络（结构就是 `Linear -> ReLU -> Linear`），分别作用在每个词上，隐藏层宽度是输入维度的 4 倍。

    这两部分都接在第 11 章的残差主干道 `x + f(x)` 上，并在进入每一部分之前先做层归一化：

    > $$x \leftarrow x + \text{attention}(\text{norm}_1(x))$$

    > $$x \leftarrow x + \text{think}(\text{norm}_2(x))$$

    箭头 $\leftarrow$ 表示“把 `x` 更新为”。**特别注意**：加回的是归一化之前的原始 `x`，不要先把 `x` 覆盖成 `norm1(x)`——`norm1(x)` 只送进注意力层，主干道上的 `x` 本身保持原样。下图展示了这个流向：蓝色的主干道把 `x` 原封不动地绕过每个子层，子层只负责把自己的增量加上去。
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

    编写 `block_step(x, norm1, attention, norm2, think)`，完成上面那两步更新。四个层都像普通函数一样直接调用。
    """)
    return


@app.function
def block_step(x, norm1, attention, norm2, think):
    # ← 把下面的 todo() 换成你的代码
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

    现在把各个部件组装起来。顺着张量形状走一遍——假设输入是一批文本，每段 `n` 个词，每个词用 `d` 个数字表示：

    | 步骤 | 形状 |
    |---|---|
    | 词的编号 | `(batch, n)` |
    | `embed`：词向量加位置向量 | `(batch, n, d)` |
    | 一个接一个的块 | `(batch, n, d)` |
    | 最后再归一化一次 | `(batch, n, d)` |
    | `out`：一个 `nn.Linear`，给词表里的每个词一个分数 | `(batch, n, vocab)` |

    每个位置输出的分数，都是模型对**下一个**词的预测。有了掩码后，位置 5 只见过位置 0 到 5 的词，因此每段文本的每一个位置同时都是一个训练样本。

    ## 练习 4：gpt_forward

    照着上表编写 `gpt_forward(ids, words, positions, blocks, norm, out)`。`blocks` 已经用 `nn.Sequential`（第 10 章）把整叠块打包成了一个层，直接调用一次即可。别忘了复用你刚写的 `embed`。
    """)
    return


@app.function
def gpt_forward(ids, words, positions, blocks, norm, out):
    # ← 把下面的 todo() 换成你的代码
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
    上面是写成 PyTorch 模块的块和模型。它们各自保存所需的子层，`forward` 分别调用你刚写的函数。`[Block(d, heads) for i in range(layers)]` 会造出一个装着所有块的列表，前面的 `*` 把列表解包成一个个独立参数交给 `nn.Sequential`——就像 `f(*[a, b])` 等同于 `f(a, b)`。

    ## 动手算：旋钮都在哪里

    训练大模型之前，先拿一个小模型对对账：`TinyGPT(10, 4, 8, 2, 1)` 的词表有 10 个词，最多 4 个位置，每个词 8 个数字，2 个头，1 个块。一个有 `r` 行、每行 `d` 个数字的 `nn.Embedding` 有 $r \times d$ 个旋钮；一个 `nn.Linear(a, b)` 有 $a \times b$ 个权重和 `b` 个偏置。对照上面两个类，算出其中四个部分的旋钮数。
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
    一共 1,066 个。你算出的四部分占了 754 个。剩下的是注意力层（`query`、`key`、`value` 三个无偏置的 $8 \times 8$ 矩阵共 192 个，加上含偏置共 72 个旋钮的 `mix`）和三个层归一化，每个 16 个旋钮（8 维各配 1 个缩放 + 1 个平移，共 16 个）。

    留意那两张“词表里每个词占一行”的表：词表 `words` 和输出层 `out`。在这个小模型里，它们只占 1,066 个旋钮中的 170 个；但换成真实大小的词表后，它们会成为模型里体积最大的部分，马上就能看到。

    ## 数据和训练循环

    数据沿用第 13 章的：相同的词表、相同的测试故事，这样分数才能放进同一张排行榜对比。下面的函数来自第 12 到 14 章，依次负责：把文本切成词、把故事转成编号、把编号切成训练窗口、计算困惑度、带温度采样一个词、带梯度裁剪的训练循环，以及评估打分。紧接着的单元格会显示词表大小、训练词数和测试词数。
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

    读一读 `train_gpt`。我们用 32 个词的窗口来训练，每批 64 个窗口，词表有 2,002 个词。`F.cross_entropy` 期望的是一张二维表：每个预测占一行，词表里每个词占一列，所以 `reshape(-1, size)` 把批大小和位置这两维合并成了一维。对一批数据来说，这张表有多少行、多少列？
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
    训练一步就能同时从 2,048 个预测里学习。LSTM 给出的预测一样多，但它必须把 32 个词按顺序一个接一个地算完；而在 Transformer 里，所有位置是并行一起算的。

    按下按钮后，代码会建一个更大的模型（每个词 128 维、4 个头、3 个块），并在全部 190 万个词上训练两轮。在普通笔记本上大约需要 3 到 7 分钟。训练完成前，依赖它的后续单元格会显示 `Ancestor stopped`，这是正常现象；如果重新运行训练单元格，需要再点一次按钮。完成后会输出三个数字：旋钮总数、最后的训练损失和测试集困惑度。
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
    大约 110 万个旋钮，测试集困惑度约 23。其中词表和 `out` 各占约 25.7 万个，合起来接近整个模型的一半，三个块一共约 59.4 万个。来看看排行榜：
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
    如果你跑过第 14 章，现在它已经是第一名了：困惑度 23 对 LSTM 的 31，而且训练耗时相近。第 13 章的统计计数模型更是远远落在后面。

    ## 练习 5：write_ids

    第 14 章的 `write` 会带着隐藏状态记忆一个词一个词地往后推。Transformer 没有这种跨步记忆：每生成一步，它都要把目前为止的文本重新读一遍，再预测下一个词。又因为位置表的行数有限，每次最多只能把最近的 `longest` 个词传给它。

    /// admonition | 🧰 工具箱：列表的末尾
    `items[-3:]` 会取出列表的最后 3 项；如果列表不足 3 项，就返回整个列表。
    ///

    编写 `write_ids(model, first, count, temperature, longest)`。它从起始词编号 `first` 出发，每次新添一个词编号，一共再添 `count` 个，最后返回完整的编号列表。每一步：

    1. 取出列表里最近的 `longest` 个编号，包装成形状为 `(1, n)` 的张量（批大小为 1），用 `torch.tensor([some_list])` 就能做到。
    2. 送进模型，得到形状为 `(1, n, vocab)` 的分数张量 `S`。
    3. 下一个词的预测就在这唯一一段文本的最后一个位置：`S[0, -1]`。用第 14 章的 `sample_word(scores, temperature)` 从这组分数中抽样一个编号，追加到列表末尾。

    生成阶段不需要训练，记得把循环放进 `torch.no_grad()` 里（第 10 章）。
    """)
    return


@app.function
def write_ids(model, first, count, temperature, longest):
    # ← 把下面的 todo() 换成你的代码
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
    现在让它从 `once` 开始续写 50 个词，温度设为 0.8。你会看到句子已经挺流畅，但情节容易跑偏——困惑度 23 大致就是这个水准。
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
    下面是最后一个块里某个注意力头的权重分布，和第 15 章一样：每一行代表一个词，每一列代表它回看的某个前面的词。右上角是空白的，因为掩码挡住了未来的词。
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
    （这里展示的是 4 个头里的第一个。）每一行的权重加起来等于 1。别指望每个头都对应某种一目了然的规则：真实训练出来的注意力分布往往很难直接读懂，这是很正常的。

    ## 接下来

    你已经亲手搭出了现代大语言模型的核心架构，而且它小到每一行都能完全看透。它和那些著名大模型之间的差别，主要在规模和数据量：大模型拥有几十亿甚至更多旋钮，也在海量得多的文本上训练过。

    不过有一件事你一直见过，却还没认真追问过：除了一个分数之外，我们怎么判断模型在某个任务上到底好不好用？**第 18 章**会探讨如何把模型迁移到新任务上，以及评估模型时那些容易让人自欺欺人的陷阱。

    ## 你现在会做的事

    - 用位置表让注意力分辨词序
    - 手算并用代码实现层归一化，并说清它的作用
    - 用两步残差连接 `x + f(x)` 搭出一个 Transformer 块
    - 从编号到分数，顺着张量形状走完整个 GPT
    - 算出模型的旋钮数量，并指出大部分旋钮分布在哪里
    - 训练一个小型 GPT，送它登上排行榜，并让它续写文本

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** Transformer 出自 *Attention Is All You Need*（Vaswani 等，2017），GPT 出自 OpenAI 自 2018 年起的系列论文。原始设计用固定的正弦波表示位置。“先归一化、再进子层”的顺序叫 *pre-norm*。实际的大模型通常还会加入 *dropout*，而且很多模型会让词向量表和输出层 `out` 共用同一张权重表，这叫 *weight tying*。
    """)
    return


if __name__ == "__main__":
    app.run()
