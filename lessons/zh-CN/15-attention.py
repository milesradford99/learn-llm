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

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.diagrams import looking
    from learn_llm.images import maps

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 15 · 注意力

    第 14 章的 LSTM 会把读到的所有信息都压进一个 128 维向量里。拿它来预测故事的下一个词，效果已经很不错了。现在换一个更难的任务：翻译。模型必须先读完一整句话，再用另一种语言把它写出来。一个很自然的思路是用两个 LSTM：**编码器**（encoder）负责把整句话读进记忆里，**解码器**（decoder）则从这份记忆出发，一个词一个词地写出译文。

    问题出在这唯一的记忆上：不管句子有多长，所有内容都得硬塞进这一个小小的向量里。想想你自己翻译长句子时会怎么做：你不会先死背整句话、再合上原文闷头写，而是写到哪儿、需要哪个词，就随时回头看一眼原文。

    这种“边写边回头看”的机制就叫**注意力**（attention），它正是 ChatGPT 中字母 T 背后的核心思想——T 代表 **Transformer**（第 16、17 章）。

    ## 一个能检查答案的任务

    课程里没有现成的双语翻译数据，所以我们换一个结构完全相同、而且对错一眼就能看出来的任务：**把一段话倒着写出来**。

    > 输入：`the dog ran home` → 输出：`home ran dog the`

    这个任务很适合做测试：第一个要写的词正好是最后读到的，最后一个要写的词则是最初读到的。它不需要模型“理解”语义，纯粹考验模型能不能把整段序列一字不差地记住。下面的单元格准备了 500 个常用词，外加一个稍后会用到的特殊词 `<start>`，还定义了一个函数，用来随机抽取指定长度的文本片段及其倒序版本。其中 `torch.flip` 沿指定维度反转张量，`rng.integers` 一次生成一批随机起点，`np.stack` 把多行叠成一张二维表（`to_words` 是第 12 章的切词函数）。
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

    解码器一次写一个词，每写完一个词，就把它作为下一步的输入。但在训练时，我们不会喂给它自己刚才猜出的词——刚开始训练时模型大多在瞎猜，第一步猜错就会把后面每一步都带偏。相反，我们在每一步都直接喂给它**正确的**上一个词。这种做法叫**教师强制**（teacher forcing），它的好处是所有位置可以一次性并行训练。因此，解码器的输入就是把正确答案整体向右挪一位，并在最前面补一个 `<start>`（编号 0）：

    ```
    答案 Y：      home ran  dog  the
    解码器输入：  <start> home ran dog
    ```

    ## 练习 1：shift_right

    /// admonition | 🧰 工具箱：表的列
    取一张表 `T`，两行分别是 `[1, 2, 3]` 和 `[4, 5, 6]`。

    - `T[:, :1]` 保留所有行，只取第 1 列：`[[1], [4]]`。结果依然是一张二维表，只是只有 1 列。
    - `T[:, :-1]` 保留所有行，去掉最后一列：`[[1, 2], [4, 5]]`。
    - `torch.zeros_like(T)` 生成一张全 0 的表，形状和数据类型都和 `T` 完全相同。
    - `torch.cat([A, B], dim=1)` 把两张表按列左右拼在一起（第 14 章用过）。
    ///

    编写 `shift_right(Y)`，其中 `Y` 是形状为 `(batch, length)` 的表。每一行去掉最后一个词，并在最前面补一个 0（`<start>` 的编号），保持形状不变：

    > `[[7, 8, 9], [4, 5, 6]]` 变成 `[[0, 7, 8], [0, 4, 5]]`

    可以把它拆成两块来拼：左边是一列全 0，右边是去掉最后一列的 `Y`。
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
    hand.stuck([
        r"""用到的工具都在上面的工具箱里：只取第 1 列的切片、`torch.zeros_like`、去掉最后一列的切片，以及 `torch.cat`。""",
        r"""一行代码就能搞定：在 `torch.cat([..., ...], dim=1)` 里把全 0 列放在前面。想要得到这列 0，先用切片取出 `Y` 的第 1 列，再对它调用 `torch.zeros_like` 生成同形状的全 0 列即可。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是模型，相当于把第 14 章的 `Reader` 拆成了两半。编码器每读一个词就输出一个状态向量，并交出最终记忆；解码器从这份最终记忆出发，每写一个词也输出一个自己的状态向量。这里先把编码器状态简记为 `K`、解码器状态简记为 `Q`，稍后讲注意力时会揭晓这两个字母的含义。

    `use` 是我们传进去的函数（就像第 11 章 `Block` 里的 `step`），用来决定怎样处理 `Q` 和 `K`：`use` 返回的是要打分的向量和注意力权重（稍后会讲到），`[0]` 取出其中的第一项。`no_attention` 原样返回 `Q`、权重返回 `None`，完全忽略 `K`，所以从编码器传给解码器的只有最后的记忆。`size` 是记忆的大小，`width` 是最后一层接收的数字个数。
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
    ## 先预测形状

    训练之前，先在纸上跟着一批数据走一遍模型。假设 `X` 里有 64 段文本，每段 6 个词，记忆大小是 128，词表共有 501 个词。逐行读一遍上面的 `forward`，填出下面几个维度大小。（`nn.LSTM` 每读一个词都会输出一个状态向量。）
    """)
    return



@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['K：词数', 'K：数字个数', 'Q：词数', 'out：分数个数'])
    return (shape_boxes,)



@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'K：词数': 6, 'K：数字个数': 128, 'Q：词数': 6, 'out：分数个数': 501}, {
        'K：词数': r"""`K` 的形状是 `(64, ?, 128)`：编码器每读一个词都会留下一个状态""",
        'K：数字个数': r"""每个状态就是一份记忆，`size` 决定了它包含多少个数字""",
        'Q：词数': r"""解码器的输入是 `shift_right(Y)`，形状和 `Y` 完全相同""",
        'out：分数个数': r"""`self.out` 会为词表里的每个词打一个分数""",
    }, LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    可见，编码器明明交出了 6 个 128 维状态，但 `no_attention` 把 `K` 直接扔掉了。真正传给解码器的只有最后那份记忆——区区一个 128 维向量。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `train_translator` 每次随机抽取 3 到 20 个词的片段来训练。`translate` 则是真正独立往外写，不再有老师提示：每一步都把解码器上一步猜出的最可能词（`torch.argmax`）作为下一步的输入。`share_correct` 会在某个固定长度下测试 500 个样本，计算写对的词所占的比例。

    按下按钮后会训练 1,500 批（大约半分钟），然后依次打印长度为 5、10、15、20 时的 `(长度, 正确率)`，例如 `0.988` 表示 98.8%。
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
    你应该会看到：长度 5 时大约 99%，长度 10 时降到约 78%，长度 15 时只有约 42%，长度 20 时跌到约 23%。句子越长表现越差，因为区区 128 个数字根本装不下整句话。这就是**瓶颈**（bottleneck）。

    ## 加权平均

    解决办法从一个很简单的想法开始。假设有三个向量，我们想按比例把它们混在一起：第一个占 50%，第二个占 30%，第三个占 20%。这就是**加权平均**（weighted average）：每个向量乘以它的**权重**，再全部加起来。

    | 行 | 向量 | 权重 | 权重 $\times$ 向量 |
    |---|---|---|---|
    | 0 | `[10, 0]` | 0.5 | `[5, 0]` |
    | 1 | `[0, 10]` | 0.3 | `[0, 3]` |
    | 2 | `[10, 10]` | 0.2 | `[2, 2]` |
    | | | **求和** | **`[7, 5]`** |

    这些权重加起来等于 1，所以结果会落在这些向量之间：谁的权重最大，结果就最偏向谁。
    如果权重是 `[1, 0, 0]`，得到的就正好是第一个向量。因此，加权平均就像一种“柔和的查找”：既可以精准挑出某一行，也可以把几行按比例融合在一起。

    ## 动手算：一次混合

    轮到你了。三个行向量分别是 `[4, 0]`、`[0, 8]` 和 `[2, 2]`，对应的权重是 0.25、0.25 和 0.5。算出加权平均得到的两个数。
    """)
    return


@app.cell(hide_code=True)
def _():
    blend_boxes = hand.boxes(['第一个数', '第二个数'])
    return (blend_boxes,)



@app.cell(hide_code=True)
def _(blend_boxes):
    hand.marked(blend_boxes, {'第一个数': 2, '第二个数': 3}, {
        '第一个数': r"""用每一行的第一个数：$0.25 \times 4 + 0.25 \times 0 + 0.5 \times 2$""",
        '第二个数': r"""用每一行的第二个数：$0.25 \times 0 + 0.25 \times 8 + 0.5 \times 2$""",
    }, LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：weighted_average

    编写 `weighted_average(a, V)`，其中权重 `a` 是一维向量、形状为 `(n,)`，矩阵 `V` 的形状为 `(n, d)`、每一行是一个向量。函数返回一个有 `d` 个数的向量。不需要写循环：用 `@` 把一维权重 `(n,)` 乘在 `(n, d)` 矩阵左边，正好把每一行乘以对应权重再相加（这就是第 05 章学过的向量乘矩阵）。

    用上面那张表的数据，权重 `[0.5, 0.3, 0.2]` 应该算出 `[7, 5]`。
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
    hand.stuck([
        r"""用 `@` 做向量乘矩阵：把一维权重乘在矩阵左边，会自动把每一行乘以对应权重再相加。""",
        r"""一行就够，把权重 `a` 写在 `@` 的左边。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 往哪里看：查询、键、值

    现在解码器终于可以回头看原文了。每写一个词时，解码器都有一个当前状态，而编码器为每个读到的词都留下了一个状态。注意力机制分三步，每一步你都见过：

    1. **打分比较。** 把解码器的当前状态和每个编码器状态做点积。点积越大，说明“这个词和当前要写的内容越相关”（第 12 章学过：点积衡量两个向量有多相似）。
    2. **分配权重。** 用 softmax 把这些分数转换成加起来等于 1 的正数**权重（注意力权重）**（第 07 章）。
    3. **加权混合。** 用这些注意力权重，对所有编码器状态做加权平均。

    **用一个具体数值例子走一遍。** 实际状态有 128 个数字，这里为了方便心算，假设每个状态只有 2 个数字。读到的三个词对应的编码器状态如下，解码器当前状态是 `q = [2, 1]`：

    | 读到的词 | 编码器状态 | 1. 比较：与 `q` 的点积 | 2. 分配权重：softmax |
    |---|---|---|---|
    | the | `[1, 0]` | $2 \times 1 + 1 \times 0 = 2$ | 0.24 |
    | dog | `[0, 1]` | $2 \times 0 + 1 \times 1 = 1$ | 0.09 |
    | ran | `[1, 1]` | $2 \times 1 + 1 \times 1 = 3$ | 0.67 |

    第 3 步用这些权重把三个状态加权混合：

    > $$0.24 \times [1, 0] + 0.09 \times [0, 1] + 0.67 \times [1, 1] = [0.91, 0.76]$$

    混合结果很接近 `ran` 的状态，同时略微掺杂了一点另外两个词的信息。下面把这一步画成图：连线越粗，表示注意力权重越大。
    """)
    return


@app.cell(hide_code=True)
def _():
    looking(['the [1, 0]', 'dog [0, 1]', 'ran [1, 1]'], [0.24, 0.09, 0.67], 'q = [2, 1]',
            '解码器状态 q 看向三个编码器状态，注意力权重分别是 0.24、0.09 和 0.67')
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    注意 softmax 的放大效应有多明显。`ran` 的分数只比 `the` 高 1 分，权重却是它的大约 2.7 倍——因为分数每高 1 分，对应的权重就会乘以 $e \approx 2.72$（第 07 章）。

    这样算出来的结果，就是专门为写当前这个词而生成的“定制版输入摘要”。这三部分的名称借用了查资料的比喻：**查询**（query，`q`）是你要找什么，**键**（key，`K`）是拿来比对的标签，**值**（value，`V`）是最终取回的内容。就像去图书馆查书：查询是你的问题，键是书脊上的书名，值是书里的内容。现在你也明白前面为什么把解码器状态记作 `Q`、把编码器状态记作 `K` 了！这里键和值恰好都是编码器状态，但它们不必相同，下一章我们就会把它们分开。

    你可能会问：解码器状态和编码器状态的点积凭什么恰好会大？一开始确实不大，全靠训练过程中的反向传播，慢慢把需要对齐的状态调成点积变大，下面马上就能看到。

    ## 动手算：打分比较

    换一个查询 `q = [1, 3]`，三个键分别是 `[2, 0]`、`[0, 1]` 和 `[1, 1]`。算出三个点积分数，再判断哪个键会分到最大的权重：0、1 还是 2。
    """)
    return



@app.cell(hide_code=True)
def _():
    score_boxes = hand.boxes(['分数 0', '分数 1', '分数 2', '权重最大的键'])
    return (score_boxes,)



@app.cell(hide_code=True)
def _(score_boxes):
    hand.marked(score_boxes, {'分数 0': 2, '分数 1': 3, '分数 2': 4, '权重最大的键': 2}, {
        '分数 0': r"""`[1, 3]` 和 `[2, 0]` 的点积：$1 \times 2 + 3 \times 0$""",
        '分数 1': r"""`[1, 3]` 和 `[0, 1]` 的点积""",
        '分数 2': r"""`[1, 3]` 和 `[1, 1]` 的点积""",
        '权重最大的键': r"""softmax 保持大小顺序不变：分数最高的，权重也最大。填这个键的编号""",
    }, LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：attention_shares

    编写 `attention_shares(q, K)`，为单个查询完成第 1、2 步。其中 `q` 的形状是 `(d,)`，键 `K` 的形状是 `(n, d)`（每行一个键）。函数返回 `n` 个注意力权重。

    `K @ q` 是矩阵乘向量（第 05 章）：它把 `K` 的每一行分别和 `q` 做点积，一次算出 `n` 个分数。调用 `torch.softmax` 时需要告诉它沿哪一维让权重之和等于 1，一维向量只有一维，也就是 `dim=0`。

    用前面表格里的数据，`q = [2, 1]` 算出的权重应该约等于 `[0.24, 0.09, 0.67]`。
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
    hand.stuck([
        r"""按顺序两步：先算出点积分数，再对分数做 softmax。""",
        r"""一行就能写完：用 `torch.softmax(...)` 包住点积分数，并指定 `dim=0`。""",
    ], LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是用你刚写的函数，为刚才那道手算题算出的注意力权重：
    """)
    return



@app.cell
def _():
    attention_shares(torch.tensor([1.0, 3.0]), torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    结果大约是 0.09、0.24 和 0.67——和前面表格里的三个权重完全一样！尽管那边的分数是 2、1、3，这边是 2、3、4。这是因为 softmax 只看分数之间的**差值**（第 07 章讲过）：给所有分数同时加上同一个数，算出来的权重不会改变。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：attend

    现在一步到位，处理一整批文本，而且每段文本里同时有多个查询。每个要写的词都有自己的查询，所以注意力权重会排成一张二维表：每写一个词占一行，每读一个词占一列。

    /// admonition | 🧰 工具箱：一批矩阵
    形状为 `(5, 2, 3)` 的张量可以看成 5 个形状为 `(2, 3)` 的矩阵，每段文本对应一个。

    - `@` 会按文本逐段相乘，作用在最后两维上：`(5, 2, 3) @ (5, 3, 4)` 得到 `(5, 2, 4)`。
    - `M.transpose(1, 2)` 交换第 1 维和第 2 维，相当于给每段文本单独做转置 `.T`：`(5, 2, 3)` 变成 `(5, 3, 2)`。
    - `torch.softmax(M, dim=2)` 让数字沿最后一维加起来等于 1，也就是让每个矩阵的每一行之和等于 1。
    ///

    为什么这里和刚才的 `K @ q` 左右反过来了？单查询时，`q` 是一维向量，写成 `K @ q`；现在把多个查询按行叠成 `Q (batch, writes, d)`，每个查询行都要和 `K (batch, reads, d)` 的每一行做点积，所以把 `K` 最后两维转置成 `(batch, d, reads)` 放到右边，得到 `Q @ K.transpose(1, 2)`，形状正好是 `(batch, writes, reads)`。

    把查询排成 $Q$ 的各行、键排成 $K$ 的各行、值排成 $V$ 的各行，注意力的三步公式就是：

    > $$\text{scores} = Q K^\top \qquad A = \text{softmax}(\text{scores}) \qquad \text{blended} = A V$$

    其中 $K^\top$ 表示把 $K$ 的行和列互换，这样分数的第 $i$ 行就正好是第 $i$ 个查询和每个键的点积。

    编写 `attend(Q, K, V)`，返回两样东西：加权混合后的向量和注意力权重 `A`：

    | | 形状 |
    |---|---|
    | `Q` | `(batch, writes, d)` |
    | `K` | `(batch, reads, d)` |
    | `V` | `(batch, reads, e)`（值可以有自己独立的维度大小） |
    | 分数以及 `A` | `(batch, writes, reads)` |
    | 混合结果 | `(batch, writes, e)` |

    每个要写的词都要把注意力分配给所有读到的词，所以权重必须沿 `reads` 这一维（最后一维）加起来等于 1。
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
    hand.stuck([
        r"""一共三行，对应上面三个公式。照着表里的形状走：`(batch, writes, d) @ (batch, d, reads)` 正好得到 `(batch, writes, reads)`。""",
        r"""`K` 的形状是 `(batch, reads, d)`，在 `@` 之前要先把它的最后两维互换。softmax 沿第 2 维（`dim=2`）计算。最后加权混合用的是 `V` 而不是 `K`，函数返回 `(混合结果, A)`。""",
    ], LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：with_attention

    只差最后一块拼图：写一个 `use` 函数传给 `Translator`，用来替换之前的 `no_attention`。可以回头看一眼 `forward` 和 `translate` 是怎样调用 `use(Q, K)` 的，以及 `no_attention` 返回了什么。

    编写 `with_attention(Q, K)`：

    1. 调用你刚写好的 `attend`。查询是解码器状态 `Q`，编码器状态 `K` 既充当键，也充当值。
    2. 把每个解码器状态和它对应的混合摘要并排拼在一起，让最后一层能同时看到两份信息。记忆是 128 维时，拼起来就是 256 维，所以最后一层的输入宽度要翻倍。
    3. 返回拼接后的向量和注意力权重。

    | | 形状 |
    |---|---|
    | `Q` | `(batch, writes, 128)` |
    | `K` | `(batch, reads, 128)` |
    | 拼好的向量 | `(batch, writes, 256)` |
    | 注意力权重 | `(batch, writes, reads)` |
    """)
    return



@app.function
def with_attention(Q, K):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "with_attention", with_attention, LOCALE)
    return



@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""调用 `attend(Q, K, K)` 会返回混合后的向量和注意力权重。`torch.cat` 可以沿指定维度把两个张量拼在一起。""",
        r"""两行代码。状态是三维张量 `(batch, writes, 128)`，需要沿最后一维（`dim=2`）拼接，把解码器状态 `Q` 放在前面。""",
    ], LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    其余条件完全不变：同样的模型、同样的数据、同样的 1,500 批、同样的记忆大小。
    在你按下按钮之前，后面的单元格会显示 `Ancestor stopped`，这是正常的。
    """)
    return


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
    你应该会看到：长度 15 以内接近 100%，长度 20 时也能达到约 95%——而不用注意力时只有 23%。记忆容量丝毫没有变大，只是它不用再硬背所有内容了：解码器写到哪里需要什么，随时回去取就行。

    ## 看看它把注意力放在哪儿

    注意力权重不仅管用，还非常直观易懂。下面是一段 12 个词的文本对应的注意力权重图：每写一个词占一行，每读一个词占一列，蓝色越深代表权重越大。图下方依次列出读到的原文和模型写出的结果。既然任务是倒着写，一个好模型在写第一个词时应该盯着最后读到的词，写最后一个词时应该盯着最早读到的词。
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
    图上应该会清晰地出现一条从右上角斜向左下角的蓝线。没有人教过模型该往哪里看：完全是每个写错的词传回来的梯度（责任），让解码器状态和编码器状态学会了自动对齐。

    注意力机制同样帮到了梯度的传递。现在从每个写出的词到每个读到的词之间都有一条直达捷径，这和第 11 章的跳跃连接、第 14 章 LSTM 的细胞通道是同一个道理。

    ## 接下来

    这里冒出一个很有意思的问题：LSTM 还留在模型里，依然只能一个词一个词地顺序读，速度很慢，因为每一步都必须等上一步算完。既然真正立大功的是注意力机制，我们还需要 LSTM 吗？如果干脆去掉 LSTM，让每个词同时直接关注句子里的其他词呢？

    这就是 **Transformer**。在**第 16 章**，你会只用注意力搭出它唯一的新层；到**第 17 章**，你会把这一层叠成完整模型，送它上排行榜，看着它一举拿下第一名。

    ## 你现在会做的事

    - 解释编码器-解码器的结构，以及为什么单个固定长度的记忆会成为瓶颈
    - 用教师强制训练解码器
    - 用 `@` 计算加权平均
    - 解释查询（query）、键（key）和值（value）的含义
    - 动手算出注意力的一步：打分、softmax 权重、加权混合
    - 为一整批数据写出注意力的三步：比较、分配权重、混合
    - 把注意力机制接入编码器-解码器模型
    - 读懂注意力权重热力图

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 用于机器翻译的注意力机制最早由 Bahdanau、Cho 和 Bengio（2014）提出；本章这种在解码器一步之后加入点积注意力的形式，更接近 Luong 等人（2015）的做法。真实的翻译系统在生成时还会使用*束搜索*（beam search），而不是每一步都只挑最可能的那个词，并用 *BLEU* 等指标来评估翻译质量。
    """)
    return


if __name__ == "__main__":
    app.run()
