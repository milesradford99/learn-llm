import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="16 · 自注意力")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 16 · 自注意力

    第 15 章最后留了个问题：既然真正起作用的是注意力，为什么还要留着 LSTM？LSTM 只能逐词地读，每一步都要等前一步，责任也要走很长的路。

    **Transformer** 彻底去掉了循环。每个词直接看它前面的词，而且同时进行。这个设计发表于 2017 年，今天所有的大语言模型都建立在它之上。

    我们分两章来学。这一章一块一块地造出它唯一的新部件：**自注意力**（self-attention）层。第 17 章把这个层叠成一个小小的 **GPT**，训练它，再放上你的排行榜。层里的每个想法你都见过：它就是第 15 章的注意力，加上四处改动。

    ## 词与词互相提问

    在第 15 章，解码器的状态向编码器的状态提问。在自注意力里，同一段文本的词互相提问：每个词都是查询，同时也是键和值。

    最简单的做法，是三个角色都直接用词的向量。我们拿三个词试试，每个向量 2 个数字。下面这张表列出所有分数：第 $i$ 行第 $j$ 列是第 $i$ 个词和第 $j$ 个词的点积，也就是第 $i$ 个词看第 $j$ 个词有多用力。
    """)
    return


@app.cell
def _():
    X_words = torch.tensor([[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]])
    X_words @ X_words.T
    return (X_words,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这张表有两个毛病。

    - **每个词主要在看自己。** 向量和自己的点积是各个数的平方和，通常是这一行里最大的：第 0 行是 4，第 2 行是 9。
    - **看是相互的。** 第 1 行第 2 列是 3，第 2 行第 1 列也是 3。第 1 个词看第 2 个词，和第 2 个词看第 1 个词一样用力，因为点积不在乎两个向量谁先谁后。

    语言不是这样的。在 `the dog ran` 里，`ran` 要找到它的主语 `dog`，但这不等于 `dog` 也同样需要 `ran`。

    所以每个词要为不同的角色准备不同的向量。它的向量经过三个不同的 `nn.Linear` 层（不带偏置，所以每个层就是一个矩阵），得到：

    - **查询**（query）：我在找什么？
    - **键**（key）：别人来找时，我能提供什么？
    - **值**（value）：被选中时，我传出什么？

    还是这三个词，这次给查询配一个随手编的矩阵，给键配另一个。第 $i$ 行第 $j$ 列现在是第 $i$ 个词的查询乘以第 $j$ 个词的键：
    """)
    return


@app.cell
def _(X_words):
    to_query_demo = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    to_key_demo = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    (X_words @ to_query_demo) @ (X_words @ to_key_demo).T
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    行和列互换之后，这张表不再相同：第 2 个词看第 0 个词的分数是 6，而第 0 个词看第 2 个词的分数是 $-6$。对角线也不再是最大的。一个词要找的和它能提供的，分开了。在真正的层里，这些矩阵都是旋钮，由训练来找。

    接下来的注意力和以前一样：比较、分配、混合各个值。所以每个词的新向量，是它认为相关的那些词传出内容的混合。

    ## 动手算：谁问谁

    两个词。`dog` 的查询是 `[1, 0]`，键是 `[0, 2]`。`ran` 的查询是 `[0, 3]`，键是 `[1, 1]`。分数总是提问者的查询乘以对方的键。把两个分数都算出来。
    """)
    return


@app.cell(hide_code=True)
def _():
    ask_boxes = hand.boxes(['dog 问 ran', 'ran 问 dog'])
    return (ask_boxes,)


@app.cell(hide_code=True)
def _(ask_boxes):
    hand.marked(ask_boxes, {'dog 问 ran': 1, 'ran 问 dog': 6}, {
        'dog 问 ran': r"""`dog` 的查询 `[1, 0]` 乘以 `ran` 的键 `[1, 1]`""",
        'ran 问 dog': r"""`ran` 的查询 `[0, 3]` 乘以 `dog` 的键 `[0, 2]`""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    两个分数不一样，这正是我们想要的。这是第 1 处改动：三个角色。后面还有三处。

    ## 改动 2：不许偷看

    语言模型要预测下一个词，所以一个词不能看到它后面的词，否则模型直接抄答案就行了。我们需要一张表，标出哪些“看”要挡住。4 个词的情况如下，`True` 表示“挡住”：

    ```
    False  True  True  True      第 0 个词只看到自己
    False False  True  True      第 1 个词看到第 0、1 个词
    False False False  True
    False False False False      第 3 个词看到自己和前面的全部
    ```

    主对角线以上全是 `True`。这张表叫**掩码**（mask）。

    ## 动手算：掩码

    一段 5 个词的文本，掩码一共挡住多少次“看”？第 2 个词（从 0 数起）连自己在内能看到几个词？
    """)
    return


@app.cell(hide_code=True)
def _():
    mask_boxes = hand.boxes(['挡住的个数', '第 2 个词能看到'])
    return (mask_boxes,)


@app.cell(hide_code=True)
def _(mask_boxes):
    hand.marked(mask_boxes, {'挡住的个数': 10, '第 2 个词能看到': 3}, {
        '挡住的个数': r"""一行一行数 `True`：第 0 个词被挡住 4 个，第 1 个词被挡住 3 个，依此类推""",
        '第 2 个词能看到': r"""第 0、1、2 个词""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：future_mask

    /// admonition | 🧰 工具箱：True 和 False 的表
    - `torch.ones(2, 3, dtype=torch.bool)` 是一张 2 行 3 列、全是 `True` 的表。用 `torch.bool` 时，1 存成 `True`，0 存成 `False`。
    - `torch.triu(T, diagonal=1)` 保留 `T` 主对角线以上的部分，其余设为 0，也就是 `False`。（“triu”是 upper triangle，上三角的缩写。）
      对一张三行分别是 `[1, 2, 3]`、`[4, 5, 6]`、`[7, 8, 9]` 的表，它给出 `[0, 2, 3]`、`[0, 0, 6]`、`[0, 0, 0]`。
    - `S.masked_fill(mask, 99.0)` 是 `S` 的一份副本，`mask` 为 `True` 的地方都存成 99。
    ///

    写 `future_mask(n)`，为 `n` 个词给出上面那样的表：`n` 行 `n` 列，只有“看向后面的词”的位置是 `True`。
    """)
    return


@app.function
def future_mask(n):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "future_mask", future_mask, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""从一张全是 `True` 的表出发，只保留主对角线以上的部分。""",
        r"""只要一行：用 `torch.triu(...)` 包住一张 `n` 乘 `n`、`dtype=torch.bool` 的全 1 表。`diagonal=1` 表示主对角线本身不保留，因为词可以看自己。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    使用掩码时，我们在 softmax 之前把被挡住的分数设为负无穷。因为 $e^{-\infty}$ 是 0，被挡住的词份额正好是 0，其余份额加起来仍然是 1。PyTorch 把负无穷写作 `-torch.inf`。

    ## 改动 3：让分数保持尺度

    两个各有 $d$ 个数的向量做点积，是把 $d$ 项加起来。根据第 09 章的规则，方差相加，所以对于普通大小的数，它的典型大小随 $\sqrt{d}$ 增长。每个词用 128 个数字时，分数比用 4 个数字时大得多。

    这有什么关系？分数之间差距一大，softmax 给出的份额就接近 0 和 1。下面的单元格来量一量。对每个大小 `d`，它随机生成一个查询和 8 个键，重复 200 次，记下 8 个份额里最大的那个：一次用原始分数，一次把分数除以 $\sqrt{d}$。每一行依次是 `d` 和两个平均值。
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    scale_table = []
    for d_demo in [4, 32, 128, 512]:
        plain = 0.0
        scaled = 0.0
        for trial in range(200):
            q_demo = torch.rand(d_demo) * 2 - 1
            K_demo = torch.rand(8, d_demo) * 2 - 1
            plain = plain + torch.max(torch.softmax(K_demo @ q_demo, dim=0)).item() / 200
            scaled = scaled + torch.max(torch.softmax(K_demo @ q_demo / np.sqrt(d_demo), dim=0)).item() / 200
        scale_table.append((d_demo, round(plain, 2), round(scaled, 2)))
    scale_table
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    用原始分数时，最大的份额从不到 0.3 一路升到接近 0.9：到 512 个数字时，还没开始训练，一个键就几乎拿走了全部。这对学习很不利。有一个份额接近 1 时，softmax 是平的，其余的词几乎收不到责任（第 07 章）。

    除以 $\sqrt{d}$ 之后，不管多大，最大的份额都停在 0.2 左右。所以自注意力就是这样做的：

    > $$\text{scores} = \frac{Q K^\top}{\sqrt{d}}$$

    ## 练习 2：attend

    /// admonition | 🧰 工具箱：从后往前数维度
    `dim=-1` 表示最后一维，`-2` 表示它前面一维。`M.transpose(-2, -1)` 交换最后两维，前面有多少维都行；`M.shape[-1]` 是最后一维的大小。这样写的代码既能用在 `(batch, words, d)` 上，也能用在中间多出一维的张量上，改动 4 就会用到。
    ///

    写 `attend(Q, K, V, mask)`。它是第 15 章的 `attend` 加上改动 2 和 3，和以前一样返回混合结果和份额：

    1. 分数：每个查询对每个键，再除以 $\sqrt{d}$，其中 $d$ 是 `K` 最后一维的大小。`np.sqrt` 可以用在普通的数上。
    2. `mask` 为 `True` 的地方，分数变成 `-torch.inf`。
    3. 份额 `A`：沿最后一维做 softmax。
    4. 用这些份额把各个值混合起来。

    每个维度都从后往前数，这样 `Q`、`K`、`V` 的形状是 `(batch, heads, words, d)` 时它也能用。
    """)
    return


@app.function
def attend(Q, K, V, mask):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "attend", attend, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""四行。第 1、3、4 行就是第 15 章的三行，把 `1` 和 `2` 换成 `-2` 和 `-1`。第 2 行用练习 1 工具箱里的 `masked_fill`。""",
        r"""掩码加在分数上，在 softmax 之前。如果在 softmax 之后再挡，份额加起来就不是 1 了。如果检查只说数值大小不对，看看有没有除以 `np.sqrt(K.shape[-1])`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 改动 4：多个头

    一轮注意力给每个词一个混合结果。但一个词可能同时想知道几件事：主语是谁？现在是什么时态？所以我们并排跑几轮，叫**头**（head），每个头有自己的查询、键和值。

    这不会增加大小。每个词用 128 个数字、有 4 个头时，每个向量切成 4 段，每段 32 个数字，每个头处理自己的一段，最后再把 4 个结果拼起来。

    切分只靠改变形状来完成。跟着一个小例子走一遍：1 段文本，2 个词，每个词 4 个数字，切给 2 个头。`torch.arange(8.0)` 是由 0.0、1.0 一直到 7.0 组成的张量，和 `range` 类似。单元格依次显示原来的张量、`reshape` 之后的张量、`transpose` 之后的张量：
    """)
    return


@app.cell
def _():
    x_trace = torch.arange(8.0).reshape(1, 2, 4)
    cut = x_trace.reshape(1, 2, 2, 2)
    swapped = cut.transpose(1, 2)
    x_trace.tolist(), cut.tolist(), swapped.tolist()
    return (x_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    把三个结果依次读一遍。

    1. 开始时形状是 `(1, 2, 4)`。第 0 个词是 `[0, 1, 2, 3]`，第 1 个词是 `[4, 5, 6, 7]`。
    2. `reshape(1, 2, 2, 2)` 把每个词切成两段。形状现在是 `(batch, words, heads, piece)`：第 0 个词有 `[0, 1]` 和 `[2, 3]` 两段。
    3. `transpose(1, 2)` 交换 `words` 和 `heads`。形状现在是 `(batch, heads, words, piece)`：第 0 个头拿到第 0 个词的 `[0, 1]` 和第 1 个词的 `[4, 5]`，也就是两个词各自的前一半。第 1 个头拿到后一半。

    为什么要交换？`@` 作用在最后两维上，把前面所有的维都当成批。把 `heads` 挪到前面、紧挨着 `batch` 之后，你的 `attend` 会把每个头当成一段独立的文本来算，一个字都不用改。练习 2 要从后往前数维度，就是为了这个。

    ## 动手算：形状

    一个形状为 `(1, 6, 12)` 的张量（1 段文本，6 个词，每个词 12 个数字）要切给 3 个头。按顺序填出结果的四个大小。
    """)
    return


@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['大小 0', '大小 1', '大小 2', '大小 3'])
    return (shape_boxes,)


@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'大小 0': 1, '大小 1': 3, '大小 2': 6, '大小 3': 4}, {
        '大小 0': r"""批留在最前面""",
        '大小 1': r"""交换之后，`heads` 排第二""",
        '大小 2': r"""然后是 `words`""",
        '大小 3': r"""每一段是 12 个数字分给 3 个头""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：split_heads

    写 `split_heads(x, heads)`，`x` 的形状是 `(batch, words, d)`，返回的形状是 `(batch, heads, words, d // heads)`，和例子里一样。（`//` 是第 11 章那种得到整数的除法。）

    `batch, words, d = x.shape` 给三个大小起了名字。

    在上面的例子里，`split_heads(x_trace, 2)` 应该给出第三个结果。
    """)
    return


@app.function
def split_heads(x, heads):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "split_heads", split_heads, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""和例子一样分两步：先 `reshape` 成 `(batch, words, heads, piece)`，再交换第 1、2 维。""",
        r"""先 reshape，再 transpose。直接 reshape 成 `(batch, heads, words, piece)`，形状是对的，但数字会进错头。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：join_heads

    注意力算完之后，要把各个头再拼回去。写 `join_heads(x)`，它撤销 `split_heads`：接收形状 `(batch, heads, words, piece)`，返回 `(batch, words, heads * piece)`。

    把那两步按相反的顺序撤销：先撤销交换，再撤销切分。`batch, heads, words, piece = x.shape` 给四个大小起了名字。

    在例子里，对第三个结果做 `join_heads`，应该得到第一个结果。
    """)
    return


@app.function
def join_heads(x):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "join_heads", join_heads, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`transpose(1, 2)` 自己就能撤销自己：把同样两维再交换一次，就换回来了。""",
        r"""两行：先给四个大小起名字，再 transpose，然后 reshape 成 `(batch, words, heads * piece)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：完整的层

    现在把四处改动合在一起。写 `self_attention_step(x, to_query, to_key, to_value, mix, heads)`。`x` 的形状是 `(batch, words, d)`。后面四个输入是层，调用层就像调用函数：`to_query(x)`。

    1. 做出查询、键和值：各自的层作用在 `x` 上，再切成多个头。
    2. 对它们运行 `attend`，掩码按这段文本的词数来做。词数是 `x.shape[1]`。
    3. 把混合结果的各个头拼回去，再送进 `mix`。这最后一层让各个头的发现能够结合起来。
    4. 返回这个结果和份额。

    | | 形状 |
    |---|---|
    | `x` | `(batch, words, d)` |
    | 查询、键、值 | `(batch, heads, words, d // heads)` |
    | 份额 | `(batch, heads, words, words)` |
    | 结果 | `(batch, words, d)` |
    """)
    return


@app.function
def self_attention_step(x, to_query, to_key, to_value, mix, heads):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "self_attention_step", self_attention_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""每一块都是你在这一章写过的函数：`split_heads`、`future_mask`、`attend`、`join_heads`。五行就够：三行做三个角色，一行 `attend`，一行返回。""",
        r"""`attend` 返回一对：混合结果和份额。只有混合结果要拼回去、再过 `mix`。份额原样返回。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是写成 PyTorch 模块的层。它的旋钮是三个角色的矩阵和 `mix`，它的 `forward` 就是你的函数。它把最新的份额存在 `self.shares` 里，第 17 章会拿出来看。之后的单元格用一个新建的层处理一段文本：5 个词，每个词 8 个数字，分给 2 个头，并显示两个形状。五个练习全部通过之前，后面的单元格会显示错误，这是正常的。
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
            out, self.shares = self_attention_step(x, self.query, self.key, self.value, self.mix, self.heads)
            return out
    return (SelfAttention,)


@app.cell
def _(SelfAttention):
    torch.manual_seed(0)
    layer_demo = SelfAttention(8, 2)
    x_demo = torch.rand(1, 5, 8)
    with torch.no_grad():
        out_demo = layer_demo(x_demo)
    out_demo.shape, layer_demo.shares.shape
    return layer_demo, x_demo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `[1, 5, 8]` 和 `[1, 2, 5, 5]`：输出和输入形状相同，所以层可以一层层叠起来。份额是 2 个头各一张 5 乘 5 的表。

    ## 注意力不识数

    还差一样东西，而且很容易漏掉。取这五个词向量，去掉掩码运行注意力。然后把词的顺序反过来，再运行一次注意力，然后把结果的顺序也反过来。每个词向同样的词问了同样的问题，所以应该得到同样的混合结果。单元格打印两次结果的最大差别：
    """)
    return


@app.cell
def _(layer_demo, x_demo):
    Q_demo = split_heads(layer_demo.query(x_demo), 2)
    K_demo2 = split_heads(layer_demo.key(x_demo), 2)
    V_demo = split_heads(layer_demo.value(x_demo), 2)
    no_mask = torch.zeros(5, 5, dtype=torch.bool)
    forwards = attend(Q_demo, K_demo2, V_demo, no_mask)[0]
    backwards = attend(torch.flip(Q_demo, dims=[2]), torch.flip(K_demo2, dims=[2]), torch.flip(V_demo, dims=[2]), no_mask)[0]
    torch.max(torch.abs(forwards - torch.flip(backwards, dims=[2]))).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    结果大约是 `1e-07`，也就是 0.0000001：只是舍入误差，所以两者相同。对注意力来说，`the dog bit the man`（狗咬了人）和 `the man bit the dog`（人咬了狗）是同一个词袋（bag of words）。LSTM 按顺序读，顺序是白送的。这里我们得自己提供。

    ## 接下来

    层已经有了。**第 17 章**会告诉它每个词站在哪里，让它的数值在很多层之后仍然稳定，把它叠成一个大约一百万个旋钮的小 GPT，再用故事来训练。

    ## 你现在会做的事

    - 说出一个词为什么需要分开的查询、键和值
    - 用掩码挡住未来
    - 说出分数为什么要除以 $\sqrt{d}$
    - 把向量切成多个头再拼回去，并说出为什么这不增加大小
    - 用这些部件搭出一个自注意力层
    - 解释为什么只有注意力时会忽略词序

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** Transformer 出自 *Attention Is All You Need*（Vaswani 等，2017），这个层在论文里叫 *scaled dot-product attention* 加 *multi-head attention*。不加掩码、词可以向两边看的层，是 BERT 用的那种（第 18 章）。
    """)
    return


if __name__ == "__main__":
    app.run()
