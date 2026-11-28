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

    第 15 章结尾留下了一个问题：既然真正起关键作用的是注意力机制，为什么还要留着 LSTM？LSTM 只能一个词接一个词地顺序阅读，每一步都得等前一步算完，训练时的梯度（误差信号）也要穿过一条很长的链路才能传回前面的词。

    **Transformer** 彻底抛弃了循环结构：句子里的每个词都能直接看向它前面的所有词，而且所有位置可以同时计算。这套架构发表于 2017 年，今天几乎所有大语言模型都建立在它之上。

    我们分两章来亲手搭出它。这一章先一步步造出它唯一的新核心部件——**自注意力**（self-attention）层；第 17 章再把这一层叠成一个微型 **GPT**，用故事数据训练它，并放上你的排行榜。这一层的核心思路你其实都见过：它就是第 15 章的注意力（算分数、用 softmax 转成权重、按权重混合值向量），再加上四处关键升级。

    ## 词与词互相提问

    在第 15 章里，是解码器的状态向编码器的状态提问。到了自注意力，变成同一段文本里的词互相提问：每个词既是提问的“查询”，同时也是供别人检索的“键”和传递信息的“值”。

    最直觉的做法，是让这三个角色都直接共用词原本的向量。我们先拿三个词试试，每个词用 2 个数字表示。下面这张表算出了每两个词之间的点积分数：第 $i$ 行第 $j$ 列代表第 $i$ 个词和第 $j$ 个词的点积，用来衡量第 $i$ 个词对第 $j$ 个词的关注程度。
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
    仔细看这张表，有两个明显的毛病：

    - **每个词主要都在盯着自己。** 一个向量和自己的点积是各分量的平方和，往往比和别的词做点积大得多：第 0 行最大的是对角线上的 4，第 2 行最大的是对角线上的 9。
    - **关注被迫完全对称。** 第 1 行第 2 列是 3，第 2 行第 1 列也是 3。第 1 个词关注第 2 个词的分数，和第 2 个词关注第 1 个词的分数一模一样——因为点积不在乎两个向量谁在前、谁在后。

    但自然语言不是对称的。在 `the dog ran` 里，动词 `ran` 需要回头找到它的主语 `dog`，但这并不意味着 `dog` 也需要用同样的方式依赖 `ran`。

    解决办法是把每个词的向量送进三个不同的 `nn.Linear` 层（不带偏置，也就是乘以三个不同的权重矩阵），拆成三个各司其职的向量：

    - **查询**（query）：我现在想找什么信息？
    - **键**（key）：别人来找信息时，我身上有什么特征可以匹配？
    - **值**（value）：如果我被选中了，我要传出去什么内容？

    还是上面那三个词，这次我们给查询配一个随手构造的矩阵，给键配另一个。现在第 $i$ 行第 $j$ 列变成了“第 $i$ 个词的查询”和“第 $j$ 个词的键”的点积：
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
    行列互换后，这张表不再对称了：第 2 个词关注第 0 个词的分数是 6，而第 0 个词关注第 2 个词的分数是 $-6$；对角线也不再一定是最大的值。一个词“想找什么”和“能提供什么”终于分开了。在真正的网络里，这几个矩阵都是可训练参数，会在训练中自动学出来。

    算出分数之后，剩下的流程和第 15 章一模一样：对每一行分数做 softmax 得到总和为 1 的注意力权重（概率），再用这些权重对各个词的值向量做加权混合。这样，每个词的新向量就融合了它认为最相关的那些词传过来的信息。

    ## 动手算：谁问谁

    假设有两个词：`dog` 的查询是 `[1, 0]`，键是 `[0, 2]`；`ran` 的查询是 `[0, 3]`，键是 `[1, 1]`。记住规则：分数永远是“提问者的查询”与“被问者的键”的点积。请把两个方向的分数都算出来。
    """)
    return


@app.cell(hide_code=True)
def _():
    ask_boxes = hand.boxes(['dog 问 ran', 'ran 问 dog'])
    return (ask_boxes,)


@app.cell(hide_code=True)
def _(ask_boxes):
    hand.marked(ask_boxes, {'dog 问 ran': 1, 'ran 问 dog': 6}, {
        'dog 问 ran': r"""`dog` 的查询 `[1, 0]` 与 `ran` 的键 `[1, 1]` 做点积""",
        'ran 问 dog': r"""`ran` 的查询 `[0, 3]` 与 `dog` 的键 `[0, 2]` 做点积""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    两个方向的分数果然不同，这正是我们想要的效果。这就是第 1 处改动：拆分出查询、键、值三个角色。接下来还有三处。

    ## 改动 2：不许偷看未来

    语言模型的任务是预测下一个词，所以每个词绝不能看到排在它后面的词，否则模型直接抄答案就行了。我们需要一张布尔表，把所有“看向后面词”的位置标出来。以 4 个词为例，`True` 表示“屏蔽掉、不许看”：

    ```
    False  True  True  True      第 0 个词只能看到自己
    False False  True  True      第 1 个词能看到第 0、1 个词
    False False False  True      第 2 个词能看到第 0、1、2 个词
    False False False False      第 3 个词能看到自己和前面的全部词
    ```

    主对角线以上的上三角区域全是 `True`。这张表叫做**掩码**（mask）。

    ## 动手算：掩码

    对于一段 5 个词的文本，这张掩码表一共屏蔽了多少个“偷看未来”的位置？第 2 个词（从 0 开始编号）连同自己在内能看到几个词？
    """)
    return


@app.cell(hide_code=True)
def _():
    mask_boxes = hand.boxes(['挡住的个数', '第 2 个词能看到'])
    return (mask_boxes,)


@app.cell(hide_code=True)
def _(mask_boxes):
    hand.marked(mask_boxes, {'挡住的个数': 10, '第 2 个词能看到': 3}, {
        '挡住的个数': r"""一行一行数 `True`：第 0 个词被屏蔽 4 个未来位置，第 1 个词 3 个，依此类推（4 + 3 + 2 + 1）""",
        '第 2 个词能看到': r"""第 0、1、2 个词""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：future_mask

    /// admonition | 🧰 工具箱：由 True 和 False 组成的表
    - `torch.ones(2, 3, dtype=torch.bool)` 会生成一张 2 行 3 列、全是 `True` 的布尔表。指定 `dtype=torch.bool` 时，1 存成 `True`，0 存成 `False`。
    - `torch.triu(T, diagonal=1)` 只保留 `T` 主对角线以上的部分，其余位置清零（布尔表里就是 `False`）。（“triu”是 upper triangle，即“上三角”的缩写。）
      例如对三行分别为 `[1, 2, 3]`、`[4, 5, 6]`、`[7, 8, 9]` 的矩阵，它返回 `[0, 2, 3]`、`[0, 0, 6]`、`[0, 0, 0]`。
    - `S.masked_fill(mask, 99.0)` 会返回 `S` 的一份副本，其中 `mask` 为 `True` 的位置全部替换成 `99.0`。
    ///

    请编写 `future_mask(n)`，为 `n` 个词生成上面那样的掩码表：形状为 `n` 行 `n` 列，只有在“看向后面的词”的位置为 `True`。
    """)
    return


@app.function
def future_mask(n):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "future_mask", future_mask, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""先构造一张全是 `True` 的方阵，再只保留主对角线以上的部分。""",
        r"""一行就能写完：用 `torch.triu(...)` 包住一张 `n` 行 `n` 列、`dtype=torch.bool` 的全 1 张量。`diagonal=1` 表示主对角线本身不屏蔽，因为每个词允许看自己。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    使用掩码的方法很巧妙：在做 softmax 之前，把所有被屏蔽位置的分数设为负无穷。因为 $e^{-\infty} = 0$，被屏蔽的词分到的注意力权重正好是 0，而其余词的权重加起来依然等于 1。在 PyTorch 里，负无穷写作 `-torch.inf`。

    ## 改动 3：给分数除以 $\sqrt{d}$

    两个各有 $d$ 个数的向量做点积，相当于把 $d$ 个乘积加在一起。按第 09 章的规律，独立项相加时方差会直接相加（方差是典型大小的平方：$d$ 项相加让方差变 $d$ 倍，开平方后典型大小就变 $\sqrt{d}$ 倍）。所以每个词用 128 个数字时，算出来的点积分数会比用 4 个数字时大得多。

    分数变大有什么问题？分数之间的差距一拉大，softmax 算出来的概率就会迅速向 0 和 1 两端集中。下面的单元格直接测给你看：对每个向量长度 `d`，它随机生成一个查询和 8 个键，重复 200 次，记下 8 个注意力权重里最大的那个——一次用原始分数，一次把分数除以 $\sqrt{d}$。每一行依次输出 `d`、原始分数的平均值和缩放后的平均值：
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
    用原始分数时，最大权重从不到 0.3 一路飙升到接近 0.9：当向量长度达到 512 时，还没开始训练，光靠随机初始化就有一个键几乎抢走了全部注意力。这对训练非常不利——还记得第 07 章吗？当某一项的概率接近 1（其余都接近 0）时，softmax 曲线会变得极平，稍微调整分数几乎改变不了概率，于是其余词几乎收不到梯度，模型很难继续学习。

    而把分数除以 $\sqrt{d}$ 之后，无论向量多长，最大权重都稳稳保持在 0.2 左右。因此自注意力的打分公式是：

    > $$\text{scores} = \frac{Q K^\top}{\sqrt{d}}$$

    ## 练习 2：attend

    /// admonition | 🧰 工具箱：从后往前数维度
    `dim=-1` 表示最后一维，`-2` 表示倒数第二维。`M.transpose(-2, -1)` 用来交换最后两维，不管前面还有多少维都能正常工作；`M.shape[-1]` 则取出最后一维的大小。这样写出来的代码，既能处理 3D 的 `(batch, words, d)`，也能直接处理改动 4 里中间多出一维的 4D 张量。
    ///

    请编写 `attend(Q, K, V, mask)`。它就是第 15 章的 `attend` 加上改动 2 和改动 3，依然返回“加权混合后的结果”和“注意力权重”：

    1. **算分数并缩放**：每个查询与每个键做点积，再除以 $\sqrt{d}$，其中 $d$ 是 `K` 最后一维的大小（`np.sqrt` 可以直接作用在普通数字上）。
    2. **套用掩码**：把 `mask` 为 `True` 的位置替换为 `-torch.inf`。
    3. **算注意力权重 `A`**：沿最后一维做 softmax。
    4. **混合值向量**：用权重 `A` 对 `V` 做加权求和，返回混合结果和 `A`。

    请务必把所有维度都从后往前数，这样当 `Q`、`K`、`V` 的形状是 4D `(batch, heads, words, d)` 时它也能直接运行。

    **关于掩码形状的说明（2D 到 4D 自动广播）：** `future_mask(n)` 返回的是 2D 表格，形状为 `(words, words)`；而在多头模式下，`scores` 是 4D 张量，形状为 `(batch, heads, words, words)`。你不需要手动把掩码扩成 4D——调用 `masked_fill` 时，PyTorch 会从最右边对齐维度，自动把这张 2D 掩码**广播**套用到这一批里的每一个样本、每一个头上。
    """)
    return


@app.function
def attend(Q, K, V, mask):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "attend", attend, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""一共四行。第 1、3、4 行就是第 15 章的那三行，只是把维度编号 `1` 和 `2` 换成 `-2` 和 `-1`；第 2 行用练习 1 工具箱里的 `masked_fill`。""",
        r"""掩码必须加在 softmax 之前的分数上。如果在 softmax 之后才屏蔽，剩下的权重加起来就不等于 1 了。如果检查提示只有数值大小偏了，检查一下有没有除以 `np.sqrt(K.shape[-1])`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 改动 4：多头注意力

    一轮注意力只能给每个词算出一种混合结果。但理解一句话时，一个词往往需要同时关注好几类信息：谁是主语？句子是什么时态？代词指代谁？所以我们并行跑好几轮注意力，每一轮叫做一个**头**（head），各自拿着自己那一份查询、键和值独立关注不同的线索。

    这样做并不会增加每个词的总维度。比如每个词有 128 个数字、分成 4 个头，就把每个词的向量切成 4 段，每段 32 个数字；4 个头各自处理自己那一段，算完后再把 4 段拼回 128 个数字。

    切分不需要额外计算，只靠改变张量形状就能完成。我们跟着一个极小的例子走一遍：1 段文本、2 个词、每个词 4 个数字，切给 2 个头。（`torch.arange(8.0)` 会生成从 `0.0` 到 `7.0` 的一维张量，和 `range` 类似。）下面的单元格依次打印原始张量、`reshape` 之后的张量，以及 `transpose` 之后的张量：
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
    把上面打印出的三个结果对照着看一遍：

    1. 一开始形状是 `(1, 2, 4)`：第 0 个词是 `[0, 1, 2, 3]`，第 1 个词是 `[4, 5, 6, 7]`。
    2. `reshape(1, 2, 2, 2)` 把每个词切成两段，形状变成 `(batch, words, heads, piece)`：第 0 个词拆成 `[0, 1]` 和 `[2, 3]`，第 1 个词拆成 `[4, 5]` 和 `[6, 7]`。
    3. `transpose(1, 2)` 交换 `words` 和 `heads` 两维，形状变成 `(batch, heads, words, piece)`：现在第 0 个头拿到了第 0 个词的 `[0, 1]` 和第 1 个词的 `[4, 5]`，正好是每个词的前一半；第 1 个头拿到了 `[2, 3]` 和 `[6, 7]`，正好是每个词的后一半。

    **为什么不能省掉 `transpose`，直接一步 `reshape(batch, heads, words, piece)`？** 因为 `reshape` 永远不会重新排列底层数字，它只是按顺序把数字重新分组。拿上面的 `[0, 1, 2, 3], [4, 5, 6, 7]` 来说，如果一步 `reshape(1, 2, 2, 2)` 并把第 1 维当成 `heads`，第 0 个头会直接吞掉整句第 0 个词 `[0, 1, 2, 3]`（完全看不到第 1 个词），第 1 个头会吞掉整个第 1 个词 `[4, 5, 6, 7]`，而不是每个词的前一半 `[0, 1]` 和 `[4, 5]`！所以必须先在词内切段（`reshape` 成 `(batch, words, heads, piece)`），再用 `transpose(1, 2)` 把 `heads` 维换到前面。

    那为什么要把 `heads` 换到前面呢？因为矩阵乘法 `@` 只作用在最后两维，把前面所有维都当成批次并行处理。把 `heads` 挪到前面、紧挨着 `batch` 之后，你刚写的 `attend` 就会把每个头当成一段独立的文本并行计算，而且 2D 掩码会自动广播到每一个头上，一行代码都不用改。练习 2 要求从后往前数维度，正是为了这一刻。

    ## 动手算：切分后的形状

    一个形状为 `(1, 6, 12)` 的张量（1 段文本，6 个词，每个词 12 个数字）要切分给 3 个头。请按顺序填出最终结果的四个维度大小。
    """)
    return


@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['大小 0', '大小 1', '大小 2', '大小 3'])
    return (shape_boxes,)


@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'大小 0': 1, '大小 1': 3, '大小 2': 6, '大小 3': 4}, {
        '大小 0': r"""批次（`batch`）留在最前面""",
        '大小 1': r"""交换之后，`heads` 排在第二维""",
        '大小 2': r"""接着是词数 `words`""",
        '大小 3': r"""每段长度是 12 个数字平均分给 3 个头""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：split_heads

    请编写 `split_heads(x, heads)`：输入 `x` 的形状是 `(batch, words, d)`，返回切分后的形状 `(batch, heads, words, d // heads)`，步骤与上面的例子完全一致。（`//` 是整除运算符，比如 `12 // 3` 等于 `4`。）

    用 `batch, words, d = x.shape` 可以把三个维度的大小分别取出并命名。

    对上面的例子调用 `split_heads(x_trace, 2)`，应该正好得到第三个结果。
    """)
    return


@app.function
def split_heads(x, heads):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "split_heads", split_heads, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""和上面的例子一样分两步：先 `reshape` 成 `(batch, words, heads, piece)`，再交换第 1、2 维。""",
        r"""记住顺序：先 `reshape`，再 `transpose`。如果直接一步 `reshape` 成 `(batch, heads, words, piece)`，形状虽然看着对，但里面的数字全进错头了。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：join_heads

    各个头分别算完注意力之后，需要把它们重新拼回完整的词向量。请编写 `join_heads(x)`，它是 `split_heads` 的逆操作：接收形状 `(batch, heads, words, piece)`，返回 `(batch, words, heads * piece)`。

    把切分时的两步按相反顺序撤销即可：先撤销维度交换，再把多头拼回一维。用 `batch, heads, words, piece = x.shape` 可以先取出四个维度的大小。

    对上面例子里的第三个结果调用 `join_heads`，应该能完整还原出第一个结果。
    """)
    return


@app.function
def join_heads(x):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "join_heads", join_heads, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`transpose(1, 2)` 自己就能撤销自己：把第 1 维和第 2 维再交换一次，就换回原来的顺序了。""",
        r"""两行搞定：先解包取出四个维度大小，再先 transpose、后 reshape 成 `(batch, words, heads * piece)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：组装完整的自注意力层

    现在把四处改动合为一体。请编写 `self_attention_step(x, to_query, to_key, to_value, mix, heads)`。其中 `x` 的形状是 `(batch, words, d)`，后面四个参数都是神经网络层，像普通函数一样调用即可（例如 `to_query(x)`）：

    1. **生成多头查询、键、值**：分别把 `x` 送进对应的层，再切分成多个头。
    2. **计算注意力**：用 `attend` 计算结果，掩码按当前文本的词数（即 `x.shape[1]`）生成。
    3. **合并并融合多头**：把混合结果的多个头拼回原状，再送进最后一层 `mix`，让不同头提取到的信息彼此融合。
    4. **返回结果**：返回融合后的张量和注意力权重。

    | 名称 | 形状 |
    |---|---|
    | `x` | `(batch, words, d)` |
    | 查询、键、值 | `(batch, heads, words, d // heads)` |
    | 注意力权重 | `(batch, heads, words, words)` |
    | 输出结果 | `(batch, words, d)` |
    """)
    return


@app.function
def self_attention_step(x, to_query, to_key, to_value, mix, heads):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "self_attention_step", self_attention_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""每一步用的都是你在这一章刚写好的函数：`split_heads`、`future_mask`、`attend`、`join_heads`。五行代码就够：三行算 Q/K/V，一行调 `attend`，一行合并返回。""",
        r"""`attend` 返回一对值：加权混合结果和注意力权重。只有混合结果需要拼回多头并送进 `mix`，注意力权重直接原样返回。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面把你的函数封装成一个标准的 PyTorch 模块。它的可训练参数是三个角色的线性层以及 `mix`，`forward` 方法直接调用你写的函数，并把最新算出的注意力权重保存在 `self.shares` 里（第 17 章会拿出来可视化）。紧接着的单元格会新建一个模块，输入一段 5 个词、每个词 8 个数字的文本（分成 2 个头），并打印输出和权重的形状。在上面五个练习全部通过之前，后面的单元格会报错，这是正常的。
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
    输出形状是 `[1, 5, 8]`，权重形状是 `[1, 2, 5, 5]`：输出与输入的形状完全一致，所以这一层可以像积木一样一层层往上叠；注意力权重则是 2 个头各有一张 5 行 5 列的表。

    ## 注意力分不清词序

    不过还漏了一件很关键、却容易被忽略的事。拿上面这 5 个词向量，关掉掩码跑一次注意力；然后把 5 个词的顺序整个倒过来，再跑一次注意力，再把输出顺序倒回去。因为每个词问的还是同一批词，算出来的混合结果应该一模一样。下面的单元格打印两次结果之间的最大差值：
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
    差值大约是 `1e-07`（也就是 0.0000001）：这只是浮点数的舍入误差，说明两次输出完全相同。在没有掩码的注意力眼里，`the dog bit the man`（狗咬了人）和 `the man bit the dog`（人咬了狗）就像同一个词袋（bag of words），根本分不清谁先谁后。LSTM 因为逐词按顺序读取，天生知道词序；而自注意力是同时看所有词的，所以我们必须显式地把位置信息加进去。

    ## 接下来

    核心层已经造好了。**第 17 章**会教模型分辨每个词排在什么位置、保证多层叠加后数值依然稳定，然后把这一层叠成一个大约一百万参数的微型 GPT，并用故事数据训练它。

    ## 你现在会做的事

    - 说清为什么每个词需要拆出独立的查询、键和值
    - 用掩码屏蔽未来的词，并理解 2D 掩码如何自动广播到 4D 分数张量
    - 解释为什么点积分数要除以 $\sqrt{d}$
    - 把词向量切成多个头再拼回去，并明白为什么不能跳过 `transpose` 一步 `reshape`
    - 用这些部件组装出完整的多头自注意力层
    - 解释为什么光靠注意力本身分不清词序

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** Transformer 出自论文 *Attention Is All You Need*（Vaswani 等，2017），其中这一层被称为*缩放点积注意力*（scaled dot-product attention）与*多头注意力*（multi-head attention）。如果去掉未来掩码、允许每个词同时看向左右两边，那就是 BERT 所使用的双向自注意力（第 18 章）。
    """)
    return


if __name__ == "__main__":
    app.run()
