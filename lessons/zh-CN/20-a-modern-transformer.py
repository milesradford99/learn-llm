import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="20 · 现代 Transformer")

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
    from learn_llm.diagrams import circle
    from learn_llm.tokenizer import Tokenizer, story_words

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 20 · 现代 Transformer

    第 17 章的模型是 2017 年的设计。今天的大模型保留了它的骨架，也就是在 $x + f(x)$ 这条路上交替的注意力和思考，但几乎每个部件后来都换成了更好的。这一章在块的内部做其中三处替换：

    1. **RMSNorm**，更简单的归一化；
    2. **旋转位置编码**（RoPE），取代位置表；
    3. **SwiGLU**，带门的思考层。

    你会把每个部件造出来，把它们拼成一个块，再检查新模型的形状和大小。然后由**第 21 章**来训练它，那里还有两处属于训练循环的替换。另外，模型现在读的是**词元**（token），用的是第 19 章的方法。

    ## 词元，以及公平的排行榜

    课程自带 2,000 次合并，是完全按第 19 章的方法在这些故事上训练出来的。在这么多文本上用你的写法训练要花不少时间，所以 `Tokenizer` 用两个标准技巧得到同样的合并，只是更快：它统计词而不是反复重读全文，并且每个词编码一次之后就记住结果。合并从不跨过一个词的开头，所以 `the` 和 ` cat` 绝不会粘成一块。

    有一条规则让排行榜保持公平。之前的模型读的是最常见的 2,000 个词，其他词都写成 `<unk>`。如果词元模型得把这些少见的词拼出来，而词模型只写一个 `<unk>`，比较就对它不公平：我们测过，这会让困惑度（perplexity）高出大约 40%。所以词元模型读的是完全相同的文本，少见的词同样是 `<unk>`，分数仍然是总意外程度除以**词**数，正如第 19 章承诺的那样。

    /// admonition | 🧰 工具箱：集合
    `set(items)` 用一个列表造出一个**集合**（set）：它能一下子回答 `word in known`，就像只有键、没有值的字典。`"dog" in set(["dog", "cat"])` 是 `True`。
    ///

    下面的单元格和以前完全一样地数词（`re.findall` 列出每一串字母和撇号，就像第 12 章的 `to_words`），要花几秒钟。
    """)
    return


@app.cell
def _():
    train_stories, test_stories = stories()
    counts = {}
    for story in train_stories:
        for word in re.findall(r"[a-z']+", story.lower()):
            counts[word] = counts.get(word, 0) + 1
    known = set(sorted(counts, key=counts.get, reverse=True)[:2000])
    tokenizer = Tokenizer.load()
    train_tokens = np.array(tokenizer.encode(story_words(train_stories, known)))
    test_words = story_words(test_stories, known)
    test_tokens = np.array(tokenizer.encode(test_words))
    word_count = sum([len(words) for words in test_words]) + len(test_words)
    tokenizer.size, len(train_tokens), len(test_tokens), word_count
    return test_tokens, tokenizer, train_tokens, word_count


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    共 2,258 个词元编号（256 个字节、2,000 次合并、`<end>` 和 `<unk>`）。测试故事变成大约 167,000 个词元，对应 159,000 个词（和以前一样，每个故事的 `<end>` 算一个词）：平均每个词略多于一个词元，因为这 2,000 个词大多是单个词元，只有少数被切成几段。

    ## 1. RMSNorm

    第 17 章的层归一化（layer normalisation）先减去均值，再除以典型大小。研究者发现减均值的作用不大，所以 **RMSNorm**（均方根归一化）省掉了这一步：把每个向量除以它的**均方根**（root mean square），再乘以向量里每个数各自学出来的**增益**（gain）：

    > $$\text{rms} = \sqrt{x^2 \text{ 的均值} + 0.000001} \qquad \text{rms\_norm}(x) = \text{增益} \times \frac{x}{\text{rms}}$$

    那个很小的数只是防止除以零。它比层归一化少一步，所以快一点，效果一样好。

    ## 动手算：一个向量

    把 $x = [1, 7]$ 归一化，两个数的增益都是 1。算出平方的均值、均方根，以及两个结果。（那个很小的数不用管。）
    """)
    return


@app.cell(hide_code=True)
def _():
    rms_boxes = hand.boxes(['平方的均值', '均方根', '第一个结果', '第二个结果'], step=0.01)
    return (rms_boxes,)


@app.cell(hide_code=True)
def _(rms_boxes):
    hand.marked(rms_boxes, {'平方的均值': 25, '均方根': 5, '第一个结果': 0.2, '第二个结果': 1.4}, {
        '平方的均值': r"""$(1 \times 1 + 7 \times 7) / 2$""",
        '均方根': r"""你第一个答案的平方根""",
        '第一个结果': r"""$1 / \text{rms}$""",
        '第二个结果': r"""$7 / \text{rms}$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    结果 0.2 和 1.4 的均方根正好是 1：除这一下就是为了这个。向量里各个数的正负和比例都没变，它的均值也没有被挪到 0。

    ## 练习 1：rms_norm

    写 `rms_norm(x, gain)`。`x` 是任意形状的张量，最后一维有 `d` 个数；`gain` 的形状是 `(d,)`。沿最后一维的每个向量都按上面的公式归一化，所以结果的形状和 `x` 一样。在代码里，那个很小的数写作 `1e-6`，放在平方根 `torch.sqrt` 的里面。

    你动手算的例子就是其中一项检查：`x = [[1.0, 7.0]]`，`gain = [1.0, 1.0]`，得到 `[[0.2, 1.4]]`。
    """)
    return


@app.function
def rms_norm(x, gain):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "rms_norm", rms_norm, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""这就是第 17 章的 `layer_norm`，只是东西更少：沿最后一维做一次 `torch.mean`，这次是对 `x * x`，而且不做减法。均值要加 `dim=-1` 和 `keepdim=True`，这样每个向量除以的才是它自己的那个数。""",
        r"""一共 2 行。第一行算出 `rms`：对均值加 `1e-6` 的结果取 `torch.sqrt`。第二行返回增益乘以 `x`，再除以 `rms`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. 旋转

    第 17 章给每个位置一个学出来的向量，这有两个弱点：如果只在 32 个位置上训练过，模型对第 33 个位置一无所知；而且“第 7 个词看第 5 个词”和“第 27 个词看第 25 个词”得分别学，尽管它们是同一种关系：往回两个词。旋转位置编码解决了这两个问题。它需要几何里的一个想法。

    **角、余弦和正弦。** 以一个点为中心，画一个半径为 1 的圆。从最右边的点 $(1, 0)$ 出发，沿着圆逆时针走。走过多少就是一个**角**（angle），记作 $\theta$（希腊字母，读作 theta）。这时你的位置有两个坐标，从圆心量起：往右多远，叫这个角的**余弦**（cosine）；往上多远，叫它的**正弦**（sine）。负数表示往左或往下。
    """)
    return


@app.cell(hide_code=True)
def _():
    circle(0.9, '一个半径为 1 的圆。一个点从 (1, 0) 出发，沿着圆走过了角 theta。它在圆心右边的距离是 cos theta，在圆心上方的高度是 sin theta。')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    所以角 $\theta$ 处的点是 $(\cos\theta, \sin\theta)$。角为 0 时，你在 $(1, 0)$。走过四分之一圈，你在顶上，$(0, 1)$。我们用**弧度**（radian）量角：一整圈是 $2\pi$（$\pi$ 大约是 3.14），大约 6.28，所以四分之一圈是 $\pi / 2$，大约 1.57。

    /// admonition | 🧰 工具箱：NumPy 里的 cos、sin 和 π
    `np.cos(angle)` 和 `np.sin(angle)` 算出这两个值，可以算一个角，也可以算一整个数组的角；`np.pi` 就是 $\pi$。
    ///
    """)
    return


@app.cell
def _():
    [(round(np.cos(angle).item(), 3), round(np.sin(angle).item(), 3)) for angle in [0, np.pi / 2, np.pi, 0.9]]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这四个点分别是最右边、顶上、最左边，以及图里的那个点：走过了 0.9 弧度。下面把它们列成一张表，后面要用：

    | 转多少 | 角 $\theta$ | $\cos\theta$ | $\sin\theta$ |
    |---|---|---|---|
    | 不转 | 0 | 1 | 0 |
    | 四分之一圈 | $\pi / 2$ | 0 | 1 |
    | 半圈 | $\pi$ | −1 | 0 |

    **旋转一对数。** 把任意一个点 $(a, b)$ 绕圆心转过角 $\theta$，几何里的规则是：

    > $$(a, b) \;\to\; (a\cos\theta - b\sin\theta,\;\; a\sin\theta + b\cos\theta)$$

    ## 动手算：转三个点

    用这条规则和上面的表。把 $(1, 0)$ 转四分之一圈。把 $(0, 1)$ 转四分之一圈。再把 $(3, 4)$ 转半圈。
    """)
    return


@app.cell(hide_code=True)
def _():
    turn_boxes = hand.boxes(['(1, 0)：新的第一个数', '(1, 0)：新的第二个数', '(0, 1)：新的第一个数', '(0, 1)：新的第二个数', '(3, 4)：新的第一个数', '(3, 4)：新的第二个数'])
    return (turn_boxes,)


@app.cell(hide_code=True)
def _(turn_boxes):
    hand.marked(turn_boxes, {'(1, 0)：新的第一个数': 0, '(1, 0)：新的第二个数': 1, '(0, 1)：新的第一个数': -1, '(0, 1)：新的第二个数': 0, '(3, 4)：新的第一个数': -3, '(3, 4)：新的第二个数': -4}, {
        '(1, 0)：新的第一个数': r"""$1 \times \cos\theta - 0 \times \sin\theta$，其中 $\cos\theta = 0$""",
        '(1, 0)：新的第二个数': r"""$1 \times \sin\theta + 0 \times \cos\theta$，其中 $\sin\theta = 1$""",
        '(0, 1)：新的第一个数': r"""$0 \times 0 - 1 \times 1$""",
        '(0, 1)：新的第二个数': r"""$0 \times 1 + 1 \times 0$""",
        '(3, 4)：新的第一个数': r"""$3 \times (-1) - 4 \times 0$""",
        '(3, 4)：新的第二个数': r"""$3 \times 0 + 4 \times (-1)$""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    对照着图看一看。最右边的点到了顶上。顶上的点到了最左边，比它原来多转了四分之一圈。而转半圈会把任何一个点送到圆心的另一侧。

    ## 练习 2：rotate

    写 `rotate(a, b, angle)`，也就是上面的规则，用 NumPy 的 `np.cos` 和 `np.sin`。它返回转过之后的一对数：先是新的第一个数，再是新的第二个数。`a` 和 `b` 可以是单个的数，也可以是形状相同的数组。

    > `rotate(1.0, 0.0, np.pi / 2)` 得到 `(0.0, 1.0)`，只差一点舍入误差
    """)
    return


@app.function
def rotate(a, b, angle):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "rotate", rotate, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""两个表达式，规则的两半各一个，中间用逗号隔开，一起返回。两个都用旧的 `a` 和旧的 `b`。""",
        r"""留意负号：只有前一半有。如果你先把新的第一个数存进 `a`，再去算第二个数，第二个数用到的 `a` 就不对了。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在来看让旋转特别适合表示位置的那个性质。回想一下，注意力分数是查询和键的点积。取两对数 `q` 和 `k`，把 `q` 转一个角，把 `k` 转另一个角，再算它们的点积。结果只取决于两个角的**差**，和角本身无关。两个都再多转 1.5，结果不变：
    """)
    return


@app.cell
def _():
    q = (0.8, -0.3)
    k = (0.5, 0.9)

    def turned_dot(angle_q, angle_k):
        qa, qb = rotate(q[0], q[1], angle_q)
        ka, kb = rotate(k[0], k[1], angle_k)
        return round((qa * ka + qb * kb).item(), 6)

    turned_dot(0.7, 0.2), turned_dot(2.2, 1.7)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    数字相同，因为把两对数一起转动，并不改变它们之间的相对位置。

    ## 旋转位置编码

    于是：把每个查询和键切成一对一对的数，第 $p$ 个位置的词，它的每对数都转过 $p$ 乘以某个角。第 7 个词问第 5 个词时，分数取决于 $7 - 5 = 2$ 步的角度。第 27 个词问第 25 个词，结果完全一样。注意力现在看到的是相对位置，完全不需要位置表，而且超出训练窗口的位置至少也能算出来（不过模型在接近训练长度时效果最好）。

    每一对数有自己的**转速**，单位是每个位置多少弧度：转得快的那些对能分清近处的位置，转得慢的能分清远处的。一个有 4 个数的向量 $[x_0, x_1, x_2, x_3]$ 有 2 对。前一半放的是每一对的第一个数，后一半放的是第二个数，所以这两对是 $(x_0, x_2)$ 和 $(x_1, x_3)$：

    | 第几对 | 它的两个数 | 转过的角 |
    |---|---|---|
    | 0 | $(x_0, x_2)$ | $\text{位置} \times \text{转速}_0$ |
    | 1 | $(x_1, x_3)$ | $\text{位置} \times \text{转速}_1$ |

    ## 练习 3：rope_one

    写 `rope_one(x, position, speeds)`，处理一个词的向量。`x` 是一个 NumPy 向量，里面的数是偶数个；`position` 是一个整数；`speeds` 是一个 NumPy 向量，长度是 `x` 的一半。它返回一个和 `x` 一样大的向量：先是所有转过之后的第一个数，再是所有转过之后的第二个数。

    转动交给你的 `rotate`，所有的对一次转完：把 `x` 的前一半当作 `a`，后一半当作 `b`，再给它一个由角组成的数组。`np.concatenate`（第 14 章）把两半重新接起来。

    > `rope_one(np.array([1.0, 0.0, 0.0, 1.0]), 1, np.array([np.pi / 2, np.pi]))` 得到 `[0, 0, 1, -1]`，只差一点舍入误差

    在这个例子里，第 0 对是 $(1, 0)$，转四分之一圈变成 $(0, 1)$；第 1 对是 $(0, 1)$，转半圈变成 $(0, -1)$。
    """)
    return


@app.function
def rope_one(x, position, speeds):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "rope_one", rope_one, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""一共有 `len(x) // 2` 对。`x` 的前一半是到这个数为止的切片，后一半是从这个数开始的切片。角是 `position * speeds`：每一对一个角。""",
        r"""可以写成 3 行：`half = len(x) // 2`，然后 `a, b = rotate(...)`，括号里是两个切片和那些角，最后返回 `np.concatenate([a, b])`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面的单元格在你的函数上检验这个性质，用的是真正的转速：最快的一对每个位置转 1 弧度，最慢的一对大约 1/10,000 弧度。（10,000 是约定俗成的取值。）它把一个有 8 个数的查询和一个键转过之后算点积，算两次：一次是位置 7 和 5，一次是位置 27 和 25。
    """)
    return


@app.cell
def _():
    rng = np.random.default_rng(0)
    q_demo, k_demo = rng.random(8), rng.random(8)
    speeds_demo = 1 / 10000 ** (np.arange(4) / 4)
    near = rope_one(q_demo, 7, speeds_demo) @ rope_one(k_demo, 5, speeds_demo)
    far = rope_one(q_demo, 27, speeds_demo) @ rope_one(k_demo, 25, speeds_demo)
    round(near.item(), 6), round(far.item(), 6)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    分数相同：往回两个词就是往回两个词，不管发生在哪里。

    下面的 `rope` 就是用在整个张量上的 `rope_one`：一次处理每段文本的每个位置，用 PyTorch 写，位置和转速都在里面算好。你不用自己写它，但有了下面两个工具，你能读懂它：

    /// admonition | 🧰 工具箱：两个索引工具
    - `x[..., :half]` 表示“最后一维的前 `half` 项，前面有什么维度都不管”。三个点代表前面所有的维度。
    - `v[:, None]` 把形状为 `(n,)` 的向量变成形状为 `(n, 1)` 的一列，`v[None, :]` 把它变成形状为 `(1, n)` 的一行。一列乘一行，得到一张形状为 `(n, m)` 的表：在这里，每个位置都和每种转速相遇。
    ///
    """)
    return


@app.function
def rope(x):
    words, size = x.shape[-2], x.shape[-1]
    half = size // 2
    speeds = 1 / 10000 ** (torch.arange(half) / half)
    angles = torch.arange(words)[:, None] * speeds[None, :]
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * torch.cos(angles) - b * torch.sin(angles), a * torch.sin(angles) + b * torch.cos(angles)], dim=-1)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：读一读 rope

    假设 `x` 的形状是 `(2, 4, 16, 32)`：2 段文本，4 个头，16 个词，每个头 32 个数。每个向量有多少对？`angles` 的两个大小，也就是它的行数和列数，各是多少？`rope` 让哪个位置上的向量保持原样？
    """)
    return


@app.cell(hide_code=True)
def _():
    rope_boxes = hand.boxes(['对数', 'angles 的行数', 'angles 的列数', '保持原样的位置'])
    return (rope_boxes,)


@app.cell(hide_code=True)
def _(rope_boxes):
    hand.marked(rope_boxes, {'对数': 16, 'angles 的行数': 16, 'angles 的列数': 16, '保持原样的位置': 0}, {
        '对数': r"""`half` 是最后一个大小除以 2""",
        'angles 的行数': r"""每个位置一行：`words` 是 `x.shape[-2]`""",
        'angles 的列数': r"""每种转速一列，一共有 `half` 种转速""",
        '保持原样的位置': r"""哪个位置让每种转速对应的角都是 0？""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. SwiGLU：带门的思考

    第 17 章的思考层是 $W_2(\text{relu}(W_1 x))$。两处改动让它更好。第一，更平滑的弯折：**SiLU**，即 $x \times \sigma(x)$，用的是第 12 章的 sigmoid。$x$ 很大时它约等于 $x$，$x$ 很负时约等于 0，和 ReLU 一样，只是没有尖角。在 PyTorch 里它是 `F.silu`。第二，类似第 14 章的门：第三个层 $W_3$ 逐个数字地决定第一个层有多少能通过。（它没有被压到 0 到 1 之间，所以只是像门，而不是严格意义上的门。）这里的 $\times$ 是逐个位置相乘：

    > $$\text{思考}(x) = W_2\big(\text{silu}(W_1 x) \times W_3 x\big)$$

    三个矩阵会比两个矩阵多出旋钮，所以隐藏层的宽度改成向量的 $\tfrac{8}{3}$ 倍，而不是 4 倍，让大小基本不变。

    ## 动手算：数一数旋钮

    我们模型的向量有 $d = 128$ 个数。第 17 章的思考层有两个矩阵：$128 \times 512$ 和 $512 \times 128$。新的隐藏层宽度是 $8 \times 128 / 3$，向下取整。它是多少？三个新矩阵一共有多少个旋钮（没有偏置）？第 17 章的那两个矩阵又有多少个（同样不算偏置）？
    """)
    return


@app.cell(hide_code=True)
def _():
    knob_boxes = hand.boxes(['隐藏层宽度', '三个矩阵的旋钮数', '原来两个矩阵的旋钮数'])
    return (knob_boxes,)


@app.cell(hide_code=True)
def _(knob_boxes):
    hand.marked(knob_boxes, {'隐藏层宽度': 341, '三个矩阵的旋钮数': 130944, '原来两个矩阵的旋钮数': 131072}, {
        '隐藏层宽度': r"""$1024 / 3$ 是 341.33…""",
        '三个矩阵的旋钮数': r"""三个矩阵每个都有 $128 \times 341$ 个旋钮""",
        '原来两个矩阵的旋钮数': r"""两个矩阵每个都有 $128 \times 512$ 个旋钮""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    两者相差不到千分之一：同样的价钱，多了一个门。

    ## 练习 4：swiglu

    写 `swiglu(x, W1, W2, W3)`，也就是上面的公式。三个都是 `nn.Linear` 层：`W1` 和 `W3` 接收 `d` 个数，给出隐藏层的宽度；`W2` 接收隐藏层的宽度，给出 `d`。`x` 的最后一维有 `d` 个数，结果也是。
    """)
    return


@app.function
def swiglu(x, W1, W2, W3):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "swiglu", swiglu, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""层像函数一样调用，比如 `W1(x)`。把公式从里往外读：两个层都看 `x`，其中一个的结果经过 `F.silu`，两者用 `*` 相乘，`W2` 再处理这个乘积。""",
        r"""只要一行，开头是 `return W2(`。SiLU 只作用在 `W1(x)` 上：`W3(x)` 是门，保持原样。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 现代的块

    现在把这三处替换装进第 17 章的块。它的计划没有变：两个部分，每个部分都在一条能绕过它的路上，每个部分之前都先做归一化。

    | 步骤 | 第 17 章 | 现在 |
    |---|---|---|
    | 归一化 | 层归一化 | `rms_norm`，用块的第一个增益 |
    | 看 | 注意力，第一个块之前加位置表 | 注意力，对查询和键做 `rope` |
    | 相加 | $x + \text{看}$ | 不变 |
    | 归一化 | 层归一化 | `rms_norm`，用块的第二个增益 |
    | 思考 | $W_2(\text{relu}(W_1 x))$ | `swiglu` |
    | 相加 | $x + \text{思考}$ | 不变 |

    ## 练习 5：modern_block_step

    写 `modern_block_step(x, gain1, look, gain2, W1, W2, W3)`，也就是右边那一列。`x` 的形状是 `(batch, words, d)`。`look` 是一个函数：它接收这种形状的张量，返回注意力的结果，形状不变；像调用别的函数一样调用它就行。两个增益的形状是 `(d,)`，三个层是 SwiGLU 的。它返回块的输出，形状和 `x` 一样。

    和第 17 章一样，后一半从前一半的结果出发：第二次归一化的是*新的* `x`。
    """)
    return


@app.function
def modern_block_step(x, gain1, look, gain2, W1, W2, W3):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "modern_block_step", modern_block_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""两行，表格的上下两半各一行。每一行都是 `x` 加上一样东西：先是归一化后的 `x` 交给 `look` 的结果，再是归一化后的 `x` 交给 `swiglu` 的结果。用上你的 `rms_norm` 和 `swiglu`。""",
        r"""第一行是 `x = x + look(rms_norm(x, gain1))`。第二行返回新的 `x`，加上把 `rms_norm(x, gain2)` 和三个层交给 `swiglu` 的结果。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面把块和模型写成 PyTorch 模块。`ModernBlock` 装着旋钮；它的 `look` 是第 17 章那种分成几个头的注意力，只是对查询和键做了 `rope`；它的 `forward` 就是你的函数。`attend` 是第 16 章的，掩码已经写在里面。有三点值得注意：

    - `nn.Parameter(tensor)` 把一个张量变成一个独立的旋钮，训练时会移动它。这里每个增益都是这样一个旋钮，起点全是 1。
    - 模型没有位置表：把 `ModernGPT` 和第 17 章的 `TinyGPT` 比一比。
    - 最后一层和注意力层都没有偏置，现代模型通常不要它们。

    五个练习全部通过之前，这两个单元格后面的那个单元格会显示错误，这是正常的。
    """)
    return


@app.function
def attend(Q, K, V):
    words = Q.shape[-2]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    mask = torch.triu(torch.ones(words, words, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1) @ V


@app.cell
def _():
    class ModernBlock(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.heads = heads
            self.gain1 = nn.Parameter(torch.ones(d))
            self.gain2 = nn.Parameter(torch.ones(d))
            self.query = nn.Linear(d, d, bias=False)
            self.key = nn.Linear(d, d, bias=False)
            self.value = nn.Linear(d, d, bias=False)
            self.mix = nn.Linear(d, d, bias=False)
            hidden = 8 * d // 3
            self.W1 = nn.Linear(d, hidden, bias=False)
            self.W2 = nn.Linear(hidden, d, bias=False)
            self.W3 = nn.Linear(d, hidden, bias=False)

        def look(self, y):
            batch, words, d = y.shape

            def heads_of(layer):
                return layer(y).reshape(batch, words, self.heads, d // self.heads).transpose(1, 2)

            blended = attend(rope(heads_of(self.query)), rope(heads_of(self.key)), heads_of(self.value))
            return self.mix(blended.transpose(1, 2).reshape(batch, words, d))

        def forward(self, x):
            return modern_block_step(x, self.gain1, self.look, self.gain2, self.W1, self.W2, self.W3)

    class ModernGPT(nn.Module):
        def __init__(self, vocab_size, d, heads, layers):
            super().__init__()
            self.words = nn.Embedding(vocab_size, d)
            self.blocks = nn.Sequential(*[ModernBlock(d, heads) for i in range(layers)])
            self.gain = nn.Parameter(torch.ones(d))
            self.out = nn.Linear(d, vocab_size, bias=False)

        def forward(self, ids):
            return self.out(rms_norm(self.blocks(self.words(ids)), self.gain))
    return (ModernGPT,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 拼得上吗？

    先不训练：先确认各个部件拼得上。下面的单元格造一个和第 17 章一样大的模型：每个词元 128 个数字，4 个头，3 个块。然后把前 64 个测试词元重复一遍，作为大小为 2 的一批喂给它。它显示分数的形状、旋钮的个数和损失。

    看结果之前先想一想：没训练过的模型一无所知，所以它的犹豫程度应该相当于在全部 2,258 个词元里选一个。按第 13 章，这时的意外程度是 $\log 2258$，大约 7.7。
    """)
    return


@app.cell
def _(ModernGPT, test_tokens, tokenizer):
    torch.manual_seed(0)
    model = ModernGPT(tokenizer.size, 128, 4, 3)
    ids_demo = torch.tensor(np.array([test_tokens[:64], test_tokens[:64]]))
    with torch.no_grad():
        S_demo = model(ids_demo)
        loss_demo = F.cross_entropy(S_demo[:, :-1].reshape(-1, tokenizer.size), ids_demo[:, 1:].reshape(-1))
    S_demo.shape, sum([p.numel() for p in model.parameters()]), round(loss_demo.item(), 2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    分数的形状是 `(2, 64, 2258)`：2 段文本，每段 64 个位置，每个位置上每个词元一个分数。旋钮大约 117 万个，和第 17 章的模型差不多。损失大约是 7.9，和预计的 7.7 很接近。（略高一点，是因为随机的分数并不完全均匀。）这是个好迹象，说明没有哪里接错：一开始的数字要是大得多，通常就是出了错。

    ## 接下来

    模型造好了，但它还没读过一个故事。**第 21 章**会好好训练它：用更干净地缩小旋钮的 **AdamW**，用学习率的**调整计划**（schedule），再加上**检查点**（checkpoint），让一次训练可以保存下来、以后接着跑。然后它就上排行榜。

    ## 你现在会做的事

    - 解释为什么词元模型必须和词模型读同样的文本
    - 用手算和代码做 RMSNorm 归一化
    - 解释角、余弦和正弦，并旋转一对数
    - 解释为什么旋转位置编码让注意力看到相对位置，并把它用在一个向量上
    - 写一个带门的 SwiGLU 层，并数出它的旋钮
    - 把各个部件拼成一个现代的块，并检查未训练模型的形状和损失

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** RMSNorm 出自 Zhang 和 Sennrich（2019），RoPE 出自 Su 等（*RoFormer*，2021），SwiGLU 出自 Shazeer（*GLU Variants Improve Transformer*，2020）。LLaMA 这样的模型正是把这些选择组合在一起。很多模型还会*绑定*（tie）第一层和最后一层，共用一张词元向量表。
    """)
    return


if __name__ == "__main__":
    app.run()
