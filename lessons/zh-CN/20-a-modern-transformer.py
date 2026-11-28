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

    第 17 章的模型用的是 2017 年的经典设计。今天的大模型依然沿用它的整体骨架——不断交替执行“注意力层”和“思考层”，并借助残差连接 $x + f(x)$ 把信息一路传下去——但里面的几乎每个部件都升级成了更好的版本。这一章我们先在 Transformer 块内部完成三处核心替换：

    1. **RMSNorm**，更简洁的归一化；
    2. **旋转位置编码**（RoPE），取代位置表；
    3. **SwiGLU**，带门控机制的思考层。

    这一章里，你会亲手实现这三个部件，把它们拼成一个现代 Transformer 块，再检查整个新模型的输出形状和参数规模。下一章（**第 21 章**）再来训练它，并补上另外两处属于训练循环的升级。此外，从现在起，模型读取的输入改成了第 19 章学过的**词元**（token）。

    ## 词元，以及公平的排行榜

    课程已经按第 19 章的方法，在这些故事上预先训练好了 2,000 次合并。直接用上一章的朴素代码在全部故事上重跑一遍会比较慢，所以这里提供的 `Tokenizer` 用了两个常见加速技巧来得到完全相同的合并结果：它先统计每个词出现的次数，而不是反复重读全文；而且每个词编码过一次就会缓存结果。合并规则从不跨过词的开头，所以 `the` 和 ` cat` 绝不会粘成一个词元。

    为了保证排行榜公平，这里定了一条规则：之前的词模型只认识最常见的 2,000 个词，其余生僻词统统替换成 `<unk>`。如果让词元模型去硬拼这些生僻词，而词模型只需猜一个 `<unk>`，那比拼就不公平了（我们实测过，这会让困惑度高出大约 40%）。因此，词元模型读的也是完全相同的文本——生僻词同样先变成 `<unk>`——而且评分标准依然是总意外程度除以**词**数，和第 19 章约定的一样。

    /// admonition | 🧰 工具箱：集合
    `set(items)` 用一个列表造出一个**集合**（set）：它能一下子回答 `word in known`，就像只有键、没有值的字典。`"dog" in set(["dog", "cat"])` 是 `True`。
    ///

    下面的单元格和之前一样先统计词频（`re.findall` 提取出每一串连续的字母和撇号，和第 12 章的 `to_words` 相同），运行大概需要几秒钟。
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

    第 17 章的层归一化（LayerNorm）先减去均值，再除以标准差来缩放数值大小。后来研究者发现，减去均值这一步其实作用不大，于是 **RMSNorm**（均方根归一化）干脆省掉了减均值：直接把向量里的每个数除以整个向量的**均方根**（RMS），再乘上每个位置各自学出来的缩放系数，叫做**增益**（gain）：

    > $$\text{rms} = \sqrt{x^2 \text{ 的均值} + 0.000001} \qquad \text{rms\_norm}(x) = \text{增益} \times \frac{x}{\text{rms}}$$

    那个极小的数 $0.000001$ 只是为了防止分母为零。RMSNorm 比层归一化少了一步计算，速度更快，效果却一样好。

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
    结果 0.2 和 1.4 的均方根正好是 1——除以 $\text{rms}$ 就是为了把向量缩放到这个尺度。向量里各个数的正负号和彼此的比例都保留了下来，均值也没有被强行平移到 0。

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

    第 17 章给每个位置配了一个单独学出来的向量，这有两个明显的短板：一是如果训练时只见过 32 个位置，模型对第 33 个位置就一无所知；二是“第 7 个词看第 5 个词”和“第 27 个词看第 25 个词”明明都是“往前看 2 个词”，模型却得把它们当成两回事分开学。**旋转位置编码（RoPE）**一口气解决了这两个问题，它背后只用到一个初中几何里很直观的想法。

    **角、余弦和正弦。** 以原点为中心，画一个半径为 1 的圆（单位圆）。从最右边的点 $(1, 0)$ 出发，沿着圆逆时针走。走过的幅度就是一个**角**，记作 $\theta$（希腊字母，读作 theta）。这时你停在圆上的位置有两个坐标：往右走的距离叫这个角的**余弦**（$\cos\theta$），往上走的距离叫它的**正弦**（$\sin\theta$）；负数分别表示往左或往下。
    """)
    return


@app.cell(hide_code=True)
def _():
    circle(0.9, '一个半径为 1 的圆。一个点从 (1, 0) 出发，沿着圆走过了角 theta。它在圆心右边的距离是 cos theta，在圆心上方的高度是 sin theta。')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    所以转过角 $\theta$ 后，点就落在 $(\cos\theta, \sin\theta)$。角度为 $0^\circ$ 时，你在起点 $(1, 0)$；逆时针转四分之一圈（$90^\circ$），你在正上方 $(0, 1)$。写代码时，数学函数习惯用**弧度**（radian）而不是角度来量角：一整圈 $360^\circ$ 对应 $2\pi$ 弧度（$\pi \approx 3.14$，一整圈约 $6.28$），所以半圈 $180^\circ$ 是 $\pi$，四分之一圈 $90^\circ$ 是 $\pi / 2 \approx 1.57$。

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
    这四个点分别是最右边、正上方、最左边，以及图里的那个点（转过了 0.9 弧度）。下面把常用的几个角列成表，后面动手算时方便对照：

    | 转多少 | 角度 | 弧度 $\theta$ | $\cos\theta$ | $\sin\theta$ |
    |---|---|---|---|---|
    | 不转 | $0^\circ$ | $0$ | $1$ | $0$ |
    | 四分之一圈 | $90^\circ$ | $\pi / 2$ | $0$ | $1$ |
    | 半圈 | $180^\circ$ | $\pi$ | $-1$ | $0$ |

    **旋转一对数。** 把任意一个点 $(a, b)$ 绕原点逆时针转过角 $\theta$，几何里的公式是：

    > $$(a, b) \;\to\; (a\cos\theta - b\sin\theta,\;\; a\sin\theta + b\cos\theta)$$

    为什么第一个坐标是减号、第二个是加号？可以把点 $(a, b)$ 想成从原点出发，先向右走 $a$ 步、再向上走 $b$ 步。向右走的 1 步转过 $\theta$ 落到 $(\cos\theta, \sin\theta)$；向上走的 1 步本身就比向右多转了 $90^\circ$，再转过 $\theta$ 就会向左上方偏过去，落到 $(-\sin\theta, \cos\theta)$。把这两段分别乘上 $a$ 和 $b$ 再加起来，就自然得到上面这个公式！

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
    对照上面的圆看一看：最右边的点转 $90^\circ$ 到了正上方；正上方的点再转 $90^\circ$ 到了最左边；而转半圈（$180^\circ$）会把任意一个点直接送到原点正对面。

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
    现在来看让旋转特别适合编码位置的核心性质。回想一下，注意力分数是查询向量（query）和键向量（key）的点积。任取两对数 `q` 和 `k`，把 `q` 转过一个角，把 `k` 转过另一个角，再算它们的点积。你会发现，结果只取决于两个角的**差值**，和各自转了多少无关——把它们同时再多转 1.5 弧度，点积丝毫不变：
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
    两个结果完全相同：把两根指针一起转过同样的角度，它们之间的夹角不变，点积自然也不变。

    ## 旋转位置编码

    于是思路就水到渠成了：把每个查询向量和键向量两两拆成许多对数，第 $p$ 个位置的词，就把它的每对数转过 $p$ 乘以对应的转速。这样一来，第 7 个词看第 5 个词时，注意力分数只取决于 $7 - 5 = 2$ 步的角度差；第 27 个词看第 25 个词，角度差也是 $2$ 步，效果完全一样。注意力机制天然看到了**相对位置**，完全不再需要位置表，而且遇到比训练窗口更长的位置时，照样能算得出旋转角（不过模型在接近训练长度时表现仍然最好）。

    每一对数都有自己的**转速**（单位是每个位置转多少弧度）：转得快的对像时钟的秒针，专门分辨相邻的近处位置；转得慢的对像时针，用来分辨隔得很远的词。以一个 4 维向量 $[x_0, x_1, x_2, x_3]$ 为例，它可以拆成 2 对。这里我们把前一半作为每对的第一个数、后一半作为第二个数，配成 $(x_0, x_2)$ 和 $(x_1, x_3)$。为什么不是把相邻的 $(x_0, x_1)$、$(x_2, x_3)$ 配在一起？纯粹是为了写代码更利落：直接用 `[:half]` 和 `[half:]` 一刀切成前后两半就行。在数学上，前后半配对和相邻配对的效果完全等价，因为点积只是把对应项相乘再求和，跟哪两项排在一起无关：

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
    下面的单元格用真实模型里的转速来验证这个性质：最快的对每个位置转 1 弧度，最慢的大约转 $1/10000$ 弧度（10,000 是沿用下来的经验底数）。举个具体的例子：当 `half = 4` 时，转速公式 `1 / 10000 ** (np.arange(4) / 4)` 算出来的 4 个转速依次是 `[1, 0.1, 0.01, 0.001]`（因为 $10000^{1/4} = 10$），每一对都比前一对慢 10 倍，等比递减地覆盖从近到远的各种跨度。这里把长度为 8 的查询和键分别旋转后求点积，比较位置 $(7, 5)$ 和位置 $(27, 25)$ 的结果：
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
    分数完全一致：只要相隔 2 个词，无论出现在句子开头还是后面，效果都一样。

    下面的 `rope` 是用 PyTorch 写成的批量版本，相当于把你的 `rope_one` 一次性应用到整批文本的所有位置上，位置编号和转速也都在里面自动算好。你不需要自己重写它，只要借助下面两个索引小工具就能轻松读懂：

    /// admonition | 🧰 工具箱：两个索引工具
    - `x[..., :half]` 表示“前面所有维度原样保留，只取最后一维的前 `half` 项”。比如对于形状为 `(batch, heads, words, size)` 的 4D 张量，省略号 `...` 就相当于 `:, :, :`，把所有文本、所有注意力头和所有词位置都原样留着。
    - `v[:, None]` 把形状为 `(n,)` 的一维向量变成 `(n, 1)` 的一列，`v[None, :]` 把它变成 `(1, n)` 的一行。一列乘一行，通过广播得到一张形状为 `(n, m)` 的二维表：这里行对应每个词的位置、列对应每种转速，第 $(p, j)$ 格正好是位置 $p$ 乘以第 $j$ 种转速得到的旋转角。
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
    ## 3. SwiGLU：带门控的思考层

    第 17 章的思考层是 $W_2(\text{relu}(W_1 x))$。两处升级能让它表现更好。第一，把带尖角的 ReLU 换成更平滑的激活函数 **SiLU**：$x \times \sigma(x)$，其中 $\sigma(x)$ 是第 12 章学过的 Sigmoid 函数，能把任意实数平滑地压到 0 到 1 之间。当 $x$ 很大时，SiLU 约等于 $x$；当 $x$ 很负时，它约等于 0，形状和 ReLU 很像，只是拐弯处平滑得多。在 PyTorch 里它叫 `F.silu`。第二，引入类似第 14 章的门控思路：多加一条支路 $W_3 x$，逐个元素地控制 $\text{silu}(W_1 x)$ 里哪些特征该放大、哪些该压低，再由 $W_2$ 汇总输出。下面的 $\times$ 表示逐元素相乘：

    > $$\text{思考}(x) = W_2\big(\text{silu}(W_1 x) \times W_3 x\big)$$

    把两个矩阵换成三个矩阵会增加参数量，因此现代模型把隐藏层宽度从原来的 4 倍缩减到 $\tfrac{8}{3}$ 倍，让总参数量基本保持不变。

    ## 动手算：数一数参数量

    我们模型的向量维度是 $d = 128$。第 17 章的思考层有两个矩阵：$128 \times 512$ 和 $512 \times 128$。新的隐藏层宽度是 $8 \times 128 / 3$ 向下取整。这个宽度是多少？三个新矩阵一共有多少个参数（即模型里的“旋钮”，不算偏置）？第 17 章那两个矩阵又有多少个（同样不算偏置）？
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
        '三个矩阵的旋钮数': r"""三个矩阵每个都有 $128 \times 341$ 个参数""",
        '原来两个矩阵的旋钮数': r"""两个矩阵每个都有 $128 \times 512$ 个参数""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    两者相差不到千分之一：几乎同样的参数量，却多了一层门控机制。

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

    现在把这三处替换装进第 17 章的 Transformer 块。整体结构没有变：依然是“先注意力、再思考”两大步，每一步执行前先做归一化，执行完再通过残差连接加回原来的 $x$。

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
    下面把现代块和完整模型封装成 PyTorch 模块。`ModernBlock` 保存了这一层的所有参数；它的 `look` 方法还是第 17 章的多头注意力，只是在算点积前对查询和键多做了一步 `rope`；它的 `forward` 直接调用你刚写的函数。`attend` 沿用第 16 章的写法，因果掩码已经内置在里面。有三处细节值得留意：

    - `nn.Parameter(tensor)` 把一个普通张量注册成可训练的参数，训练时优化器会自动更新它。这里每个增益都是一个独立参数，初始值全为 1。
    - 整个模型彻底去掉了位置表：可以把 `ModernGPT` 和第 17 章的 `TinyGPT` 对照着看。
    - 注意力层和最后的输出层都去掉了偏置项（`bias=False`），现代大模型普遍这样做。

    在上面五个练习全部通过之前，后面测试运行的单元格会报错，这是正常现象。
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

    先别急着训练，先确认各个部件能顺畅拼在一起。下面的单元格搭起一个和第 17 章规模相近的模型：每个词元 128 维，4 个注意力头，3 个块；再把前 64 个测试词元复制成两份，作为一批（batch = 2）喂给它，打印出输出分数的形状、总参数量和初始损失。

    看结果之前先估一估：没训练过的模型一无所知，基本相当于在全部 2,258 个词元里随机瞎猜，每个词元的均匀概率为 $1/2258$。按第 13 章交叉熵损失的公式，这时的损失应该是 $-\log(1/2258) = \log 2258 \approx 7.7$。
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
    输出分数的形状是 `(2, 64, 2258)`：2 段文本，每段 64 个位置，每个位置给出全部 2,258 个词元的得分。总参数量大约 117 万，和第 17 章的模型相差无几。初始损失大约是 7.9，非常接近估算的 7.7（略高一点是因为随机初始化的分数不会完全相等）。这说明整条流水线接线无误——如果刚起步的损失远大于这个数，通常意味着哪里写错了。

    ## 接下来

    模型已经搭好，只是还没读过任何故事。**第 21 章**会正式训练它，并引入更干净的权重衰减优化器 **AdamW**、学习率**调度策略**（schedule），以及用来保存和续训进度的**检查点**（checkpoint）。训完之后，它就会登上排行榜。

    ## 你现在会做的事

    - 解释为什么词元模型必须和词模型读同样的文本
    - 用手算和代码做 RMSNorm 归一化
    - 解释角、余弦和正弦，并旋转一对数
    - 解释为什么旋转位置编码让注意力看到相对位置，并把它用在一个向量上
    - 写一个带门控的 SwiGLU 层，并算出它的参数量
    - 把各个部件拼成一个现代的块，并检查未训练模型的形状和损失

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** RMSNorm 出自 Zhang 和 Sennrich（2019），RoPE 出自 Su 等（*RoFormer*，2021），SwiGLU 出自 Shazeer（*GLU Variants Improve Transformer*，2020）。LLaMA 这样的模型正是把这些选择组合在一起。很多模型还会*绑定*（tie）第一层和最后一层，共用一张词元向量表。
    """)
    return


if __name__ == "__main__":
    app.run()
