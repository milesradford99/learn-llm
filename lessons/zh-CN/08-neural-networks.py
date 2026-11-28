import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="08 · 神经网络")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    from learn_llm import hand, images, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.diagrams import grids, network, pipeline
    from learn_llm.plot import chart

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 08 · 神经网络

    第 07 章的模型能认对大约 90% 的测试数字。就算训练得再久，准确率也只会慢慢爬到大约 91%，然后停滞不前。问题不在训练方法，而在于模型**能**表达什么。
    每个数字只有一个模板，一张图的分数就是它各个像素的加权和。加权和只能说“这里有墨水就加分，那里有墨水就减分”，
    却说不出“4 的顶部要么开口，**要么**封口”——这是两种不同的形状，但都是 4。单靠一个模板，只能把两种形状混成一个模模糊糊的 4，哪一种都对不太上；
    更糟的是，一张两处都有墨水的图反而会拿最高分，可现实中根本没有哪个 4 长那样。

    这一章，你会补上缺的那块拼图——让直线能**拐弯**的**激活函数 ReLU**，并搭出一个**神经网络**（neural network）。

    ## 让直线拐个弯：ReLU

    回想第 04 章米娅遇到的那次热浪：气温升高时冰淇淋销量一路上涨，可到了大约 33 °C 后反而开始下降。一条直线跟不上这种先升后降的变化。现在来认识最简单的拐弯函数，叫 **ReLU**：

    > $z$ 为正数时 $\operatorname{relu}(z) = z$，否则为 $0$

    比如 $\operatorname{relu}(-3) = 0$，$\operatorname{relu}(0) = 0$，$\operatorname{relu}(2) = 2$。它在 $z \le 0$ 时一路贴在 0 上，到 $z = 0$ 处拐一个角，接着像直线一样斜着往上走：
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((-4, 4), (-1, 4), "z", "relu(z)", curve=[(-4, 0), (0, 0), (4, 4)], dots=[(-3, 0), (0, 0), (2, 2)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    你可以把它想成靠在墙边的一条斜坡：墙左边一路贴地平着走，过了墙才开始往上爬。单独看它平平无奇，但把它**加**到一条直线上，就能让直线在你选定的位置拐弯。
    下面是一条在 33 °C 处拐弯的折线（−105、12 和 −20 这几个数是手工挑的）：

    > $$\text{猜测} = -105 + 12 \times t - 20 \times \operatorname{relu}(t - 33)$$

    低于 33 °C 时，$t - 33$ 是负数，ReLU 输出 0，公式就是直线 $-105 + 12 \times t$。高于 33 °C 时，ReLU 输出 $t - 33$，前面乘上 $-20$ 就把直线往下掰：
    超过 33 °C 之后，气温每再升高 1 °C，销量就会变化 $12 - 20 = -8$。
    """)
    return


@app.cell(hide_code=True)
def _():
    _temps = [14, 17, 20, 22, 25, 28, 31, 33, 35, 37, 39]
    _sales = [63, 90, 137, 148, 193, 234, 257, 288, 280, 262, 236]
    _bent = [(t / 10, -105 + 12 * t / 10 - 20 * max(0, t / 10 - 33)) for t in range(120, 411)]
    _loss = np.mean([(-105 + 12 * t - 20 * max(0, t - 33) - s) ** 2 for t, s in zip(_temps, _sales)])
    mo.vstack([
        chart((10, 42), (0, 320), "气温（°C）", "卖出的冰淇淋（个）",
              dots=list(zip(_temps, _sales)), curve=_bent),
        mo.md(f"这条折线在 11 天上的损失：**{_loss:.0f}**。"
              "最好的直线大约是 810；第 04 章的曲线大约是 283。"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    只拐一个弯，就已经胜过第 04 章的曲线了。你自己来调一调这个折点：拐弯放在哪里、拐弯后斜率改变多少。下方的损失会告诉你每种选择拟合得有多好。
    """)
    return


@app.cell(hide_code=True)
def _():
    bend_at = mo.ui.slider(24, 39, value=33, label="拐弯的位置（°C）", show_value=True)
    bend_turn = mo.ui.slider(-30, 0, step=2, value=-20, label="拐弯后斜率的变化", show_value=True)
    mo.vstack([bend_at, bend_turn])
    return bend_at, bend_turn


@app.cell(hide_code=True)
def _(bend_at, bend_turn):
    _temps = [14, 17, 20, 22, 25, 28, 31, 33, 35, 37, 39]
    _sales = [63, 90, 137, 148, 193, 234, 257, 288, 280, 262, 236]
    _a, _k = bend_at.value, bend_turn.value
    _loss = np.mean([(-105 + 12 * t + _k * max(0, t - _a) - s) ** 2 for t, s in zip(_temps, _sales)])
    mo.vstack([
        chart((10, 42), (0, 320), "气温（°C）", "卖出的冰淇淋（个）", dots=list(zip(_temps, _sales)),
              curve=[(t / 10, -105 + 12 * t / 10 + _k * max(0, t / 10 - _a)) for t in range(120, 411)]),
        mo.md(r"猜测 $= -105 + 12 \times t {sign} {size} \times \operatorname{{relu}}(t - {at})$，损失 **{loss:.0f}**".format(
            sign="-" if _k < 0 else "+", size=abs(_k), at=_a, loss=_loss)),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    把斜率变化调成 0，折点就消失了，又变回一条直线。把折点移到 24 °C，线拐得太早。在所有滑块组合里，33 °C、斜率变化 −20 时损失最低，大约是 37。
    梯度下降会替我们自动搜索，而且是一次把所有折点一起调好。

    折点越多（每个都有自己的位置和拐弯幅度），就能一段一段地拼出任何形状。
    这就是神经网络的核心想法：许多简单的部件，每个带一个 ReLU 拐弯，再把它们组合在一起。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：这条折线怎么算

    用上面那条折线，$\text{猜测} = -105 + 12 \times t - 20 \times \operatorname{relu}(t - 33)$。30 °C 时它猜多少？40 °C 时呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    bend_boxes = hand.boxes(['30 度时的猜测', '40 度时的猜测'])
    return (bend_boxes,)


@app.cell(hide_code=True)
def _(bend_boxes):
    hand.marked(bend_boxes, {'30 度时的猜测': 255, '40 度时的猜测': 235}, {
        '30 度时的猜测': r"""在折点以下，$\operatorname{relu}(30 - 33) = 0$，所以就是 $-105 + 12 \times 30$""",
        '40 度时的猜测': r"""在折点以上，$\operatorname{relu}(40 - 33) = 7$：$-105 + 12 \times 40 - 20 \times 7$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：relu

    /// admonition | 🧰 工具箱：两个数里较大的那个
    `np.maximum(A, 2)` 逐个位置把数组里的每个数和 2 比较，留下较大的那个：
    `np.maximum(np.array([1, 5, -3]), 2)` 得到 `[2, 5, 2]`。结果的形状和 `A` 相同。

    别和 `np.max` 弄混：`np.max` 是从数组里挑出最大的那一个数。
    ///

    写出 `relu(Z)`，`Z` 可以是任意形状的数组：正数保持不变，其余的都变成 0。

    比如，`[[-2, 0, 3], [1.5, -0.5, 0]]` 应该得到 `[[0, 0, 3], [1.5, 0, 0]]`。（记得用英文输入法。）
    """)
    return


@app.function
def relu(Z):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("08", "relu", relu, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""ReLU 留下的是每个数和 0 当中较大的那个。""",
        r"""只要一行，用 `np.maximum`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 隐藏层

    现在回到手写数字。我们不再从像素一步跳到十个分数，而是在中间多走一步，先算出 64 个数：

    1. $Z = X W_1 + b_1$：对像素做 64 个加权和，和第 06 章算分数一模一样，只是换成了 64 组权重。`W1` 的形状是 `(784, 64)`，相当于 64 个模板；`Z` 的形状是 `(n, 64)`。
    2. $H = \operatorname{relu}(Z)$：给每个加权和过一遍 ReLU（就像上面用 $\operatorname{relu}(t - 33)$ 让直线拐弯一样，$b_1$ 决定了每个折点的位置）。和模板对不上的（加权和为负）直接归零；对得上的，则输出它吻合的程度。$H$ 的每一列都是一个小**探测器**（detector）：它说明自己要找的笔画特征在图里有多明显，没出现时就保持沉默。
    3. $S = H W_2 + b_2$：十个分数，每个都是这 64 个探测器的加权和。`W2` 的形状是 `(64, 10)`。

    `H` 叫作**隐藏层**（hidden layer）：它既不是输入图片，也不是最终分数，只待在两者之间。整个结构就叫**神经网络**。
    这样一来，判断“是不是 4”时，网络就能表达“探测器 12（顶部开口）**或者**探测器 40（顶部封口）”（编号只是举个例子）：哪个探测器亮起来就加上它那一份分，沉默的就什么也不加。

    下面把这三步画成流程图，并标出有 `n` 张图片时每一步的形状：
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([('X', '(n, 784)'), ('Z', '(n, 64)'), ('H', '(n, 64)'), ('S', '(n, 10)')],
             ['@ W1 + b1', 'relu', '@ W2 + b2'],
             'X 经过 W1 和 b1 得到 Z，relu 得到 H，再经过 W2 和 b2 得到分数 S')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 小世界里的“这个或那个”

    把开头的问题缩小到 $3 \times 3$ 的图片上看。假设一根“竖杠”既可以画在**最左**一列，也可以画在**最右**一列：两种形状都算竖杠；
    但如果左右两列**都**有墨水，就不算一根竖杠。

    如果像第 06 章那样用取平均做单个模板，左右两列的权重都是 0.5。每根竖杠有 3 个有墨的像素踩中模板的一半，得 1.5 分；
    可两列都有墨的那张图一共踩中 6 个，得 3 分，反而是最高分。单纯的加权和只会不断把墨水累加起来，所以在对的位置墨越多，分数就一定越高。

    现在换成两个探测器。探测器 A 的权重在左列是 $+1$，在右列是 $-1$，偏置 $b = -2$；探测器 B 和它左右对调：
    """)
    return


@app.cell(hide_code=True)
def _():
    _a = [[1, 0, -1]] * 3
    _b = [[-1, 0, 1]] * 3
    _marks = lambda grid: {(r, c): "blue" if v > 0 else "red" for r in range(3) for c, v in enumerate(grid[r]) if v}
    mo.vstack([
        grids([
            {"name": "左边的竖杠", "values": [[1, 0, 0]] * 3, "marks": {(r, 0): "blue" for r in range(3)}}, "   ",
            {"name": "右边的竖杠", "values": [[0, 0, 1]] * 3, "marks": {(r, 2): "blue" for r in range(3)}}, "   ",
            {"name": "两列都有（不是竖杠）", "values": [[1, 0, 1]] * 3,
             "marks": {(r, c): "blue" for r in range(3) for c in (0, 2)}},
        ], "三张图片：左列一根竖杠，右列一根竖杠，两列都有墨"),
        grids([
            {"name": "探测器 A 的权重", "values": _a, "marks": _marks(_a)}, "   ",
            {"name": "探测器 B 的权重", "values": _b, "marks": _marks(_b)},
        ], "探测器 A：左列加 1，右列减 1；探测器 B 反过来"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    “竖杠”的分数是 $\operatorname{relu}(z_A) + \operatorname{relu}(z_B)$。对每张图片算一遍：

    | 图片 | $z_A$ | $z_B$ | $h_A$ | $h_B$ | 网络的分数 | 单模板的分数 |
    |---|---|---|---|---|---|---|
    | 左边的竖杠 | $3 - 0 - 2 = 1$ | $0 - 3 - 2 = -5$ | 1 | 0 | **1** | 1.5 |
    | 右边的竖杠 | $-5$ | $1$ | 0 | 1 | **1** | 1.5 |
    | 两列都有 | $3 - 3 - 2 = -2$ | $-2$ | 0 | 0 | **0** | 3 |

    每个探测器只对自己的形状亮起来，其余时候保持沉默；ReLU 把“沉默”变成干干净净的 0，而不会留下一个到处拉低分数的负数。
    网络表达出了“左边**或**右边，但不能两边都有”，这是任何单个模板都做不到的。这就是隐藏层带来的能力。

    算出的分数照样送进第 07 章的 softmax 和交叉熵。下面把它们再列出来，连同第 06 章的 `accuracy` 和第 07 章的 `one_hot` 一起——都已经写好了，不用改动：
    """)
    return


@app.function
def softmax(S):
    S = S - np.max(S, axis=1, keepdims=True)
    E = np.exp(S)
    return E / np.sum(E, axis=1, keepdims=True)


@app.function
def cross_entropy(P, Y):
    return -np.mean(np.sum(Y * np.log(P), axis=1))


@app.function
def accuracy(S, labels):
    return np.mean(np.argmax(S, axis=1) == labels)


@app.function
def one_hot(labels):
    Y = np.zeros((len(labels), 10))
    for i in range(len(labels)):
        Y[i, labels[i]] = 1
    return Y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：一张图片走过一个小网络

    给 784 个像素写代码之前，先用 2 个像素算一遍。只有一张图片 `x = [1, 2]`，两个探测器，两个分数。所有的偏置 `b` 都是 0，权重如下：

    > $$W_1 = \begin{bmatrix} 1 & -3 \\ 1 & 1 \end{bmatrix} \qquad W_2 = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$$

    $W_1$ 的第 0 列是探测器 0 的权重，所以 $z_0 = 1 \times 1 + 2 \times 1 = 3$。把其余几个数算出来：另一个加权和、两个隐藏层输出、两个最终分数。

    下面把同一个网络画成图。每条箭头上写着一个权重：把箭头起点的值乘以箭头上的数，再把汇进同一个方框的数加在一起，就得到这个方框的加权和。
    """)
    return


@app.cell(hide_code=True)
def _():
    tiny_edges = [(0, 0, 0, "1"), (0, 0, 1, "-3"), (0, 1, 0, "1"), (0, 1, 1, "1"),
                  (1, 0, 0, "relu"), (1, 1, 1, "relu"),
                  (2, 0, 0, "1"), (2, 0, 1, "0"), (2, 1, 0, "2"), (2, 1, 1, "1")]
    tiny_heads = ["像素 x", "加权和 z", "探测器 h", "分数 s"]
    network([["x0 = 1", "x1 = 2"], ["z0 = 3", "z1 = ?"], ["h0 = ?", "h1 = ?"], ["s0 = ?", "s1 = ?"]],
            tiny_edges, "一个有 2 个像素、2 个探测器、2 个分数的网络；权重写在箭头上", tiny_heads)
    return tiny_edges, tiny_heads


@app.cell(hide_code=True)
def _():
    forward_boxes = hand.boxes(['z1', 'h0', 'h1', 's0', 's1'])
    return (forward_boxes,)


@app.cell(hide_code=True)
def _(forward_boxes):
    hand.marked(forward_boxes, {'z1': -1, 'h0': 3, 'h1': 0, 's0': 3, 's1': 0}, {
        'z1': r"""`x` 和 $W_1$ 第 1 列的点积：$1 \times (-3) + 2 \times 1$""",
        'h0': r"""$\operatorname{relu}(z_0)$，而 $z_0$ 是 3""",
        'h1': r"""$\operatorname{relu}(z_1)$：加权和为负时得到 0""",
        's0': r"""`h` 和 $W_2$ 第 0 列的点积：$h_0 \times 1 + h_1 \times 2$""",
        's1': r"""`h` 和 $W_2$ 第 1 列的点积：$h_0 \times 0 + h_1 \times 1$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _(forward_boxes, tiny_edges, tiny_heads):
    mo.stop(not hand.done(forward_boxes, {'z1': -1, 'h0': 3, 'h1': 0, 's0': 3, 's1': 0}),
            mo.md("*五个格子都填对之后，这里的图会把数填上。*"))
    network([["x0 = 1", "x1 = 2"], ["z0 = 3", "z1 = -1"], ["h0 = 3", "h1 = 0"], ["s0 = 3", "s1 = 0"]],
            tiny_edges, "同一个网络，每个值都填上了", tiny_heads)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：forward

    写出 `forward(X, W1, b1, W2, b2)`（forward 即“前向传播”）。它对所有图片一次性完成这三步，并按 `Z`、`H`、`S` 的顺序返回全部三个结果
    （后面算梯度时还要用到 `Z` 和 `H`）。直接调用你刚才写的 `relu`。

    在上面的小网络上，取 `X = [[1, 2]]`（只有一行的矩阵，对应这一张图），它应该返回 `[[3, -1]]`、`[[3, 0]]` 和 `[[3, 0]]`。
    """)
    return


@app.function
def forward(X, W1, b1, W2, b2):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("08", "forward", forward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""每一步一行，顺序和流程图一样：加权和、ReLU、加权和。""",
        r"""加权和用 `@`，加上偏置 `b` 用 `+`。返回三个结果，用逗号隔开。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 为什么不能少了 ReLU

    如果去掉 ReLU，第 3 步算的就是 `(X @ W1 + b1) @ W2 + b2`。先不管偏置 `b`：`@` 和普通乘法一样满足结合律，先算哪一对都行，
    所以 `(X @ W1) @ W2 = X @ (W1 @ W2)`，而 `W1 @ W2` 合并起来不过是一个 `(784, 10)` 的矩阵——又变回了每个数字一个模板。没有 ReLU 这一步拐弯，两层叠在一起并不比一层强。
    用几个小的随机矩阵验证一下这种结合律（单元格里的新工具紧接着就会解释）：
    """)
    return


@app.cell
def _():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (3, 4))
    B = rng.normal(0, 1, (4, 5))
    C = rng.normal(0, 1, (5, 2))
    (A @ B) @ C - A @ (B @ C)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    结果几乎全是 0：不管先算哪一对，得到的都是同一个矩阵。像 `2.22044605e-16` 这样的数，意思是 $2.2 \times 0.0000000000000001$，只是最后一位上的舍入误差；
    零后面的 `e+00` 表示“乘以 1”，也就是 0。

    这个单元格用到了一样新工具：**随机数**。`np.random.default_rng(0)` 会创建一个随机数生成器；`0` 是起点（叫作*种子*，seed），
    保证每次运行这一页都生成完全相同的随机数。`rng.normal(0, 1, (3, 4))` 返回一个 `(3, 4)` 的数组，里面的数分布在 0 附近，
    大多落在 −2 到 2 之间（`0` 是中心，`1` 是典型幅度）。

    我们需要用随机数来初始化网络。如果每个参数都从 0 开始，64 个隐藏探测器就会完全一模一样，而且训练下去也永远一模一样。
    所以我们从很小的随机权重起步：`W1` 的典型幅度是 0.05，`W2` 是 0.18。（第 09 章会解释这两件事：为什么不能全 0，以及为什么挑这两个幅度。）
    下一个单元格用种子 1 生成初始参数，这样你的起点和这里看到的完全一致。

    ## 数据，以及第一次前向传播
    """)
    return


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    X = train_images.reshape(10000, 784) / 255
    X_test = test_images.reshape(2000, 784) / 255
    Y = one_hot(train_labels)
    start_rng = np.random.default_rng(1)
    W1_start = start_rng.normal(0, 0.05, (784, 64))
    b1_start = np.zeros(64)
    W2_start = start_rng.normal(0, 0.18, (64, 10))
    b2_start = np.zeros(10)
    Z0, H0, S0 = forward(X, W1_start, b1_start, W2_start, b2_start)
    Z0.shape, H0.shape, S0.shape, cross_entropy(softmax(S0), Y)
    return (
        H0, S0, W1_start, W2_start, X, X_test, Y, Z0, b1_start, b2_start,
        test_labels, train_labels,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    形状分别是 `(10000, 64)`、`(10000, 64)` 和 `(10000, 10)`，损失约 2.41，比瞎猜时的 2.303 还略差一点——随机参数自然只能做出随机猜测。

    ## 经过两层求斜率

    现在来求梯度。参数要经过一串步骤才会影响最终的损失（`W1` → `Z` → `H` → `S` → 损失），第 03 章讲过链条怎么处理：倒着往回走，一步步把责任传回去。
    我们从末尾出发，给每个量的责任前面加上 `d`：`dH` 是 `H` 的责任，`dW1` 是 `W1` 的责任（斜率），依此类推。

    **分数。** 损失和第 07 章一样，是 `softmax(S)` 的交叉熵，所以分数的责任直接沿用第 07 章的结果：$G = (P - Y) / n$，其中 $P = \operatorname{softmax}(S)$。

    **`W2` 和 `b2`。** `S = H @ W2 + b2` 和第 07 章的 `X @ W + b` 结构完全一样，只是把输入从 `X` 换成了 `H`。
    （快速回忆一下第 07 章为什么是 `X.T @ G`：每个参数的斜率，都是“所有图上对应的输入”和“所有图上对应分数的责任”做点积；把输入矩阵转置后列变行，`@` 就能一次算出所有点积。）
    所以这里结论完全相同：$dW_2 = H^\top G$（写成代码就是 `H.T @ G`），$db_2$ 是 $G$ 每一列纵向求和。

    **隐藏层 $H$。** 这一步是本章新内容：第 07 章从未需要求**输入**的斜率。来看其中一个隐藏值 `H[i, j]`（第 $i$ 张图、第 $j$ 个探测器）。它参与了第 $i$ 张图的全部十个分数：
    展开就是 $S[i, c] = H[i, 0] \times W_2[0, c] + \ldots + H[i, 63] \times W_2[63, c] + b_2[c]$。
    按第 03 章的乘法规则（在 $a \times c$ 中，$a$ 每变 1，结果就变 $c$），`H[i, j]` 每变 1，`S[i, c]` 就变 `W2[j, c]`，这就是局部斜率。
    一个值分出十条箭头，**责任相加**（第 03 章），因此：

    > `H[i, j]` 的责任 = $G[i, 0] \times W_2[j, 0] + G[i, 1] \times W_2[j, 1] + \ldots + G[i, 9] \times W_2[j, 9]$

    这是 `G` 的第 `i` 行和 `W2` 的第 `j` 行的点积，也就是 `G` 的第 `i` 行和 `W2.T` 的第 `j` **列**的点积。
    回忆矩阵乘法：`A @ B` 的 `[i, j]` 项正是 `A` 的第 `i` 行与 `B` 的第 `j` 列的点积，所以这就是 `G @ W2.T` 的 `[i, j]` 项！
    对所有 `i` 和 `j` 一次性算出：$dH = G W_2^\top$（即 `G @ W2.T`），形状是 `(n, 10) @ (10, 64) = (n, 64)`，和 `H` 严丝合缝。

    放到 2 张图、2 个探测器、2 个数字的极简例子里一眼看清（沿用上面小网络的 $W_2 = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$，转置后 $W_2^\top = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}$）：
    `G` 的第 0 行（蓝）与 `W2.T` 的第 1 列（红）做点积，结果放进 `dH[0, 1]`：$1 \times 2 + (-1) \times 1 = 1$。
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "G (2, 2)", "values": [[1, -1], [0, 2]],
         "rows": ["图片 0", "图片 1"], "cols": ["数字 0", "数字 1"],
         "marks": {(0, 0): "blue", (0, 1): "blue"}},
        "@",
        {"name": "W2.T (2, 2)", "values": [[1, 2], [0, 1]],
         "rows": ["数字 0", "数字 1"], "cols": ["探测器 0", "探测器 1"],
         "marks": {(0, 1): "red", (1, 1): "red"}},
        "=",
        {"name": "dH (2, 2)", "values": [[1, 1], [0, 2]],
         "rows": ["图片 0", "图片 1"], "cols": ["探测器 0", "探测器 1"],
         "marks": {(0, 1): "box"}},
    ], "G 的第 0 行（蓝）与 W2.T 的第 1 列（红）做点积，得到 dH[0, 1]：1 乘 2 加 -1 乘 1 等于 1")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **ReLU 激活。** `H = relu(Z)` 是逐个位置计算的。来看 $\operatorname{relu}(z)$ 的斜率：$z > 0$ 时 $\operatorname{relu}(z) = z$，斜率是 1；$z < 0$ 时 $\operatorname{relu}(z) = 0$，斜率是 0
    （正好在 0 处是个拐角，我们取斜率 0）。所以 ReLU 就像一扇闸门：在 $Z > 0$ 的位置把责任原样放行，其余位置全部拦住：

    > `dZ = dH * (Z > 0)`

    注意这里是**逐个位置相乘**的 `*`，不是矩阵乘法 `@`；`Z > 0` 会对每个位置给出 `True` 或 `False`，做乘法时会自动当成 `1` 和 `0`。
    像这样用来放行一部分位置、挡住另一部分位置的 0/1 数组，叫作**掩码**（mask）。如果某个探测器对一张图保持沉默，它就不对这张图承担责任——把它的权重轻推一点点，输出也不会有任何变化。以 2 张图、3 个探测器为例：
    """)
    return


@app.cell(hide_code=True)
def _():
    _z = [[2.0, -1.0, 0.5], [-0.3, 1.2, -2.0]]
    _m = [[1 if v > 0 else 0 for v in row] for row in _z]
    _dh = [[0.4, -0.6, 0.1], [0.3, -0.2, 0.5]]
    _blocked = {(r, c): "red" for r in range(2) for c in range(3) if not _m[r][c]}
    grids([
        {"name": "Z", "values": _z, "rows": ["第 0 张", "第 1 张"], "marks": _blocked},
        "→",
        {"name": "掩码 Z > 0", "values": _m, "marks": _blocked},
        "   ",
        {"name": "dH", "values": _dh},
        "×",
        {"name": "掩码", "values": _m, "marks": _blocked},
        "=",
        {"name": "dZ", "values": [[d if m else 0 for d, m in zip(dr, mr)] for dr, mr in zip(_dh, _m)], "marks": _blocked},
    ], "Z 为负的地方掩码是 0（红），dH 乘以掩码后，那里的责任就被挡住了")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `Z` 里为负的位置（红）对应的探测器保持沉默，掩码为 0，`dH` 传到这里就被截停；其余位置则原样通过。

    **`W1` 和 `b1`。** `Z = X @ W1 + b1` 又是第 07 章的同一套公式，只需把责任换成 `dZ`：$dW_1 = X^\top dZ$（即 `X.T @ dZ`），$db_1$ 是 $dZ$ 每一列纵向求和。

    这就是整个神经网络的反向传播。每一步用的都是第 03 章的链式法则；矩阵运算只是把它一次性批处理到所有图片和所有参数上。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：责任走过小网络

    回到那个小网络：`x = [1, 2]`，`Z = [3, -1]`，`H = [3, 0]`。假设两个分数的责任算出来是 `G = [1, -1]`（取整数是为了好算）。
    把责任一步一步往回传。只有一张图片时，每个公式都很短：

    - $dW_2[j, c] = H[j] \times G[c]$
    - $dH[j]$ 是 $G$ 和 $W_2$ 第 $j$ 行的点积
    - $dZ$ 在 $Z$ 为正的地方等于 $dH$，其余地方是 0
    - $dW_1[i, j] = x[i] \times dZ[j]$
    """)
    return


@app.cell(hide_code=True)
def _():
    backward_boxes = hand.boxes(['dW2[0, 0]', 'dW2[1, 0]', 'dH[0]', 'dH[1]', 'dZ[0]', 'dZ[1]', 'dW1[1, 0]'])
    return (backward_boxes,)


@app.cell(hide_code=True)
def _(backward_boxes):
    hand.marked(backward_boxes, {'dW2[0, 0]': 3, 'dW2[1, 0]': 0, 'dH[0]': 1, 'dH[1]': 1, 'dZ[0]': 1, 'dZ[1]': 0, 'dW1[1, 0]': 2}, {
        'dW2[0, 0]': r"""$H[0] \times G[0]$""",
        'dW2[1, 0]': r"""$H[1] \times G[0]$，而探测器 1 是沉默的""",
        'dH[0]': r"""$W_2$ 的第 0 行是 `[1, 0]`：$1 \times 1 + (-1) \times 0$""",
        'dH[1]': r"""$W_2$ 的第 1 行是 `[2, 1]`：$1 \times 2 + (-1) \times 1$""",
        'dZ[0]': r"""$Z[0]$ 是正的，所以责任原样传过去""",
        'dZ[1]': r"""$Z[1]$ 是负的，所以责任被挡住""",
        'dW1[1, 0]': r"""$x[1] \times dZ[0]$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _(backward_boxes, tiny_edges, tiny_heads):
    mo.stop(not hand.done(backward_boxes, {'dW2[0, 0]': 3, 'dW2[1, 0]': 0, 'dH[0]': 1, 'dH[1]': 1, 'dZ[0]': 1, 'dZ[1]': 0, 'dW1[1, 0]': 2}),
            mo.md("*七个格子都填对之后，这里的图上会出现各个责任。*"))
    network([["x0 = 1", "x1 = 2"], [["z0 = 3", "责任 1"], ["z1 = -1", "责任 0"]],
             [["h0 = 3", "责任 1"], ["h1 = 0", "责任 1"]], [["s0 = 3", "责任 1"], ["s1 = 0", "责任 -1"]]],
            tiny_edges, "小网络，每个值下面用红字写着它的责任", tiny_heads)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    看看探测器 1。它对这张图片保持沉默，所以从它连出去的权重不承担责任（`dW2[1, 0]` 是 0）；掩码又挡住了它的责任，不让它继续往回传（`dZ[1]` 是 0），
    所以它自己在 $W_1$ 里的权重也不承担责任。对这张图片来说，只有探测器 0 在学习。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：backward

    写出 `backward(X, Y, Z, H, S, W2)`（backward 即“反向传播”），按 `dW1, db1, dW2, db2` 的顺序返回四个斜率。和正文一样，从末尾倒着算回去：

    | 步骤 | 公式 | 形状乘积 |
    |---|---|---|
    | 分数的责任 | $G = (P - Y) / n$，其中 $P = \operatorname{softmax}(S)$，$n$ 是图片数 `len(X)` | `(n, 10)` |
    | 第二层 | $dW_2 = H^\top G$，$db_2$ 是 $G$ 每一列纵向求和 | `(64, n) @ (n, 10)` → `(64, 10)`；`(10,)` |
    | 隐藏值 | $dH = G W_2^\top$ | `(n, 10) @ (10, 64)` → `(n, 64)` |
    | ReLU 激活 | $dZ$ 是 $dH$ 乘以掩码 `Z > 0`（逐个位置相乘 `*`，不是 `@`） | `(n, 64) * (n, 64)` → `(n, 64)` |
    | 第一层 | $dW_1 = X^\top dZ$，$db_1$ 是 $dZ$ 每一列纵向求和 | `(784, n) @ (n, 64)` → `(784, 64)`；`(64,)` |

    写成代码时，矩阵乘法 $A B$ 写作 `A @ B`，转置 $M^\top$ 写作 `M.T`（所以 $H^\top G$ 对应 `H.T @ G`，$G W_2^\top$ 对应 `G @ W2.T`），每一列纵向求和是 `np.sum(M, axis=0)`（第 07 章）。
    留意每次 `@` 里中间那个维度都会正好消掉：`(64, n) @ (n, 10)` 得到 `(64, 10)`，而 `(n, 10) @ (10, 64)` 得到 `(n, 64)`——这能帮你一眼自查谁该转置、谁放前面。记得用英文输入法。
    """)
    return


@app.function
def backward(X, Y, Z, H, S, W2):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("08", "backward", backward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""表里每一行对应一两行代码，顺序和表一样。边写边对照表里的形状。""",
        r"""每个 `@` 里的先后顺序很重要：是 `H.T @ G`，却是 `G @ W2.T`。如果下面的轻推检查只有 `W1` 对不上，看看掩码。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 用轻推来检查

    整个网络一下子引入了这么多新公式，最好用数值方法验算一下。在训练开始前，轻轻推一下第一层的一个权重 `W1[406, 0]`（从像素 406，也就是第 14 行第 14 列、图片正中间，连向探测器 0），看损失怎么变；
    再对第二层的一个权重 `W2[5, 3]` 做同样的验算。（这个单元格需要完成全部三个练习。下面 `forward(...)[2]` 里的 `[2]` 是取第 3 个返回值，也就是分数 `S`——Python 从 0 开始数，`forward` 返回 `(Z, H, S)`。）
    """)
    return


@app.cell
def _(H0, S0, W1_start, W2_start, X, Y, Z0, b1_start, b2_start):
    h = 0.0001
    dW1, db1, dW2, db2 = backward(X, Y, Z0, H0, S0, W2_start)
    base = cross_entropy(softmax(S0), Y)

    nudge1 = np.zeros((784, 64))
    nudge1[406, 0] = h
    S1 = forward(X, W1_start + nudge1, b1_start, W2_start, b2_start)[2]

    nudge2 = np.zeros((64, 10))
    nudge2[5, 3] = h
    S2 = forward(X, W1_start, b1_start, W2_start + nudge2, b2_start)[2]

    [(cross_entropy(softmax(S1), Y) - base) / h, dW1[406, 0],
     (cross_entropy(softmax(S2), Y) - base) / h, dW2[5, 3]]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第 0、1 项是 `W1[406, 0]` 的轻推结果和公式结果；第 2、3 项是 `W2[5, 3]` 的。两层算出来的结果，前几位有效数字都完全吻合（−0.01332… 和 −0.01333…；
    两个都是 0.0029739…）——毕竟轻推只是近似，不可能精确到最后一位。（角落里的像素斜率正好是 0，什么也检不出来，所以我们挑了正中间的像素。）
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：一共有多少个参数

    训练之前，先数一数这次要调多少个参数。`W1` 的形状是 `(784, 64)`，`b1` 有 64 个数，`W2` 的形状是 `(64, 10)`，`b2` 有 10 个数。
    `W1` 里有多少个参数？整个网络一共有多少个？
    """)
    return


@app.cell(hide_code=True)
def _():
    count_boxes = hand.boxes(['W1', '整个网络'])
    return (count_boxes,)


@app.cell(hide_code=True)
def _(count_boxes):
    hand.marked(count_boxes, {'W1': 50176, '整个网络': 50890}, {
        'W1': r"""$784 \times 64$""",
        '整个网络': r"""`W1`，加 64，加 $64 \times 10$，再加 10""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第 07 章的模型有 7,850 个参数。这个网络是它的六倍多，而且几乎全集中在第一层。

    ## 训练

    和第 07 章一样的训练循环，只是要更新的参数从两组变成四组。训练 300 步、每步扫完 10,000 张图需要一点时间（最多大约一分钟），所以这里加了一个按钮，等你点下再开始跑。单元格第一行的 `mo.stop(...)` 就是让它先停住等你按按钮。
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="训练网络（最多可能要一分钟）")
    train_button
    return (train_button,)


@app.cell
def _(W1_start, W2_start, X, X_test, Y, b1_start, b2_start, test_labels, train_button, train_labels):
    mo.stop(not train_button.value, mo.md("*按下按钮开始训练。*"))
    W1, b1, W2, b2 = W1_start, b1_start, W2_start, b2_start
    for step in range(300):
        Z, H, S = forward(X, W1, b1, W2, b2)
        gW1, gb1, gW2, gb2 = backward(X, Y, Z, H, S, W2)
        W1 = W1 - 0.5 * gW1
        b1 = b1 - 0.5 * gb1
        W2 = W2 - 0.5 * gW2
        b2 = b2 - 0.5 * gb2
    S_train = forward(X, W1, b1, W2, b2)[2]
    S_test = forward(X_test, W1, b1, W2, b2)[2]
    cross_entropy(softmax(S_train), Y), accuracy(S_train, train_labels), accuracy(S_test, test_labels)
    return (W1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    输出的三个数分别是训练损失、训练准确率和测试准确率。最后一个约 0.944，也就是认对了大约 **94%** 的测试图片，
    而同样训练这么久的单层模型大约是 91%：加上隐藏层后，认错的比例一下少了大约 40%。

    ## 探测器在找什么

    `W1` 的每一列都是一个探测器的 784 个权重，画成一张 $28 \times 28$ 的图（蓝色：这里有墨水会让它亮起来；红色：这里有墨水会让它沉默）。下面是前 16 个：
    """)
    return


@app.cell(hide_code=True)
def _(W1):
    images.weight_maps([W1[:, j] for j in range(16)], [f"探测器 {j}" for j in range(16)], size=72)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    它们看起来比第 07 章的模板杂乱得多，而这种杂乱本身就很说明问题。靠近边缘的地方，几乎每张图片都是白纸，很多像素在所有图里都是 0，其余的也几乎总是 0：
    这些位置的参数几乎收不到梯度（斜率 `X.T @ dZ` 乘的正是这些 0），所以基本一直停在最初的随机值上。越往中间、墨水越多的地方，就能看出平滑一些的色块：每个探测器自己学会了对某一段笔画做出反应，没人教过它该找什么。
    第二层则负责学会：把哪些探测器组合起来，就对应哪一个数字。

    ## 下一步

    网络已经能跑出不错的效果，但训练很慢：300 步里的每一步都要扫完 10,000 张图片。我们还没说清为什么初始化幅度要挑 0.05 和 0.18。
    在**第 09 章**里，你会学到让训练又快又稳的办法：小批量训练、更聪明的更新规则、有依据的初始化，以及怎样确认模型在新数据上的表现真的靠得住。

    ## 你现在会做的事

    - 说清楚为什么加权和表达不了“这个或那个”，以及 ReLU 怎样解决这个问题
    - 搭一个带隐藏层的神经网络：`X @ W1 + b1`、ReLU、`H @ W2 + b2`
    - 说清楚为什么没有 ReLU 的两层并不比一层强
    - 用矩阵在整个网络里做反向传播：`H.T @ G`、`G @ W2.T`、ReLU 掩码、`X.T @ dZ`
    - 用轻推验算网络的梯度，并完成训练

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**这种网络叫作*多层感知机*（MLP）；ReLU 是 *rectified linear unit*（线性整流单元）的缩写。
    “把许多带折点的简单函数加起来能逼近任何形状”这一结论叫作*万能近似定理*。
    """)
    return


if __name__ == "__main__":
    app.run()
