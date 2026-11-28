import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="09 · 让训练靠得住")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np

    from learn_llm import hand, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.diagrams import grids
    from learn_llm.plot import chart

    LOCALE = "zh-CN"
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 09 · 让训练靠得住

    第 08 章的网络准确率已经达到了约 94%，但还留着几个没交代清楚的尾巴：为什么一开始把旋钮（参数）的大小选成 0.05 和 0.18？每走一步都要算完整个 10,000 张图片，导致 300 步就花了约 10 秒，能不能更快？还有，第 02 章里米娅预测冰淇淋销量时，`b` 挪得慢吞吞的，因为同一个学习率没法同时照顾好所有旋钮——从那以后，我们一直只能靠提前缩放输入数据来绕开这个麻烦。

    这一章我们把这些尾巴逐个收干净：有理有据的起点、**小批量**（mini-batch）、两种更聪明的更新规则（**动量**和 **Adam**），以及如何发现模型是在“死记硬背”而不是真正学会。现代每一次模型训练都离不开这些基本功，课程最后的大语言模型也不例外。

    下面是第 08 章的函数，可以直接用：
    """)
    return


@app.function
def softmax(S):
    S = S - np.max(S, axis=1, keepdims=True)
    E = np.exp(S)
    return E / np.sum(E, axis=1, keepdims=True)


@app.function
def forward(X, W1, b1, W2, b2):
    Z = X @ W1 + b1
    H = np.maximum(Z, 0)
    return Z, H, H @ W2 + b2


@app.function
def backward(X, Y, Z, H, S, W2):
    G = (softmax(S) - Y) / len(X)
    dZ = (G @ W2.T) * (Z > 0)
    return X.T @ dZ, np.sum(dZ, axis=0), H.T @ G, np.sum(G, axis=0)


@app.function
def accuracy(S, labels):
    return np.mean(np.argmax(S, axis=1) == labels)


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    X = train_images.reshape(10000, 784) / 255
    X_test = test_images.reshape(2000, 784) / 255
    Y = np.eye(10)[train_labels]
    X.shape, Y.shape
    return X, X_test, Y, test_labels, train_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：三个 NumPy 小工具
    - `np.eye(10)` 是对角线上为 1 的 $10 \times 10$ 表格，它的第 $k$ 行正好是数字 $k$ 的 one-hot 行。所以上面单元格里的 `np.eye(10)[train_labels]` 是第 07 章 `one_hot` 的简便写法：给它一串标签，就为每张图片挑出对应的一行。（用列表一次挑出多行，会在“小批量”一节讲。）
    - `np.sqrt` 是平方根：`np.sqrt(9)` 是 3，因为 $3 \times 3 = 9$。
    - `np.abs` 取绝对值（去掉负号）：`np.abs(np.array([-2, 5]))` 会得到 `[2, 5]`。因此 `np.max(np.abs(W))` 就是 `W` 里最大的绝对值，不管正负。
    ///

    ## 有多分散？方差

    要选起始大小，我们需要一种衡量一组数有多分散的办法。先求它们的均值，再算出每个数离均值的距离并平方（这样正负不会互相抵消，和第 01 章一样），然后求这些平方的均值：

    > $$\textbf{方差}\text{（variance）} = (\text{数} - \text{均值})^2 \text{ 的均值}$$

    比如 `[1, 3]`：均值是 2，到均值的距离是 $-1$ 和 $1$，平方后都是 1，所以方差是 1。方差的平方根就是这些距离的**典型大小**（在 NumPy 里用 `np.sqrt` 计算）。如果一组数以 0 为中心，那它们的均值就是 0，方差也就直接等于平方的均值。

    为什么要算这个？因为只看均值会掩盖很多信息。两个班的一次小测平均分都是 4 分（满分 8 分）：A 班的分数是 `[3, 4, 4, 5]`，B 班是 `[1, 4, 4, 7]`。把它们画在数轴上，每个分数都用虚线连到均值：
    """)
    return


@app.cell(hide_code=True)
def _():
    _a, _b = [3, 4, 4, 5], [1, 4, 4, 7]
    _dots = [(x, 2 + 0.15 * (i == 2)) for i, x in enumerate(_a)] + [(x, 1 + 0.15 * (i == 2)) for i, x in enumerate(_b)]
    _segs = [(x, 2, 4, 2) for x in _a] + [(x, 1, 4, 1) for x in _b]
    chart((0, 8), (0.5, 2.5), "分数（上面一行是 A 班，下面一行是 B 班）", "班级",
          dots=_dots, segments=_segs + [(4, 0.5, 4, 2.5)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    均值相同，分散程度却天差地别。A 班到 4 的距离是 $-1, 0, 0, 1$，平方后是 $1, 0, 0, 1$，方差为 $2 / 4 = 0.5$；B 班的距离是 $-3, 0, 0, 3$，平方后是 $9, 0, 0, 9$，方差为 $18 / 4 = 4.5$。图里一眼看到的离散程度，方差用一个数就概括出来了。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：一个方差

    取 `[2, 4, 4, 6]` 这四个数。算出它们的均值，再算出方差。
    """)
    return


@app.cell(hide_code=True)
def _():
    variance_boxes = hand.boxes(['均值', '方差'])
    return (variance_boxes,)


@app.cell(hide_code=True)
def _(variance_boxes):
    hand.marked(variance_boxes, {'均值': 4, '方差': 2}, {
        '均值': r"""四个数加起来，除以 4""",
        '方差': r"""到均值的距离是 $-2, 0, 0, 2$。把它们平方，再求均值""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：variance

    按上面的公式写出 `variance(v)`，其中 `v` 是一个 NumPy 数组。数组的加减乘除都是逐元素进行的，`** 2` 表示平方。

    输入 `[1, 3]` 应返回 1，输入刚才算过的 `[2, 4, 4, 6]` 应返回 2。
    """)
    return


@app.function
def variance(v):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "variance", variance, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三步：均值，每个数到均值的距离的平方，再对这些平方求均值。""",
        r"""只要一行。不要用 `np.sqrt`：那样得到的是典型大小，不是方差。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 有依据的起点

    **为什么不能从 0 开始？** 如果每个旋钮都从 0 起步，每个隐藏探测器算出的结果完全一样（全是 0），分到的责任完全一样，更新步子也完全一样，它们就会永远一模一样。更糟的是：反向传播时，隐藏层拿到的梯度是第 08 章算过的 `dH = G @ W2.T`（其中 `G` 是输出层的误差）；如果 `W2` 也全是 0，乘出来 `dH` 就全是 0，根本没有任何梯度能传回第一层的 `W1` 和 `b1`。想象 64 个工人站在同一个起点，拿到同样的指令，挨同样的批评改正——他们永远不会分头去干不同的活。下面的单元格把一个全 0 初始化的网络训练 100 步：
    """)
    return


@app.cell
def _(X, X_test, Y, test_labels):
    W1z, b1z, W2z, b2z = np.zeros((784, 64)), np.zeros(64), np.zeros((64, 10)), np.zeros(10)
    for _step in range(100):
        Zz, Hz, Sz = forward(X, W1z, b1z, W2z, b2z)
        g1, gb1, g2, gb2 = backward(X, Y, Zz, Hz, Sz, W2z)
        W1z, b1z, W2z, b2z = W1z - 0.5 * g1, b1z - 0.5 * gb1, W2z - 0.5 * g2, b2z - 0.5 * gb2
    np.max(np.abs(W1z)), accuracy(forward(X_test, W1z, b1z, W2z, b2z)[2], test_labels)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第一层旋钮的最大绝对值仍然正好是 0（`np.max(np.abs(...))` 算的就是最大绝对值），测试准确率大约 12%：只有 `b2`（输出分数的起始高度）学到了东西，`b1` 也收不到任何梯度，所以模型对每张图片都猜同一个数字，和瞎猜差不多。随机的起始旋钮能打破这种完全对称的僵局，让每个探测器一开始就各不相同。

    **那起始值该多大？** 看一个探测器对一张图片的加权和：$z = w_0 \times x_0 + w_1 \times x_1 + \ldots$，共有 $n$ 个输入，旋钮是典型大小为 $s$ 的随机数（方差为 $s^2$；例如 `rng.normal(0, 0.05, …)` 抽出的数典型大小是 0.05，方差是 $0.05^2$）。这些旋钮是**互相独立**的：每个都是单独抽出来的，互不影响。关于互相独立的随机数相加，有一条非常好用的规律：**和的方差等于各自方差之和**。

    再看每一项 $w \times x$：固定一张图片来看，像素 $x$ 是确定的常数，只有旋钮 $w$ 是随机的。把一个随机数乘以常数 $x$，它的分散程度放大 $x$ 倍，方差就会放大 $x^2$ 倍——拿 `[-1, +1]`（方差为 1）心算一下就明白：每个数乘以 3 变成 `[-3, +3]`，平方后是 `[9, 9]`，方差变成 $9 = 3^2$。因此，每一项 $w \times x$ 的方差就是 $s^2 \times x^2$。

    “方差相加”听起来有点抽象，用最小的例子来验证：抛硬币。一枚硬币正面算 $+1$，反面算 $-1$，均值是 0，方差是 $(1^2 + (-1)^2) / 2 = 1$。两枚硬币相加，有四种等可能的结果：

    | 第一枚 | 第二枚 | 和 | 和的平方 |
    |---|---|---|---|
    | +1 | +1 | +2 | 4 |
    | +1 | −1 | 0 | 0 |
    | −1 | +1 | 0 | 0 |
    | −1 | −1 | −2 | 4 |

    和的均值依然是 0，所以方差就是平方的均值：$8 / 4 = 2$，正好等于 $1 + 1$。分散程度既不会互相抵消，也不会相乘，而是直接相加。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 手算：三枚硬币

    三枚硬币有 8 种可能性一样大的结果。其中两种的和是 $+3$ 或 $-3$；另外六种的和是 $+1$ 或 $-1$。和的方差是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    coin_boxes = hand.boxes(["三枚硬币的方差"])
    return (coin_boxes,)


@app.cell(hide_code=True)
def _(coin_boxes):
    hand.marked(coin_boxes, {"三枚硬币的方差": 3}, {
        "三枚硬币的方差": r"""均值是 0，所以求平方的均值：两种结果的平方是 9，六种的平方是 1，一共 8 种结果""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    三枚硬币，方差是 3：每枚贡献 1。回到探测器：把 $n$ 项的方差加起来，得到 $s^2 \times (x_0^2 + x_1^2 + \ldots)$，而 $n$ 个平方之和正好等于 $n$ 乘以它们的均值：

    > $$z \text{ 的方差} \approx n \times s^2 \times (x^2 \text{ 的均值})$$

    在第一张图片上验证一下：用 2,000 组不同的随机旋钮，每组 784 个，大小 0.05：
    """)
    return


@app.cell
def _(X):
    rng = np.random.default_rng(0)
    x = X[0]
    z_samples = []
    for _trial in range(2000):
        z_samples.append(rng.normal(0, 0.05, 784) @ x)
    z_samples = np.array(z_samples)
    variance(z_samples), 784 * 0.05 ** 2 * np.mean(x * x)
    return (z_samples,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    实测约 0.14，预测 0.135，吻合得很好（微小差别来自 2,000 次随机抽样的波动）。接下来经过 ReLU 弯折。这批 `z` 以 0 为中心（每个旋钮取正、取负的可能性一样），大约一半是负的，此时平方的均值就是方差。ReLU 把负半边清零后，均值不再是 0（严格来说平方的均值不再等于方差），但我们真正关心的是信号幅度大小（平方的均值），负半边被清掉一半后正好减半：
    """)
    return


@app.cell
def _(z_samples):
    np.mean(np.maximum(z_samples, 0) ** 2), np.mean(z_samples ** 2) / 2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    0.073 对 0.071，果然正好减半。所以经过一层之后，输出的平方均值变成 $n \times s^2 \times (x^2 \text{ 的均值}) / 2$。这一层的输出就是下一层的输入，也就是说这个平方均值会成为下一层的“$x^2$ 的均值”。要想让信号幅度一层一层传下去既不放大也不缩小，前面的系数就必须等于 1：$n \times s^2 / 2 = 1$，也就是

    > $s^2 = 2 / n$，所以起始的典型大小是 $s = \sqrt{2 / n}$

    第一层有 $n = 784$ 个输入，$\sqrt{2 / 784} \approx 0.05$；第二层有 $n = 64$ 个输入，$\sqrt{2 / 64} \approx 0.18$——这正是第 08 章里的那两个数。这条初始化规则叫作 **He 初始化**，以研究者何恺明（Kaiming He）的名字命名。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：一个起始大小

    某一层有 200 个输入。按上面的规则，它的起始旋钮的方差 $s^2$ 应该是多少？典型大小 $s$ 呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    he_boxes = hand.boxes(['s 的平方', 's'], step=0.01)
    return (he_boxes,)


@app.cell(hide_code=True)
def _(he_boxes):
    hand.marked(he_boxes, {'s 的平方': 0.01, 's': 0.1}, {
        's 的平方': r"""$2 / n$，其中 $n = 200$""",
        's': r"""第一个答案的平方根""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    它的威力在更深的网络里才看得出来。下面让图片依次经过 6 层、每层 64 个 ReLU 探测器，对比三种不同的起始大小。表里的数字是每一层输出信号的典型幅度 $\sqrt{\text{平方的均值}}$（输入像素一开始约为 0.33）。和第 08 章一样，Python 用科学计数法 `e` 表示很小的数：`1.18e-05` 就是 $1.18 \times 0.00001 = 0.0000118$，`e-` 后面的数字表示小数点向左移动几位。
    """)
    return


@app.cell(hide_code=True)
def _(X):
    _rows = ["| 起始大小 | 第 1 层 | 2 | 3 | 4 | 5 | 6 |", "|---|---|---|---|---|---|---|"]
    for _name, _size in (("0.01", lambda n: 0.01), ("0.3", lambda n: 0.3), (r"$\sqrt{2 / n}$", lambda n: np.sqrt(2 / n))):
        _r = np.random.default_rng(0)
        _h = X[:1000]
        _cells = []
        for _layer in range(6):
            _n = _h.shape[1]
            _h = np.maximum(_h @ _r.normal(0, _size(_n), (_n, 64)), 0)
            _cells.append(f"{np.sqrt(np.mean(_h ** 2)):.3g}")
        _rows.append(f"| {_name} | " + " | ".join(_cells) + " |")
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    起点太小，信号几层之后就衰减到几乎为 0：前向没信号，反向传播时梯度也要乘以同样小的旋钮，同样消失得无影无踪。起点太大，信号又会一层比一层暴涨（从第 1 层的 1.8 涨到第 6 层的 24，放大了约 13 倍）。只有 $\sqrt{2 / n}$ 能让信号一路保持平稳。这就是深度网络必须讲究初始化的原因。

    这就像一排人传话：如果每个人都说得比听到的轻一点，传到第六个人就只剩谁也听不清的耳语；如果每个人都喊得更响一点，最后就震耳欲聋。合适的起始大小，让每一层都以刚刚好的音量把信息传下去。

    ## 小批量

    第 08 章里，每走一步都要用全部 10,000 张图片算一次斜率。但斜率本来就是所有图片的**均值**，随机抽 100 张算出来的均值，已经是全部 10,000 张均值的一个不错估计——虽然粗糙一点，计算量却只有原来的百分之一。所以我们宁可用很多“小而粗糙”的步子，代替少数“大而精确”的步子：

    1. 把图片顺序随机打乱。
    2. 切成一**批**一批（batch），比如每批 100 张。
    3. 每批更新一步，只用这 100 张图片算出的斜率。

    把所有批次完整过一遍叫作一个**轮次**（epoch）：10,000 张图片按每批 100 张切分，一个轮次就能更新 100 步，总计算量和原来走 1 步全量更新差不多。

    这就像民意调查：不用挨家挨户问遍全国，随机抽一千个人，结果就八九不离十。下面我们挑出一个刚初始化的网络里斜率最大的旋钮（在 `W2` 中），对比用全部 10,000 张图片算出的斜率，和用五个随机 100 张小批量算出的结果：
    """)
    return


@app.cell(hide_code=True)
def _(X, Y):
    _r = np.random.default_rng(3)
    _k = [_r.normal(0, np.sqrt(2 / 784), (784, 64)), np.zeros(64), _r.normal(0, np.sqrt(2 / 64), (64, 10)), np.zeros(10)]
    _Z, _H, _S = forward(X, *_k)
    _full = backward(X, Y, _Z, _H, _S, _k[2])[2]
    _j, _c = np.unravel_index(np.argmax(np.abs(_full)), _full.shape)
    _rows = ["| 用了哪些图片 | 这个旋钮的斜率 |", "|---|---|", f"| 全部 10,000 张 | **{_full[_j, _c]:.4f}** |"]
    for _i in range(5):
        _pick = _r.permutation(10000)[:100]
        _Zb, _Hb, _Sb = forward(X[_pick], *_k)
        _rows.append(f"| 随机小批量 {_i + 1}（100 张） | {backward(X[_pick], Y[_pick], _Zb, _Hb, _Sb, _k[2])[2][_j, _c]:.4f} |")
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每个小批量都有偏差，有的偏得还不少，偶尔甚至会指反方向（比如这里的第 5 个）。但平均来看它们指向正确的方向，工作量却只有百分之一：多走几步之后，一会儿偏左、一会儿偏右的误差大体上就互相抵消了。
    """)
    return





@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：打乱和切分
    - `rng.permutation(n)` 给出 $0, 1, \ldots, n - 1$ 这些数的一个随机顺序。
    - `range(0, n, size)` 从 0 开始，每次跳 `size`：`range(0, 10, 4)` 给出 0、4、8。
    - `a[start:stop]` 是一个**切片**（slice）：位置 `start`、`start + 1`……一直到 `stop` 之前（不含 `stop`）的那些项。`a[4:8]` 有 4 项。如果 `stop` 超过了末尾，切片就到末尾为止。
    - 数组可以用一整个位置列表来取数：`X[[3, 0, 7]]` 是由 `X` 的第 3、0、7 行组成的矩阵。所以如果 `batch` 里放的是位置，`X[batch]` 就取出这一批的图片。
    ///

    ## 动手算：切成小批

    10 张图片，每批 4 张。一共切成多少批？最后一批有几张？
    """)
    return


@app.cell(hide_code=True)
def _():
    cut_boxes = hand.boxes(['批数', '最后一批的张数'])
    return (cut_boxes,)


@app.cell(hide_code=True)
def _(cut_boxes):
    hand.marked(cut_boxes, {'批数': 3, '最后一批的张数': 2}, {
        '批数': r"""切片分别从位置 0、4、8 开始""",
        '最后一批的张数': r"""只剩下位置 8 和 9""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    画出来是这样。假设打乱之后，十张图片的顺序如下。这个顺序的切片 `[0:4]`、`[4:8]` 和 `[8:12]` 就是三个批次；最后一个切片要到位置 12，
    可顺序到 10 就结束了，所以它就停在那里：
    """)
    return


@app.cell(hide_code=True)
def _():
    _order = [7, 2, 9, 0, 5, 1, 8, 3, 6, 4]
    grids([
        {"name": "打乱后的顺序：rng.permutation(10)", "values": [_order], "cols": [str(i) for i in range(10)],
         "marks": {**{(0, c): "blue" for c in range(4)}, **{(0, c): "box" for c in range(4, 8)},
                   **{(0, c): "red" for c in range(8, 10)}}},
        "→",
        {"name": "第 0 批", "values": [_order[0:4]], "marks": {(0, c): "blue" for c in range(4)}},
        {"name": "第 1 批", "values": [_order[4:8]], "marks": {(0, c): "box" for c in range(4)}},
        {"name": "第 2 批", "values": [_order[8:10]], "marks": {(0, c): "red" for c in range(2)}},
    ], "十个位置打乱后的顺序，切成 4、4、2 个一批")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：batches

    写出 `batches(n, size, rng)`。它把位置 $0$ 到 $n - 1$ 打乱一次，再把打乱后的顺序依次切成每段 `size` 个位置的切片，最后把这些切片放进一个列表返回，每个切片是一个数组。

    `n = 10`、`size = 4` 时，列表里应该有三个数组，分别有 4、4、2 个位置，合起来 0 到 9 的每个位置正好出现一次。
    """)
    return


@app.function
def batches(n, size, rng):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "batches", batches, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""只打乱一次，在循环之前。然后用 `for` 循环遍历各个起点 `range(0, n, size)`，每次把打乱后顺序的一个切片 `append` 进去。""",
        r"""切的是打乱后的顺序，不是 `range(n)`。`return` 在循环之后，和 `for` 对齐。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是用这些部件拼成的完整训练函数。它用 He 初始化设定起始大小，每个轮次里按小批量逐批更新。（代码里提前出现的 `decay * W1` 和 `decay * W2` 会在最后一节讲，现在 `decay` 默认传 0，先忽略即可。另外，对二维矩阵用 `len(X)` 得到的是它的行数。）

    它后面的单元格训练 5 个轮次，并用 `time.perf_counter()`（一个以秒计的时钟）计时。

    /// admonition | 🧰 工具箱：把列表拆成几个输入
    调用时在列表前面加 `*`，就把它拆开：设 `pair = [3, 4]`，`print(*pair)` 就是 `print(3, 4)`。所以 `forward(X_test, *knobs)` 把 `knobs` 里的四组旋钮当作四个单独的输入传进去。
    ///
    """)
    return


@app.function
def train(X, Y, epochs, batch_size, rate, decay, seed):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2 / 784), (784, 64))
    b1 = np.zeros(64)
    W2 = rng.normal(0, np.sqrt(2 / 64), (64, 10))
    b2 = np.zeros(10)
    for epoch in range(epochs):
        for batch in batches(len(X), batch_size, rng):
            Z, H, S = forward(X[batch], W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward(X[batch], Y[batch], Z, H, S, W2)
            W1 = W1 - rate * (dW1 + decay * W1)
            b1 = b1 - rate * db1
            W2 = W2 - rate * (dW2 + decay * W2)
            b2 = b2 - rate * db2
    return W1, b1, W2, b2


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：一共多少步

    读一读 `train`。下一个单元格用 5 个轮次、每批 100 张来调用它，数据是那 10,000 张图片。它一共走多少步，也就是旋钮一共改变多少次？它总共看了多少张图片？
    """)
    return


@app.cell(hide_code=True)
def _():
    step_boxes = hand.boxes(['步数', '看过的图片数'])
    return (step_boxes,)


@app.cell(hide_code=True)
def _(step_boxes):
    hand.marked(step_boxes, {'步数': 500, '看过的图片数': 50000}, {
        '步数': r"""里面的循环每一批跑一次：5 个轮次，每个轮次 100 批""",
        '看过的图片数': r"""每个轮次把全部 10,000 张图片过一遍""",
    }, LOCALE)
    return


@app.cell
def _(X, X_test, Y, test_labels):
    started = time.perf_counter()
    knobs = train(X, Y, 5, 100, 0.5, 0.0, 0)
    seconds = time.perf_counter() - started
    accuracy(forward(X_test, *knobs)[2], test_labels), seconds
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    只跑了 5 个轮次，准确率就达到了约 **94%**，追平了第 08 章 300 步全量训练的效果。第二个数是耗时：不到 1 秒，而第 08 章花了约 10 秒。500 个小批量步子看过的图片总数，只相当于 5 步全量更新；虽然每一步粗糙些，但更新次数多了 100 倍。现代大模型全都是这样训练的——没人会在几十亿条数据上算完一遍总斜率才挪一步。

    ## 动量：记住方向

    回忆第 02 章的米娅卖冰淇淋问题：我们用直线 `w * 气温 + b` 预测冰淇淋销量，最优直线大约是 $w \approx 11.9$、$b \approx -107$。直接用原始气温训练时，`w` 的斜率很大（误差要乘以几十度的气温），`b` 的斜率却很小，所以照顾 `w` 不发散的学习率会让 `b` 像蜗牛一样慢。现在把两个旋钮打包成一个向量 `k = [w, b]`，从第 02 章的起点 `[6, 0]` 出发，用差不多能承受的最大学习率 `0.0016` 走 500 步普通梯度下降，看看结果：
    """)
    return


@app.function
def mia_gradient(k):
    t = np.array(temps)
    s = np.array(sales)
    e = k[0] * t + k[1] - s
    return np.array([np.mean(2 * e * t), np.mean(2 * e)])


@app.function
def mia_loss(k):
    e = k[0] * np.array(temps) + k[1] - np.array(sales)
    return np.mean(e * e)


@app.cell
def _():
    k_plain = np.array([6.0, 0.0])
    for _step in range(500):
        k_plain = k_plain - 0.0016 * mia_gradient(k_plain)
    k_plain, mia_loss(k_plain)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    走了 500 步，`b` 才挪到约 $-11$（离最优直线的 $-107$ 还差得远），损失依然高达 640 左右。

    `b` 的每一步虽然很小，但始终指向**同一个方向**。**动量**（momentum）正好利用了这一点：它维护一个不断累积的**速度**，把最近的斜率叠加起来，同时每一步稍微忘掉一点过去：

    > $\text{速度} = 0.9 \times \text{速度} + \text{斜率}$，然后 $\text{旋钮} = \text{旋钮} - \text{学习率} \times \text{速度}$

    如果斜率一直不变，速度就会持续增大，直到不再变化，也就是 $V = 0.9 \times V + \text{斜率}$。两边同时减去 $0.9 \times V$，得到 $0.1 \times V = \text{斜率}$，即 $V = 10 \times \text{斜率}$：沿着稳定方向前进时，速度最多能加快 10 倍。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：速度是怎样攒起来的

    速度从 0 开始，假设每一步的斜率都是 1。走 1 步、2 步、3 步之后，速度各是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    momentum_boxes = hand.boxes(['1 步之后', '2 步之后', '3 步之后'], step=0.01)
    return (momentum_boxes,)


@app.cell(hide_code=True)
def _(momentum_boxes):
    hand.marked(momentum_boxes, {'1 步之后': 1, '2 步之后': 1.9, '3 步之后': 2.71}, {
        '1 步之后': r"""$0.9 \times 0 + 1$""",
        '2 步之后': r"""$0.9 \times$ 第一个答案，再加 1""",
        '3 步之后': r"""$0.9 \times$ 第二个答案，再加 1""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    它朝着 10 爬上去，也就是上面算出的稳定值：一开始很快，然后越来越慢：
    """)
    return


@app.cell(hide_code=True)
def _():
    _v, _points = 0.0, [(0, 0.0)]
    for _step in range(1, 61):
        _v = 0.9 * _v + 1
        _points.append((_step, _v))
    chart((0, 60), (0, 11), "斜率一直是 1 的步数", "速度", curve=_points, dots=_points[1:4],
          segments=[(0, 10, 60, 10)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    就像一个球沿着长长的缓坡往下滚：每次推力不大，但它攒下了前面推出来的速度，于是越滚越快，直到达到最高速度（这里是单次推力的 10 倍）。

    ## 练习 3：momentum_step

    写出 `momentum_step(w, velocity, grad, rate)`，完成上面那两个公式：先用旧的速度和斜率 `grad` 算出新的速度，再算出新的旋钮。
    返回新的旋钮和新的速度。它对单个旋钮能用，对一整个数组的旋钮也能用（逐元素计算）。
    """)
    return


@app.function
def momentum_step(w, velocity, grad, rate):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "momentum_step", momentum_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""两行：一行算速度，一行返回这一对。""",
        r"""旋钮移动的是 `rate` 乘以**新的**速度，所以要先更新速度。""",
    ], LOCALE)
    return


@app.cell
def _():
    k_momentum = np.array([6.0, 0.0])
    velocity = np.zeros(2)
    for _step in range(500):
        k_momentum, velocity = momentum_step(k_momentum, velocity, mia_gradient(k_momentum), 0.003)
    k_momentum, mia_loss(k_momentum)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    同样走 500 步（学习率 `0.003`，差不多是动量在这里不发散的上限），损失从普通步子的 640 一口降到了约 41。

    ## Adam：每个旋钮一个步长

    动量帮了大忙，但 $w$ 和 $b$ 仍然共用同一个学习率。**Adam** 则给每个旋钮单独配一套步长尺度。它先记录最近斜率的**滑动平均**：$m = 0.9 \times m + 0.1 \times \text{斜率}$。这相当于把动量的速度乘了 0.1，乘完它就是真正的平均——如果斜率一直不变，$m$ 最终会稳定在斜率本身（把 $m = 0.9 \times m + 0.1 \times \text{斜率}$ 两边减去 $0.9 \times m$，即得 $0.1 \times m = 0.1 \times \text{斜率}$），而不是斜率的 10 倍。与此同时，Adam 还记录斜率**平方**的滑动平均，再除以它的平方根（也就是这个旋钮最近的典型斜率大小）：

    > $$\text{步长} = \text{学习率} \times \frac{\text{平均斜率}}{\sqrt{\text{平均平方斜率}}}$$

    这样一来，斜率很大的旋钮（`w`）除完后步子变小，斜率很小的旋钮（`b`）除完后步子变大，每个旋钮都能以差不多的节奏前进。第 1 步时，平均斜率就是斜率本身，分母 $\sqrt{\text{斜率}^2}$ 是它的绝对值，两者相除不是 $+1$ 就是 $-1$，所以每个旋钮第 1 步都正好移动一个 `rate` 的距离；往后随着滑动平均趋于平稳，旋钮越接近最优直线，步子就会自动缩小。后面的单元格会用学习率 `1.0` 跑完 500 步（途中 `b` 在第 100 步就到了约 $-42$，第 300 步到了约 $-99$）。这其实和我们在第 04、05 章手工缩放气温的效果一模一样，只不过现在由 Adam 对每个旋钮全自动完成。

    还有一个细节：两个滑动平均都从 0 起步，所以开头几步会明显偏小。不妨假设每一步斜率都是 $g$：第 1 步 $m_1 = 0.1g = (1 - 0.9^1)g$，第 2 步 $m_2 = 0.9 \times 0.1g + 0.1g = 0.19g = (1 - 0.9^2)g$。一眼就能看出，第 $t$ 步时 $m$ 总是被缩成了原来的 $1 - 0.9^t$！所以 Adam 的修正办法非常直接：把 $m$ 除以 $1 - 0.9^t$、把 $v$ 除以 $1 - 0.999^t$（代码里乘方写作 `**`，比如 `0.9 ** t`），其中 $t$ 是从 1 开始计数的步数。这样就能把偏小的开头正好补齐；而随着 $t$ 变大，$0.9^t$ 很快趋近于 0，修正系数也就自然退化成除以 1。（分母末尾加上的 `1e-8`，即 0.00000001，是为了防止万一斜率为 0 时发生除以 0 的错误。）
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面是 Adam 完整的一步，$g$ 是斜率，$t$ 是第几步：

    > $$m = 0.9 \times m + 0.1 \times g \qquad v = 0.999 \times v + 0.001 \times g^2$$

    > $$\hat{m} = \frac{m}{1 - 0.9^t} \qquad \hat{v} = \frac{v}{1 - 0.999^t}$$

    > $$w \leftarrow w - \text{rate} \times \frac{\hat{m}}{\sqrt{\hat{v}} + 0.00000001}$$

    $\hat{m}$ 和 $\hat{v}$（读作“m 帽”和“v 帽”）是修正之后的平均。

    ## 动手算：Adam 的第一步

    一个旋钮的第一步：$m$ 和 $v$ 从 0 开始，$t = 1$，斜率 $g = 4$，学习率是 0.5。把公式一个个算下来。最后一格填旋钮移动了多少。（那个很小的 0.00000001 不用管。）
    """)
    return


@app.cell(hide_code=True)
def _():
    adam_boxes = hand.boxes(['m', 'v', 'm 帽', 'v 帽', '旋钮移动了'], step=0.001)
    return (adam_boxes,)


@app.cell(hide_code=True)
def _(adam_boxes):
    hand.marked(adam_boxes, {'m': 0.4, 'v': 0.016, 'm 帽': 4, 'v 帽': 16, '旋钮移动了': 0.5}, {
        'm': r"""$0.9 \times 0 + 0.1 \times 4$""",
        'v': r"""$0.999 \times 0 + 0.001 \times 4^2$""",
        'm 帽': r"""$m / (1 - 0.9)$""",
        'v 帽': r"""$v / (1 - 0.999)$""",
        '旋钮移动了': r"""$\text{rate} \times \hat{m} / \sqrt{\hat{v}}$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    旋钮正好移动了 `0.5`，刚好等于学习率，和前面说的一样。在脑子里把斜率换成 4,000 再算一遍：答案还是 0.5。

    ## 练习 4：adam_step

    写出 `adam_step(w, m, v, grad, t, rate)`，按顺序完成上面的公式。返回新的旋钮和两个平均，也就是没有修正过的 `m` 和 `v`，因为下一步要用的是它们。
    在代码里，乘方写作 `**`，`1e-8` 就是 0.00000001。

    检查之后，有一个单元格会在米娅的问题上跑 500 步 Adam。`range(1, 501)` 数的是 1、2……500，所以 `t` 从 1 开始。
    """)
    return


@app.function
def adam_step(w, m, v, grad, t, rate):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "adam_step", adam_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""五行，每个公式一行：`m`、`v`、两个修正后的平均（另起名字），最后是 `return`。""",
        r"""不要把修正后的平均存回 `m` 和 `v`：返回的是没有修正过的。如果得到 `nan`，多半是 `np.sqrt(...) + 1e-8` 外面少了括号。""",
    ], LOCALE)
    return


@app.cell
def _():
    k_adam = np.array([6.0, 0.0])
    m = np.zeros(2)
    v = np.zeros(2)
    for t in range(1, 501):
        k_adam, m, v = adam_step(k_adam, m, v, mia_gradient(k_adam), t, 1.0)
    k_adam, mia_loss(k_adam)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    同样走 500 步，Adam 稳稳落在了最优直线上：$w \approx 11.9$，$b \approx -107$，损失降到约 26.8——而且用的是原始气温，没做任何缩放。如今几乎所有大模型（包括本课程后面的 Transformer）都是用 Adam 或它的变体训练的。

    下面画出三种规则在这 500 步里 `b` 的变化轨迹（每种都用各自不发散的最大学习率），虚线是最佳值 $-107$：
    """)
    return


@app.cell(hide_code=True)
def _():
    try:
        _paths = {"普通步子": [], "动量": [], "Adam": []}
        _k = np.array([6.0, 0.0])
        for _step in range(500):
            _k = _k - 0.0016 * mia_gradient(_k)
            _paths["普通步子"].append((_step + 1, _k[1]))
        _k, _vel = np.array([6.0, 0.0]), np.zeros(2)
        for _step in range(500):
            _k, _vel = momentum_step(_k, _vel, mia_gradient(_k), 0.003)
            _paths["动量"].append((_step + 1, _k[1]))
        _k, _m, _v = np.array([6.0, 0.0]), np.zeros(2), np.zeros(2)
        for _t in range(1, 501):
            _k, _m, _v = adam_step(_k, _m, _v, mia_gradient(_k), _t, 1.0)
            _paths["Adam"].append((_t, _k[1]))
        _out = chart((0, 500), (-130, 10), "步数", "b", series=list(_paths.items()), segments=[(0, -107, 500, -107)])
    except Exception:
        _out = mo.md("*完成练习 3 和 4 后，这里会画出三条路径。*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    普通步子慢如蜗牛，500 步走完 `b` 才刚起步；动量攒起速度，走完了大半程；Adam 则直奔虚线而去，因为它从不让 `w` 巨大的斜率拖慢 `b` 的脚步。

    ## 是学会了，还是背下来了？

    最后一个问题。让同一个网络只用 **500** 张图片训练 200 个轮次，对比它在这 500 张上的准确率和在 2,000 张测试图片上的准确率。（下面的单元格跑两次：`decay` 为 0 一次，为 0.003 一次，后面会解释。输出依次是：训练准确率、测试准确率、加上衰减后的测试准确率。）
    """)
    return


@app.cell
def _(X, X_test, Y, test_labels, train_labels):
    small = train(X[:500], Y[:500], 200, 50, 0.5, 0.0, 0)
    small_decay = train(X[:500], Y[:500], 200, 50, 0.5, 0.003, 0)
    [
        accuracy(forward(X[:500], *small)[2], train_labels[:500]),
        accuracy(forward(X_test, *small)[2], test_labels),
        accuracy(forward(X_test, *small_decay)[2], test_labels),
    ]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    在训练用的 500 张图片上 100% 正确，在新图片上却只有约 87%。网络的旋钮多到足以把这 500 张图片**死记硬背**下来，连划痕和污渍都记住了，而背下来的细节到了新笔迹上根本派不上用场。这叫作**过拟合**（overfitting）：训练成绩看着满分，却代表不了模型在新数据上的真实水平。这就是我们总要留出测试集的原因。

    这就像学生不学解题方法，光背去年考卷的答案：做去年的卷子能拿满分，遇到今年的卷子就卡壳了。下面是一次类似的训练（同样的 500 张图片，100 个轮次），每个轮次结束后都记录一遍两个准确率：
    """)
    return


@app.cell(hide_code=True)
def _(X, X_test, Y, test_labels, train_labels):
    try:
        _r = np.random.default_rng(0)
        _k = [_r.normal(0, np.sqrt(2 / 784), (784, 64)), np.zeros(64), _r.normal(0, np.sqrt(2 / 64), (64, 10)), np.zeros(10)]
        _seen, _new = [], []
        for _epoch in range(1, 101):
            for _batch in batches(500, 50, _r):
                _Z, _H, _S = forward(X[_batch], *_k)
                _g = backward(X[_batch], Y[_batch], _Z, _H, _S, _k[2])
                _k = [_w - 0.5 * _d for _w, _d in zip(_k, _g)]
            _seen.append((_epoch, accuracy(forward(X[:500], *_k)[2], train_labels[:500])))
            _new.append((_epoch, accuracy(forward(X_test, *_k)[2], test_labels)))
        _out = chart((0, 100), (0.6, 1.0), "轮次", "准确率",
                     series=[("用来学习的 500 张", _seen), ("2,000 张新图片", _new)])
    except Exception:
        _out = mo.md("*完成练习 2 后，这里会画出两条曲线。*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    最初几个轮次，两条曲线一起往上爬，说明网络学到的是所有手写数字共有的规律。随后它们分道扬镳：训练曲线一路冲到 100%，测试曲线却在 90% 下方停滞不前。从那以后多学到的东西，全都是这 500 张特定图片的死细节，对新图片毫无帮助。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：读懂这两个分数

    这个网络把 500 张训练图片全部认对（100%），2,000 张测试图片认对了 87.25%。两组图片各认错了多少张？
    """)
    return


@app.cell(hide_code=True)
def _():
    fit_boxes = hand.boxes(['认错的训练图片', '认错的测试图片'])
    return (fit_boxes,)


@app.cell(hide_code=True)
def _(fit_boxes):
    hand.marked(fit_boxes, {'认错的训练图片': 0, '认错的测试图片': 255}, {
        '认错的训练图片': r"""全对，就一张也没错""",
        '认错的测试图片': r"""2,000 的百分之 $100 - 87.25 = 12.75$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **权重衰减**（weight decay）是一种常用的对策：在损失里加上一项对大权重的小小惩罚，大小是 $(\text{decay} / 2) \times (\text{所有权重平方的和})$。这里的权重只包括 `W1` 和 `W2`——各个 `b` 通常叫作*偏置*，一般不加惩罚，所以叫*权重*衰减。微调某一个权重 $w$ 时，只有它自己的平方会变，其他项不受影响。按第 02 章的规则，$w^2$ 的斜率是 $2 \times w$，因此 $(\text{decay} / 2) \times w^2$ 的斜率就是 $(\text{decay} / 2) \times 2 \times w = \text{decay} \times w$，更新规则就变成：

    > $\text{权重} = \text{权重} - \text{学习率} \times (\text{斜率} + \text{decay} \times \text{权重})$

    这就是前面 `train` 里 `+ decay * W1` 的来历：每一步都会顺手把每个权重往 0 拉回一点点。只有当数据持续需要某个权重发挥作用时，它才会留在较大的数值上，没必要的权重则会被压小。在上面这个 500 张图的小例子里，权重衰减的效果很有限（测试准确率从 87.25% 提到 87.4%，2,000 张里多对了 3 张），但在长时间训练的大模型里它非常管用。不过，对付过拟合最根本的办法其实更简单：**增加数据**——同一个网络放到全部 10,000 张图片上训练，测试准确率就达到了约 94%。

    ## 下一步

    深度学习库做的核心工作，你现在都已经亲手实现了一遍：前向计算、矩阵反向传播、合理初始化、小批量训练、动量和 Adam。在**第 10 章**里，你会认识这个领域最常用的库 **PyTorch**——看看它如何替你完成这一切，再亲手验证它的计算结果和你写的一模一样。

    ## 你现在会做的事

    - 用方差以及它的平方根（典型大小）衡量一组数的分散程度
    - 说清楚为什么全 0 初始化行不通，并推出 He 初始化的起点 $\sqrt{2 / n}$
    - 用打乱的小批量和轮次来训练，熟练使用切片和 `rng.permutation`
    - 用动量（累积最近斜率的速度）加快训练
    - 用 Adam 给每个旋钮自适应步长，并理解开头的偏差修正
    - 从训练成绩和测试成绩识别过拟合，并用权重衰减或更多数据来应对

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**小批量训练叫作*随机梯度下降*（SGD）。起点 $\sqrt{2 / n}$ 叫作 *He*（或 *Kaiming*）初始化。*Dropout* 和*批归一化*也是这一类工具；Transformer 那几章会用到*层归一化*。
    """)
    return


if __name__ == "__main__":
    app.run()
