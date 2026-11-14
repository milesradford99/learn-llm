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

    LOCALE = "zh-CN"
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 09 · 让训练靠得住

    第 08 章的网络准确率达到了大约 94%，但还有几处没交代清楚。我们给旋钮选的起始大小 0.05 和 0.18 没说理由。每一步都要看全部 10,000 张图片，
    所以 300 步花了大约 10 秒。再往前，第 02 章里米娅的 `b` 挪得很慢，因为一个学习率没法适合所有旋钮；从那以后我们一直靠换算输入来绕开这个问题。

    这一章把这些问题一个一个解决：有依据的起点、**小批量**（mini-batch）、两种更聪明的步进规则——**动量**（momentum）和 **Adam**，以及怎样识别一个只会死记、没有真正学会的模型。
    现代的每一次训练都离不开这些做法，包括本课程最后的语言模型。

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
    - `np.eye(10)` 是对角线上为 1 的 $10 \times 10$ 表格，它的第 $k$ 行正好是数字 $k$ 的 one-hot 行。所以上面单元格里的 `np.eye(10)[train_labels]` 是第 07 章 `one_hot` 的简便写法：给它一串标签，就为每张图片挑出这样的一行。（用列表一次挑出好几行，会在“小批量”一节讲。）
    - `np.sqrt` 是平方根：`np.sqrt(9)` 是 3，因为 $3 \times 3 = 9$。
    - `np.abs` 去掉负号：`np.abs(np.array([-2, 5]))` 是 `[2, 5]`。所以 `np.max(np.abs(W))` 就是 `W` 里最大的那个大小，不管正负。
    ///

    ## 有多分散？方差

    要选起始大小，我们需要一种衡量一组数有多分散的办法。先求它们的均值，再求每个数离均值的距离并平方（这样正负不会抵消，和第 01 章一样），
    然后求这些平方的均值：

    > $$\textbf{方差}\text{（variance）} = (\text{数} - \text{均值})^2 \text{ 的均值}$$

    对 `[1, 3]`：均值是 2，距离是 −1 和 1，平方后是 1 和 1，所以方差是 1。方差的平方根就是这些距离的**典型大小**（typical size）。NumPy 里用 `np.sqrt`
    求平方根：9 的平方根是 3，因为 $3 \times 3 = 9$。对以 0 为中心的数，均值是 0，方差就是平方的均值。
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

    按上面的公式写出 `variance(v)`，`v` 是一个 NumPy 数组。数组的算术是逐个位置进行的，`** 2` 是平方。

    `[1, 3]` 应该得到 1，你刚才算的 `[2, 4, 4, 6]` 应该得到 2。
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

    **为什么不从 0 开始？** 如果每个旋钮都从 0 开始，每个隐藏的探测器算出的东西完全一样（全是 0），得到的责任完全一样，更新也完全一样。
    它们会永远一模一样。更糟的是：`W2` 也是 0 时，第 08 章的 `dH = G @ W2.T` 全是 0，所以根本没有责任能传到 `W1`。
    下面的单元格把一个从 0 开始的网络训练 100 步：
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
    第一层最大的旋钮仍然正好是 0（`np.max(np.abs(...))` 就是最大的那个大小），测试准确率大约 12%：
    只有 `b2`（分数的起始高度）学到了东西（`b1` 也收不到任何责任），所以它对每张图片都给出同一个数字，和瞎猜差不多。随机的起始旋钮能打破这种完全相同的局面：每个探测器一开始就各不相同。

    **该多大？** 看一个探测器对一张图片的加权和：$z = w_0 \times x_0 + w_1 \times x_1 + \ldots$，共 $n$ 个输入，旋钮是典型大小为 $s$ 的随机数（所以方差是 $s^2$；
    `rng.normal(0, 0.05, …)` 给出的数典型大小是 0.05，方差是 $0.05^2$）。这些旋钮是**互相独立**的：每个都是单独抽出来的，不看其他旋钮。
    一个很有用的事实：互相独立的随机数相加，**和的方差等于各自方差之和**。每一部分 $w \times x$（图片是固定的，只有旋钮是随机的）的方差是 $s^2 \times x^2$
    （把一个随机数乘以 $x$，它的分散程度也乘以 $x$，方差就乘以 $x^2$）。把 $n$ 部分加起来得到 $s^2 \times (x_0^2 + x_1^2 + \ldots)$，
    而这 $n$ 个平方的和等于 $n$ 乘以它们的均值：

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
    实测约 0.14，预测 0.135：规则成立（小小的差别来自 2,000 次抽样的运气）。然后是弯折。这些 `z` 以 0 为中心（每个旋钮取正、取负的可能性一样），
    所以大约一半是负的，而且它们平方的均值就是方差。ReLU 把为负的那一半 `z` 变成 0，保留为正的那一半，所以平方的均值应该减半：
    """)
    return


@app.cell
def _(z_samples):
    np.mean(np.maximum(z_samples, 0) ** 2), np.mean(z_samples ** 2) / 2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    0.073 对 0.071：差不多是一半。所以经过一层之后，平方的均值是 $n \times s^2 \times (x^2 \text{的均值}) / 2$。这一层的输出就是下一层的输入，
    所以这个平方的均值就成了下一层的“$x^2$ 的均值”。要让它一层一层保持不变，前面的系数必须是 1：$n \times s^2 / 2 = 1$，也就是

    > $s^2 = 2 / n$，所以起始的典型大小是 $s = \sqrt{2 / n}$

    第一层有 $n = 784$ 个输入：$\sqrt{2 / 784} \approx 0.05$。第二层 $n = 64$：$\sqrt{2 / 64} \approx 0.18$。正是第 08 章的那两个数。这条起始规则叫作
    **He 初始化**，以研究者何恺明（Kaiming He）的名字命名。
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
    它的重要性在更深的网络里才看得出来。下面，图片依次经过 6 层、每层 64 个 ReLU 探测器，起始大小有三种选择。表里的数是每一层之后数值的 $\sqrt{\text{平方的均值}}$（输入的像素一开始约为 0.33）。和第 08 章一样，Python 用 `e` 来写很小的数：`1.18e-05` 表示 $1.18 \times 0.00001$，也就是 0.0000118；`e-` 后面的数表示小数点向左移几位。
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
    起点太小，数值几层之后就缩小到几乎为 0：没有信号，而且（反向时责任也要乘以同样小的旋钮）也没有责任。起点太大，数值就一层比一层大（从第 1 层之后的 1.8 涨到第 6 层之后的 24，大了约 13 倍）。
    $\sqrt{2 / n}$ 让它们保持稳定。所以，深度网络的起点必须有讲究。

    ## 小批量

    第 08 章的每一步都用全部 10,000 张图片来算一个斜率。可斜率是对所有图片求出的**均值**，而随机挑 100 张图片的均值，已经是全部 10,000 张的均值的
    一个不错的估计：粗糙一些，却便宜 100 倍。所以我们用很多小而粗糙的步子，代替少数大而精确的步子：

    1. 把图片打乱成随机的顺序。
    2. 把这个顺序切成一**批**一批（batch），比如每批 100 张。
    3. 每批走一步，只用这 100 张图片的斜率。

    把所有批次都走一遍叫作一个**轮次**（epoch）：10,000 张图片、每批 100 张，一个轮次就是 100 步，花的却是一步全量计算的代价。
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
    下面是用这些部件搭成的完整训练函数。它按 He 初始化选起始大小，每个轮次里每批走一步。（`decay` 先不用管；在最后一节之前它都是 0。对矩阵用 `len(X)` 得到的是它的行数。）

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
    五个轮次就达到了大约 **94%**，和第 08 章的 300 步全量计算一样。第二个数是花掉的秒数：不到一秒，而第 08 章大约要 10 秒。
    500 个小步看过的图片和 5 个全量步一样多。每个小步都更粗糙，但步数多了 100 倍。
    每个大模型都是这样训练的：没有人会在一个有几十亿样本的数据集上整体算一次斜率。

    ## 动量：记住方向

    回到第 02 章的问题。在米娅的原始气温上，`w` 的斜率很大（误差要乘以气温，而气温有几十度），`b` 的斜率很小，所以对 `w` 足够小的学习率会让 `b` 挪得很慢。
    把两个旋钮装进一个向量 `k = [w, b]`，从第 02 章的起点 `[6, 0]` 出发，用差不多是不发散的最大学习率（0.0016）走普通的步子，看看会怎样：
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
    走了 500 步，`b` 才到大约 −11（最好的直线 `b` 约是 −107），损失还在 640 左右。

    `b` 的每一步都很小，但它们都指向**同一个方向**。动量就利用这一点：它保留一个不断累积的**速度**（velocity），把最近的斜率加起来，
    每一步都忘掉一点过去：

    > $\text{速度} = 0.9 \times \text{速度} + \text{斜率}$，然后 $\text{旋钮} = \text{旋钮} - \text{学习率} \times \text{速度}$

    如果斜率保持不变，速度会一直增大，直到不再变化，也就是 $V = 0.9 \times V + \text{斜率}$。两边同时减去 $0.9 \times V$：$0.1 \times V = \text{斜率}$，所以 $V = 10 \times \text{斜率}$：
    稳定的前进最多能快 10 倍。
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
    它朝着 10 爬上去，也就是上面算出的稳定值：一开始很快，然后越来越慢。

    ## 练习 3：momentum_step

    写出 `momentum_step(w, velocity, grad, rate)`，完成上面那两个公式：先用旧的速度和斜率 `grad` 算出新的速度，再算出新的旋钮。
    返回新的旋钮和新的速度。它对单个旋钮能用，对一整个数组的旋钮也能用（逐个位置计算）。
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
    同样 500 步（学习率 0.003，差不多是动量在这里不发散的最大值）：损失降到大约 41，而普通的步子还在 640。

    ## Adam：每个旋钮一个步长

    动量有帮助，但 $w$ 和 $b$ 仍然共用一个学习率。**Adam** 给每个旋钮一个自己的尺度。它保留最近斜率的**平均**：$m = 0.9 \times m + 0.1 \times \text{斜率}$。
    这就是动量的速度乘以 0.1，乘了 0.1 它才是真正的平均：如果斜率保持不变，$m$ 最终就等于斜率本身（把 $m = 0.9 \times m + 0.1 \times \text{斜率}$ 两边同时减去 $0.9 \times m$：$0.1 \times m = 0.1 \times \text{斜率}$），
    而不是斜率的 10 倍。与此同时，Adam 还保留斜率**平方**的平均，并除以它的平方根，也就是这个旋钮最近的典型斜率大小：

    > $$\text{步长} = \text{学习率} \times \frac{\text{平均斜率}}{\sqrt{\text{平均平方斜率}}}$$

    斜率很大的旋钮（`w`），除完之后步子变小；斜率很小的旋钮（`b`），步子变大。每个旋钮最后都以差不多的速度前进：第一步时，每个旋钮正好移动 `rate`（学习率）那么多（这时平均斜率就是斜率本身，$\sqrt{\text{斜率}^2}$是它的大小，两者之比是 +1 或 −1）。之后两个平均逐渐稳定下来，旋钮越接近最好的直线，步子就越小。下面的单元格只显示学习率 1.0 走完 500 步后的结果；在这途中，`b` 第 100 步时约为 −42，第 300 步时约为 −99。
    这正是我们在第 04、05 章换算气温时手工做的事，现在自动完成，而且是一个旋钮一个旋钮地做。

    一个细节：两个平均都从 0 开始，所以前几步时它们都太小（走了一步之后，平均斜率只有 $0.1 \times \text{斜率}$）。Adam 的修正办法是除以 $1 - 0.9^t$ 和
    $1 - 0.999^t$（代码里乘方要写 `**`：`0.9 ** t`），其中 $t$ 从 1 开始数步数：第 1 步之后分别除以 0.1 和 0.001，正好把缩小的部分补回来；$t$ 越大，这个修正就越接近除以 1。
    （`1e-8` 也就是 0.00000001，是为了永远不会除以 0。）
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
    旋钮正好移动了学习率那么多，0.5，和前面说的一样。在脑子里把斜率换成 4,000 再算一遍：答案还是 0.5。

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
    同样走 500 步，Adam 就停在了最好的那条直线上：`w` 约 11.9，`b` 约 −107，损失约 26.8，用的是原始气温，没有做任何换算。
    如今几乎所有大模型都是用 Adam（稍加改动）训练的，包括本课程后面的 Transformer。

    ## 是学会了，还是背下来了？

    最后一个遗留问题。让同一个网络只用 **500** 张图片训练 200 个轮次，然后比较它在这 500 张上的准确率和在 2,000 张测试图片上的准确率。
    （下面的单元格训练两次：`decay` 为 0 一次，为 0.003 一次，后面会解释。它依次显示：训练准确率、测试准确率、加了衰减后的测试准确率。）
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
    在自己的 500 张训练图片上 100% 正确，在新图片上却只有大约 87%。网络的旋钮多到足以把这 500 张图片**背下来**，连划痕和污渍一起，
    而背下来的细节到了新的笔迹上就派不上用场。这叫作**过拟合**（overfitting）：训练成绩看起来完美，却说明不了模型在新数据上的表现。
    这就是我们总要留出测试数据的原因。
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
    **权重衰减**（weight decay）是一种常用的对策。给损失加上一个对大旋钮的小小惩罚：$(\text{decay} / 2) \times (\text{所有权重平方的和})$，这里的权重指 `W1` 和 `W2`
    里的数（各个 $b$，通常叫作*偏置*（bias），一般不管，所以叫*权重*衰减）。轻推一个权重 $w$，只有它自己的平方会变，其他的不变。按第 02 章的规则，$w^2$ 的斜率是 $2 \times w$，
    所以 $(\text{decay} / 2) \times w^2$ 的斜率是 $(\text{decay} / 2) \times 2 \times w = \text{decay} \times w$，步进规则变成

    > $\text{权重} = \text{权重} - \text{学习率} \times (\text{斜率} + \text{decay} \times \text{权重})$

    这就是 `train` 里的 `+ decay * W1`：每一步还会让每个权重朝 0 收缩一点，一个权重只有在数据一直需要它时，才能保持很大。
    在这里效果很小：上面的最后一个数是 87.4%，对比 87.25%，2,000 张里只多对了 3 张。在训练很久的大模型里，权重衰减的作用更大。最有力的办法其实更简单：**更多数据**。同一个网络用全部 10,000 张图片训练时，达到了大约 94%。

    ## 下一步

    深度学习库做的所有事情，你现在都亲手做过了：正向计算、矩阵上的反向传播、好的起点、小批量、动量和 Adam。在**第 10 章**里，
    你会认识 **PyTorch**，这是这个领域里大多数人用的库。你会看到它替你完成这一切，再检查它的答案和你的是否一致。

    ## 你现在会做的事

    - 用方差、以及它的平方根（典型大小）来衡量分散程度
    - 说清楚为什么从 0 开始不行，并推出起点 $\sqrt{2 / n}$
    - 用打乱的小批量和轮次来训练，会用切片和 `rng.permutation`
    - 用动量（把最近的斜率累加起来的速度）加快训练
    - 用 Adam 让每个旋钮有自己的步伐，包括它开头的修正
    - 从训练成绩和测试成绩认出过拟合，并用权重衰减或更多数据来对付它

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**小批量训练叫作*随机梯度下降*（SGD）。起点 $\sqrt{2 / n}$ 叫作 *He*（或 *Kaiming*）初始化。*Dropout* 和*批归一化*
    也是这一类工具；Transformer 那几章会用到*层归一化*。
    """)
    return


if __name__ == "__main__":
    app.run()
