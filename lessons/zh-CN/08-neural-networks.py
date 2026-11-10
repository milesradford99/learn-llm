import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="08 · 神经网络")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    from learn_llm import images, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.plot import chart

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 08 · 神经网络

    第 06 章的模型能认对大约 90% 的测试数字。训练得再久一点，它会慢慢爬到大约 91%，然后就停住了。问题不在训练，而在于模型**能**表达什么。
    每个数字只有一个模板，图片的分数是其像素的加权和。加权和能说“这里有墨水就加分，那里有墨水就减分”，
    却说不出“4 的顶部要么是开口的，**要么**是封口的”：两种不同的形状，都是 4。单单一个模板只能把两种形状混成一个模糊的 4，哪种都对不上；
    而一张两处都有墨水的图片反而会得到最高分，尽管没有哪个 4 长那样。

    这一章，你会补上缺的那一样东西：一个**弯折**（bend），并搭出一个**神经网络**（neural network）。

    ## 一个弯折

    回想第 04 章米娅遇到的那次热浪：销量随气温上升，到大约 33 °C 后开始下降。直线跟不上这种变化。现在来认识最简单的弯折，叫 **ReLU**：

    > $z$ 为正数时 $\operatorname{relu}(z) = z$，否则为 $0$

    比如 $\operatorname{relu}(-3) = 0$，$\operatorname{relu}(0) = 0$，$\operatorname{relu}(2) = 2$。它先是平的（等于 0），在 $z = 0$ 处拐一个角，然后像直线一样上升。单独看它没什么意思。但把它**加**到一条直线上，就得到一条在你选定的地方拐弯的线。
    下面是一条在 33 °C 处拐弯的线（−105、12 和 −20 这几个数是手工挑的）：

    > $$\text{猜测} = -105 + 12 \times t - 20 \times \operatorname{relu}(t - 33)$$

    33 °C 以下，$t - 33$ 是负数，ReLU 给出 0，这就是直线 $-105 + 12 \times t$。33 °C 以上，ReLU 给出 $t - 33$，加到直线上，前面的 $- 20 \times$ 让线往下弯：
    从那以后，每升高一度，变化是 $12 - 20 = -8$。
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
        mo.md(f"这条弯折线在 11 天上的损失：**{_loss:.0f}**。"
              "最好的直线大约是 810；第 04 章的曲线大约是 283。"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    一个弯折就已经胜过第 04 章的曲线了。弯折越多（每个都在自己的位置、有自己的大小），就能一段一段地跟上任何形状。
    这就是神经网络的核心想法：许多简单的部件，每个带一个弯折，加在一起。

    在 NumPy 里，`np.maximum(Z, 0)` 逐个位置把每个数和 0 比较，留下较大的那个：正好是对整个数组一次做 ReLU。

    ## 练习 1：relu

    写出 `relu(Z)`：`return np.maximum(Z, 0)`。（记得用英文输入法。）
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
    mo.md(r"""
    ## 隐藏层

    现在回到数字。我们不再从像素直接走到十个分数，而是在中间多走一步，先算出 64 个数：

    1. `Z = X @ W1 + b1`：像素的 64 个加权和，和第 06 章的分数一模一样，只是有 64 个。`W1` 的形状是 `(784, 64)`：64 个模板。
       `Z` 的形状是 `(n, 64)`。
    2. `H = relu(Z)`：给每一个加上弯折，就像上面 $\operatorname{relu}(t - 33)$ 让直线拐弯一样（$b_1$ 决定每个弯折的位置）。和模板吻合得差的（加权和为负）给出 0；
       吻合得好的，输出它吻合的程度。$H$ 的每一列都是一个小**探测器**（detector）：
       它说明自己的图案在图片里出现得有多明显，没出现时就保持沉默。
    3. `S = H @ W2 + b2`：十个分数，每个都是这 64 个探测器的加权和。`W2` 的形状是 `(64, 10)`。

    `H` 叫作**隐藏层**（hidden layer）：它既不是输入，也不是答案，只存在于两者之间。整个结构就叫**神经网络**。
    现在，4 的分数可以说，比如“探测器 12（开口的顶部）**或者**探测器 40（封口的顶部）”（编号是随便举的）：每个亮起来的探测器加上自己那一份，沉默的探测器什么也不加。

    分数接下来照样经过第 06 章的 softmax 和交叉熵。下面把它们再写一遍，还有第 06 章的 `accuracy` 和 `one_hot`。它们都已经写好了，不要改动：
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
    ## 练习 2：正向计算

    写出 `forward(X, W1, b1, W2, b2)`，返回全部三步的结果（算斜率时还要用到 `Z` 和 `H`）：

    1. `Z = X @ W1 + b1`
    2. `H = relu(Z)`
    3. `S = H @ W2 + b2`
    4. `return Z, H, S`

    （`forward` 是“向前”，也就是正向计算。）
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
    mo.md(r"""
    ## 为什么弯折很重要

    如果没有 ReLU，第 3 步算的就是 `(X @ W1 + b1) @ W2 + b2`。先不管那些 `b`：`@` 和普通乘法一样，先算哪一对都行，
    所以 `(X @ W1) @ W2 = X @ (W1 @ W2)`，而 `W1 @ W2` 就是一个 `(784, 10)` 的矩阵：又回到每个数字一个模板。没有弯折的两层，并不比一层强。
    用几个小的随机矩阵验证一下这种分组（这个单元格里的新工具，紧接着就会解释）：
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
    全都（几乎）是 0：两种分组方式得到同一个矩阵。像 `2.22044605e-16` 这样的数，意思是 $2.2 \times 0.0000000000000001$，是最后一位上的舍入误差；
    零后面的 `e+00` 表示“乘以 1”：就是 0。

    这个单元格用到了一样新东西：**随机数**。`np.random.default_rng(0)` 做出一个随机数生成器；`0` 是一个起点（叫作*种子*（seed）），
    它让每次运行这一页时都生成同样的数。`rng.normal(0, 1, (3, 4))` 给出一个 `(3, 4)` 的数组，里面是散布在 0 附近的随机数，
    大多在 −2 到 2 之间（`0` 是中心，`1` 是典型大小）。

    我们需要随机数来给网络一个起点。如果每个旋钮都从 0 开始，每个隐藏的探测器就会一模一样，而且永远一模一样。
    所以我们从很小的随机旋钮开始：`W1` 的典型大小是 0.05，`W2` 是 0.18。（第 09 章会解释这两件事：为什么不从 0 开始，以及为什么是这两个大小。）
    下一个单元格用自己的生成器，种子是 1，所以你的起点和这里描述的一样。

    ## 数据，以及第一次正向计算
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
    形状是 `(10000, 64)`、`(10000, 64)` 和 `(10000, 10)`，损失约 2.41，比什么都不懂时的 2.303 还差一点：随机的旋钮做出的是随机的猜测。

    ## 经过两层求斜率

    现在来求梯度。旋钮要经过一串步骤才影响到损失（`W1` → `Z` → `H` → `S` → 损失），而第 03 章告诉过我们链条该怎么处理：倒着走，把责任传回去。
    我们从末尾开始，一步一步来。我们给每个责任的名字前面加一个 `d`：`dH` 是 `H` 的责任，`dW1` 是 `W1` 的责任（斜率），依此类推。

    **分数。** 损失是 `softmax(S)` 的交叉熵，和第 06 章一样，所以分数的责任就是第 06 章的 `G = (P - Y) / n`，其中 $P = \operatorname{softmax}(S)$。

    **`W2` 和 `b2`。** `S = H @ W2 + b2` 正是第 06 章的 `X @ W + b`，只是用 `H` 代替了 `X`。所以结论相同：`dW2 = H.T @ G`，
    `db2 = np.sum(G, axis=0)`。

    **$H$。** 这一步是新的：第 06 章从来不需要求**输入**的斜率。取一个隐藏值 `H[i, j]`（图片 $i$、探测器 $j$）。它出现在图片 $i$
    的全部十个分数里：写出来就是 $S[i, c] = H[i, 0] \times W_2[0, c] + \ldots + H[i, 63] \times W_2[63, c] + b_2[c]$，所以按第 03 章讲过的乘法那条规则（在 $a \times c$ 里，$a$ 每变 1，结果就变 $c$），
    `H[i, j]` 每变 1，`S[i, c]` 就变 `W2[j, c]`：这就是局部斜率。
    一个值连着十条箭头：**责任相加**（第 03 章）。所以

    > `H[i, j]` 的责任 = $G[i, 0] \times W_2[j, 0] + G[i, 1] \times W_2[j, 1] + \ldots + G[i, 9] \times W_2[j, 9]$

    这是 `G` 的第 `i` 行和 `W2` 的第 `j` 行的点积，也就是 `G` 的第 `i` 行和 `W2.T` 的第 `j` **列**的点积。因为 `A @ B` 的 `[i, j]` 项就是 `A` 的第 `i` 行和 `B` 的第 `j` 列的点积，
    这正是 `G @ W2.T` 的 `[i, j]` 项。对所有 `i` 和 `j` 一次算出：
    `dH = G @ W2.T`，形状是 `(n, 10) @ (10, 64) = (n, 64)`，和 `H` 相同。

    **弯折。** `H = relu(Z)`，逐个位置计算。$\operatorname{relu}(z)$ 的斜率：$z$ 为正时 $\operatorname{relu}(z) = z$，斜率 1；$z$ 为负时 $\operatorname{relu}(z) = 0$，斜率 0。
    （正好在 0 处有个拐角；我们取斜率 0。）所以弯折在 $Z > 0$ 的地方把责任原样传过去，在其他地方把它挡住：

    > `dZ = dH * (Z > 0)`

    $Z > 0$ 对每个位置给出 `True`/`False`，乘上它时，`True` 算 1，`False` 算 0。像这样用来放行一些值、挡住另一些值的数组，叫作**掩码**（mask）。一个对某张图片保持沉默的探测器，对这张图片不承担责任：
    把它的旋钮轻推一点点，什么也不会改变。

    **`W1` 和 `b1`。** `Z = X @ W1 + b1` 又是第 06 章的那一步，责任换成 `dZ`：`dW1 = X.T @ dZ`，`db1 = np.sum(dZ, axis=0)`。

    这就是整个网络的反向传播。每一步都是第 03 章的规则；矩阵只是把它一次用到所有图片、所有旋钮上。

    ## 练习 3：反向传播

    写出 `backward(X, Y, Z, H, S, W2)`，返回 `dW1, db1, dW2, db2`：

    1. `G = (softmax(S) - Y) / len(X)`
    2. `dW2 = H.T @ G` 和 `db2 = np.sum(G, axis=0)`
    3. `dH = G @ W2.T`
    4. `dZ = dH * (Z > 0)`
    5. `dW1 = X.T @ dZ` 和 `db1 = np.sum(dZ, axis=0)`
    6. `return dW1, db1, dW2, db2`

    （`backward` 是“向后”，也就是反向传播。记得用英文输入法。）
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
    mo.md(r"""
    ## 用轻推来检查

    整个网络用了这么多新公式，值得检查一下。在训练开始前，轻推第一层的一个旋钮 `W1[406, 0]`（从像素 406，也就是第 14 行第 14 列、图片正中间，通到探测器 0），看损失怎么变；
    再对第二层的一个旋钮 `W2[5, 3]` 做同样的事。（这个单元格需要练习 2 和 3。）
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
    第 0、1 项是 `W1[406, 0]` 的轻推结果和公式结果；第 2、3 项是 `W2[5, 3]` 的。两层里，它们去掉前面的 0 之后，前三四位有效数字都一致（−0.01332… 和 −0.01333…；
    两个都是 0.0029739…）：轻推永远不是完全精确的。（角落里的像素斜率正好是 0，什么也检查不出来，所以我们挑了中间的一个。）

    ## 训练

    和第 06 章同样的循环，只是旋钮从两组变成四组。300 步、每步看全部 10,000 张图片，要花一些时间（最多可能要一分钟），所以训练要等你按下按钮。训练单元格的第一行 `mo.stop(...)` 只是让它等你按按钮。
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
    这三个数是训练损失、训练准确率和测试准确率。最后一个约是 0.944，也就是认对了大约 **94%** 的测试图片，
    而同样训练这么久的单层模型大约是 91%：隐藏层把错误减少了大约 40%。

    ## 探测器在找什么

    `W1` 的每一列是一个探测器的 784 个旋钮，画成一张 $28 \times 28$ 的图（蓝色：这里有墨水会让它亮起来；红色：这里有墨水会让它沉默）。下面是前 16 个：
    """)
    return


@app.cell(hide_code=True)
def _(W1):
    images.weight_maps([W1[:, j] for j in range(16)], [f"探测器 {j}" for j in range(16)], size=72)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    它们看起来比第 06 章的模板杂乱得多，而这种杂乱本身就很说明问题。靠近边缘的地方，几乎每张图片都是白纸，很多像素在每张图片里都是 0，其余的也几乎总是 0：
    它们的旋钮几乎不承担责任（斜率 `X.T @ dZ` 乘的正是这些 0），（基本上）一直保留着随机的初始值。越往中间、墨水越多的地方，能看出平滑一些的色块：每个探测器学会了对某一段笔画做出反应，没有人告诉它该找什么。
    第二层学的是：哪几种探测器混在一起就是哪个数字。

    ## 下一步

    它能用，但训练很慢：300 步里的每一步都看了全部 10,000 张图片。我们还没说清为什么起始大小是 0.05 和 0.18。
    在**第 09 章**里，你会学到让训练又快又可靠的办法：小批量、更聪明的步进规则、有依据的起点，以及怎样确认模型在新数据上的表现靠得住。

    ## 你现在会做的事

    - 说清楚为什么加权和表达不了“这个或那个”，以及弯折怎样解决这个问题
    - 搭一个带一个隐藏层的神经网络：`X @ W1 + b1`、ReLU、`H @ W2 + b2`
    - 说清楚为什么没有弯折的两层并不比一层强
    - 用矩阵在网络里做反向传播：`H.T @ G`、`G @ W2.T`、ReLU 的掩码、`X.T @ dZ`
    - 用轻推检查网络的梯度，并训练它

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**这种网络叫作*多层感知机*（MLP）；ReLU 是 *rectified linear unit*（线性整流单元）的缩写。
    “弯折加起来能逼近任何形状”这一事实叫作*万能近似定理*。
    """)
    return


if __name__ == "__main__":
    app.run()
