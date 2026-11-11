import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="07 · 为分类准备的损失")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    from learn_llm import hand, images, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.plot import chart

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 07 · 为分类准备的损失

    第 06 章最后得到的是取均值做成的模板：测试数字认对 62%，而且有四分之一的图片都被猜成 8。
    那些旋钮从没有为了区分不同的数字而调整过。这一章让梯度下降来做这件事，就像当初调米娅的直线一样。它需要的是一个损失，
    而造这个损失要用到两样新的数学工具 $e^x$ 和 $\log$。你会像第 02 章那样，亲手推出它们的斜率。

    下面是你在第 06 章写的两个函数（已经写好）。之后是图片，和以前一样展开成行，旋钮全是 0，以及这个模型的准确率。
    """)
    return


@app.function
def guess(S):
    return np.argmax(S, axis=1)


@app.function
def accuracy(S, labels):
    return np.mean(guess(S) == labels)


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    X = train_images.reshape(10000, 784) / 255
    X_test = test_images.reshape(2000, 784) / 255
    W_zero = np.zeros((784, 10))
    b_zero = np.zeros(10)
    S_zero = X @ W_zero + b_zero
    accuracy(S_zero, train_labels)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    0.0969：所有分数都是 0 时，模型对每张图片都猜 0，只在那 969 张 0 上猜对。

    ## 为什么准确率不能当损失

    我们想让梯度下降改进旋钮，所以需要一个能给出有用斜率的损失。准确率给不出这样的斜率：把一个旋钮轻推一点点，分数也只变一点点，
    最大的那个分数几乎总还是最大。没有哪个猜测改变，准确率就一动不动：它的斜率几乎处处是 0，步进规则也就改变不了旋钮。

    我们需要一个平滑的衡量标准：不只是**哪个**数字赢了，还要知道模型对每个数字**有多确定**。如果能把分数变成加起来等于 1 的份额（share），
    比如“3 占 0.7，8 占 0.2，其余占 0.1”，那么轻推一下，份额也会跟着变一点点，这就有了斜率。

    ## 从分数到份额：$e^s$

    分数可以是负数，份额不行。所以先把每个分数变成正数，而且要保持大小顺序（分数高的，变完还要更高）。用的工具是一个特别的数 **e**
    （约等于 2.718）的乘方。

    乘方你在第 00 章见过：`2 ** 3` 就是 $2 \times 2 \times 2$。乘方有一条好用的规则：$2^{3 + 2} = 2^3 \times 2^2$（两边都是五个 2 相乘）。
    人们把乘方推广到所有的数，包括 0、负数和小数，推广的方式只有一种，就是让这条规则继续成立。比如：$e^3 \times e^0$ 必须等于 $e^{3 + 0} = e^3$，
    所以 $e^0 = 1$；$e^1 \times e^{-1}$ 必须等于 $e^0 = 1$，所以 $e^{-1} = 1 / e$；$e^{0.5} \times e^{0.5}$ 必须等于 $e^1$，所以 $e^{0.5}$ 是自己乘自己等于 e 的那个数
    （约 1.65）。用同样的办法还能得到一条规则：$e^{a - b} \times e^b = e^a$，所以 **$e^{a - b} = e^a / e^b$**。我们写作 $e^s$，
    在 NumPy 里是 `np.exp(s)`。下面是分数从 -3 到 3 时的样子：
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((-3, 3), (0, 20), "分数 s", "eˢ",
          curve=[(x / 20, float(np.exp(x / 20))) for x in range(-60, 61)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    注意三点：$e^s$ 永远是正的；分数越高，$e^s$ 越大；分数每加 1，$e^s$ 就乘以大约 2.718。（为什么用 e 而不用 2？因为用 e 时斜率最简单，
    下面你会看到确切的原因。）

    然后**把每一个都除以总和**，让它们加起来等于 1。分数是 `[2, 1, 0.1]` 时：$e^2 = 7.39$，$e^1 = 2.72$，$e^{0.1} = 1.11$，总和 11.21，
    所以份额是 $7.39 / 11.21 = 0.659$、$0.242$ 和 $0.099$。这个做法（先 $e^{\square}$，再除以总和）叫作 **softmax**。
    得到的份额就是模型的**概率**（probability）：它对每个数字有多确定。

    有一个实际问题：`np.exp(1000)` 对计算机来说太大了，会得到 `inf`（无穷大），而 `inf / inf` 不是一个数（NumPy 会显示成 `nan`，
    是英文“not a number”的缩写）。解决办法：在算 $e^{\square}$ 之前，先让每个分数都减去最大的那个分数 $m$。这完全不改变份额：根据上面的规则，
    $e^{s - m} = e^s / e^m$，所以每个 $e^{\square}$ 都除以了同一个 $e^m$，各部分和总和都除以同一个数，份额不变。减完之后，最大的分数是 0，
    它的 $e^{\square}$ 是 1，就不会大到算不了了。（如果漏掉这一步，NumPy 还会打印一条提到“overflow”（溢出）的警告：说的就是这个问题。）

    ### 行、列和 `keepdims`

    对分数矩阵，我们一行一行地做：每张图片有自己的份额。`np.max(S, axis=1)` 给出每一行最大的数，`np.sum(E, axis=1)` 给出每一行的总和，
    每张图片一个数。加上 `keepdims=True`，结果会保持“列”的形状，是 `(10000, 1)` 而不是 `(10000,)`。这样的一列会横着广播到每一行上：
    设 `m = np.max(S, axis=1, keepdims=True)`，`S - m` 就让每张图片的十个分数都减去它自己最大的分数（第 05 章是拿一行去减每一行；这里是拿一列去减每一列）。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：份额

    三个分数：`[2, 1, 0]`。保留两位小数，$e^2 = 7.39$，$e^1 = 2.72$，$e^0 = 1$。算出它们的总和，再算出第一个分数和最后一个分数的份额，保留两位小数。
    """)
    return


@app.cell(hide_code=True)
def _():
    share_boxes = hand.boxes(['总和', '第一个份额', '最后一个份额'], step=0.01)
    return (share_boxes,)


@app.cell(hide_code=True)
def _(share_boxes):
    hand.marked(share_boxes, {'总和': 11.11, '第一个份额': 0.67, '最后一个份额': 0.09}, {
        '总和': r"""三个数加起来""",
        '第一个份额': r"""$7.39 \div \text{总和}$""",
        '最后一个份额': r"""$1 \div \text{总和}$""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：softmax

    写出 `softmax(S)`，`S` 是分数矩阵，每张图片一行。每一行有自己的份额：

    1. 每个分数都减去它那一行最大的分数。
    2. 用 `np.exp` 对结果求 $e^{\square}$。
    3. 每个数都除以它那一行的总和。

    第 1 步和第 3 步都需要“每行一个数”，并且能横着广播到这一行上。`keepdims=True` 就是干这个的。

    检查下面的那个单元格会在一张图片的分数 `[[2, 1, 0.1]]` 上试试你的函数，结果应该约为 `[[0.659, 0.242, 0.099]]`。注意两层方括号：这是只有一行（一张图片）的矩阵，因为 softmax 是一行一行地算的。
    """)
    return


@app.function
def softmax(S):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("07", "softmax", softmax, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`np.max(S, axis=1, keepdims=True)` 是每一行最大的分数，排成一列。每一行的总和用 `np.sum`，加上同样两个设置。""",
        r"""如果两张图片的检查没过、一张图片的却过了，说明少了一个 `keepdims=True`。如果 `[[1000, 1000]]` 得到 `nan`，说明少了第 1 步。""",
    ], LOCALE)
    return


@app.cell
def _():
    softmax(np.array([[2, 1, 0.1]]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 损失：意外程度

    正确答案是第一个时，像 `[0.659, 0.242, 0.099]` 这样的份额该怎么打分？我们想要的损失是：模型把全部份额（1.0）都给了正确数字时为 0，
    而这个份额越接近 0，损失就越大，而且没有上限：一个对错误答案非常确定的模型，应该付出很大的代价。

    **log** 正好做到这一点。$\log$ 是 $e^{\square}$ 的逆运算：$\log(e^a) = a$，所以 $\log x$ 回答的是“e 的几次方等于 x？”（在 NumPy 里是 `np.log(x)`）。$\log 1 = 0$（因为 $e^0 = 1$），
    份额小于 1 时 log 是负的（只有 e 的负数次方才小于 1），份额越接近 0，它就越往下走，没有尽头。所以我们用正确数字份额的 log 的**相反数**，叫作**意外程度**（surprise）：

    | 正确数字的份额 | 1 | 0.9 | 0.5 | 0.1 | 0.01 |
    |---|---|---|---|---|---|
    | 意外程度 $-\log p$ | 0 | 0.105 | 0.693 | 2.303 | 4.605 |
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((0, 1), (0, 5), "正确数字的份额 p", "意外程度 −log p",
          curve=[(x / 200, float(-np.log(x / 200))) for x in range(2, 201)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    损失就是所有图片意外程度的均值，叫作**交叉熵**（cross-entropy）。

    要在份额矩阵里挑出正确数字的份额，我们用一个形状相同的矩阵 `Y`：正确数字那一列是 1，其余全是 0（这样的一行叫 **one-hot**，即“独热”行）。
    这样 `Y * P` 在每一行里只留下正确数字的份额，`np.sum(Y * P, axis=1)` 就给出每张图片一个份额。下面这个辅助函数根据标签做出 `Y`：
    先是一个全 0 的矩阵，然后对每张图片，在第 `i` 行、第 `labels[i]` 列放一个 1。
    """)
    return


@app.function
def one_hot(labels):
    Y = np.zeros((len(labels), 10))
    for i in range(len(labels)):
        Y[i, labels[i]] = 1
    return Y


@app.cell
def _(test_labels, train_labels):
    Y = one_hot(train_labels)
    train_labels[0], Y[0]
    return (Y,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第一张图片是 6，所以它的 one-hot 行只在位置 6（从 0 数起，也就是第 7 个数）上有一个 1。数后面的点（比如 `1.`）表示这些数是按小数存的。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：两张图片的损失

    两张图片。模型给第一张的正确数字 0.5 的份额，给第二张的正确数字 0.1 的份额。在上面的表里查出这两个意外程度，再算出损失：两者的均值。
    """)
    return


@app.cell(hide_code=True)
def _():
    loss_boxes = hand.boxes(['第一个意外程度', '第二个意外程度', '损失'], step=0.001)
    return (loss_boxes,)


@app.cell(hide_code=True)
def _(loss_boxes):
    hand.marked(loss_boxes, {'第一个意外程度': 0.693, '第二个意外程度': 2.303, '损失': 1.498}, {
        '第一个意外程度': r"""表里份额为 0.5 的那一项""",
        '第二个意外程度': r"""表里份额为 0.1 的那一项""",
        '损失': r"""两个意外程度相加，除以 2""",
    }, LOCALE, tolerance=0.002)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：cross_entropy

    写出 `cross_entropy(P, Y)`：意外程度的均值，也就是**交叉熵**。`P` 是份额，每张图片一行；`Y` 是那些 one-hot 行。

    1. `np.log` 是逐个位置计算的，所以 `np.log(P)` 是和 `P` 形状相同的矩阵。
    2. 乘以 `Y` 之后，每一行只留下正确数字的那一项。再沿每一行求和，每张图片得到一个数：它正确数字的 $\log p$。
    3. 损失是这些数的均值的相反数。

    把上面那两张图片写成 `P = [[0.5, 0.5], [0.1, 0.9]]` 和 `Y = [[1, 0], [1, 0]]`，结果应该是 1.498。
    """)
    return


@app.function
def cross_entropy(P, Y):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("07", "cross_entropy", cross_entropy, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`np.sum(M, axis=1)` 沿每一行相加，每张图片得到一个数。""",
        r"""只要一行：一个负号，再对 `Y * np.log(P)` 每行的和求 `np.mean`。负号不能少：份额的 log 是负的，而损失应该是正的。""",
    ], LOCALE)
    return


@app.cell
def _(S_zero, Y):
    cross_entropy(softmax(S_zero), Y)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    所有分数都是 0 时，每个数字的份额都一样，是 1/10，所以每张图片的意外程度都是 $-\log 0.1 = 2.303$。这就是一个什么都不懂的模型的损失。
    训练应该让它降下来。

    ## $e^x$ 和 $\log x$ 的斜率

    做梯度下降需要损失的斜率，而损失是由 $e^{\square}$ 和 $\log$ 搭起来的。所以，像第 02 章那样，我们用轻推来求它们的斜率。

    **$e^x$。** 把 $x$ 轻推一个很小的 $h$。根据乘方规则，$e^{x + h} = e^x \times e^h$，所以变化量是 $e^x \times e^h - e^x$。两部分都含有 $e^x$（第二部分是 $e^x \times 1$），
    所以变化量是 $e^x \times (e^h - 1)$，比率（变化量 $\div$ $h$）是

    > $$e^x \times \frac{e^h - 1}{h}$$

    $(e^h - 1) / h$ 这一部分完全不依赖 $x$。同样的步骤对任何底数都成立：底数是 2 时，这一部分是 $(2^h - 1) / h$。下面是底数 2、3 和 e 在 $h$ 越来越小时的值：

    | 底数 | `h = 0.01` | `h = 0.0001` |
    |---|---|---|
    | 2 | 0.6956 | 0.6932 |
    | 3 | 1.1047 | 1.0987 |
    | e | 1.0050 | 1.0001 |

    和第 02 章一样，$h$ 越小，比率越逼近一个固定的数：底数 2 约是 0.693，底数 3 约是 1.099，底数 e 正好是 **1**。
    这就是 e 的特别之处；其实 e 正是被**选来**让这个数等于 1 的底数。所以比率逼近 $e^x \times 1$：

    > $e^x$ 的斜率就是 $e^x$ 自己。

    **$\log x$。** 设 $y = \log x$，也就是 $x = e^y$。把 $y$ 轻推一个很小的 $k$：根据刚求出的斜率（变化量 $\approx$ 斜率 $\times$ 轻推量），$x$ 大约增加 $e^y \times k = x \times k$。
    也就是说，$x$ 增加 $x \times k$ 时，$\log x$ 增加 $k$。那么 $x$ 每变化 1，$\log x$ 变化多少？

    > $\log x$ 的斜率是 $k / (x \times k) = 1 / x$。

    ## 损失对每个分数的斜率

    为了简短，假设一共只有三个数字（0、1、2）。取一张图片，它的三个分数是 `s0, s1, s2`，正确数字是 0。记总和 $T = e^{s_0} + e^{s_1} + e^{s_2}$，正确数字的份额是 $p_0 = e^{s_0} / T$，
    意外程度是 $L = -\log(e^{s_0} / T)$。

    先补一条 log 规则。任取两个正数 $x$ 和 $y$，把它们写成 e 的乘方：$x = e^a$、$y = e^b$，也就是 $a = \log x$、$b = \log y$。
    那么 $x / y = e^a / e^b = e^{a - b}$（前面那条规则），而“e 的几次方等于 $x / y$？”的答案是 $a - b$。所以
    $\log(x / y) = a - b = \log x - \log y$。用到意外程度上：

    > $L = -(\log(e^{s_0}) - \log T) = -\log(e^{s_0}) + \log T = -s_0 + \log T$

    （前面的负号让括号里两部分都变号；$\log(e^{s_0}) = s_0$，因为 log 是 $e^{\square}$ 的逆运算。）

    现在用第 02、03 章的规则（和的斜率相加；链条相乘，责任乘局部斜率）来求斜率：

    - **对 $s_0$**（正确数字）：$-s_0$ 这部分的斜率是 −1。$\log T$ 这部分：$\log T$ 对 $T$ 的斜率是 $1 / T$，$T$ 对 $s_0$ 的斜率是 $e^{s_0}$
      （另外两部分 $e^{s_1}$ 和 $e^{s_2}$ 里没有 $s_0$：斜率 0），合起来是 $e^{s_0} / T = p_0$。合计：**$p_0 - 1$**。
    - **对 $s_1$**（错误数字）：$-s_0$ 里没有 $s_1$，斜率 0。$\log T$ 给出 $e^{s_1} / T = p_1$。合计：**$p_1$**。$s_2$ 也一样：$p_2$。

    用一句话说：**正确数字是份额减 1，其他数字就是份额本身。** 用 one-hot 行 `y` 来写，就是 `p - y`。梯度下降朝斜率的反方向走，
    所以它把正确数字的分数**往上**推（它的斜率是负的），把其他每个分数**往下**压，压多少和它被错分到的份额成正比。

    用第 02 章的办法，在分数 `[2, 1, 0.1]` 上轻推检查一下：
    """)
    return


@app.cell
def _():
    def surprise(s):
        return -np.log(np.exp(s[0]) / np.sum(np.exp(s)))

    s3 = np.array([2.0, 1.0, 0.1])
    h = 0.0001
    nudged = [
        (surprise(s3 + np.array([h, 0, 0])) - surprise(s3)) / h,
        (surprise(s3 + np.array([0, h, 0])) - surprise(s3)) / h,
        (surprise(s3 + np.array([0, 0, h])) - surprise(s3)) / h,
    ]
    nudged
    return (h,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约是 `-0.341`、`0.242` 和 `0.099`：也就是份额 `[0.659, 0.242, 0.099]`，正确数字的那个减去 1。公式成立。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：一张图片的斜率

    一张写着 1 的图片。模型给数字 0、1、2 的份额是 `[0.7, 0.2, 0.1]`：它相当有把握，却猜错了。它的意外程度对三个分数的斜率各是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    slope_boxes = hand.boxes(['分数 0', '分数 1', '分数 2'], step=0.1)
    return (slope_boxes,)


@app.cell(hide_code=True)
def _(slope_boxes):
    hand.marked(slope_boxes, {'分数 0': 0.7, '分数 1': -0.8, '分数 2': 0.1}, {
        '分数 0': r"""错误的数字：斜率就是它的份额""",
        '分数 1': r"""正确的数字：它的份额减 1""",
        '分数 2': r"""错误的数字：斜率就是它的份额""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    按大小算，最大的斜率属于正确的数字：梯度下降会用力把它的分数往上推。而拿到 0.7 的那个错误数字，被往下压的力度比拿到 0.1 的那个大得多。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 所有图片、所有旋钮

    对 `n` 张图片的平均损失，每张图片的斜率都要除以 `n`。所以损失对所有分数的斜率，一次写出来就是

    > `G = (P - Y) / n`

    一个形状为 `(n, 10)` 的矩阵，每张图片一行，每个数字一列：`G[i, c]` 是分数 `S[i, c]` 的责任。

    接下来从分数到旋钮，思路和第 05 章一样（那里是责任向量在前：`(2 * e) @ X`；这里的形状要求另一种排法，下面会推出来）。看旋钮 `W[j, c]`（像素 $j$、数字 $c$）：在每张图片里，数字 $c$ 的分数都有一项是像素 $j$ 乘以 `W[j, c]`。
    所以它的斜率是：对每张图片，取分数的责任 `G[i, c]` 乘以局部斜率 `X[i, j]`（在 $X[i, j] \times W[j, c]$ 里，`W[j, c]` 每变 1，分数变 `X[i, j]`），
    再全部加起来，因为 `W[j, c]` 连着 $n$ 个分数，它们的责任要相加。这正是 **$X$ 的第 $j$ 列**和
    **$G$ 的第 $c$ 列**的点积。

    我们要对每个 `j`、每个 `c` 都算出来：一个 `(784, 10)` 的矩阵，和 `W` 形状相同。如果 `X` 的列变成行，这就是矩阵 `@` 矩阵了。
    把列变成行叫作**转置**（transpose），写作 `X.T`：`X.T` 的第 `j` 行就是 `X` 的第 `j` 列，所以 `X.T` 的形状是 `(784, n)`。于是

    > `X.T @ G` 的形状是 `(784, n) @ (n, 10)` = `(784, 10)`，其中 `[j, c]` 这一项 = `X` 的第 `j` 列和 `G` 的第 `c` 列的点积

    这正是每个旋钮的斜率。而 `b[c]` 的斜率是 `G` 第 `c` 列的总和（每张图片的分数里，`b[c]` 都是原样（乘以 1）加进去的）：
    `np.sum(G, axis=0)`，也就是沿着每一列**往下**加。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：gradient

    写出 `gradient(X, Y, W, b)`，把 `W` 的斜率和 `b` 的斜率作为一对返回：

    1. 份额 `P`：对分数做 softmax。
    2. 各个分数的责任，$G = (P - Y) / n$。图片数 $n$ 是 `len(X)`，也就是 `X` 的行数。
    3. `W` 的斜率，由 `X.T` 和 `G` 得到；`b` 的斜率，是 `G` 每一列往下的和。

    `W` 的斜率必须和 `W` 形状相同，`b` 的斜率必须和 `b` 形状相同。
    """)
    return


@app.function
def gradient(X, Y, W, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("07", "gradient", gradient, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""分数是 `X @ W + b`，和第 06 章一样。""",
        r"""`X.T @ G` 的形状是 `(784, n) @ (n, 10)`。`np.sum(G, axis=0)` 沿着列往下加。两个结果用逗号隔开，一起返回。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    最后在真实的图片上轻推检查一次：在全 0 的起点，取像素 300、数字 3 的那个旋钮。`nudge` 除了第 300 行第 3 列是 `h` 之外全是 0，
    所以 `W_zero + nudge` 只轻推了这一个旋钮。（它用到练习 1 到 3。）
    """)
    return


@app.cell
def _(W_zero, X, Y, b_zero, h):
    nudge = np.zeros((784, 10))
    nudge[300, 3] = h
    before = cross_entropy(softmax(X @ W_zero + b_zero), Y)
    after = cross_entropy(softmax(X @ (W_zero + nudge) + b_zero), Y)
    dW_zero, db_zero = gradient(X, Y, W_zero, b_zero)
    (after - before) / h, dW_zero[300, 3]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    两个都约是 0.0108。公式和轻推的结果一致。

    ## 训练

    梯度下降，和第 05 章一样：从 0 开始，学习率 0.5，走 100 步。之后我们在训练图片上测量损失和准确率，再在**测试**图片上测准确率，
    这些图片模型在学习时从没见过。测试准确率才反映真实水平：它说明模型面对新的笔迹表现如何。（这个单元格要算几秒钟，需要完成全部三个练习。）
    """)
    return


@app.cell
def _(X, X_test, Y, test_labels, train_labels):
    W = np.zeros((784, 10))
    b = np.zeros(10)
    for step in range(100):
        dW, db = gradient(X, Y, W, b)
        W = W - 0.5 * dW
        b = b - 0.5 * db
    train_loss = cross_entropy(softmax(X @ W + b), Y)
    train_acc = accuracy(X @ W + b, train_labels)
    test_acc = accuracy(X_test @ W + b, test_labels)
    train_loss, train_acc, test_acc
    return W, b


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    损失从 2.303 降到大约 0.38，模型现在能认对大约 0.90（**90%**）的测试图片，而一开始只有 10%。它能读懂从没见过的笔迹了。第 06 章取均值做成的模板只有 62%。
    这里训练准确率和测试准确率几乎一样：这么简单的模型没法死记训练图片。更大的模型就能，第 09 章会展示那是什么样子。

    ## 模型学到了什么

    $W$ 的每一列是一个数字的 784 个旋钮，每个像素一个，所以可以把它画成一张 $28 \times 28$ 的图：**蓝色**是墨水会让这个数字的分数升高的地方，
    **红色**是墨水会让分数降低的地方。第 06 章的模板只有蓝色。这些模板还学会了哪里的墨水对一个数字**不利**。
    """)
    return


@app.cell(hide_code=True)
def _(W):
    images.weight_maps([W[:, c] for c in range(10)], list(range(10)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    能看出各个数字的大致轮廓：0 是一个蓝色的圈，中间的洞是红的，因为 0 在那里没有墨水。1 是中间一道蓝色的竖笔。每一列都是一个**模板**（template），
    图片的分数就是它和模板有多吻合。

    下面是一些模型认错的测试图片，以及它的猜测：
    """)
    return


@app.cell(hide_code=True)
def _(W, X_test, b, test_images, test_labels):
    _guesses = np.argmax(X_test @ W + b, axis=1)
    _wrong = np.nonzero(_guesses != test_labels)[0][:8]
    images.digits(test_images[_wrong], [f"{test_labels[i]} → {_guesses[i]}" for i in _wrong])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    （每张图下面的文字是“真实数字 → 猜测”。）有些确实很难认；另一些是写得偏了或斜了的数字，单单一个模板应付不了。

    ## 下一步

    模板只能说“这里有墨水好，那里有墨水不好”，再把它们加起来。它说不出“7 的一笔在这里**或**那里，但不会两处都有”。
    在**第 08 章**里，你会在中间叠上一层由简单部件组成的隐藏层，模型就变成了**神经网络**。

    ## 你现在会做的事

    - 用 softmax 把分数安全地变成概率
    - 说出准确率为什么不能当损失，并用交叉熵（平均意外程度）衡量一个分类器
    - 用轻推推出 $e^x$ 的斜率（它自己）和 $\log x$ 的斜率（$1 / x$）
    - 推出交叉熵的梯度 $p - y$，并用 `X.T @ G` 一次算出所有旋钮的斜率
    - 训练一个认数字的分类器，并在从没见过的图片上测试它

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**这个模型叫作 *softmax 回归*，也叫*多项逻辑回归*；这里的 $\log$ 是*自然对数*，常写作 `ln`。
    """)
    return


if __name__ == "__main__":
    app.run()
