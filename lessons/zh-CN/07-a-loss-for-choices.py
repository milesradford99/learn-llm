import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="07 · 为分类准备的损失")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    from learn_llm import hand, images, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.diagrams import bars, grids, network, pipeline
    from learn_llm.plot import chart

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 07 · 为分类准备的损失

    上一章里，我们用 `S = X @ W + b` 给每张手写数字图片打分（每张图展平成一行 784 个像素，参数 `W` 有 10 列、对应 0 到 9 十个数字，`b` 是 10 个偏置），然后直接把每个数字的所有图片求平均当成了模板。这套模板在测试集上只有 62% 的准确率，而且把四分之一的图片都猜成了 8——因为这些参数从来没有为了“把数字区分开”而真正训练过。

    这一章，我们让梯度下降直接来调 `W` 和 `b`，就像前面调米娅卖冰淇淋的直线一样。梯度下降需要一个**损失函数**，而构造它需要两样新工具：$e^x$ 和 $\log x$。别担心没学过微积分，你会像第 02 章那样，一步步亲手推出它们的斜率。

    下面已经放好了你在第 06 章写过的两个函数。接着我们把图片展平成行，把参数全初始化为 0，看看这个“什么都没学”的模型准确率是多少。
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
    结果是 `0.0969`：因为所有分数都是 0，模型对每张图都默认猜 0，刚好蒙对了那 969 张真正的 0。

    ## 为什么不能直接拿准确率当损失？

    梯度下降要想知道“参数该往哪边调”，损失函数就必须有**斜率**。但准确率偏偏给不出有用的斜率：当你把某个参数稍微调大一点点，分数只会变化一点点，原本最高分通常还是最高分——模型的最终猜测根本没变，准确率也一动不动。换句话说，准确率的斜率几乎处处是 0，梯度下降根本无从下手。

    拿一张写着“1”的图片来看，假设只有 0 和 1 两个选项。把数字 0 的分数固定在 1，慢慢把数字 1 的分数从 -1 调到 3。左边是这张图上的准确率，右边是一个平滑的指标，表示模型有多确定它是 1（马上就会构造它）：
    """)
    return


@app.cell(hide_code=True)
def _():
    _xs = [x / 50 for x in range(-50, 151)]
    mo.hstack([
        chart((-1, 3), (0, 1), "数字 1 的分数", "准确率",
              curve=[(x, 1.0 if x > 1 else 0.0) for x in _xs]),
        chart((-1, 3), (0, 1), "数字 1 的分数", "数字 1 的概率",
              curve=[(x, float(np.exp(x) / (np.exp(x) + np.exp(1)))) for x in _xs]),
    ], widths="equal")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    准确率就像一个**电灯开关**：平时死死停在 0，直到数字 1 的分数超过数字 0 的那一瞬间才突然跳到 1，之后又死死停在 1。在平直的地方，参数微调一点毫无反应，斜率是 0，梯度下降完全不知道该往哪边走。而右边的曲线像一个**无级调光旋钮**：分数每动一点，曲线就跟着动一点，处处都有斜率指引方向。

    所以我们需要一个平滑的衡量方式：不只看**谁**拿了最高分，还要看模型对每个数字**有多大的把握**。如果能把原始分数转换成加起来等于 1 的**概率（probability）**——比如“3 的概率 0.7，8 的概率 0.2，其余共 0.1”——那么参数每微调一点，概率就会平滑变化一点，梯度下降就有路可走了。

    ## 从分数到概率：$e^s$ 与 softmax

    原始分数可以是负数，但概率不能是负数。所以我们先要把每个分数变成正数，同时保持大小顺序不变（原来分数高的，变完依然更高）。这里用到的工具是一个特别的数字 **$e$**（约等于 2.718）的乘方。

    乘方你在第 00 章见过：`2 ** 3` 就是 $2 \times 2 \times 2$。乘方有一条非常好用的规律：$2^{3 + 2} = 2^3 \times 2^2$（两边都是 5 个 2 相乘）。数学上把乘方推广到 0、负数和小数时，唯一原则就是让这条规律继续成立：

    - $e^3 \times e^0 = e^{3 + 0} = e^3$，所以 **$e^0 = 1$**；
    - $e^1 \times e^{-1} = e^0 = 1$，所以 **$e^{-1} = 1 / e \approx 0.37$**；
    - $e^{0.5} \times e^{0.5} = e^1$，所以 **$e^{0.5} = \sqrt{e} \approx 1.65$**；
    - 同理，因为 $e^{a - b} \times e^b = e^a$，所以 **$e^{a - b} = e^a / e^b$**。

    我们把 $e$ 的 $s$ 次方写作 $e^s$，在 NumPy 里就是 `np.exp(s)`。它在分数从 -3 到 3 时的曲线长这样：
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
    看几个具体数值更有手感：

    | 分数 $s$ | −2 | −1 | 0 | 1 | 2 | 3 |
    |---|---|---|---|---|---|---|
    | $e^s$ | 0.14 | 0.37 | 1 | 2.72 | 7.39 | 20.09 |

    分数每加 1，结果就乘以 2.718；每减 1，就除以 2.718。不管分数负得多厉害，$e^s$ 也只是越来越贴近 0，永远大于 0。

    记住它的三个特点：永远是正数；分数越大结果越大；分数每涨 1，结果大约翻 2.718 倍。（至于为什么偏偏选 $e$ 而不是 2？因为底数是 $e$ 时斜率最干净，下面推斜率时你一眼就会明白。）

    变成正数之后，下一步只要**把每个 $e^s$ 除以它们的总和**，就能让所有数加起来正好等于 1。比如三个分数 `[2, 1, 0.1]`：$e^2 = 7.39$，$e^1 = 2.72$，$e^{0.1} = 1.11$，总和是 11.21，各自除以总和得到 `0.659`、`0.242`、`0.099`。这套“先算 $e^s$，再除以总和”的做法，就叫 **softmax**，算出来的就是模型预测的**概率**。

    拖一拖下面的滑块，直观感受分数、$e^s$ 和最终概率怎么联动：
    """)
    return


@app.cell(hide_code=True)
def _():
    play_scores = mo.ui.array([mo.ui.slider(-3, 3, step=0.5, value=v, label=f"数字 {i} 的分数", show_value=True)
                               for i, v in enumerate([2, 1, 0])])
    play_scores.vstack()
    return (play_scores,)


@app.cell(hide_code=True)
def _(play_scores):
    _s = np.array(play_scores.value, dtype=float)
    _e = np.exp(_s)
    mo.hstack([
        mo.vstack([mo.md(r"$e^s$"), bars([(f"数字 {i}", v) for i, v in enumerate(_e)], "每个分数的 e 次方", digits=2)]),
        mo.vstack([mo.md(r"概率：每个 $e^s$ 除以总和 " + f"{_e.sum():.2f}"),
                   bars([(f"数字 {i}", v) for i, v in enumerate(_e / _e.sum())], "每个数字的概率", top=1.0)]),
    ], widths="equal")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    可以试试这三件事：

    - 把三个分数调成一样：每个概率都是 $1/3$，模型完全拿不定主意。
    - 把其中一个分数调高 1：它的 $e^s$ 瞬间放大 2.72 倍，把其他数字的概率吸走一大块。
    - 先设成 `2, 1, 0` 记下概率，再一起加 1 变成 `3, 2, 1`：概率**纹丝不动**！因为所有分数同加一个常数，每个 $e^s$ 都会乘上同一个倍数，除以总和时刚好约掉。真正决定概率的，只是分数之间的**差值**。

    这个性质顺手帮我们解决了一个工程坑：`np.exp(1000)` 大到超出了计算机能存的上限，会直接溢出成 `inf`（无穷大），接着 `inf / inf` 就会算出 `nan`（not a number，无效数值）。

    解决办法非常巧妙：在算 $e^s$ 之前，先把每个分数都减去它们当中最大的那个数 $m$。根据前面的规则 $e^{s - m} = e^s / e^m$，分子和分母同时除以了同一个 $e^m$，最终概率完全不变。减完之后，最大的分数变成 0（$e^0 = 1$），其余全是负数，彻底告别溢出警告（overflow）。

    比如 `[1000, 999, 998]` 减去最大值 `1000` 后变成 `[0, -1, -2]`，和 `[2, 1, 0]` 的差值一模一样，算出的概率同样是 `0.67, 0.24, 0.09`，而且 $e^0 = 1$、$e^{-1} = 0.37$、$e^{-2} = 0.14$ 全是又小又安全的数。

    ### 按行计算与 `keepdims=True`

    面对一整批图片组成的分数矩阵 `S`，我们需要**逐行**算 softmax，每张图各算各的概率。`np.max(S, axis=1)` 会找出每一行的最大值，`np.sum(E, axis=1)` 会算出每一行的总和，都是每张图一个数。关键在于加上 `keepdims=True`：它会把结果保留成“竖着的一列”，形状是 `(10000, 1)`，而不是被拍扁的一维数组 `(10000,)`。

    为什么必须留成一列？回忆一下第 05 章的广播（broadcasting）规则：一维的 `(3,)` 会被当成一行、竖着套用到 `(2, 3)` 的每一行上；而二维的 `(2, 1)` 是一列、会**横着**套用到 `(2, 3)` 的每一列上。所以写 `m = np.max(S, axis=1, keepdims=True)` 后，`S - m` 就会让每张图的各个分数各自减去**它自己那一行的最大值**：
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "S  (2, 3)", "values": [[2, 1, 0], [5, 7, 6]], "rows": ["第 0 张", "第 1 张"],
         "marks": {(0, 0): "box", (1, 1): "box"}},
        "−",
        {"name": "m  (2, 1)", "values": [[2], [7]], "marks": {(0, 0): "blue", (1, 0): "red"}},
        "=",
        {"name": "S − m  (2, 3)", "values": [[0, -1, -2], [-2, 0, -1]],
         "marks": {**{(0, c): "blue" for c in range(3)}, **{(1, c): "red" for c in range(3)}}},
    ], "每一行最大的分数保存成一列 m，再从这一行的每个分数里减去")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    可以看到，`m` 竖着存了每一行的最大值（在 `S` 中用黑框标出）：第 0 张图的 `2` 减在第 0 行每个数上（蓝色），第 1 张图的 `7` 减在第 1 行每个数上（红色）。如果不加 `keepdims=True`，`m` 会变成形状为 `(2,)` 的 `[2, 7]`，NumPy 会把它当成**一行**去配 `S` 每行的 3 个数，直接报错（更糟的是，万一图片数刚好也等于类别数，它甚至不报错，而是悄悄算错！）。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：概率

    三个分数 `[2, 1, 0]`。保留两位小数，$e^2 = 7.39$，$e^1 = 2.72$，$e^0 = 1$。请先算出它们的总和，再算出第一个分数和最后一个分数的概率（保留两位小数）。
    """)
    return


@app.cell(hide_code=True)
def _():
    share_boxes = hand.boxes(['总和', '第一个概率', '最后一个概率'], step=0.01)
    return (share_boxes,)


@app.cell(hide_code=True)
def _(share_boxes):
    hand.marked(share_boxes, {'总和': 11.11, '第一个概率': 0.67, '最后一个概率': 0.09}, {
        '总和': r"""把三个数加起来""",
        '第一个概率': r"""$7.39 \div \text{总和}$""",
        '最后一个概率': r"""$1 \div \text{总和}$""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：softmax

    写出 `softmax(S)`，输入是分数矩阵 `S`（每张图片占一行），返回每张图对应的概率矩阵：

    1. 每一行的分数都减去该行的最大值。
    2. 用 `np.exp` 计算每个数的 $e^s$。
    3. 每一行的每个数都除以该行的总和。

    第 1 步和第 3 步都需要“每行一个数、横着广播到整行”，别忘了加上 `keepdims=True`。

    检查框下面的单元格会用单张图片的分数 `[[2, 1, 0.1]]` 测试你的函数，结果应约为 `[[0.659, 0.242, 0.099]]`。注意外面的两层方括号：它表示一个只有 1 行（1 张图）的二维矩阵，因为 softmax 是按行计算的。
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
        r"""`np.max(S, axis=1, keepdims=True)` 会按行取最大值并保持成一列；每行求和用 `np.sum`，加上同样的两个参数。""",
        r"""如果 1 张图的测试过了、2 张图的没过，说明漏掉了 `keepdims=True`；如果 `[[1000, 1000]]` 算出 `nan`，说明少了第 1 步减最大值。""",
    ], LOCALE)
    return


@app.cell
def _():
    softmax(np.array([[2, 1, 0.1]]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 损失：意外程度（cross-entropy）

    假设正确答案是第一个数字，模型给出了概率 `[0.659, 0.242, 0.099]`，我们该怎么评价它猜得好不好？理想的损失函数要满足两个条件：模型把全部概率（`1.0`）都压在正确答案上时，损失为 0；而它给正确答案的概率越接近 0，损失就飙升得越高、没有上限——越是“自信满满地猜错”，代价就该越惨重。

    **$\log$（对数）**天生就具备这个脾气。$\log$ 是 $e^a$ 的逆运算：$\log(e^a) = a$，换句话说，$\log x$ 回答的是“$e$ 的几次方等于 $x$？”（在 NumPy 里是 `np.log(x)`）。因为 $e^0 = 1$，所以 $\log 1 = 0$；当概率 $p < 1$ 时，$\log p$ 是负数（只有 $e$ 的负次方才小于 1），而且 $p$ 越接近 0，它就跌得越深、永无止境。在前面加一个负号把负数翻成正数，就得到了**意外程度（surprise）**：

    | 正确数字的概率 $p$ | 1 | 0.9 | 0.5 | 0.1 | 0.01 |
    |---|---|---|---|---|---|
    | 意外程度 $-\log p$ | 0 | 0.105 | 0.693 | 2.303 | 4.605 |

    留意最后两列：概率从 `0.1` 跌到 `0.01`（缩小 10 倍），意外程度恰好从 `2.303` 涨到 `4.605`（稳稳加了一个 `2.303`）；如果再缩小 10 倍到 `0.001`，它就变成 `6.908`。每错得离谱 10 倍，代价就再多加 2.3。
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((0, 1), (0, 5), "正确数字的概率 p", "意外程度 −log p",
          curve=[(x / 200, float(-np.log(x / 200))) for x in range(2, 201)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    把所有图片的意外程度取平均，就是分类任务最常用的损失——**交叉熵（cross-entropy）**。

    怎么从概率矩阵 `P` 里把每张图“正确数字”对应的概率挑出来？我们准备一张和 `P` 形状一样的矩阵 `Y`：每行只在正确数字那一列填 1，其余全填 0（这种编码叫 **one-hot**）。把 `Y` 和 `P` 逐个位置相乘（`Y * P`），每行就只剩下正确数字的概率，再按行求和（`np.sum(Y * P, axis=1)`），就能得到每张图对应的那个概率。比如 3 个数字、一张 0 和一张 2：
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "P", "values": [[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]], "rows": ["第 0 张", "第 1 张"],
         "cols": ["0", "1", "2"]},
        "×",
        {"name": "Y", "values": [[1, 0, 0], [0, 0, 1]], "cols": ["0", "1", "2"],
         "marks": {(0, 0): "blue", (1, 2): "blue"}},
        "=",
        {"name": "Y × P", "values": [[0.7, 0, 0], [0, 0, 0.4]], "cols": ["0", "1", "2"],
         "marks": {(0, 0): "blue", (1, 2): "blue"}},
        "→",
        {"name": "每行求和", "values": [[0.7], [0.4]], "marks": {(0, 0): "box", (1, 0): "box"}},
    ], "概率 P 和 one-hot 矩阵 Y 逐个位置相乘，每行只留下正确数字的概率；按行求和得到 0.7 和 0.4")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `Y` 里的 0 把所有错误数字的概率都清零了，按行一加就刚好拎出正确答案那一项。下面的辅助函数根据标签生成 `Y`：先建一张全 0 矩阵，再对第 `i` 张图，把第 `i` 行、第 `labels[i]` 列设为 1。
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
    第一张图片是 6，所以它的 one-hot 行只在下标 6（从 0 数起，第 7 个位置）上是 1。数字后面的小数点（如 `1.`）表示它们以浮点数保存。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：两张图片的损失

    假设有两张图片：模型给第一张的正确数字打出了 `0.5` 的概率，给第二张的正确数字打出了 `0.1` 的概率。请从上面的表格查出这两个意外程度，再算出它们的均值（即损失）。
    """)
    return


@app.cell(hide_code=True)
def _():
    loss_boxes = hand.boxes(['第一个意外程度', '第二个意外程度', '损失'], step=0.001)
    return (loss_boxes,)


@app.cell(hide_code=True)
def _(loss_boxes):
    hand.marked(loss_boxes, {'第一个意外程度': 0.693, '第二个意外程度': 2.303, '损失': 1.498}, {
        '第一个意外程度': r"""表里概率为 0.5 对应的那一项""",
        '第二个意外程度': r"""表里概率为 0.1 对应的那一项""",
        '损失': r"""把两个意外程度相加，再除以 2""",
    }, LOCALE, tolerance=0.002)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：cross_entropy

    写出 `cross_entropy(P, Y)`，计算所有图片的平均意外程度（即**交叉熵**）。`P` 是概率矩阵（每张图一行），`Y` 是对应的 one-hot 矩阵：

    1. `np.log` 会逐个位置取对数，所以 `np.log(P)` 的形状和 `P` 完全相同。
    2. 乘以 `Y` 后，每行只留下正确数字的那一项；再按行求和，每张图就得到一个数：正确数字的 $\log p$。
    3. 对这些数求均值，再取相反数（加负号），就是损失。

    用上面那两张图测试：`P = [[0.5, 0.5], [0.1, 0.9]]`、`Y = [[1, 0], [1, 0]]`，结果应为 `1.498`。
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
        r"""`np.sum(M, axis=1)` 会按行相加，每张图得到一个数。""",
        r"""一行就能写完：前面加一个负号，然后对 `Y * np.log(P)` 按行求和的结果取 `np.mean`。别漏掉负号——概率的 log 是负数，而损失必须是正数。""",
    ], LOCALE)
    return


@app.cell
def _(S_zero, Y):
    cross_entropy(softmax(S_zero), Y)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    所有分数都是 0 时，10 个数字平分概率，每个都是 $1/10$，每张图的意外程度都是 $-\log 0.1 = 2.303$。这就是模型“瞎猜”时的基准损失，接下来训练的目标就是把它压下去。

    ## $e^x$ 和 $\log x$ 的斜率

    要做梯度下降，我们需要算出损失的斜率；而这个损失是由 $e^x$ 和 $\log x$ 搭出来的。和第 02 章一样，我们用“把输入稍微增加一点点 $h$，看输出变化多少”的办法，一步步把它们的斜率推出来。

    **先看 $e^x$。** 把 $x$ 增加一个很小的 $h$。根据乘方规则，$e^{x + h} = e^x \times e^h$，所以输出的变化量是 $e^x \times e^h - e^x$。把公因式 $e^x$ 提出来，变化量就是 $e^x \times (e^h - 1)$，再除以 $h$ 得到斜率比率：

    > $$e^x \times \frac{e^h - 1}{h}$$

    你会发现 $\frac{e^h - 1}{h}$ 这一项**完全不含 $x$**！这套推导对任何底数都成立（换底数 2 就是 $\frac{2^h - 1}{h}$）。来看看 $h$ 越缩越小时，底数 2、3 和 $e$ 各自的比率会逼近什么数：

    | 底数 | `h = 0.01` | `h = 0.0001` |
    |---|---|---|
    | 2 | 0.6956 | 0.6932 |
    | 3 | 1.1047 | 1.0987 |
    | $e$ | 1.0050 | 1.0001 |

    $h$ 越小，这个比率越逼近一个固定常数：底数 2 约是 `0.693`，底数 3 约是 `1.099`，而底数 $e$ 恰好就是 **1**！其实数学家当初定义 $e \approx 2.718$，正是为了挑出“让这个比率正好等于 1”的底数。于是斜率比率逼近 $e^x \times 1$：

    > $e^x$ 的斜率就是 $e^x$ 自己。

    看图最直观：站在这条曲线的任何一点上，它有多高，坡度就有多陡。在 $x = 1$ 处，曲线高 `2.72`，切线斜率也是 `2.72`；在 $x = 2$ 处，曲线高 `7.39`，切线斜率也是 `7.39`。
    """)
    return


@app.cell(hide_code=True)
def _():
    _e = float(np.exp(1))
    chart((-1, 2.6), (0, 14), "x", "eˣ",
          curve=[(x / 50, float(np.exp(x / 50))) for x in range(-50, 131)],
          lines=[(_e, 0.0), (_e ** 2, -(_e ** 2))],
          dots=[(1, _e), (2, _e ** 2)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **再看 $\log x$。** 我们先拿一个具体数字算一遍，马上就能明白。已知 $e^2 \approx 7.39$，反过来就是 $\log 7.39 \approx 2$。现在把 $y = 2$ 往上挪一点点 $k = 0.01$（变成 $2.01$）：刚才已经知道 $e^y$ 在 $y = 2$ 处的斜率就是 $7.39$，所以 $x = e^y$ 大约会涨 $7.39 \times 0.01 = 0.0739$。

    现在换个角度反过来看：当 $x$ 涨了 `0.0739` 时，$\log x$ 只涨了 `0.01`。所以“$\log x$ 的变化量 $\div$ $x$ 的变化量”就是 $0.01 / 0.0739 = 1 / 7.39$！

    把 `7.39` 换成任意正数 $x$（设 $y = \log x$，即 $x = e^y$），道理一模一样：$y$ 增加很小的 $k$，$x$ 就大约增加 $e^y \times k = x \times k$，因此：

    > $\log x$ 的斜率是 $\frac{k}{x \times k} = \frac{1}{x}$。

    ## 损失对每个分数的斜率

    为了方便看清，先假设一共只有 3 个数字（0、1、2），拿一张正确答案是 0 的图片来推。设它的三个分数是 $s_0, s_1, s_2$，记总和 $T = e^{s_0} + e^{s_1} + e^{s_2}$，那么正确数字的概率是 $p_0 = e^{s_0} / T$，意外程度是 $L = -\log(e^{s_0} / T)$。

    先把这个式子化简一步，只需用到一条对数规律：**$\log(x / y) = \log x - \log y$**。道理很简单：把 $x, y$ 写成 $x = e^a, y = e^b$（即 $a = \log x, b = \log y$），前面知道 $x / y = e^{a - b}$（比如 $e^5 / e^2 = e^3$，所以 $\log(e^5 / e^2) = 3 = 5 - 2$）。用这条规律把 $L$ 拆成很清爽的两项：

    > $$L = -(\log(e^{s_0}) - \log T) = -\log(e^{s_0}) + \log T = -s_0 + \log T$$

    （括号外的负号让里面两项都变号；$\log(e^{s_0}) = s_0$，因为 $\log$ 和 $e^s$ 刚好互逆抵消。）

    ### 顺手回顾：反向传责任的两条规则（第 03 章）

    把计算画成方框和箭头，每个箭头上标着**局部斜率**（箭头起点的数每变 1，终点的数变多少）。最终损失 $L$ 自己的责任是 1，往回推时只需两条规则：

    - **沿着一条路，相乘。** 如果 $u \to v$ 的局部斜率是 3，而 $v$ 的责任是 2，那 $u$ 沿这条路分到的责任就是 $3 \times 2 = 6$。
    - **路分叉的地方，相加。** 如果 $u$ 分叉通往两个框，一条路分到 3，另一条路分到 4，总责任就是 $3 + 4 = 7$。

    现在把 $L = -s_0 + \log T$ 画成图，每条箭头的局部斜率都标上去：
    """)
    return


@app.cell(hide_code=True)
def _():
    network(
        [[["s0"], ["s1"], ["s2"]], [["-s0"], ["T", "总和"]], [["L", "-s0 + log T"]]],
        [(0, 0, 0, "-1"), (0, 0, 1, "exp(s0)"), (0, 1, 1, "exp(s1)"), (0, 2, 1, "exp(s2)"),
         (1, 0, 0, "1"), (1, 1, 0, "1/T")],
        "三个分数通往 -s0 和总和 T，再相加得到 L = -s0 + log T。s0 有两条路通往 L（斜率分别是 -1 和 exp(s0)/T），s1 和 s2 各只有一条路经过 T。",
        ["分数", "两块", "意外程度"],
        at=0.45,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    看着图从右往左一乘一加，答案直接就出来了（$L$ 的责任是 1）：

    - **对正确数字的分数 $s_0$**：从 $s_0$ 出发有**两条路**，各自相乘再相加：
      - 上面那条（$s_0 \to -s_0 \to L$）：$(-1) \times 1 = -1$；
      - 下面那条（$s_0 \to T \to L$）：$e^{s_0} \times \frac{1}{T} = \frac{e^{s_0}}{T} = p_0$；
      - 两条路合起来：**$p_0 - 1$**。
    - **对错误数字的分数 $s_1$**：只有下面那**一条路**：$e^{s_1} \times \frac{1}{T} = p_1$。$s_2$ 完全一样，是 **$p_2$**。

    总结成一句话，漂亮得不可思议：**正确数字的斜率是“概率减 1”，其余每个数字的斜率就是“概率本身”**。写成 one-hot 向量形式，就是 `p - y`。梯度下降往斜率的反方向走：正确数字的斜率是负的，所以它的分数会被**推高**；其他错误数字的斜率是正的，会被**压低**，而且被错分到的概率越大，压得越狠。

    我们用第 02 章的数值试探法，在分数 `[2, 1, 0.1]` 上验算一下：
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
    结果约为 `-0.341`、`0.242`、`0.099`——正好就是概率 `[0.659, 0.242, 0.099]` 把正确数字那一项减去 1。公式分毫不差！
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：一张图片的斜率

    一张写着 1 的图片，模型给数字 0、1、2 算出的概率分别是 `[0.7, 0.2, 0.1]`（相当自信，可惜猜错了）。它的意外程度对这三个分数的斜率各是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    slope_boxes = hand.boxes(['分数 0', '分数 1', '分数 2'], step=0.1)
    return (slope_boxes,)


@app.cell(hide_code=True)
def _(slope_boxes):
    hand.marked(slope_boxes, {'分数 0': 0.7, '分数 1': -0.8, '分数 2': 0.1}, {
        '分数 0': r"""错误数字：斜率就是它的概率""",
        '分数 1': r"""正确数字：它的概率减 1""",
        '分数 2': r"""错误数字：斜率就是它的概率""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    看绝对值大小，斜率最大的是正确数字 1，梯度下降会铆足劲把它的分数往上拉；而错拿了 `0.7` 概率的数字 0，被往下压的力度也远大于只拿了 `0.1` 的数字 2。

    这就像一场拔河：正确数字被往上拉的力是 $1 - p$（离满分 1 还差多少），每个错误数字被往下压的力是 $p$（它不该拿却拿走的部分）。等正确数字拿满概率，拉力自然归零。拖动滑块看看这张图一步步更新的过程（学习率设为 1）：
    """)
    return


@app.cell(hide_code=True)
def _():
    pull_steps = mo.ui.slider(0, 6, value=0, label="走了几步", show_value=True)
    pull_steps
    return (pull_steps,)


@app.cell(hide_code=True)
def _(pull_steps):
    _s = np.log(np.array([0.7, 0.2, 0.1]))
    _y = np.array([0.0, 1.0, 0.0])
    for _ in range(pull_steps.value):
        _p = np.exp(_s) / np.exp(_s).sum()
        _s = _s - (_p - _y)
    _p = np.exp(_s) / np.exp(_s).sum()
    mo.hstack([
        mo.vstack([mo.md("概率"), bars([(f"数字 {i}", v) for i, v in enumerate(_p)], "概率", top=1.0)]),
        mo.vstack([mo.md("斜率：概率减 one-hot"),
                   bars([(f"数字 {i}", v) for i, v in enumerate(_p - _y)], "斜率", top=1.0)]),
        mo.md(rf"意外程度：**{-np.log(_p[1]):.3f}**"),
    ], widths=[2, 2, 1])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    只走一步，数字 1 的概率就从 `0.2` 跳到约 `0.5`；走满 6 步就超过 `0.9`，意外程度也从 `1.609` 迅速掉向 0。正确概率越大，各项斜率就越小：模型离答对越近，调整就越轻柔。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 从分数到参数（旋钮）

    不过，梯度下降真正要调整的不是分数 `S`，而是模型里的参数：`W` 里的 $784 \times 10$ 个数，以及 `b` 里的 10 个数。下面是一次算完所有 $n$ 张图的完整流程，每个方框下面标着它的矩阵形状：
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("X", "(n, 784)"), ("S", "(n, 10)"), ("P", "(n, 10)"), ("L", "一个数")],
             ["@ W + b", "softmax", "交叉熵"],
             "X 乘以 W 再加 b 得到分数 S；softmax 得到概率 P；交叉熵得到损失 L")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    上一节我们把责任从**一张**图的意外程度传回了它的分数，现在还差最后两步：总损失 $L$ 是所有图片意外程度的**均值**（不只是单张图），而且我们要把责任继续传给参数 `W` 和 `b`。用到的依然是刚才那两条规则——**沿路相乘，分叉相加**。我们缩到一个极小的世界里，把每一个数字都看得清清楚楚。

    ### 一个小到能看清每个数的世界

    假设只有 2 张图片，每张只有 3 个像素，数字也只有 0 和 1 两种。有墨水的像素记为 1，空白记为 0。图片 0 是一个 0，墨水在像素 0 和像素 2；图片 1 是一个 1，墨水在像素 1 和像素 2。假设模型给图片 0 的概率是 `[0.6, 0.4]`（猜对了，但不太稳），给图片 1 的概率是 `[0.8, 0.2]`（很自信，但猜错了）。

    先看计算的后半段：4 个分数、2 个意外程度、1 个总损失。`S[i, c]` 表示图片 $i$ 对数字 $c$ 的分数，总损失 $L = (\text{意外}_0 + \text{意外}_1) / 2$。箭头上是局部斜率，方框里红字是责任：
    """)
    return


@app.cell(hide_code=True)
def _():
    blame_edges = [(0, 0, 0, "-0.4"), (0, 1, 0, "0.4"), (0, 2, 1, "0.8"), (0, 3, 1, "-0.8"),
                   (1, 0, 0, "1/2"), (1, 1, 0, "1/2")]
    blame_heads = ["分数", "意外程度", "损失"]
    network([[["S[0,0]", "责任 -0.2"], ["S[0,1]", "责任 0.2"], ["S[1,0]", "责任 ?"], ["S[1,1]", "责任 ?"]],
             [["意外 0", "责任 1/2"], ["意外 1", "责任 1/2"]], [["L", "责任 1"]]],
            blame_edges, "四个分数通往两个意外程度，再通往损失 L。通往 L 的每个箭头局部斜率是二分之一；"
            "图片 0 的分数出发的箭头局部斜率是 -0.4 和 0.4，图片 1 的是 0.8 和 -0.8",
            blame_heads)
    return blame_edges, blame_heads


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 第 1 步：从总损失到每张图的意外程度

    因为 $L = (\text{意外}_0 + \text{意外}_1) / 2$，把“意外 0”稍微增加一点点 $h$，总损失 $L$ 只变化 $h / 2$，所以通往 $L$ 的两条箭头局部斜率都是 $\frac{1}{2}$。$L$ 的责任是 1，沿路相乘，每张图的意外程度各分到 $\frac{1}{2} \times 1 = \frac{1}{2}$。推广到 $n$ 张图片时，$L$ 是求和再除以 $n$，每张图的意外程度就各分到 $\frac{1}{n}$。

    ### 第 2 步：从意外程度到各个分数

    上一节已经推出了单张图的意外程度对各个分数的局部斜率：概率减去 one-hot，即 $p - y$。图片 0 是数字 0，概率是 `[0.6, 0.4]`、one-hot 是 `[1, 0]`，所以 `S[0,0]` 的局部斜率是 $0.6 - 1 = -0.4$，`S[0,1]` 是 $0.4 - 0 = 0.4$（正是图里箭头上的数）。再沿路乘上意外程度的责任 $\frac{1}{2}$：

    > $$S[0, 0] \text{ 的责任} = -0.4 \times \tfrac{1}{2} = -0.2 \qquad S[0, 1] \text{ 的责任} = 0.4 \times \tfrac{1}{2} = 0.2$$

    每个分数只影响自己那张图的意外程度，没有分叉，直接用这一条路就行。现在轮到你算图片 1 的两个分数：
    """)
    return


@app.cell(hide_code=True)
def _():
    score_blame_boxes = hand.boxes(["S[1, 0]", "S[1, 1]"], step=0.1)
    return (score_blame_boxes,)


@app.cell(hide_code=True)
def _(score_blame_boxes):
    hand.marked(score_blame_boxes, {"S[1, 0]": 0.4, "S[1, 1]": -0.4}, {
        "S[1, 0]": r"""图片 1 是数字 1，one-hot 是 `[0, 1]`：局部斜率是 $0.8 - 0$，再乘以它对应意外程度的责任 $\frac{1}{2}$""",
        "S[1, 1]": r"""局部斜率是 $0.2 - 1$，再乘以它对应意外程度的责任 $\frac{1}{2}$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _(blame_edges, blame_heads, score_blame_boxes):
    mo.stop(not hand.done(score_blame_boxes, {"S[1, 0]": 0.4, "S[1, 1]": -0.4}),
            mo.md("*两个格子都填对后，下面的图会自动补全。*"))
    network([[["S[0,0]", "责任 -0.2"], ["S[0,1]", "责任 0.2"], ["S[1,0]", "责任 0.4"], ["S[1,1]", "责任 -0.4"]],
             [["意外 0", "责任 1/2"], ["意外 1", "责任 1/2"]], [["L", "责任 1"]]],
            blame_edges, "同一张图，每个分数的责任都填上了", blame_heads)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 把所有分数的责任收进一张表 `G`

    每个分数的算法完全一致：（概率减 one-hot）再除以图片数 $n$。我们把它们摆成一张和分数 `S` 形状完全相同的矩阵 `G`（每张图一行、每个数字一列）：

    > $$G = (P - Y) / n$$

    `G[i, c]` 就是分数 `S[i, c]` 分到的责任。在我们这个 2 图小世界里：
    """)
    return


@app.cell(hide_code=True)
def _():
    _cols = ["数字 0", "数字 1"]
    grids([
        {"name": "P", "values": [[0.6, 0.4], [0.8, 0.2]], "rows": ["图片 0", "图片 1"], "cols": _cols},
        "−",
        {"name": "Y", "values": [[1, 0], [0, 1]], "cols": _cols},
        "=",
        {"name": "P − Y", "values": [[-0.4, 0.4], [0.8, -0.8]], "cols": _cols},
        "÷ 2 =",
        {"name": "G", "values": [[-0.2, 0.2], [0.4, -0.4]], "cols": _cols,
         "marks": {(r, c): "red" for r in range(2) for c in range(2)}},
    ], "概率 P 减去 one-hot 矩阵 Y，再除以图片数 2，得到四个分数的责任 G")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    最难的一关已经过了！从分数责任 `G` 再往回推一步到参数 `W` 和 `b`，只需再用一次同样的两条规则。

    ### 第 3 步：从分数到一个参数

    每个数字在每个像素位置上都有一个参数：`W[j, c]` 对应像素 $j$ 和数字 $c$，所以在这个 3 像素、2 数字的小世界里，`W` 的形状是 `(3, 2)`。再看一眼两张图的像素矩阵 `X`（每张图占一行）：
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X", "values": [[1, 0, 1], [0, 1, 1]],
         "rows": ["图片 0", "图片 1"], "cols": ["像素 0", "像素 1", "像素 2"]},
    ], "像素 X：图片 0 在像素 0 和 2 有墨水，图片 1 在像素 1 和 2 有墨水")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    先拿参数 `W[2, 0]` 来举例。第 $i$ 张图对数字 0 的打分公式是：

    > $$S[i, 0] = X[i, 0] \times W[0, 0] + X[i, 1] \times W[1, 0] + X[i, 2] \times W[2, 0] + b[0]$$

    可以看到，`W[2, 0]` 同时参与了**两张**图对数字 0 的打分——从它出发伸出两条箭头。每条箭头的局部斜率，就是它乘着的那个像素值 `X[i, 2]`（因为在 $x \times w$ 里，$w$ 每增加 1，结果就增加 $x$）。两张图在像素 2 处都有墨水（都是 1），所以两条箭头的局部斜率都是 1：
    """)
    return


@app.cell(hide_code=True)
def _():
    network([[["W[2,0]", "责任 ?"]], [["S[0,0]", "责任 -0.2"], ["S[1,0]", "责任 0.4"]]],
            [(0, 0, 0, "X[0,2] = 1"), (0, 0, 1, "X[1,2] = 1")],
            "参数 W[2,0] 通往两张图片的数字 0 分数，每个箭头的局部斜率都是 1",
            ["参数", "分数"], at=0.55)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    有分叉就相加，每条路相乘：

    > $$W[2, 0] \text{ 的斜率} = 1 \times (-0.2) + 1 \times 0.4 = 0.2$$

    如果某张图在这个像素位置是空白的呢？看 `W[1, 0]`：图片 0 在像素 1 没有墨水（`X[0, 1] = 0`），所以那条箭头的局部斜率是 0，传不回任何责任；只有图片 1 起作用：$0 \times (-0.2) + 1 \times 0.4 = 0.4$。也就是说，**一张图只有在某个像素有墨水时，才会更新这个像素对应的参数**。
    """)
    return


@app.cell(hide_code=True)
def _():
    network([[["W[1,0]", "责任 0.4"]], [["S[0,0]", "责任 -0.2"], ["S[1,0]", "责任 0.4"]]],
            [(0, 0, 0, "X[0,1] = 0"), (0, 0, 1, "X[1,1] = 1")],
            "参数 W[1,0]：通往图片 0 分数的箭头局部斜率是 0，通往图片 1 分数的是 1",
            ["参数", "分数"], at=0.55)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    把刚才算 `W[2, 0]` 的过程放到矩阵图里看：黑框是参数 `W[2, 0]`，蓝色是它乘过的像素（`X` 的第 2 列），红色是它影响到的分数和对应责任（`G` 的第 0 列）。
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X", "values": [[1, 0, 1], [0, 1, 1]], "rows": ["图片 0", "图片 1"],
         "cols": ["像素 0", "像素 1", "像素 2"], "marks": {(0, 2): "blue", (1, 2): "blue"}},
        "@",
        {"name": "W", "values": [["W[0,0]", "W[0,1]"], ["W[1,0]", "W[1,1]"], ["W[2,0]", "W[2,1]"]],
         "cols": ["数字 0", "数字 1"], "marks": {(2, 0): "box"}},
        "→",
        {"name": "S", "values": [["S[0,0]", "S[0,1]"], ["S[1,0]", "S[1,1]"]],
         "cols": ["数字 0", "数字 1"], "marks": {(0, 0): "red", (1, 0): "red"}},
        "→",
        {"name": "G", "values": [[-0.2, 0.2], [0.4, -0.4]],
         "cols": ["数字 0", "数字 1"], "marks": {(0, 0): "red", (1, 0): "red"}},
    ], "参数 W[2,0] 乘的是每张图片的像素 2，影响每张图片数字 0 的分数")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：再算三个参数

    用同样的方法算出下面三个参数的斜率。对任意 `W[j, c]`：把每张图在像素 $j$ 的值，乘上那张图对数字 $c$ 的分数责任 `G[i, c]`，全部加起来即可。
    """)
    return


@app.cell(hide_code=True)
def _():
    knob_boxes = hand.boxes(["W[0, 0]", "W[1, 1]", "W[2, 1]"], step=0.1)
    return (knob_boxes,)


@app.cell(hide_code=True)
def _(knob_boxes):
    hand.marked(knob_boxes, {"W[0, 0]": -0.2, "W[1, 1]": -0.4, "W[2, 1]": -0.2}, {
        "W[0, 0]": r"""像素 0、数字 0：$X[0, 0] \times G[0, 0] + X[1, 0] \times G[1, 0]$，只有图片 0 在像素 0 有墨水""",
        "W[1, 1]": r"""像素 1、数字 1：只有图片 1 在像素 1 有墨水，所以只有 `S[1, 1]` 的责任算数""",
        "W[2, 1]": r"""像素 2、数字 1：两张图在像素 2 都有墨水，把数字 1 的两个责任加起来""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    梯度下降会往斜率的**反方向**更新参数，看看每个参数怎么变，完全合情合理：

    - `W[0, 0]` **变大**：只有图片 0（真值是 0）在像素 0 有墨水，以后像素 0 有墨水就更偏向猜 0。
    - `W[1, 0]`（前面例子，斜率 `0.4`）**变小**：只有图片 1（真值是 1）在像素 1 有墨水，而模型刚才给了它 `0.8` 的“是 0”概率——以后像素 1 有墨水，模型会少往 0 上猜。
    - `W[1, 1]` **变大**：同样是图片 1 的像素 1，它作为“是 1”只拿到了 `0.2` 的概率，所以模型会调大 `W[1, 1]`，让像素 1 的墨水更偏向 1。
    - `W[2, 1]` 稍微**变大**：两张图在像素 2 都有墨水，两个责任 $0.2 + (-0.4)$ 部分抵消，留下一个较小的负斜率。

    ### 发现规律：两列做点积

    回头看你刚才算 `W[j, c]` 乘的是哪两组数：一组是每张图的像素 $j$（也就是 **`X` 的第 $j$ 列**），另一组是每张图数字 $c$ 的责任（也就是 **`G` 的第 $c$ 列**）。两两相乘再求和——正好就是这两列的**点积**！以 `W[2, 1]` 为例：

    | 图片 | 像素 2（`X` 的第 2 列） | 数字 1 的责任（`G` 的第 1 列） | 乘积 |
    |---|---|---|---|
    | 0 | 1 | 0.2 | 0.2 |
    | 1 | 1 | −0.4 | −0.4 |
    | **合计** | | | **−0.2** |
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X", "values": [[1, 0, 1], [0, 1, 1]], "rows": ["图片 0", "图片 1"],
         "cols": ["像素 0", "像素 1", "像素 2"], "marks": {(0, 2): "blue", (1, 2): "blue"}},
        "·",
        {"name": "G", "values": [[-0.2, 0.2], [0.4, -0.4]],
         "cols": ["数字 0", "数字 1"], "marks": {(0, 1): "red", (1, 1): "red"}},
        "=",
        {"name": "W[2,1] 的斜率", "values": [[-0.2]], "marks": {(0, 0): "box"}},
    ], "X 的第 2 列和 G 的第 1 列做点积，得到参数 W[2,1] 的斜率：1 乘 0.2 加 1 乘 -0.4")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 把列翻成行：转置（transpose）

    我们要一次性算出所有像素 $j$、所有数字 $c$ 的点积，得到一张和 `W` 形状相同的斜率矩阵。回忆第 06 章的矩阵乘法 `A @ B`：它是拿左边矩阵的每一**行**，去和右边矩阵的每一**列**做点积。所以我们只要先把 `X` 的每一列翻成行，就能直接用 `@` 一次算完！

    把矩阵的列翻成行、行翻成列，叫作**转置（transpose）**，在 NumPy 里写作 `X.T`：`X.T` 的第 $j$ 行，就是 `X` 原来的第 $j$ 列。原来 `(2, 3)` 的 `X` 转置后变成 `(3, 2)`：
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X", "values": [[1, 0, 1], [0, 1, 1]], "rows": ["图片 0", "图片 1"],
         "cols": ["像素 0", "像素 1", "像素 2"], "marks": {(0, 2): "blue", (1, 2): "blue"}},
        "→",
        {"name": "X.T", "values": [[1, 0], [0, 1], [1, 1]], "rows": ["像素 0", "像素 1", "像素 2"],
         "cols": ["图片 0", "图片 1"], "marks": {(2, 0): "blue", (2, 1): "blue"}},
    ], "转置 X：X 的第 2 列变成 X.T 的第 2 行")
    return


@app.cell
def _():
    X_tiny = np.array([[1, 0, 1], [0, 1, 1]])
    X_tiny.T
    return (X_tiny,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在 `X.T @ G` 就会拿 `X.T` 的第 $j$ 行（所有图的像素 $j$）去和 `G` 的第 $c$ 列（所有图对数字 $c$ 的责任）做点积，一步算出所有参数的斜率！核对一下形状：`(3, 2) @ (2, 2)` 得到 `(3, 2)`，和 `W` 严丝合缝。
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X.T", "values": [[1, 0], [0, 1], [1, 1]], "rows": ["像素 0", "像素 1", "像素 2"],
         "cols": ["图片 0", "图片 1"], "marks": {(2, 0): "blue", (2, 1): "blue"}},
        "@",
        {"name": "G", "values": [[-0.2, 0.2], [0.4, -0.4]], "cols": ["数字 0", "数字 1"],
         "marks": {(0, 1): "red", (1, 1): "red"}},
        "=",
        {"name": "W 的斜率", "values": [[-0.2, 0.2], [0.4, -0.4], [0.2, -0.2]],
         "rows": ["像素 0", "像素 1", "像素 2"], "cols": ["数字 0", "数字 1"],
         "marks": {(2, 1): "box"}},
    ], "X.T 乘以 G，一次算出所有参数的斜率，形状和 W 相同：X.T 的第 2 行和 G 的第 1 列点积，填进第 2 行第 1 列")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    你刚才手算的三个答案就在 `[0, 0]`、`[1, 1]` 和 `[2, 1]`，前面两个例子在 `[2, 0]` 和 `[1, 0]`。用 NumPy 跑一下：
    """)
    return


@app.cell
def _(X_tiny):
    G_tiny = np.array([[-0.2, 0.2], [0.4, -0.4]])
    X_tiny.T @ G_tiny
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    至于偏置 `b`：`b[c]` 是原样（系数为 1）加进每张图的数字 $c$ 分数里的，所以它的斜率就是把 `G` 的第 $c$ 列全部加起来——即 `np.sum(G, axis=0)`（沿着每一列**纵向**求和）。在小世界里，数字 0 是 $-0.2 + 0.4 = 0.2$，数字 1 是 $0.2 + (-0.4) = -0.2$。

    ### 放大到真实尺寸

    换成真正的 10,000 张手写数字图片，公式**一个字都不用改**，只是矩阵变大了：`X` 是 `(n, 784)`，转置后 `X.T` 是 `(784, n)`；`G` 是 `(n, 10)`。于是：

    > `X.T @ G` 的形状是 `(784, n) @ (n, 10)` = `(784, 10)`，正好是 `W` 的形状

    里面的 `[j, c]` 项就是像素 $j$、数字 $c$ 对应参数的斜率。写完下面这道练习，就能把这些斜率画成 $28 \times 28$ 的图像来看看。
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X.T  (784, n)", "values": [["·", "·", "⋯", "·"], ["·", "·", "⋯", "·"],
                                             ["⋮", "⋮", "", "⋮"], ["·", "·", "⋯", "·"]],
         "rows": ["像素 0", "像素 1", "⋮", "像素 783"], "cols": ["图 0", "图 1", "⋯", "图 n-1"],
         "marks": {(1, c): "blue" for c in range(4)}},
        "@",
        {"name": "G  (n, 10)", "values": [["·", "·", "⋯", "·"], ["·", "·", "⋯", "·"],
                                          ["⋮", "⋮", "", "⋮"], ["·", "·", "⋯", "·"]],
         "rows": ["图 0", "图 1", "⋮", "图 n-1"], "cols": ["数字 0", "数字 1", "⋯", "数字 9"],
         "marks": {(r, 3): "red" for r in range(4)}},
        "=",
        {"name": "W 的斜率  (784, 10)", "values": [["·", "·", "⋯", "·"], ["·", "·", "⋯", "·"],
                                                  ["⋮", "⋮", "", "⋮"], ["·", "·", "⋯", "·"]],
         "rows": ["像素 0", "像素 1", "⋮", "像素 783"], "cols": ["数字 0", "数字 1", "⋯", "数字 9"],
         "marks": {(1, 3): "box"}},
    ], "真实尺寸：形状为 784 乘 n 的 X.T，乘以形状为 n 乘 10 的 G，得到 W 的斜率，784 乘 10")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：gradient

    写出 `gradient(X, Y, W, b)`，返回一对结果：`W` 的斜率和 `b` 的斜率。步骤如下：

    1. 算出分数矩阵，用 `softmax` 得到概率矩阵 `P`。
    2. 算出每个分数的责任 $G = (P - Y) / n$，其中图片数 $n$ 就是 `len(X)`（`X` 的行数）。
    3. 用 `X.T` 和 `G` 算出 `W` 的斜率；对 `G` 沿每一列纵向求和，得到 `b` 的斜率。

    返回的 `W` 斜率形状要和 `W` 一致，`b` 斜率形状要和 `b` 一致。
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
        r"""`X.T @ G` 的形状是 `(784, n) @ (n, 10)`；`np.sum(G, axis=0)` 会沿着每一列纵向求和。两个结果用逗号隔开一起 `return`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    最后在真实图片上做一次数值验算：在全 0 起点处，挑出像素 300、数字 3 对应的那个参数。`nudge` 除了第 300 行第 3 列是极小值 `h` 之外全是 0，所以 `W_zero + nudge` 只单独微调了这一个参数。（运行前需先完成练习 1 到 3。）
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
    return (dW_zero,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    两边都约等于 `0.0108`，公式算出的梯度和数值验算完全吻合！

    ### 把第一步的梯度画成图

    `dW_zero` 的第 $c$ 列是数字 $c$ 在 784 个像素上的斜率，所以同样可以还原成 $28 \times 28$ 的图像。下面画出的是每个数字斜率的**相反数**，也就是梯度下降第一步会把参数往哪边拉：**蓝色**表示调大，**红色**表示调小。
    """)
    return


@app.cell(hide_code=True)
def _(dW_zero):
    images.weight_maps([-dW_zero[:, c] for c in range(10)], list(range(10)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    仅仅走完第一步，每个数字就已经浮现出模糊的轮廓！在全 0 起点，所有概率都是 `0.1`，所以某个数字比平均图片墨水更多的位置会被拉成蓝色（调大），墨水更少的位置会被拉成红色（调小）。一次 `X.T @ G` 矩阵乘法，就同时算出了全部 10 张图。

    ## 正式训练

    用和第 05 章一样的梯度下降循环：从全 0 出发，学习率 `0.5`，迭代 100 步。训练结束后，我们先看训练集上的损失和准确率，再看**测试集**上的准确率——测试集里的图片模型在训练时从未见过，它才能真实反映模型识别新笔迹的能力。（这个单元格大约运行几秒，需先完成全部三个练习。）
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
    损失从 `2.303` 一路掉到约 `0.38`，测试集准确率从 10% 飙升到约 `0.90`（**90%**）——远远超过第 06 章平均模板的 62%，它真的能认出从未见过的手写笔迹了！你会发现训练准确率和测试准确率几乎一样高：这么简单的单层模型还不够大，背不下来训练图片；等后面第 09 章换成更大的模型，就会遇到“死记硬背（过拟合）”的问题。

    ## 模型到底学到了什么？

    `W` 的每一列装着某个数字的 784 个参数，把它们画回 $28 \times 28$ 的图像：**蓝色**代表这里有墨水会给该数字**加分**，**红色**代表这里有墨水会**扣分**。第 06 章的均值模板只有蓝色（加分），而梯度下降训出的模板还学会了“哪里不该有墨水”。
    """)
    return


@app.cell(hide_code=True)
def _(W):
    images.weight_maps([W[:, c] for c in range(10)], list(range(10)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每个数字的特征一目了然：0 是一个蓝色圆环、中间空心处涂满红色（因为 0 中间不该有墨水）；1 是正中一道蓝色的竖笔。每一列都是一张**模板（template）**，图片打分就是在衡量它和模板有多贴合。

    最后看几张模型认错的测试图片（下方标注为“真实数字 → 模型猜测”）：
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
    有些字迹确实连人都难辨认，但另一些只是写得偏了一点或歪了一点——单靠一张固定位置的模板，应付不了这种变化。

    ## 下一步

    单张模板只能做“这里有墨水加分、那里有墨水扣分”的简单累加，表达不了“7 的那一横要么在偏左、要么在偏右”这种组合判断。在**第 08 章**，你会在输入和输出之间叠上一层隐藏层，模型就会真正进化成**神经网络**。

    ## 本章收获

    - 明白为什么不能用准确率当损失，会用 **softmax** 把分数安全地转成概率，并用**交叉熵**（平均意外程度）衡量分类效果
    - 亲手推出 $e^x$ 的斜率（它自己）和 $\log x$ 的斜率（$1 / x$）
    - 画出计算图推出交叉熵对分数的斜率 $p - y$，并用转置乘法 `X.T @ G` 一次算出所有参数的梯度
    - 从零训练出一个测试准确率达 90% 的手写数字分类器

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**这一章的模型在教科书里叫 *softmax 回归*（softmax regression），也叫*多项逻辑回归*（multinomial logistic regression）；这里的 $\log$ 是以 $e$ 为底的*自然对数*，数学里常写作 `ln`。
    """)
    return


if __name__ == "__main__":
    app.run()
