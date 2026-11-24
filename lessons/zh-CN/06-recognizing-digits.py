import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="06 · 认数字")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    from learn_llm import hand, images, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.diagrams import grids

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 06 · 认数字

    到目前为止，我们做的模型都只猜一个数值：卖出多少个冰淇淋。这一章换一种任务：模型看一张手写数字的图片，判断它究竟是 0、1、2……还是 9。
    这种从有限几个选项里“做选择题”的任务，叫**分类**（classification）。这一章，你会把图片转换成数字、给每个候选数字打分，并亲手做出第一个分类器；到第 07 章，再让机器自己学出一个更准的。

    ## 图片

    课程自带 12,000 张手写数字图片，来自著名的 MNIST 数据集。`load_digits()` 会把它们分成四份返回给你：
    10,000 张用来学习的图片和它们对应的正确答案（叫**标签**，label），以及另外 2,000 张带标签的图片——先留在一边，等模型做好了再拿来考试。
    """)
    return


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    train_images.shape
    return test_images, test_labels, train_images, train_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这里的形状有三个数 `(10000, 28, 28)`（上面显示为包含三个数的列表：10000、28 和 28），意思是 10,000 张图片，每张都是 28 行、28 列的方格。方格里的每个数代表图片上的一个点，叫**像素**（pixel）：
    0 是白纸，255 是纯黑墨水，中间的数则是不同深浅的灰。

    /// admonition | 🧰 工具箱：取出其中一项
    想从数组（或列表）里取出某一项，就在方括号里写它的编号，从 0 开始数：`train_images[0]` 是第一张图片，`train_labels[0]` 是它的标签。

    对矩阵，先写行号、再写列号：比如矩阵 `M` 里，`M[2, 5]` 是行号为 2、列号为 5 的数（从 0 数起，也就是第 3 行第 6 列）。你也可以分两次写括号——`M[0][14]` 和 `M[0, 14]` 是等价的两种写法，前者是“先取第 0 行，再从这一行里取第 14 项”。

    只给矩阵一个编号，得到的就是整整一行：`train_images[0]` 是第一张图片（一个 $28 \times 28$ 的矩阵），所以 `train_images[0][14]`（也可以写成 `train_images[0, 14]`）就是它第 14 行的所有像素，正好横穿图片正中间：
    ///
    """)
    return


@app.cell
def _(train_images, train_labels):
    train_labels[0], train_images[0][14]
    return


@app.cell(hide_code=True)
def _(train_images, train_labels):
    mo.vstack([
        mo.md(r"""
    标签是 6（NumPy 显示单个数时会顺手标出它的数据类型：`np.uint8(6)` 表示 0 到 255 之间的整数，后面还会见到表示小数的 `np.float64(...)`；我们只看括号里的数值就行）；
    这一行除了笔画经过的地方，其余都是空白（0）。下面是用像素值画出来的前 8 张图片和它们的标签：
    """),
        images.digits(train_images[:8], train_labels[:8]),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    不过在计算机眼里，图片并没有“笔画”，只是一堆数。下面是第一张图片里的一小块：6 行 12 列，取自数字 6 下半部分绕圈的地方。
    每一格就是一个像素的数值，超过 128（墨多于纸）的格子标成了蓝色。眯起眼睛看，能从这些数字里看出那个圆弧的轮廓。
    """)
    return


@app.cell(hide_code=True)
def _(train_images):
    _patch = train_images[0][15:21, 9:21]
    grids([{"name": "第 0 张图片，第 15 到 20 行，第 9 到 20 列", "values": _patch.tolist(),
            "rows": [f"第 {r} 行" for r in range(15, 21)], "cols": [str(c) for c in range(9, 21)],
            "marks": {(r, c): "blue" for r in range(6) for c in range(12) if _patch[r, c] > 128}}],
          "第一张图片里 6 行 12 列的一小块：像素值从 0 到 255，深色的涂了颜色，描出 6 下方那个圈")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 每张图片展开成一长行

    我们的模型习惯处理一维向量，所以先把每个 $28 \times 28$ 的方格展开成一行 784 个数（$28 \times 28 = 784$）：先放第 0 行，接着拼上第 1 行，依此类推。
    `reshape` 干的就是这件事——数值和顺序都不变，只换个排列形状。我们还把所有像素都除以 255，让它们落在 0（白）到 1（黑）之间，
    理由和第 04、05 章缩放气温一样：数值较小、量级统一，才好配合同一个学习率。处理完后，训练集和测试集的形状分别是 `(10000, 784)` 和 `(2000, 784)`，也就是每张图片占一行、每行 784 个数。

    用一张 $3 \times 3$ 的小图“7”来看展开的过程（1 表示墨，0 表示纸）：三行首尾相接，先是第 0 行（蓝），再是第 1 行，最后是第 2 行（红），一共 9 个数。
    信息一点也没丢——只要知道宽度是 3，随时能把这 9 个数折回原来的方格。
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "一张 3 × 3 的图片", "values": [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
         "rows": ["第 0 行", "第 1 行", "第 2 行"],
         "marks": {**{(0, c): "blue" for c in range(3)}, **{(2, c): "red" for c in range(3)}}},
        "→",
        {"name": "同一张图片，展开成一行 9 个数", "values": [[1, 1, 1, 0, 0, 1, 0, 0, 1]],
         "cols": [str(i) for i in range(9)],
         "marks": {**{(0, c): "blue" for c in range(3)}, **{(0, c): "red" for c in range(6, 9)}}},
    ], "一张 3 × 3 的“7”展开成一行 9 个数：第 0 行、第 1 行、第 2 行依次排开")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    真正的图片也是同样的道理，只不过换成了 28 行、每行 28 个数：展开后的长行里，第 0 到 27 号位置是图片最上面一行，第 28 到 55 号是第二行，以此类推。
    """)
    return


@app.cell
def _(test_images, train_images):
    X = train_images.reshape(10000, 784) / 255
    X_test = test_images.reshape(2000, 784) / 255
    X.shape, X_test.shape
    return X, X_test


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 分数就像一张镂空模板

    先不看真实图片，我们来搭一个迷你世界：图片只有 $3 \times 3$ 大小，“数字”也只有两个——竖杠 **|** 和横杠 **—**。
    给每个数字配一张 $3 \times 3$ 的参数网格（模板），大小和图片一模一样，就像一张镂空板（stencil）：这个数字该有墨的地方填 1，其他地方填 0。

    一张图片对某个数字的**分数**怎么算？把每个像素乘以同一位置上的参数，再全部加起来。因为这里的参数只有 1 和 0，分数其实就是数一数：
    有多少个有墨的像素正好落在模板的 1 上。下面是用一根竖杠分别对两张模板打分：
    """)
    return


@app.cell(hide_code=True)
def _():
    _bar = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    _stand = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
    _lie = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
    _ink = {(r, 1): "blue" for r in range(3)}
    mo.vstack([
        grids([
            {"name": "图片", "values": _bar, "marks": _ink}, "×",
            {"name": "| 的参数", "values": _stand, "marks": {(r, 1): "red" for r in range(3)}}, "→",
            {"name": "| 的分数", "values": [[3]], "marks": {(0, 0): "box"}},
        ], "竖杠对 | 的模板：三个有墨的像素都落在 1 上，分数 3"),
        grids([
            {"name": "图片", "values": _bar, "marks": _ink}, "×",
            {"name": "— 的参数", "values": _lie, "marks": {(1, c): "red" for c in range(3)}}, "→",
            {"name": "— 的分数", "values": [[1]], "marks": {(0, 0): "box"}},
        ], "竖杠对 — 的模板：只有中间那个像素落在 1 上，分数 1"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **|** 得 3 分，**—** 得 1 分，所以模型判断它是“|”。墨水（蓝）和模板（红）重叠的地方越多，分数就越高。

    ### 手算：一个 T

    现在换一张 **T** 的图片：最上面一整行有墨，中间一列也有墨。请你算一算它对两张模板各自得几分。
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([{"name": "一个 T", "values": [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
            "marks": {(0, 0): "blue", (0, 1): "blue", (0, 2): "blue", (1, 1): "blue", (2, 1): "blue"}}],
          "一张 3 × 3 的 T：最上面一行和中间一列有墨")
    return


@app.cell(hide_code=True)
def _():
    t_boxes = hand.boxes(["| 的分数", "— 的分数"])
    return (t_boxes,)


@app.cell(hide_code=True)
def _(t_boxes):
    hand.marked(t_boxes, {"| 的分数": 3, "— 的分数": 1}, {
        "| 的分数": r"""| 的模板在中间一列：T 在这三个格子里有几个是有墨的？""",
        "— 的分数": r"""— 的模板在中间一行：T 在这三个格子里有几个是有墨的？""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    “同一位置相乘再加起来”，其实就是第 05 章学过的点积。把图片和模板都展开成一行 9 个数，打分就是求它们的点积。
    有两个数字时，把两张展开后的模板竖着并排放进矩阵 $W$，做成它的两**列**，形状是 `(9, 2)`；把图片一行一行叠起来，做成矩阵 $X$。
    于是 $X$ 里每张图片的那一行，和 $W$ 里每个数字的那一列做点积，就得到对应的分数——这正好是 `X @ W`。下面是竖杠和 T 的计算过程：
    """)
    return


@app.cell(hide_code=True)
def _():
    _cols = [str(i) for i in range(9)]
    grids([
        {"name": "X  (2, 9)", "values": [[0, 1, 0, 0, 1, 0, 0, 1, 0], [1, 1, 1, 0, 1, 0, 0, 1, 0]],
         "rows": ["竖杠", "T"], "cols": _cols, "marks": {(1, c): "blue" for c in range(9)}},
        "@",
        {"name": "W  (9, 2)", "values": [[0, 0], [1, 0], [0, 0], [0, 1], [1, 1], [0, 1], [0, 0], [1, 0], [0, 0]],
         "rows": _cols, "cols": ["|", "—"], "marks": {(r, 1): "red" for r in range(9)}},
        "=",
        {"name": "S  (2, 2)", "values": [[3, 1], [3, 1]], "rows": ["竖杠", "T"], "cols": ["|", "—"],
         "marks": {(1, 1): "box"}},
    ], "X 的两行是展开的竖杠和 T；W 的两列是 | 和 — 的模板；S = X @ W 每张图片、每个数字一个分数。T 那一行和 — 那一列的点积是 1")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    框出的分数，就是 $X$ 里“T”那一行（蓝）和 $W$ 里“—”那一列（红）的点积：也就是 T 对 — 的分数，正好是你刚才算出的 1。
    真实的手写数字完全一样，只是规模更大：像素从 9 个变成 784 个，数字从 2 个变成 10 个，而且参数（旋钮）可以是任意实数，不再局限于 0 或 1。

    ## 十个分数，每个数字一个

    在第 05 章里，一组参数把一天的天气数据变成一个预测值。现在我们需要十个数——每个数字对应一个**分数**（score），所以给每个数字都配一组专属的 784 个参数（外加一个专属的偏置 `b`）。
    数字 3 的分数，就是图片与数字 3 那组参数的点积：如果这组参数在 3 常有墨水的地方取值很大，真正的 3 就会拿到高分。

    把十组参数并排放成矩阵 `W` 的十**列**，形状是 `(784, 10)`；十个 `b` 放进形状为 `(10,)` 的向量 `b`。于是

    > `S = X @ W + b`

    一次就能算出所有图片对所有数字的分数。矩阵 `@` 矩阵，就是拿第一个矩阵的**每一行**和第二个矩阵的**每一列**逐一做点积：
    第 $i$ 张图片对数字 $c$ 的分数，就是 $X$ 的第 $i$ 行和 $W$ 的第 $c$ 列的点积。形状规则还是老样子：
    `(10000, 784) @ (784, 10)` 得到 `(10000, 10)`——中间挨着的两个数（784 和 784）必须相等，算完后消掉。最后的 `+ b` 会通过广播（broadcasting），把每个数字的 `b`
    加到对应那一列的每一张图片上：拿一个 $2 \times 2$ 的小例子来说，`[[3, 1], [3, 1]] + [10, 20]` 会把 `10` 加到第 0 列、`20` 加到第 1 列，得到 `[[13, 21], [13, 21]]`。

    /// admonition | 🧰 工具箱：全 0 数组
    我们先让所有参数都从 0 起步。`np.zeros((784, 10))` 会创建一个该形状的全 0 数组（形状要写成括号包起来的一对数），`np.zeros(10)` 则是一个包含 10 个 0 的一维向量。
    ///
    """)
    return


@app.cell
def _(X):
    W_zero = np.zeros((784, 10))
    b_zero = np.zeros(10)
    S_zero = X @ W_zero + b_zero
    S_zero.shape
    return S_zero, W_zero, b_zero


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：形状规则

    往下走之前，先在一个小例子上练练手。假设 `X` 里有 5 张图片，形状是 `(5, 784)`，`W` 和上面一样，形状是 `(784, 10)`。
    `X @ W` 有多少行、多少列？`W` 里一共有多少个参数？
    """)
    return


@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['行数', '列数', 'W 的参数个数'])
    return (shape_boxes,)


@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'行数': 5, '列数': 10, 'W 的参数个数': 7840}, {
        '行数': r"""每张图片一行分数""",
        '列数': r"""每个数字一个分数""",
        'W 的参数个数': r"""784 行，每行 10 个参数""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 做出猜测

    模型的猜测，就是分数**最高**的那个数字。下面是随手编的一组分数：4 张图片，为了方便观察，先只看 3 个数字。

    | 图片 | 数字 0 的分数 | 数字 1 的分数 | 数字 2 的分数 | 标签 |
    |---|---|---|---|---|
    | 0 | 1 | 3 | 2 | 1 |
    | 1 | 0 | 0 | 5 | 2 |
    | 2 | 4 | 1 | 1 | 1 |
    | 3 | 2 | 6 | 3 | 1 |

    图片 0 的最高分是 3，落在数字 1 那一列，所以模型猜它是 1。它的标签正好也是 1，这次猜对了。

    ## 动手算：猜测与准确率

    请填出图片 1、2、3 的猜测结果，再算出**准确率**（accuracy）：这 4 张图片里，猜对的占多大比例（写成 0 到 1 之间的小数）。
    """)
    return


@app.cell(hide_code=True)
def _():
    table_boxes = hand.boxes(['图片 1', '图片 2', '图片 3', '准确率'], step=0.01)
    return (table_boxes,)


@app.cell(hide_code=True)
def _(table_boxes):
    hand.marked(table_boxes, {'图片 1': 2, '图片 2': 0, '图片 3': 1, '准确率': 0.75}, {
        '图片 1': r"""这一行里最大的数在哪一列""",
        '图片 2': r"""这一行里最大的数在哪一列""",
        '图片 3': r"""这一行里最大的数在哪一列""",
        '准确率': r"""数一数有几张图片的猜测和标签相同（图片 0 是其中一张），再除以 4""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：guess

    /// admonition | 🧰 工具箱：最大的数在哪里？
    `np.argmax(v)` 返回向量里最大那个数的**位置**（索引）：比如 `np.argmax(np.array([3, 9, 4]))` 的结果是 1。

    对矩阵 `M`，`np.argmax(M, axis=1)` 会**横向沿着每一行**找最大值的位置，每一行给出一个结果；如果换成 `axis=0`，则是**纵向沿着每一列**往下找。
    遇到并列最大时，`np.argmax` 返回最先出现的那个位置。
    ///

    写出 `guess(S)`。`S` 是分数矩阵，每张图片占一行，每个数字占一列；请你为每张图片返回猜测的数字。

    在上面那张表上，它应该返回 `[1, 2, 0, 1]`。
    """)
    return


@app.function
def guess(S):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("06", "guess", guess, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""一行里最大分数的位置**就是**猜测的数字，因为第 3 列放的就是数字 3 的分数。""",
        r"""只要一行，用 `np.argmax` 沿着每一行找。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：accuracy

    /// admonition | 🧰 工具箱：比较数组
    `==`（两个等号）用来判断“是否相等”，而单个 `=` 用来赋值。用在数组上时，它会逐个位置比较：
    `np.array([1, 2, 0]) == np.array([1, 5, 0])` 会得到 `[True, False, True]`。

    对这样的布尔数组求 `np.mean` 时，每个 `True` 当作 1、每个 `False` 当作 0，因此结果就是 `True` 所占的比例：这里是 $2 / 3$。
    ///

    写出 `accuracy(S, labels)`：计算猜对的图片所占的比例。可以直接调用你刚才写的 `guess`。

    在上面那张表上，标签为 `[1, 2, 1, 1]` 时，它应该返回 0.75。
    """)
    return


@app.function
def accuracy(S, labels):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("06", "accuracy", accuracy, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三步：先算出每张图的猜测，再拿去和标签逐个比较，最后求平均。""",
        r"""一行就够：对 `guess(S) == labels` 求均值。""",
    ], LOCALE)
    return


@app.cell
def _(S_zero, train_labels):
    accuracy(S_zero, train_labels)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    所有分数都是 0 时，十个数字处处平手，`np.argmax` 只好每次都选第 0 个位置——模型把每张图片都猜成了 0。结果只有图片本来就是 0 时才蒙对，
    准确率只有 0.0969（10,000 张里对了 969 张），大约十分之一，跟瞎猜差不多。

    ## 亲手做参数：模板

    还没开始训练，我们能不能先手工凑出一组比瞎猜强得多的参数？回想一下分数的算式：它是图片和某个数字那组参数的点积。
    图片在参数大的地方有墨水，得分就高。由此可以想到一个很直觉的办法：把数字 3 的参数做成一张**典型的 3**——把训练集里所有的 3 按像素逐位求平均。
    大多数 3 都有墨水的地方，平均出来参数就大；3 几乎不落笔的空白处，参数就接近 0。这样做出的一列参数，就叫**模板**（template）。

    ## 练习 3：template

    /// admonition | 🧰 工具箱：用 True 和 False 筛选行
    在矩阵的方括号里放一个由 `True` 和 `False` 组成的数组（长度等于行数），就能只留下标着 `True` 的那些行。
    比如 `M` 的三行是 `[1, 2]`、`[3, 4]` 和 `[5, 6]`，那么 `M[np.array([True, False, True])]` 取出的就是 `[1, 2]` 和 `[5, 6]`。

    这个布尔数组通常来自一次比较。设 `kinds = np.array([7, 9, 7])`，`M[kinds == 7]` 取出的还是这两行。

    `np.mean(M, axis=0)` 会**纵向沿着每一列**求平均，每一列留下一个数；对刚才挑出的两行，结果是 `[3, 4]`。

    记住这两个方向的一个小口诀：`axis=1` 是**横向**压扁每一行（`(n, 10) -> (n,)`，每张图留 1 个数）；`axis=0` 是**纵向**压扁每一列（`(n, 784) -> (784,)`，每个像素留 1 个平均数）。
    ///

    写出 `template(X, labels, c)`：计算 `X` 中所有标签等于 `c` 的行的平均值。`X` 每行是一张图片，所以结果是一个一维向量，长度等于一张图片的像素数。

    比如 `X` 的三行是 `[1, 2]`、`[3, 4]`、`[5, 6]`，标签是 `[7, 9, 7]`，那么 `template(X, labels, 7)` 应该返回 `[3, 4]`。
    """)
    return


@app.function
def template(X, labels, c):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("06", "template", template, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""两步：先筛出标签为 `c` 的所有行，再沿着每一列纵向求平均。""",
        r"""用 `labels == c` 标出要留下的行；求平均时必须写 `axis=0`，否则会把整块矩阵压成单个数字。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下一个单元格（代码已隐藏）会调用你的函数，为 0 到 9 这十个数字各做一张模板，并排拼成 `W` 的十列，所有 `b` 依然保持为 0。
    运行后会输出两个准确率：一个是训练集上的，一个是**测试集**上的——也就是做模板时**从未见过**的那 2,000 张图片。
    （这个单元格和后面两个单元格都需要先完成练习 1 到 3。）
    """)
    return


@app.cell(hide_code=True)
def _(X, X_test, test_labels, train_labels):
    W_hand = np.stack([template(X, train_labels, c) for c in range(10)], axis=1)
    b_hand = np.zeros(10)
    accuracy(X @ W_hand + b_hand, train_labels), accuracy(X_test @ W_hand + b_hand, test_labels)
    return W_hand, b_hand


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    两个准确率都在 0.62 左右：一下从瞎猜的 10% 跳到了 62%，而且完全没有经过训练，只是简单取了平均。其中测试集准确率才最说明问题——它衡量的是模型面对陌生笔迹时的真实本事。
    下面把这十张模板画出来看看，参数越大的地方蓝色越深：
    """)
    return


@app.cell(hide_code=True)
def _(W_hand):
    images.weight_maps([W_hand[:, c] for c in range(10)], list(range(10)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    它们看起来就像十个模糊的手写数字——本来也就是求平均叠出来的。

    不过这套做法有个明显的软肋。下一个单元格会统计在训练集上每个数字各被猜了多少次（从左到右依次是 0 到 9）。实际上，每个数字的图片都差不多各有 1,000 张。
    """)
    return


@app.cell(hide_code=True)
def _(W_hand, X, b_hand):
    [int(n) for n in np.bincount(guess(X @ W_hand + b_hand), minlength=10)]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    模型竟然把大约 2,750 张图片猜成了 8、把大约 2,100 张猜成了 0，而 5 总共只被猜中过一次。原因在于 0 和 8 的笔画又粗又铺得开，模板里的正数又多又大，无论拿哪张图去算点积都很容易蹭出高分；
    相比之下，5 的模板颜色偏淡，几乎每次都被 0 和 8 压过一头，大部分 5 最后都被错认成了 0 或 8。

    回到我们的迷你世界，一眼就能看明白。假设再加第三个符号 **+**，它的模板同时包含竖杠和横杠，一共有 5 个 1（而不是 3 个）。这时，一根工整的竖杠对 **|** 得 3 分，对 **+** 同样得 3 分，已经打平了。
    要是这根竖杠写得稍微潦草一点，中间偏左多蹭出一个有墨的像素，它对 **|** 还是 3 分、对 **—** 得 2 分，对 **+** 却涨到了 4 分——直接被判成了 **+**。
    模板里的 1 越多，就越容易“通吃”各种图片的笔画，白白占尽便宜。真实数字里的 0 和 8 吃的正是这个红利。
    """)
    return


@app.cell(hide_code=True)
def _():
    _sloppy = [[0, 1, 0], [1, 1, 0], [0, 1, 0]]
    _ink = {(0, 1): "blue", (1, 0): "blue", (1, 1): "blue", (2, 1): "blue"}
    grids([
        {"name": "潦草的 |", "values": _sloppy, "marks": _ink}, "×",
        {"name": "| 的参数", "values": [[0, 1, 0], [0, 1, 0], [0, 1, 0]], "marks": {(r, 1): "red" for r in range(3)}},
        "→", {"name": "分数", "values": [[3]]}, "    ",
        {"name": "潦草的 |", "values": _sloppy, "marks": _ink}, "×",
        {"name": "+ 的参数", "values": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
         "marks": {(0, 1): "red", (1, 0): "red", (1, 1): "red", (1, 2): "red", (2, 1): "red"}},
        "→", {"name": "分数", "values": [[4]], "marks": {(0, 0): "box"}},
    ], "多出一个像素的竖杠，对 | 的模板得 3 分，对笔画更密的 + 的模板却得了 4 分")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    求平均的做法每次只盯着单个数字自己，从不关心真正关键的问题：
    哪些位置的参数才能把 5 和 8 **区分开**？

    ## 下一步

    一组好参数既要抬高正确数字的得分，**又要**压低其他数字的得分。靠人手去调这 7,850 个参数显然不现实，但梯度下降可以——前提是得给它一个平滑、能指明优化方向的损失函数。
    准确率本身没法胜任这个角色。**第 07 章**会解释背后的原因，把得分转换成概率并构造出一个真正好用的分类损失，让机器自己学出一套参数模板，准确率一举突破 90%。

    ## 你现在会做的事

    - 用 `[i]`、`[i, j]` 和 `[i][j]` 从数组里取数，把二维图像展开成一维长行
    - 做矩阵乘法，并用形状规则检查结果维度
    - 用 `argmax` 把分数变成猜测，并计算分类准确率
    - 用布尔数组（`True` / `False`）筛选矩阵里的行，分清 `axis=0` 和 `axis=1` 的方向
    - 用平均模板搭出一个分类器，并在从未见过的测试集上评估它
    - 说清楚为什么简单取平均做出的模板会有明显短板

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**这些图片来自 *MNIST*（LeCun、Cortes 和 Burges）。用每一类的均值作为模板来做判断的分类器，在机器学习里叫作*最近质心分类器*（nearest-centroid classifier）或*模板匹配分类器*（template-matching classifier）。
    """)
    return


if __name__ == "__main__":
    app.run()
