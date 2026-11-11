import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="06 · 认数字")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    from learn_llm import hand, images, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 06 · 认数字

    到目前为止，每个模型都只猜一个数：卖出多少个冰淇淋。这一章，模型要看一张手写数字的图片，说出它是哪个数字：0、1、2……还是 9。
    这叫**分类**（classification）：从一张固定的答案表里选出一个。这一章，你要把图片变成数、给它们打分，
    并亲手造出第一个分类器。到第 07 章，机器会自己学出一个更好的。

    ## 图片

    课程自带 12,000 个手写数字，来自一个叫 MNIST 的著名数据集。`load_digits()` 把它们分成四份交给你：
    10,000 张用来学习的图片、它们的正确数字（叫**标签**（label）），以及另外 2,000 张带标签的图片，先放在一边，最后用来测试模型。
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
    形状有三个数 `(10000, 28, 28)`（上面显示成一个有三个数的列表：10000、28 和 28）。也就是 10,000 张图片，每张是 28 行、28 列的方格。方格里的每个数是图片上的一个点，叫**像素**（pixel）：
    0 是白纸，255 是黑色墨水，中间的数是不同深浅的灰。

    /// admonition | 🧰 工具箱：取出其中一项
    要从数组（或列表）里取出一项，就在方括号里写它的位置，从 0 开始数：`train_images[0]` 是第一张图片，`train_labels[0]` 是它的标签。
    对矩阵，先写行、再写列：对一个矩阵 $M$，`M[2, 5]` 是行号为 2、列号为 5 的数（从 0 数起，也就是第三行第六列）。只给矩阵一个位置，得到的是整整一行：
    `train_images[0]` 是第一张图片（一个 $28 \times 28$ 的矩阵），所以 `train_images[0][14]` 是它行号为 14 的那一行，一条横穿图片中间的线：
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
    标签是 6（NumPy 显示单个数时会把它的类型写在外面：`np.uint8(6)` 是 0 到 255 之间的整数，后面的 `np.float64(...)` 是小数；只看括号里的数就行）；
    这一行除了笔画经过的地方，都是白的（0）。下面是用这些像素值画出来的前 8 张图片，以及它们的标签：
    """),
        images.digits(train_images[:8], train_labels[:8]),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 每张图片展开成一长行

    我们的模型处理的是向量，所以把每个 $28 \times 28$ 的方格展开成一行 784 个数（$28 \times 28 = 784$）：先是图片的第 0 行，接着第 1 行，依此类推。
    `reshape` 做的就是这件事：数还是那些数，顺序也不变，只改变形状。我们还把它们都除以 255，让每个像素都在 0（白）到 1（黑）之间，
    理由和第 04、05 章换算气温一样：数值小、大小又相近，才适合用同一个学习率。下面的形状是 `(10000, 784)` 和 `(2000, 784)`：每张图片一行 784 个数。
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
    ## 十个分数，每个数字一个

    在第 05 章里，一组旋钮把一天的输入变成一个猜测。现在我们要十个数，每个数字一个**分数**（score），所以给每个数字配一组自己的 784 个旋钮
    （再加一个自己的 `b`）。数字 3 的分数，就是图片和数字 3 那组旋钮的点积：如果这组旋钮在 3 常有墨水的地方取值很大，3 的分数就会高。

    把十组旋钮并排放成矩阵 `W` 的十**列**，形状是 `(784, 10)`；十个 `b` 放进形状为 `(10,)` 的向量 `b`。于是

    > `S = X @ W + b`

    一次就算出所有图片的所有分数。矩阵 `@` 矩阵是让第一个矩阵的**每一行**和第二个矩阵的**每一列**做点积：
    第 $i$ 张图片对数字 $c$ 的分数，就是 $X$ 的第 $i$ 行和 $W$ 的第 $c$ 列的点积。形状规则还是第 05 章那条：
    `(10000, 784) @ (784, 10)` 得到 `(10000, 10)`；中间挨着的两个数（784 和 784）相同，算完就消掉。然后 $+ b$ 把每个数字的 $b$
    加到它那一列上，每张图片都加（第 05 章的广播）。

    /// admonition | 🧰 工具箱：全是 0 的数组
    先让所有旋钮都从 0 开始。`np.zeros((784, 10))` 做出一个这个形状、全是 0 的数组（形状写成一对数，放在括号里），`np.zeros(10)` 则是一个有 10 个 0 的向量。
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

    继续之前，先在一个小一点的例子上试试这条规则。假设 `X` 里有 5 张图片，所以形状是 `(5, 784)`，`W` 和上面一样，是 `(784, 10)`。
    `X @ W` 有多少行、多少列？`W` 里一共有多少个旋钮？
    """)
    return


@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['行数', '列数', 'W 的旋钮数'])
    return (shape_boxes,)


@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'行数': 5, '列数': 10, 'W 的旋钮数': 7840}, {
        '行数': r"""每张图片一行分数""",
        '列数': r"""每个数字一个分数""",
        'W 的旋钮数': r"""784 行，每行 10 个旋钮""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 猜测

    模型的猜测，就是分数**最高**的那个数字。下面是随手编的分数：4 张图片，为了简单只有 3 个数字。

    | 图片 | 数字 0 的分数 | 数字 1 的分数 | 数字 2 的分数 | 标签 |
    |---|---|---|---|---|
    | 0 | 1 | 3 | 2 | 1 |
    | 1 | 0 | 0 | 5 | 2 |
    | 2 | 4 | 1 | 1 | 1 |
    | 3 | 2 | 6 | 3 | 1 |

    图片 0 最高的分数是 3，在数字 1 那一列，所以模型猜 1。它的标签也是 1，所以这次猜对了。

    ## 动手算：猜测和准确率

    填出图片 1、2、3 的猜测。再填**准确率**（accuracy）：4 张图片里，猜测和标签相同的占多大比例，写成 0 到 1 之间的数。
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

    /// admonition | 🧰 工具箱：最大的在哪里？
    `np.argmax(v)` 给出向量里最大那个数的**位置**：`np.argmax(np.array([3, 9, 4]))` 是 1。

    对矩阵 `M`，`np.argmax(M, axis=1)` 会**沿着每一行**来找，每一行给出一个位置。`axis=0` 则是沿着每一列往下找。
    有两个数并列最大时，`np.argmax` 取第一个。
    ///

    写出 `guess(S)`。`S` 是分数矩阵，每张图片一行，每个数字一列。为每张图片返回一个猜测的数字。

    在上面那张表上，它应该得到 `[1, 2, 0, 1]`。
    """)
    return


@app.function
def guess(S):
    # ← replace todo() below with your code
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
    `==`（两个等号）问的是“相等吗？”，而一个 `=` 是存值。用在数组上，它逐个位置比较：
    `np.array([1, 2, 0]) == np.array([1, 5, 0])` 得到 `[True, False, True]`。

    对这样的数组求 `np.mean` 时，每个 `True` 算 1、每个 `False` 算 0，所以得到的是 `True` 所占的比例：这里是 $2 / 3$。
    ///

    写出 `accuracy(S, labels)`：猜测和标签相同的图片所占的比例。用上你的 `guess`。

    在上面那张表上，标签是 `[1, 2, 1, 1]` 时，它应该得到 0.75。
    """)
    return


@app.function
def accuracy(S, labels):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("06", "accuracy", accuracy, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三步：先得到各个猜测，再和标签比较，最后求均值。""",
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
    所有分数都是 0 时，处处都是平局，`np.argmax` 会选第一个位置：模型对每张图片都猜 0。只有图片真的是 0 时它才猜对，
    比例是 0.0969（10,000 张里有 969 张）。大约十分之一，和瞎猜一样。

    ## 亲手定旋钮：模板

    还没开始学习，我们能不能先选一组比瞎猜强的旋钮？回想一下分数是什么：图片和某个数字那组旋钮的点积。
    图片在旋钮大的地方有墨水，分数就高。所以有个简单的主意：让数字 3 的旋钮就是一个**典型的 3**，也就是把训练图片里所有的 3 逐个像素取均值。
    大多数 3 都有墨水的像素，旋钮就大；3 在那里是空白的像素，旋钮就接近 0。这样的一列旋钮叫**模板**（template）。

    ## 练习 3：template

    /// admonition | 🧰 工具箱：用 True 和 False 挑出行
    在矩阵的方括号里放一个由 `True` 和 `False` 组成的数组，每一行对应一个，得到的就只有标着 `True` 的那些行。
    设 `M` 的三行是 `[1, 2]`、`[3, 4]` 和 `[5, 6]`，那么 `M[np.array([True, False, True])]` 里是 `[1, 2]` 和 `[5, 6]`。

    这个数组通常来自一次比较。设 `kinds = np.array([7, 9, 7])`，`M[kinds == 7]` 给出的还是这两行。

    `np.mean(M, axis=0)` 是**沿着每一列往下**求均值：每一列一个数。对刚才挑出的两行，结果是 `[3, 4]`。
    ///

    写出 `template(X, labels, c)`：`X` 里标签为 `c` 的那些行的均值。`X` 每张图片一行，所以结果是一个向量，图片有多少个像素，它就有多少个数。

    设 `X` 的三行是 `[1, 2]`、`[3, 4]`、`[5, 6]`，标签是 `[7, 9, 7]`，`template(X, labels, 7)` 应该得到 `[3, 4]`。
    """)
    return


@app.function
def template(X, labels, c):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("06", "template", template, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""两步：先挑出标签是 `c` 的行，再沿着每一列往下求均值。""",
        r"""`labels == c` 标出了要的行。要让均值沿着列往下算，得写 `axis=0`：不写的话，得到的是所有数合在一起的一个均值。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下一个单元格（代码是隐藏的）用你的函数为十个数字各做一个模板，并排放成 `W` 的十列。每个 `b` 都保持为 0。
    它显示两个准确率：一个在训练图片上，一个在**测试**图片上，也就是做模板时**没有**用到的那 2,000 张。
    （这个单元格和后面两个单元格需要完成练习 1 到 3。）
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
    两个都是大约 0.62：认对 62%，而瞎猜只有 10%。这里完全没有学习，只是取了均值。测试准确率才反映真实水平：它说明模型面对从没见过的笔迹表现如何。
    下面把十个模板画成图片，旋钮大的地方是蓝色：
    """)
    return


@app.cell(hide_code=True)
def _(W_hand):
    images.weight_maps([W_hand[:, c] for c in range(10)], list(range(10)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    它们看起来像模糊的数字，事实上也正是如此。

    现在看它的弱点。下一个单元格数一数：在训练图片上，每个数字被猜了多少次，从左边的数字 0 到右边的数字 9。每个数字大约有 1,000 张图片。
    """)
    return


@app.cell(hide_code=True)
def _(W_hand, X, b_hand):
    [int(n) for n in np.bincount(guess(X @ W_hand + b_hand), minlength=10)]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    模型把大约 2,750 张图片猜成 8，把大约 2,100 张猜成 0，而 5 只猜了一次。0 和 8 的模板墨水最多，铺得又开，所以几乎给什么都打高分。
    大多数 5 最后被猜成了 0 和 8：5 的模板比较淡，几乎每次都被它们比下去。取均值是把每个数字单独拿出来看，它从来不问真正要紧的问题：
    哪些旋钮能把 5 和 8 **区分开**？

    ## 下一步

    好的旋钮既要抬高正确数字的分数，**又要**压低其他数字的分数。没有人能这样亲手调好 7,850 个旋钮。梯度下降可以，只要给它一个能往下走的损失。
    准确率当不了这个损失。**第 07 章**会说明为什么，造出一个管用的损失，再让机器自己去找模板。它们能达到 90%。

    ## 你现在会做的事

    - 用 `[i]` 和 `[i, j]` 从数组里取数，把一个方格展开成一行
    - 用矩阵乘矩阵，并检查形状
    - 把分数变成猜测，并用准确率来衡量
    - 用 `True` 和 `False` 挑出矩阵里的行
    - 用模板搭出一个分类器，并在从没见过的图片上测试它
    - 说出取均值得到的模板差在哪里

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**这些图片来自 *MNIST*（LeCun、Cortes 和 Burges）。每一类用一个均值做成的分类器，叫作*最近质心*分类器，也叫*模板匹配*分类器。
    """)
    return


if __name__ == "__main__":
    app.run()
