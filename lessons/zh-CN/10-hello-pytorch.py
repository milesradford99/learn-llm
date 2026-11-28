import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="10 · 你好，PyTorch")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.diagrams import cards, grids, pipeline

    LOCALE = "zh-CN"
    laptop_speed()
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 10 · 你好，PyTorch

    到现在为止，训练的每个环节都是你亲手写出来的：正向计算、自动记账的求导引擎、矩阵的反向传播、小批量训练、Adam 优化器。不过在实际工作中，没人会为每个新模型把这些轮子从头造一遍。大家都会用**库（library）**——也就是别人写好并充分测试过的现成工具箱，就像你之前一直在用的 NumPy。

    这个领域里用得最多的库就是 **PyTorch**。这一章其实不是在学一门全新的知识——PyTorch 做的事你早就门儿清了，因为你亲手造过一遍。接下来我们一块一块地认识它，每看一块，就拿它的计算结果和你手写的代码对一对。只要数字严丝合缝，你就能放心地用它，而且心里清楚它底层到底在干什么。

    ## 张量：PyTorch 的数组

    PyTorch 里对应 NumPy 数组的概念叫**张量（tensor）**。名字听着挺唬人，说白了就是“带形状的数字数组”。你在 NumPy 里学过的大部分操作都能直接照搬：`@`、`*`、`.T`、`.shape`、索引和切片的用法几乎一模一样。

    各种维度的张量你其实都见过了：单独一个数，形状是空的 `()`；一串数（向量，比如某一天的三个输入特征），形状里有一个数；一张表（矩阵，比如第 05 章那 8 天的数据），形状里有两个数；一叠表，形状里有三个数——第 06 章的 10,000 张 $28 \times 28$ 图片，形状就是 `(10000, 28, 28)`，就像一本厚书，每一页印着一张表格。
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "()", "values": [[7]]}, "   ",
        {"name": "(3,)", "values": [[17, 1, 0]]}, "   ",
        {"name": "(2, 3)", "values": [[14, 0, 0], [17, 1, 0]]}, "   ",
        {"name": "(2, 2, 3)：第 0 页", "values": [[1, 0, 1], [0, 1, 0]]}, " ",
        {"name": "第 1 页", "values": [[0, 0, 1], [1, 1, 0]]},
    ], "形状为空、3、2 乘 3、2 乘 2 乘 3 的张量：一个数、一行、一张表、两张叠起来的表")
    return


@app.cell
def _():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    u = torch.tensor([10.0, 20.0])
    A @ u, A.shape, torch.mean(A)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `torch.tensor(...)` 用列表创建张量，就像 `np.array(...)`。`torch.mean` 对应 `np.mean`，而 `torch.sum`、`torch.exp`、`torch.log` 也都和你猜的一样。

    /// admonition | 🧰 工具箱：老工具，新名字
    PyTorch 里的工具大多是你已经认识的，只是有些换了名字。这一章里有四个名字不太一样：

    | 你认识的 | PyTorch |
    |---|---|
    | `axis=1` | `dim=1`（dimension，维度） |
    | `relu(Z)`，第 08 章的弯折 | `torch.relu(Z)` |
    | `default_rng(0)`，让随机数可重现 | `torch.manual_seed(0)` |
    | `rng.permutation(n)` | `torch.randperm(n)` |

    还有一个新工具：`x.item()` 可以把只装着单个数字的张量取出成普通的 Python 数字。
    ///

    /// admonition | 🧰 工具箱：数字有多精确？
    计算机用固定大小的内存来存每个数字，内存大小用**位（bit）**计量。NumPy 默认每个数占 64 位，能保证大约 16 位有效数字准确；PyTorch 默认用 32 位，能保证大约 7 位——内存省一半，训练模型也绰绰有余。正因为精度不同，同一个算式的最后几位可能会有微小差别。下面用两种方式算 $0.1 \times 3$。（两边都不正好等于 0.3，这很正常：计算机用二进制存数，0.1 没法精确表示，就像十进制里 $1 / 3$ 写成 0.333… 永远写不完，总得在某一位截断一样。）
    ///
    """)
    return


@app.cell
def _():
    (torch.tensor(0.1) * 3).item(), 0.1 * 3
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：数字的种类
    张量里存的是哪种数字，叫**数据类型（dtype）**。最常见的有三种：

    | dtype | 存什么 | 用在哪里 |
    |---|---|---|
    | `torch.float32` | 小数，32 位 | 几乎所有地方 |
    | `torch.float64` | 小数，64 位 | 需要精确对比的时候，比如下面的验证 |
    | `torch.long` | 整数 | 分类标签和索引位置 |

    调用 `torch.tensor` 时可以指定 dtype，比如 `torch.tensor(labels, dtype=torch.long)`。直接从 NumPy 数组创建的张量会保留原数组的 dtype（通常是 64 位）。PyTorch 不允许 `@` 两侧的 dtype 不一致，所以要是你遇到包含 `must have the same dtype` 的报错，原因就在这里。
    ///

    ## 你的引擎，已经内置

    先重温一下第 04 章的例子：米娅卖冰淇淋的第 1 天（`t = 14`，`s = 63`），取 `w = 10`、`b = -60`。当时你算出的梯度（责任）分别是 476 和 34。
    """)
    return


@app.cell
def _():
    w = torch.tensor(10.0, requires_grad=True)
    b = torch.tensor(-60.0, requires_grad=True)
    e = w * 14 + b - 63
    L = e * e
    L.backward()
    w.grad.item(), b.grad.item()
    return b, w


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    结果正是 476 和 34，和你算的一模一样。回忆一下第 04 章你亲手写的自动求导引擎：每次做加减乘除时，引擎都会在账本（`book`）里记下这次运算的输入是谁、局部斜率是多少；调用反向传播时，引擎倒着翻账本，把斜率一路乘回去并累加到每个参数的 `.grad` 上。对比一下你的引擎和 PyTorch：

    | 你的引擎（第 04 章） | PyTorch |
    |---|---|
    | `Value(10)` | `torch.tensor(10.0, requires_grad=True)` |
    | `add(a, b, book)`、`mul(a, b, book)` | `a + b`、`a * b` |
    | `book`，由你手动维护 | 自动记录，藏在后台 |
    | `backward(loss, book)` | `loss.backward()` |
    | `v.data`、`v.grad` | `v.item()`、`v.grad` |

    `requires_grad=True` 的意思是“这是一个要调的参数（旋钮）：凡是由它算出来的每一步都要记账”。对这样的张量做 `+` 或 `*` 时，PyTorch 会自动记下输入来源和局部斜率，和你的 `mul` 做的事情完全一样。接着调用 `backward()`，它就会倒着翻账本，把梯度填进每个参数的 `.grad` 里。`w.grad` 是一个只装着单个数的张量，用 `.item()` 就能取出普通的 Python 数字。

    这本账虽然藏在后台，但你依然能顺藤摸瓜看到它。每个由参数算出来的张量，都附带一张“我是哪一步算出来的”便条，而那一步又指向它前面的步骤。从 `L` 出发，每次沿着一个输入往回追溯，就能看到下面这条链（卡片标题是 PyTorch 内部真实的操作名，直接从这次计算里读出来的）：
    """)
    return


@app.cell(hide_code=True)
def _():
    _w = torch.tensor(10.0, requires_grad=True)
    _b = torch.tensor(-60.0, requires_grad=True)
    _L = (_w * 14 + _b - 63) ** 2
    _names, _step = [], _L.grad_fn
    while _step is not None:
        _names.append(type(_step).__name__)
        _step = _step.next_functions[0][0] if _step.next_functions else None
    _meaning = [["L = e × e", "你的 square"], ["e = g − 63", "你的 sub"], ["g = p + b", "你的 add"],
                ["p = w × 14", "你的 mul"], ["把责任加进", "w.grad"]]
    cards([{"title": name, "lines": lines} for name, lines in zip(_names, _meaning)],
          "PyTorch 为米娅这一天藏起来的账本，从 L 倒着读：乘方、减法、加法、乘法，最后一步把责任加进旋钮 w", arrows="→")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这就是第 04 章的账本，只不过从末尾倒着读：`backward()` 走的正是这条链。（`PowBackward0` 对应把 `e * e` 写成乘方 `** 2` 的那一步；最后一张卡片 `AccumulateGrad` 则是把传回来的梯度**加**进 `w.grad` 的那一步。先记住这个“加”字。）

    还记得第 04 章引擎里的那个坑吗？因为梯度（责任）是**累加**到张量上的，所以每次必须从 0 开始。PyTorch 也是这样，而且是有意这么设计的。下面我们重新算一遍损失，再调一次 `backward()`，但故意不清空 `.grad`：
    """)
    return


@app.cell
def _(b, w):
    L_again = (w * 14 + b - 63) ** 2
    L_again.backward()
    w.grad.item(), b.grad.item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    翻倍了：变成 952 和 68。（再运行一次这个单元格，它还会继续叠加变大；想重头来过，重新运行前面带 `L.backward()` 的单元格即可。）新算出的梯度直接加到了旧梯度上。所以训练的每一步都必须先把旧梯度清零。忘了清零是写 PyTorch 时最经典的坑，后面很快就会见到专门帮我们清零的工具。

    （顺带一提，这里我们必须重新算一遍 `L_again`。为了省内存，PyTorch 走完一次反向传播就会把账本销毁，所以每一步都需要重新做一次正向计算，和第 04 章一样。）
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：PyTorch 应该算出什么？

    问 PyTorch 之前，先自己手算一个答案用来核对。取直线参数 `w = 1`、`b = 0`，两天的数据是 `xs = [1, 2]`、`ys = [2, 4]`。
    两天的误差分别是 $1 \times 1 + 0 - 2 = -1$ 和 $1 \times 2 + 0 - 4 = -2$。按第 03 章的公式，均方误差对 `w` 的斜率是 $2 \times e \times x$ 的均值，对 `b` 的斜率是 $2 \times e$ 的均值。
    """)
    return


@app.cell(hide_code=True)
def _():
    gradient_boxes = hand.boxes(['w 的斜率', 'b 的斜率'])
    return (gradient_boxes,)


@app.cell(hide_code=True)
def _(gradient_boxes):
    hand.marked(gradient_boxes, {'w 的斜率': -5, 'b 的斜率': -3}, {
        'w 的斜率': r"""$2 \times (-1) \times 1$ 和 $2 \times (-2) \times 2$ 的均值""",
        'b 的斜率': r"""$2 \times (-1)$ 和 $2 \times (-2)$ 的均值""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：torch_gradient

    在第 04 章，你的 `engine_gradient` 写了大约二十行。现在用张量把同样的事再写一遍。`torch_gradient(w, b, xs, ys)` 接收两个普通小数 `w`、`b` 和两个列表，返回均方误差对 `w` 和 `b` 的两个斜率（也是普通 Python 数字）。对照上面的表格来做：

    1. 把 `w` 和 `b` 变成需要记账的参数张量。
    2. 把两个列表变成张量 `x` 和 `y`，并指定 `dtype=torch.float32`。（如果不指定，整数列表会变成整数张量，没法和浮点参数混在一起算。）
    3. 一次算出所有天的误差。只装着一个数的张量 `w` 会自动广播到整个向量 `x` 上。
    4. 用 `torch.mean` 算出误差平方的均值作为损失。
    5. 从损失开始反向传播，再用 `.item()` 取出并返回两个斜率。

    对你刚才手算的例子，`torch_gradient(1.0, 0.0, [1, 2], [2, 4])` 应该正好给出你算出来的那两个数。
    """)
    return


@app.function
def torch_gradient(w, b, xs, ys):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("10", "torch_gradient", torch_gradient, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""大约八行简短的代码。`w` 和 `b` 这两个变量名可以直接复用：用 `w = torch.tensor(w, ...)` 把普通数字替换成带记账功能的张量。""",
        r"""只有对损失调用 `backward()` 之后，梯度才会填进 `w.grad` 和 `b.grad` 里。如果遇到 dtype 报错，检查第 2 步。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 三种方法，一个答案

    现在来个更复杂的：第 08 章的两层神经网络。下面已经放好了你在第 08、09 章用 NumPy 写好的 `softmax`、`forward` 和 `backward`。接下来我们取同样的 100 张图片，分别用你亲手推出的公式和 PyTorch 来算这个网络所有参数的梯度，看看结果是不是一致。
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


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    X_np = train_images.reshape(10000, 784) / 255
    X_test_np = test_images.reshape(2000, 784) / 255
    rng = np.random.default_rng(0)
    W1_np = rng.normal(0, np.sqrt(2 / 784), (784, 64))
    b1_np = np.zeros(64)
    W2_np = rng.normal(0, np.sqrt(2 / 64), (64, 10))
    b2_np = np.zeros(10)
    X_np.shape, W1_np.shape
    return (
        W1_np,
        W2_np,
        X_np,
        X_test_np,
        b1_np,
        b2_np,
        test_labels,
        train_labels,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：scores

    用张量写出正向计算。`scores(X, W1, b1, W2, b2)` 就是第 08 章的 `forward`：先做一次加权和、经过弯折（激活），再做一次加权和。PyTorch 里的弯折函数叫 `torch.relu`。只需要返回最终分数。

    和第 08 章不同，这次不用把中间结果 `Z` 和 `H` 返回出来。以前保留它们，是因为手写的 `backward` 要用；现在 PyTorch 的账本会自动替你记着。
    """)
    return


@app.function
def scores(X, W1, b1, W2, b2):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("10", "scores", scores, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`@` 和 `+` 用在张量上，和用在数组上一样。""",
        r"""一行就够。弯折只包住第一次加权和。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    算损失还需要一个新工具。

    /// admonition | 🧰 工具箱：现成的交叉熵
    `F.cross_entropy(S, labels)` 就是第 07 章的交叉熵损失，但省了两件事：它直接接收原始**分数**（内部会自动做 softmax），**标签**也直接传整数就行，不用转成 one-hot 向量。`F` 是 PyTorch 中 `torch.nn.functional` 模块的通用简称，这里已经提前帮你导入好了（`torch`、`nn` 和 `np` 也一样）。
    ///

    下面的单元格在 100 张图片上做一次完整对比：前半段跑你手写的 NumPy 版本；后半段把同样的权重和数据装进张量（从 NumPy 数组创建的张量会自动保留 64 位精度，所以两边精度完全一致），让 PyTorch 自动算梯度。最后一行输出全部 50,890 个梯度里相差最大的那一个。

    /// admonition | 🧰 工具箱：两种简写
    单元格里用到了两种新写法：`[torch.tensor(k, requires_grad=True) for k in (…)]` 是 Python 的列表推导式，意思是“对括号里的每一项执行一次前面的表达式，把结果收集成列表”，相当于一个带 `append` 的简短 `for` 循环。另外，`.numpy()` 可以把张量转回 NumPy 数组；`max` 是 Python 内置函数，返回两个数里较大的那个。
    ///
    """)
    return


@app.cell
def _(W1_np, W2_np, X_np, b1_np, b2_np, train_labels):
    X_100, labels_100 = X_np[:100], train_labels[:100]
    Y_100 = np.eye(10)[labels_100]
    Z_np, H_np, S_np = forward(X_100, W1_np, b1_np, W2_np, b2_np)
    mine = backward(X_100, Y_100, Z_np, H_np, S_np, W2_np)

    knobs_t = [torch.tensor(k, requires_grad=True) for k in (W1_np, b1_np, W2_np, b2_np)]
    S_t = scores(torch.tensor(X_100), *knobs_t)
    F.cross_entropy(S_t, torch.tensor(labels_100, dtype=torch.long)).backward()

    biggest = 0
    for m, k in zip(mine, knobs_t):
        biggest = max(biggest, np.max(np.abs(m - k.grad.numpy())).item())
    biggest
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    结果末尾大约是 `e-17`。和第 09 章一样，这表示小数点往左移 17 位——也就是 0.00000000000000003，纯属浮点舍入误差。你在第 08 章一项一项亲手推出的公式，算出来的和 PyTorch 账本算出来的完全是同一个东西。别忘了这些梯度你还用“轻推法”验算过，这就等于三条互相独立的路通到了同一个答案：数值轻推、手推公式、PyTorch 自动求导。

    ## 把层打包成对象

    网络**层（layer）**一多，手动维护 `W1, b1, W2, b2` 这些变量就很繁琐。PyTorch 把每一层的参数打包成了一个现成的对象：
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    layer = nn.Linear(784, 64)
    layer.weight.shape, layer.bias.shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    （`torch.Size` 只是 PyTorch 打印张量形状的格式。）

    `nn.Linear(784, 64)` 会创建一个接收 784 个输入、产出 64 个输出的线性层，内部自带一个权重矩阵 `layer.weight` 和一个偏置向量 `layer.bias`。它们一创建好就已经做了随机初始化，并且默认开启了 `requires_grad=True`。（`torch.manual_seed(0)` 用来固定随机数，作用和 `default_rng(0)` 一样。）

    注意一个小约定：创建 `nn.Linear(784, 64)` 时，括号里的参数顺序是 `(输入个数, 输出个数)`，但 PyTorch 内部存 `layer.weight` 时用的形状却是转置过来的 `(64, 784)`——也就是 `(输出个数, 输入个数)`。这和我们之前的习惯正好倒过来：我们的 `W1` 是 `(784, 64)`，每个探测器占一列；而 PyTorch 把每个探测器放在一行里。里面的数字完全一样，只是横竖翻了个面。以 3 个输入、2 个探测器为例，标蓝的是探测器 1 的参数：
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "我们的 W  (3, 2)", "values": [[1, 4], [2, 5], [3, 6]], "rows": ["输入 0", "输入 1", "输入 2"],
         "cols": ["探测器 0", "探测器 1"], "marks": {(r, 1): "blue" for r in range(3)}},
        "   ",
        {"name": "layer.weight  (2, 3)", "values": [[1, 2, 3], [4, 5, 6]], "rows": ["探测器 0", "探测器 1"],
         "cols": ["输入 0", "输入 1", "输入 2"], "marks": {(1, c): "blue" for c in range(3)}},
    ], "我们的 W 把每个探测器的旋钮放在一列里；layer.weight 把它们放在一行里：同样的数，转置了一下")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    因此 `layer.weight.T` 就是我们熟悉的 `W`。接着，只要像调用函数一样写 `layer(X)`，PyTorch 就会自动帮你算出 `X @ layer.weight.T + layer.bias`（也就是我们之前的 `Z`）。在 Python 里，把一个对象直接加括号当函数调用，刚开始看可能有点怪，其实它就是 PyTorch 提供的一种简写，意思是“把输入送进这一层，跑一遍它内部的计算”。

    `nn.Sequential(...)` 可以把多个层按顺序串起来：上一层的输出自动变成下一层的输入。`nn.ReLU()` 就是弯折（激活），它也做成了一个层，只是里面没有任何需要训练的参数。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：make_model

    写出 `make_model(hidden)`，返回一个拥有 `hidden` 个探测器的第 08 章网络。它是一个包含三层的 `nn.Sequential`：先从 784 个像素映射到 `hidden` 个探测器，再经过弯折，最后从 `hidden` 个探测器映射到 10 个分类分数。

    /// admonition | 🧰 工具箱：获取模型的参数
    `model.parameters()` 会依次返回模型里所有的参数张量（每层的权重和偏置各占一个）；`p.numel()` 则返回某一个张量里一共包含多少个数字。检查之后的单元格就用这两个方法来统计你的模型一共有多少个参数。
    ///
    """)
    return


@app.function
def make_model(hidden):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("10", "make_model", make_model, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`nn.Sequential(...)` 里放三层，用逗号隔开：一个 `nn.Linear`，一个 `nn.ReLU()`，再一个 `nn.Linear`。""",
        r"""第一层的输出个数和最后一层的输入个数都是 `hidden`，也就是你这个函数的输入，不是固定的 64。""",
    ], LOCALE)
    return


@app.cell
def _():
    torch.manual_seed(0)
    model_demo = make_model(64)
    knob_count = 0
    for p in model_demo.parameters():
        knob_count = knob_count + p.numel()
    model_demo, knob_count
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    PyTorch 会逐层打印出模型结构（`bias=True` 表示这一层带有偏置），后面跟着我们的统计结果：一共 50,890 个参数——$784 \times 64$ 个权重、64 个偏置、$64 \times 10$ 个权重、10 个偏置。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：探测器减半

    `make_model(32)` 会有多少个参数？先算第一层（权重和偏置），再算整个模型。
    """)
    return


@app.cell(hide_code=True)
def _():
    count_boxes = hand.boxes(['第一层', '整个模型'])
    return (count_boxes,)


@app.cell(hide_code=True)
def _(count_boxes):
    hand.marked(count_boxes, {'第一层': 25120, '整个模型': 25450}, {
        '第一层': r"""$784 \times 32$ 个权重和 32 个偏置""",
        '整个模型': r"""再加上最后一层：$32 \times 10$ 个权重和 10 个偏置""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 你的 Adam，已经内置

    **优化器（optimizer）**是专门管理所有参数、并按规则更新它们的对象。下面我们把第 09 章手写的 `adam_step` 和 PyTorch 内置的 `torch.optim.Adam` 并排跑一遍，解决米娅的冰淇淋问题：使用原始气温，都从 `[6, 0]` 出发，学习率 1.0，跑 500 步。
    """)
    return


@app.function
def adam_step(w, m, v, grad, t, rate):
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * grad * grad
    m_hat = m / (1 - 0.9 ** t)
    v_hat = v / (1 - 0.999 ** t)
    return w - rate * m_hat / (np.sqrt(v_hat) + 1e-8), m, v


@app.function
def mia_gradient(k):
    t = np.array(temps)
    s = np.array(sales)
    e = k[0] * t + k[1] - s
    return np.array([np.mean(2 * e * t), np.mean(2 * e)])


@app.cell
def _():
    k_mine = np.array([6.0, 0.0])
    m_mine = np.zeros(2)
    v_mine = np.zeros(2)
    for t_mine in range(1, 501):
        k_mine, m_mine, v_mine = adam_step(k_mine, m_mine, v_mine, mia_gradient(k_mine), t_mine, 1.0)

    k_torch = torch.tensor([6.0, 0.0], dtype=torch.float64, requires_grad=True)
    t_days = torch.tensor(temps, dtype=torch.float64)
    s_days = torch.tensor(sales, dtype=torch.float64)
    optimizer_mia = torch.optim.Adam([k_torch], lr=1.0)
    for _step in range(500):
        optimizer_mia.zero_grad()
        e_days = k_torch[0] * t_days + k_torch[1] - s_days
        torch.mean(e_days * e_days).backward()
        optimizer_mia.step()

    k_mine, k_torch.detach().numpy()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    两组参数完全一致，显示出来的每一位数字都一模一样。PyTorch 的 Adam 用的正是你写过的 0.9、0.999 和 `1e-8`。

    /// admonition | 🧰 工具箱：不带账本的数字
    `t.detach()` 返回和 `t` 数值相同的张量，但不再挂着账本。上面单元格在调用 `.numpy()` 之前必须先 `.detach()`，因为 PyTorch 不允许直接把正在记账的张量转成 NumPy 数组。
    ///

    上面循环里的这三个调用，是所有 PyTorch 训练循环的核心：

    - `optimizer.zero_grad()` 把所有参数的旧梯度清零（这就是前面提到用来防坑的工具）；
    - `loss.backward()` 算出每个参数的梯度并填进 `.grad`；
    - `optimizer.step()` 根据 `.grad` 更新每个参数。

    把正向计算放进前两步之间，就构成了完整的四步配方：清零、正向、反向、更新。
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("zero_grad()", ".grad 归零"), ("model(X)", "分数、损失"),
              ("backward()", "填好 .grad"), ("step()", "更新参数")],
             ["然后", "然后", "然后"],
             "训练的一步：清空旧梯度，算出损失，反向填好梯度，更新参数；下一批再从清空开始")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下一批数据到来时，再从最左边重新走一遍。这门课里所有的训练循环——一直到最后的大语言模型——本质上都是把这张流程图放进一个 `for` 循环里。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 找出错误

    下面是有人赶时间写出的一步训练代码：

    ```
    def hasty_step(model, optimizer, X, labels):
        loss = F.cross_entropy(model(X), labels)
        loss.backward()
        optimizer.step()
        return loss
    ```

    它跑起来不会报错，第一步甚至还是对的。对照上面的四步配方：清零（1）、正向（2）、反向（3）、更新（4），这四步里漏了哪一步？
    """)
    return


@app.cell(hide_code=True)
def _():
    bug_boxes = hand.boxes(['少了第几样'])
    return (bug_boxes,)


@app.cell(hide_code=True)
def _(bug_boxes):
    hand.marked(bug_boxes, {'少了第几样': 1}, {
        '少了第几样': r"""正向计算就是第一行里的 `model(X)`。找一找把 `.grad` 清空的那个调用""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    从第二步开始，每个参数的 `.grad` 里存的就不再是当前这一步的梯度，而是新梯度**加上**前面所有步骤残留的旧梯度，就像前面翻倍成 952 一样。参数虽然还在动，但方向早就偏了，步子也迈得太大。

    拿米娅的第 1 天用具体数字看看：走 8 步普通梯度下降（`torch.optim.SGD` 就是第 02 章的更新规则：参数减去学习率乘以斜率），一边每步清零，一边从不清零。下表列出了每一步的 `w`、`w.grad` 和损失：
    """)
    return


@app.cell(hide_code=True)
def _():
    _rows = {}
    for _clear in (True, False):
        _w = torch.tensor(10.0, requires_grad=True)
        _b = torch.tensor(-60.0, requires_grad=True)
        _opt = torch.optim.SGD([_w, _b], lr=0.001)
        _rows[_clear] = []
        for _step in range(8):
            if _clear:
                _opt.zero_grad()
            _loss = (_w * 14 + _b - 63) ** 2
            _loss.backward()
            _rows[_clear].append(f"{_w.item():.2f} | {_w.grad.item():.0f} | {_loss.item():.1f}")
            _opt.step()
    _lines = ["| 步 | `w`（每步清空） | `w.grad`（每步清空） | 损失（每步清空） "
              "| `w`（从不清空） | `w.grad`（从不清空） | 损失（从不清空） |",
              "|---|---|---|---|---|---|---|"]
    for _i, (_a, _c) in enumerate(zip(_rows[True], _rows[False])):
        _lines.append(f"| {_i + 1} | {_a} | {_c} |")
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每步清零时，损失一路稳步下降，从 289 降到接近 0。如果从不清零，每一步的新梯度都会叠在旧梯度上。前两步里，累积下来的梯度刚好还指向正确方向，损失甚至降得更快；可到了第 3 步，`w` 已经很接近理想值了，累积梯度却还带着前几步的“余威”继续把它往小里推，结果 `w` 冲过了头，损失一路反弹回近 300，直到新的反向梯度把旧账抵消掉，方向才勉强扭转回来。参数就这样来回震荡，怎么也稳不下来。

    ## 练习 4：train_step

    写出 `train_step(model, optimizer, X, labels)`：在一批数据上按正确顺序完成一步训练。用 `F.cross_entropy` 计算模型输出分数与标签之间的损失，最后把损失作为普通的 Python 数字返回。
    """)
    return


@app.function
def train_step(model, optimizer, X, labels):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("10", "train_step", train_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""六行简短的代码：清零、算分数、算损失、反向传播、更新参数，最后 `return`。""",
        r"""模型可以像函数一样直接调用：`model(X)`。用 `.item()` 把只含一个数的损失张量转成普通数字。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 完整的一次训练

    下面是第 09 章的训练函数，用刚才造好的组件重写了一遍。`torch.randperm` 就是 PyTorch 版的 `rng.permutation`。按轮次（epoch）和批次（batch）遍历的循环和你写的一样，其余细节全都收进了 `train_step` 里。
    """)
    return


@app.function
def train(model, X, labels, epochs, batch_size, rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            loss = train_step(model, optimizer, X[batch], labels[batch])
    return loss


@app.function
def accuracy(S, labels):
    return torch.mean((torch.argmax(S, dim=1) == labels).float()).item()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：读懂 `accuracy` 与 `with torch.no_grad():`
    上面 `accuracy` 的那一行套了四层调用，从里往外拆成四步就一目了然了：

    1. `torch.argmax(S, dim=1)`：在每行 10 个分数里找出最高分所在的列号，也就是模型预测的数字；
    2. `== labels`：把预测结果和真实标签逐个对比，得到一串 `True`/`False`；
    3. `.float()`：把 `True`/`False` 转成小数 `1.0`/`0.0`（因为 `torch.mean` 只能对小数求平均）；
    4. `torch.mean(...).item()`：对这串 `1.0` 和 `0.0` 求平均得到正确率，再用 `.item()` 取出成普通的 Python 小数。

    下面单元格里还会见到 `with torch.no_grad():`。Python 的 `with` 语句块用来“临时开启某种模式，离开缩进块后自动恢复”——这里它会在缩进块内暂时关掉自动记账。评估测试集不需要算梯度，关掉记账既省时间又省内存。
    ///

    下面的单元格把数据转成 32 位张量，用 Adam 以学习率 0.002 训练 10 轮，然后在测试集上计算准确率。第二个输出数字是训练耗时（秒），和第 09 章一样。
    """)
    return


@app.cell
def _(X_np, X_test_np, test_labels, train_labels):
    X = torch.tensor(X_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    labels = torch.tensor(train_labels, dtype=torch.long)
    labels_test = torch.tensor(test_labels, dtype=torch.long)

    torch.manual_seed(0)
    model = make_model(64)
    started = time.perf_counter()
    train(model, X, labels, 10, 100, 0.002)
    seconds = time.perf_counter() - started
    with torch.no_grad():
        test_score = accuracy(model(X_test), labels_test)
    test_score, seconds
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    测试集准确率大约 95%，耗时最多几秒钟，而你总共只写了几行代码。这和第 09 章用的是同一个网络结构、同一批图片，只是记账和更新规则交给了 PyTorch。

    ## GPU 是什么？

    到现在为止，所有计算都是在电脑的主处理器 **CPU** 上完成的。CPU 里有几个非常全能的“高材生”；而 **GPU**（图形处理器，最初是为了同时给屏幕上几百万个像素点上色而设计的）里塞着成千上万个结构简单的“小帮手”。这正好对上了我们的胃口：矩阵乘法本质上就是成千上万道一模一样的小加法。
    打个比方，要算完一万道简单的加法题：请几位数学教授来算，虽然每个人手速都很快，但如果找一千名小学生坐满礼堂、每人只算十道，大家同时开工，很快就能比教授们先做完。对于大型模型，GPU 往往能快上 10 到 100 倍。

    这门课默认安装的是轻巧的纯 CPU 版 PyTorch，确保每一章都能在普通笔记本电脑上顺利运行。课程里没有哪一章非用 GPU 不可；如果哪一章用 GPU 会有明显加速，我们会特别说明。运行下面的单元格，看看 PyTorch 在你这台电脑上能不能找到 GPU：
    """)
    return


@app.cell
def _():
    torch.cuda.is_available()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    显示 `False` 很正常（在 Apple 芯片的 Mac 上，PyTorch 有另外的加速选项）。如果你确实有 GPU，代码需要改动的地方也非常少。

    /// admonition | 🧰 工具箱：搬到 GPU 上
    张量和模型都存放在某个**设备（device）**上，同一次计算涉及的所有数据必须在同一个设备里。用 `model.to("cuda")` 和 `X.to("cuda")` 就能把它们搬到 GPU 上（`cuda` 是最常见的 GPU 计算平台名称），其余代码一行都不用改。
    ///

    ## 接下来

    现在的网络把一张图片看成 784 个互不相干的独立数字：只要手写数字稍微挪动几个像素，在网络眼里就成了一张完全不同的图。在**第 11 章**，你会学到一个专门为图像设计的巧思：**卷积（convolution）**——也就是在图片上滑动的小探测器。既然记账已经交给 PyTorch 了，接下来你就能把精力全放在模型设计上。

    ## 你现在会做的事

    - 创建张量、选择合适的 dtype，并看懂 `must have the same dtype` 这类报错
    - 使用 `requires_grad`、`backward()` 和 `.grad`，并用自己写过的自动求导引擎解释它们背后的原理
    - 解释为什么每步训练前都必须把梯度清零
    - 用 `nn.Linear`、`nn.ReLU` 和 `nn.Sequential` 搭建神经网络
    - 用优化器完成标准训练循环：清零、正向、反向、更新
    - 用 `with torch.no_grad():` 在不记账的模式下评估模型
    - 说清楚 GPU 的优势以及“设备（device）”的概念

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** PyTorch 官方文档见 https://pytorch.org/docs/。`backward()` 背后的自动求导引擎叫 *autograd*，下一章要认识的 `nn.Module` 是所有网络层的基类。`nn.Linear` 内部自带一套参数初始化规则，思路和第 09 章的 He 初始化很接近。
    """)
    return


if __name__ == "__main__":
    app.run()
