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

    LOCALE = "zh-CN"
    laptop_speed()
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 10 · 你好，PyTorch

    到现在为止，训练的每个环节都是你亲手写的：正向计算、记账的引擎、穿过矩阵的反向步骤、小批量、Adam。但没有人会为每个模型把这些从头再写一遍。大家用的是**库**（library），也就是别人写好并测试过的工具箱，就像你已经在用的 NumPy。

    这个领域里用得最多的库叫 **PyTorch**。这一章不是在学一门新学问：PyTorch 做的事你已经知道，因为你亲手造过。所以我们一块一块地看它，每次都拿它的答案和你的答案比一比。数字对得上，你就可以放心用这个工具，也知道它内部在做什么。

    ## 张量：PyTorch 的数组

    PyTorch 里相当于 NumPy 数组的东西叫**张量**（tensor）。名字听着唬人，其实张量就是一个有形状的数字数组。你学过的大部分内容都能照搬：`@`、`*`、`.T`、`.shape`、索引和切片的用法都一样。
    """)
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
    `torch.tensor(...)` 用列表造出一个张量，就像 `np.array(...)`。`torch.mean` 就是 `np.mean`，`torch.sum`、`torch.exp`、`torch.log` 也都和你猜的一样。只有一个名字不同：NumPy 写 `axis=1` 的地方，PyTorch 写 `dim=1`（dimension，维度），意思完全相同。

    **关于精度。** 计算机用固定大小的内存存每个数字，内存大小用**位**（bit）来计量。NumPy 通常每个数用 64 位，能保证大约 16 位数字是准的。PyTorch 通常用 32 位，能保证大约 7 位：内存省一半，训练也完全够用。所以同一个算式，最后几位可能不一样。下面用两种方式算 $0.1 \times 3$。（两边都不正好是 0.3。这很正常：计算机用二进制存数，0.1 没法精确存下来。）
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
    存的是哪一种数字，叫**数据类型**（dtype）。你会遇到三种：

    | dtype | 存什么 | 用在哪里 |
    |---|---|---|
    | `torch.float32` | 带小数点的数，32 位 | 几乎所有地方 |
    | `torch.float64` | 带小数点的数，64 位 | 需要精确比较的时候，比如下面 |
    | `torch.long` | 整数 | 标签和位置 |

    `torch.tensor` 可以指定 dtype，比如 `torch.tensor(labels, dtype=torch.long)`。从 NumPy 数组造出的张量会保留数组原来的 dtype，通常是 64 位。PyTorch 不允许 `@` 的两边 dtype 不同，所以如果你看到提示 `must have the same dtype` 的报错，原因就在这里。
    ///

    ## 你的引擎，已经内置

    这是第 04 章做过的事：米娅的第 1 天（`t = 14`，`s = 63`），`w = 10`，`b = -60`。你当时算出的责任是 476 和 34。
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
    476 和 34，和你的一样。对比一下你写的：

    | 你的引擎（第 04 章） | PyTorch |
    |---|---|
    | `Value(10)` | `torch.tensor(10.0, requires_grad=True)` |
    | `add(a, b, book)`、`mul(a, b, book)` | `a + b`、`a * b` |
    | `book`，由你来填 | 自动填写，你看不到 |
    | `backward(loss, book)` | `loss.backward()` |
    | `v.data`、`v.grad` | `v.item()`、`v.grad` |

    `requires_grad=True` 的意思是“这是一个旋钮：凡是由它算出来的东西都要记账”。对这样的张量做的每一次 `+` 和 `*` 都会被记下来，连同来源和局部斜率，和你的 `mul` 做的一样。`backward()` 再倒着读账本，把 `.grad` 填好。`w.grad` 是一个只装着一个数的张量，`.item()` 把它变成普通的 Python 数字。

    你的引擎有一个坑：责任是**累加**到框上的，所以必须从 0 开始。PyTorch 也一样，而且是有意这样设计的。下面我们重新算一遍损失，再走一次反向，但不清空 `.grad`：
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
    翻倍了：952 和 68。（再运行一次这个单元格，它还会继续变大；运行带 `L.backward()` 的那个单元格就能重新开始。）新的责任加到了旧的上面。所以训练的每一步都要先清掉旧的责任。忘了这一点是 PyTorch 里最经典的错误，后面有专门的工具来做这件事。

    （注意 `L_again` 必须从头再算一遍。PyTorch 走完反向就把账本扔掉，以节省内存，所以每一步都需要一次新的正向计算，和第 04 章一样。）
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：PyTorch 应该给出什么？

    问 PyTorch 之前，先自己算出一个答案来对照。取 `w = 1`、`b = 0` 的直线，以及两天的数据：`xs = [1, 2]`，`ys = [2, 4]`。
    误差是 $1 \times 1 + 0 - 2 = -1$ 和 $1 \times 2 + 0 - 4 = -2$。按第 03 章，均方误差对 `w` 的斜率是 $2 \times e \times x$ 的均值，对 `b` 的斜率是 $2 \times e$ 的均值。
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

    在第 04 章，你的 `engine_gradient` 写了大约二十行。现在用张量把同样的事再做一遍。`torch_gradient(w, b, xs, ys)` 接收普通的数 `w`、`b` 和两个列表，
    返回均方误差的两个斜率，也是普通的数。照着上面那张对照表来：

    1. 把 `w` 和 `b` 变成旋钮：让 PyTorch 给它们记账的张量。
    2. 把两个列表变成张量 `x` 和 `y`，加上 `dtype=torch.float32`。（不然，整数列表会变成整数张量，没法和旋钮混在一起算。）
    3. 一次算出所有天的误差。只装着一个数 `w` 的张量会广播到整个向量 `x` 上。
    4. 损失是误差平方的均值，用 `torch.mean`。
    5. 从损失往回走，再用 `.item()` 返回两个斜率。

    对你刚才手算的例子，`torch_gradient(1.0, 0.0, [1, 2], [2, 4])` 应该给出你算出的那两个数。
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
        r"""大约八行短短的代码。`w` 和 `b` 这两个名字可以接着用：`w = torch.tensor(w, ...)` 把普通的数换成了旋钮。""",
        r"""只有对损失运行过 `backward()` 之后，斜率才会出现在 `w.grad` 和 `b.grad` 里。如果报 dtype 的错，检查第 2 步。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 三种方法，一个答案

    现在来个大的：第 08 章的网络。下面是你在第 08、09 章写的 `softmax`、`forward` 和 `backward`，用 NumPy 写的，已经完成。我们要算同一个网络在同样 100 张图片上的斜率，一次用你亲手推出的公式，一次用 PyTorch。
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

    用张量写出正向计算。`scores(X, W1, b1, W2, b2)` 就是第 08 章的 `forward`：加权和、弯折、再一次加权和。PyTorch 的弯折是 `torch.relu`。只返回分数。

    和第 08 章相比，这次不用保留 `Z` 和 `H`。当初保留它们，是为了让 `backward` 能用，现在账本替你记着了。
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
    算损失还需要一个新工具。`F.cross_entropy(S, labels)` 就是第 07 章的交叉熵，但多了两点方便：它接收原始的**分数**（softmax 由它自己做），**标签**用整数表示，不用 one-hot 行。`F` 是 PyTorch 里 `torch.nn.functional` 这一部分的通用简称。它已经替你加载好了，`torch`、`nn` 和 `np` 也一样。

    下面的单元格在 100 张图片上做完整的对比。前半段是你亲手写的版本。后半段把同样的数字放进张量（对 NumPy 数组用 `torch.tensor` 会保留 64 位的 dtype，所以两边精度相同），让 PyTorch 算斜率。最后一行显示 50,890 个斜率中最大的差别。

    /// admonition | 🧰 工具箱：两种简写
    单元格里有两种新写法。`[torch.tensor(k, requires_grad=True) for k in (…)]` 对每一项运行一次表达式，造出一个列表：这是带 `append` 的循环的简写。`.numpy()` 把张量变回 NumPy 数组。`max` 是 Python 自带的：取两个数中较大的那个。
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
    这个数结尾大约是 `e-17`。和第 09 章一样，意思是小数点往左移 17 位：0.00000000000000003，只是舍入误差。你在第 08 章一项一项推出的公式，和 PyTorch 账本算出的是同一个东西。这些斜率你还用轻推检查过，所以这是三条互相独立的路，通到了同一个答案：轻推、你的公式、PyTorch。

    ## 把层变成对象

    手写 `W1, b1, W2, b2`，**层**（layer）一多就很累。PyTorch 把一层的旋钮打包成一个对象：
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
    （`torch.Size` 只是 PyTorch 打印形状的方式。）

    `nn.Linear(784, 64)` 造出一个 784 个输入、64 个输出的层，里面装着一个权重矩阵和一个偏置向量。它们已经是随机的，并且带着 `requires_grad=True`。（`torch.manual_seed(0)` 让随机数可以重现，相当于 `default_rng(0)`。）

    有一点和我们的习惯相反：我们的 `W1` 是 `(784, 64)`，每个探测器占一列；PyTorch 存的是 `(64, 784)`，每个探测器占一行。数字是一样的，只是转置了。调用 `layer(X)` 算的是 `X @ layer.weight.T + layer.bias`，也就是我们的 `Z`。把对象当函数来调用，看起来有点怪，其实只是运行它内部计算的简写。

    `nn.Sequential(...)` 把层串起来：每一层的输出是下一层的输入。`nn.ReLU()` 就是弯折，一个没有旋钮的层。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：make_model

    写出 `make_model(hidden)`，返回第 08 章的网络，有 `hidden` 个探测器。它是一个装着三层的 `nn.Sequential`：从 784 个像素到探测器，弯折，再从探测器到 10 个分数。

    检查之后的单元格会向模型要它的旋钮。`model.parameters()` 把它们一个个给出来，`p.numel()` 是其中一个里面有多少个数。
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
    PyTorch 逐层打印出模型（`bias=True` 只是说这一层有偏置），后面是我们的计数：50,890 个旋钮：$784 \times 64$ 个权重，64 个偏置，$64 \times 10$ 个权重，10 个偏置。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：探测器减半

    `make_model(32)` 会有多少个旋钮？先算第一层（权重和偏置），再算整个模型。
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

    **优化器**（optimizer）是一个对象，它拿着所有旋钮，并执行一条步进规则。下面把你第 09 章的 `adam_step` 和 `torch.optim.Adam` 放在一起，解决米娅的问题：原始气温，同样从 `[6, 0]` 出发，学习率 1.0，500 步。
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
    旋钮完全相同，显示出来的每一位都一样。（前面要加 `.detach()`，因为 PyTorch 不肯把正在记账的张量变成 NumPy 数组。它给出同样的数字，但不带账本。）PyTorch 的 Adam 用的也是你敲过的 0.9、0.999 和 `1e-8`。

    这个循环里的三个调用，是所有 PyTorch 训练循环的核心：

    - `optimizer.zero_grad()` 清掉所有旋钮上的旧责任（这就是前面说的那个工具）；
    - `loss.backward()` 填好每个旋钮的 `.grad`；
    - `optimizer.step()` 根据 `.grad` 移动每个旋钮。

    再加上夹在前两步之间的正向计算，就是完整的配方：清空、正向、反向、走一步。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 找出错误

    下面是有人匆忙写出的一步训练：

    ```
    def hasty_step(model, optimizer, X, labels):
        loss = F.cross_entropy(model(X), labels)
        loss.backward()
        optimizer.step()
        return loss
    ```

    它运行起来不报错，第一步甚至是对的。拿它和那套步骤比一比：清零（1）、正向（2）、反向（3）、走一步（4）。四样里少了哪一样？
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
    从第二步开始，每个旋钮的 `.grad` 里装的是新的责任**加上**前面几步剩下的全部，就像上面翻倍成 952 那样。旋钮在动，但方向不对，也走得太远。

    ## 练习 4：train_step

    写出 `train_step(model, optimizer, X, labels)`：在一批数据上正确地训练一步，整套步骤按顺序做完。损失是模型的分数和标签的 `F.cross_entropy`。
    把损失作为普通的数返回。
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
        r"""六行短短的代码：清零、分数、损失、反向、走一步，还有 `return`。""",
        r"""模型像函数一样调用：`model(X)`。`.item()` 把只有一个数的损失张量变成普通的数。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 完整的一次训练

    下面是第 09 章的训练函数，用刚才造好的东西重写。`torch.randperm` 就是 PyTorch 版的 `rng.permutation`。按轮次和批次的循环和你写的一样，其余的都进了 `train_step`。
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
    `accuracy` 用在张量上要小改一下：`.float()` 把 `True`/`False` 变成 1 和 0，因为 `torch.mean` 只能对带小数点的数求平均。

    下面的单元格把数据变成 32 位张量，用 Adam 以学习率 0.002 训练 10 轮，然后在测试图片上打分。有两点是新的：

    - `torch.no_grad()`：写在 `with torch.no_grad():` 下面、缩进的那些行运行时不记账。打分不需要责任，这样能省时间和内存。
    - 第二个数字是用掉的秒数，和第 09 章一样。
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
    测试图片上大约 95%，最多几秒钟，而你只写了寥寥几行。这和第 09 章的网络是同样的设计，用的也是同样的图片，只是记账和步进规则由 PyTorch 来做。

    ## GPU 是什么？

    到现在为止，所有数字都是电脑的主处理器 **CPU** 算出来的。CPU 里有几个能力很强的工人。**GPU**（图形处理器，本来是为同时给屏幕上几百万个点上色而造的）里有成千上万个简单的工人，这正适合我们的工作：矩阵乘法就是成千上万个相同的小求和。对大模型来说，GPU 可能快 10 到 100 倍。

    这门课安装的是小巧的纯 CPU 版 PyTorch，这样每一章都能在任何笔记本上运行。课程里没有哪一章必须用 GPU；能从 GPU 受益的章节会特别说明。下面的单元格问问 PyTorch 在这里看到了什么：
    """)
    return


@app.cell
def _():
    torch.cuda.is_available()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    显示 `False` 是正常的（在 Apple 芯片的 Mac 上，PyTorch 有类似的选项）。如果你真有 GPU，要改的地方很少：张量和模型都放在某个**设备**（device）上，一次计算涉及的所有东西必须在同一个设备上。`model.to("cuda")` 和 `X.to("cuda")` 把它们搬过去（`cuda` 是常见的 GPU 系统的名字），其余代码不用动。

    ## 接下来

    你的网络把一张图片看成 784 个互不相干的数字：数字稍微挪一下，对网络来说就是另一张图片了。在**第 11 章**，你会用一个专为图片设计的想法来解决这个问题：**卷积**（convolution），也就是在图片上滑动的小探测器。记账交给 PyTorch，你就能专心思考设计了。

    ## 你现在会做的事

    - 创建张量，选择 dtype，看懂 `must have the same dtype` 这类报错
    - 使用 `requires_grad`、`backward()` 和 `.grad`，并用你的引擎的说法解释它们各做什么
    - 解释为什么每一步之前都要清掉责任
    - 用 `nn.Linear`、`nn.ReLU` 和 `nn.Sequential` 搭建网络
    - 用优化器训练：清空、正向、反向、走一步
    - 用 `torch.no_grad()` 打分
    - 说出 GPU 带来了什么，设备是什么

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** PyTorch 的文档在 https://pytorch.org/docs/。`backward()` 背后的引擎叫 *autograd*，下一章你会见到的 `nn.Module` 是所有层的基础。`nn.Linear` 有自己的旋钮起始规则，思路和第 09 章的 He 初始化相近。
    """)
    return


if __name__ == "__main__":
    app.run()
