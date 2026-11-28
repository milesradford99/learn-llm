import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="11 · 用卷积看世界")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.images import digits as show_digits
    from learn_llm.images import maps

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 11 · 用卷积看世界

    到现在为止，你的网络都把图片当成一长串互不相干的数字：第 300 个像素有自己的旋钮，却没有任何机制告诉网络，第 301 个像素就紧挨在它旁边。这招之前行得通，是因为数据里的手写数字全都规规矩矩地摆在正中间。真实照片可不是这样——一只猫不管蹲在画面的左上角还是右下角，都是猫。

    这一章，你会先亲眼看到这个弱点，然后用一个想法修好它：让一个小探测器滑过整张图片。接着再学第二个想法——**残差（residual）**，它让网络能叠得很深。正是这两个想法，把计算机视觉从玩具问题带到了能认出几千种物体的实用阶段。

    下面准备好了第 10 章的工具，直接拿来用：
    """)
    return


@app.function
def train_step(model, optimizer, X, labels):
    optimizer.zero_grad()
    S = model(X)
    loss = F.cross_entropy(S, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


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


@app.function
def count_knobs(model):
    total = 0
    for p in model.parameters():
        total = total + p.numel()
    return total


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 会移动的数字

    为了让问题显露出来，我们把数字打散：把每张 $28 \times 28$ 的图片，贴到一张 $40 \times 40$ 空白画布的随机位置上：
    """)
    return


@app.function
def shift_digits(images, size, seed):
    rng = np.random.default_rng(seed)
    canvas = np.zeros((len(images), size, size))
    for i in range(len(images)):
        r = rng.integers(0, size - 27)
        c = rng.integers(0, size - 27)
        canvas[i, r:r + 28, c:c + 28] = images[i] / 255
    return canvas


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：随机整数与二维切片
    - `rng.integers(0, n)` 生成一个从 0 到 $n$（不含 $n$）的随机整数。
    - 切片可以同时在两个方向上切：`M[1:3, 0:2]` 取出 `M` 里第 1、2 行、第 0、1 列的 $2 \times 2$ 方块。上面单元格里的 `canvas[i, r:r + 28, c:c + 28]`，就是第 $i$ 张画布里左上角位于第 $r$ 行、第 $c$ 列的 $28 \times 28$ 区域；往里面赋值，就相当于把数字贴了上去。
    ///
    """)
    return


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    X_np = shift_digits(train_images, 40, 0)
    X_test_np = shift_digits(test_images, 40, 1)
    X = torch.tensor(X_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    labels = torch.tensor(train_labels, dtype=torch.long)
    labels_test = torch.tensor(test_labels, dtype=torch.long)
    show_digits(X_np[:6] * 255, train_labels[:6])
    return X, X_test, labels, labels_test, test_images, train_images


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在用第 10 章那样的网络来试试，为了公平，我们把规模加大到 1,600 个输入、256 个探测器。`nn.Flatten()` 这一层会把每张 $40 \times 40$ 的图片拉平为一行 1,600 个数字，作用和 `reshape` 一样。下面会输出三个数：旋钮总数、训练集准确率、测试集准确率。
    """)
    return


@app.cell
def _(X, X_test, labels, labels_test):
    torch.manual_seed(0)
    flat_model = nn.Sequential(nn.Flatten(), nn.Linear(1600, 256), nn.ReLU(), nn.Linear(256, 10))
    train(flat_model, X, labels, 10, 100, 0.002)
    with torch.no_grad():
        flat_train = accuracy(flat_model(X), labels)
        flat_test = accuracy(flat_model(X_test), labels_test)
    count_knobs(flat_model), flat_train, flat_test
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    超过 412,000 个旋钮，训练集准确率将近 100%，遇到新图片却掉到约 88%。这个落差就是第 09 章见过的**过拟合（overfitting）**，现在一眼就能看出原因：每个旋钮只管画布上的一个固定位置。网络在左上角学会了一道笔画，对右下角同样的一道笔画毫无帮助——同一个形状换个位置就得从头再学一遍。旋钮又多得用不完，它干脆把训练图片死记硬背了下来。

    ## 会滑动的探测器

    更好的办法是这样的：把探测器做小（比如 $3 \times 3$ 个旋钮），不再给图片的每个区域各配一套探测器，而是让同一个探测器滑过整张图片。每停在一个位置，就取它下面盖住的 $3 \times 3$ 小块做一次点积——对应位置相乘，再全部加起来。这个小小的旋钮方格叫**卷积核（kernel）**，滑动它的操作叫**卷积（convolution）**。

    我们手算一遍。下面是一张 $4 \times 4$ 的图片，数字在中间从 0 跳到 1，右边是一个专门比较左右两边的卷积核：

    ```
    图片              卷积核
    0  0  1  1       -1  0  1
    0  0  1  1       -1  0  1
    0  0  1  1       -1  0  1
    0  0  1  1
    ```

    把卷积核放在左上角的小块上（第 0–2 行，第 0–2 列）：

    3 行里每一行都是 $(-1 \times 0) + (0 \times 0) + (1 \times 1) = 1$，加起来 $1 + 1 + 1 = \boldsymbol{3}$。

    向右滑一步（第 1–3 列）：每一行都是 $(-1 \times 0) + (0 \times 1) + (1 \times 1) = 1$，还是 **3**。

    全是 0 或全是 1 的小块都会算出 0，因为 $-1$ 和 $+1$ 刚好抵消。所以结果只在数字从左到右发生跳变的地方才大：这个卷积核就是一个**竖直边缘探测器**。

    $3 \times 3$ 的卷积核放在 $4 \times 4$ 的图片上，横着能停 2 个位置，竖着也能停 2 个位置，所以输出是 $2 \times 2$。为什么是 2？把宽 4、核宽 3 拿出来数一下：卷积核的左边缘只能停在位置 0（盖住第 0–2 列）或位置 1（盖住第 1–3 列），再往右就出界了；从 0 数到 $4 - 3 = 1$，一共 $1 + 1 = 2$ 个位置。推广到一般情况：高度为 $h$ 的图片配边长为 $k$ 的卷积核，输出就有 $h - k + 1$ 行（同理，宽度为 $w$ 时输出 $w - k + 1$ 列）。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：滑动卷积核

    轮到你了。卷积核不变，换一张图片：

    ```
    图片             卷积核
    0  0  0  1       -1  0  1
    0  0  1  1       -1  0  1
    0  1  1  1       -1  0  1
    1  1  1  1
    ```

    在一小块的每一行里，这个卷积核算出的是右边的数减去左边的数。算出输出的四个数。
    再加一问：$28 \times 28$ 的图片配 $5 \times 5$ 的卷积核，输出有多少行？
    """)
    return


@app.cell(hide_code=True)
def _():
    slide_boxes = hand.boxes(['左上', '右上', '左下', '右下', '28 配 5 时的行数'])
    return (slide_boxes,)


@app.cell(hide_code=True)
def _(slide_boxes):
    hand.marked(slide_boxes, {'左上': 2, '右上': 2, '左下': 2, '右下': 1, '28 配 5 时的行数': 24}, {
        '左上': r"""第 0 到 2 行、第 0 到 2 列：三行分别得到 0、1、1""",
        '右上': r"""第 0 到 2 行、第 1 到 3 列：三行分别得到 1、1、0""",
        '左下': r"""第 1 到 3 行、第 0 到 2 列：三行分别得到 1、1、0""",
        '右下': r"""第 1 到 3 行、第 1 到 3 列：三行分别得到 1、0、0""",
        '28 配 5 时的行数': r"""$h - k + 1$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：convolve

    写出 `convolve(image, kernel)`，其中 `image` 是一个二维矩阵，`kernel` 是一个方形矩阵：

    1. 先取出尺寸：`image.shape` 给出高和宽，`kernel.shape[0]` 是卷积核的边长 $k$。
    2. 创建一个全 0 的输出矩阵，大小为 $h - k + 1$ 行、$w - k + 1$ 列。
    3. 用两层嵌套循环走遍输出的每个位置：`r` 遍历行，`c` 遍历列。
    4. 在每个位置取出对应小块——图片里左上角在 `(r, c)` 的 $k \times k$ 方块（和上面 `canvas[i, r:r + 28, c:c + 28]` 一样的二维切片）。把这一小块和卷积核逐个位置相乘，用 `np.sum` 全部加起来，存进 `out[r, c]`。
    5. 返回输出矩阵。

    在上面手算过的例子上，它应该得到 `[[3, 3], [3, 3]]`。
    """)
    return


@app.function
def convolve(image, kernel):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "convolve", convolve, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`h, w = image.shape` 给两个大小起了名字。循环是 `for r in range(...)`，里面再套 `for c in range(...)`，要做的事写在两层循环的里面。""",
        r"""那一小块是 `image[r:r + k, c:c + k]`，行的部分和列的部分之间用逗号隔开。`return` 在两层循环之后。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在把它用到真实的手写数字上：左边是原图，中间是竖直边缘探测器，右边把同一个卷积核转置（`.T`）来找水平边缘。蓝色表示正响应，红色表示负响应：
    """)
    return


@app.cell
def _(train_images):
    digit = train_images[0] / 255
    edge_v = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    edge_h = edge_v.T
    response_v = convolve(digit, edge_v)
    response_h = convolve(digit, edge_h)
    maps([digit, response_v, response_h], ["picture", "vertical edges", "horizontal edges"], 140)
    return digit, edge_v, response_v


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    卷积的输出本身也像一张图，标出了某种形状出现在哪些位置，所以叫**特征图（feature map）**。它有两个非常讨喜的性质。

    /// admonition | 🧰 工具箱：平移一个数组
    `np.roll(a, 1)` 把数组里的每一项往后挪一位，最后一项绕回最前面：`np.roll(np.array([1, 2, 3]), 1)` 得到 `[3, 1, 2]`。对二维矩阵加上 `axis=1`，就是沿着列的方向向右平移。
    ///

    **它跟着图片一起动。** 用 `np.roll` 把数字向右挪 3 列，特征图也跟着挪 3 列。“先平移再卷积”和“先卷积再平移”的最大差别正好是 0：
    """)
    return


@app.cell
def _(digit, edge_v, response_v):
    moved = np.roll(digit, 3, axis=1)
    np.max(np.abs(convolve(moved, edge_v) - np.roll(response_v, 3, axis=1)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **它只需要极少的旋钮。** 不管图片多大，一个 $3 \times 3$ 卷积核都只有 9 个旋钮，因为同一组 9 个旋钮会在每个位置反复复用。在一个角落学到的规律，换到任何角落都同样管用。

    ## 缩小：最大池化

    一个数字不只是零散的边缘，而是按特定方式拼在一起的边缘。为了看清更大的图案，我们先把特征图缩小，再做下一轮卷积。**最大池化（max pooling）** 把特征图切成不重叠的 $2 \times 2$ 方块，每块只保留最大的那个数。它相当于说“这个特征大概在这一带出现过”，故意忘掉精确像素位置，同时把高和宽减半。这样下一层的 $3 \times 3$ 卷积核，在原图上看到的范围就宽了一倍。

    ```
    1  3  0  2
    5  2  1  0      →     5  2
    0  0  4  1            1  6
    1  0  6  2
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：给一张图做池化

    给下面这张图做池化：写出每个 $2 \times 2$ 方块里最大的数。

    ```
    2  0  1  1
    0  3  0  1
    7  1  0  0
    2  2  0  5
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    pool_boxes = hand.boxes(['左上', '右上', '左下', '右下'])
    return (pool_boxes,)


@app.cell(hide_code=True)
def _(pool_boxes):
    hand.marked(pool_boxes, {'左上': 3, '右上': 1, '左下': 7, '右下': 5}, {
        '左上': r"""2、0、0、3 里最大的""",
        '右上': r"""1、1、0、1 里最大的""",
        '左下': r"""7、1、2、2 里最大的""",
        '右下': r"""0、0、0、5 里最大的""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：整除（丢掉小数部分）
    `//` 是直接舍去小数部分的整除：`7 // 2` 得 3，`8 // 2` 得 4。
    ///

    ## 练习 2：max_pool

    写出 `max_pool(M)`，把矩阵 `M` 的高和宽各缩小一半。输出的行数和列数用 `//` 来算（如果原图行数是奇数，多出来的最后一行直接舍去）。

    沿用 `convolve` 的骨架：先建好全 0 的输出矩阵，再用两层循环走遍输出的每个位置。输出位置 `(r, c)` 对应的方块，从 `M` 的第 $2 \times r$ 行、第 $2 \times c$ 列开始，高 2 行、宽 2 列，用 `np.max` 取出其中最大的数填进去。

    对上面正文里的例子，它应该得到 `[[5, 2], [1, 6]]`；对你刚才手算的例子，应该得到你算出的四个数。
    """)
    return


@app.function
def max_pool(M):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "max_pool", max_pool, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""输出有 `h // 2` 行、`w // 2` 列，循环也按这两个数来走。""",
        r"""方块是这样的切片：行从 `2 * r` 到 `2 * r + 2`，列同样用 `c` 来写。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 让网络自己学卷积核

    刚才的边缘探测器是我们亲手设计的。在神经网络里，卷积核里的 9 个数字就是旋钮，交给一贯的反向传播和 Adam 自己去学。

    /// admonition | 🧰 工具箱：PyTorch 现成的卷积层
    在 PyTorch 里，带可学习卷积核的卷积层写成 `nn.Conv2d(1, 1, 3)`，读作“输入 1 个通道，输出 1 个通道，卷积核大小 3”。送进来的图片必须是 4 维形状：**`(图片数, 通道数, 高, 宽)`**。**通道（channel）**就是每张图片自带的“特征图层数”：灰度数字只有 1 层明暗信息，所以是 1 个通道；第一层卷积如果算出 8 张特征图，对下一层来说就是 8 个通道。`X.unsqueeze(1)` 在第 1 号维度插入一个长度为 1 的新维度，于是 `(10000, 40, 40)` 变成 `(10000, 1, 40, 40)`。
    ///

    为了确认 PyTorch 算出来的和我们刚才手写的一模一样，我们把你的卷积核塞进它的层里对比一下（`.data` 可以直接读写张量里的纯数字，不记入自动求导账本）：
    """)
    return


@app.cell
def _(digit, edge_v, response_v):
    conv = nn.Conv2d(1, 1, 3, bias=False)
    conv.weight.data = torch.tensor(edge_v).reshape(1, 1, 3, 3)
    digit_4d = torch.tensor(digit).reshape(1, 1, 28, 28)
    with torch.no_grad():
        response_t = conv(digit_4d)
    response_t.shape, np.max(np.abs(response_t.numpy()[0, 0] - response_v))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    输出形状的最后两位是 $26 \times 26$，刚好符合公式 $28 - 3 + 1$。第二个数是它和你手写结果的最大差别：0（最多只有微小的浮点舍入误差）。再补上两块拼图，就能搭起完整的卷积网络了。

    **多通道与多卷积核。** 一层卷积可以同时学很多个卷积核，每个卷积核输出 1 个通道（也就是 1 张特征图）。比如 `nn.Conv2d(1, 8, 3)` 有 8 个卷积核，把 1 个输入通道变成 8 个输出通道。

    那下一层 `nn.Conv2d(8, 16, 3)` 接过这 8 个通道后，到底是怎么算的？把这 8 个通道想象成 8 张对齐叠在一起的透明胶片。此时，**第 1 个卷积核**不再是单薄的一片 $3 \times 3$，而是长成 **8 片 $3 \times 3$**（共 $8 \times 3 \times 3 = 72$ 个权重参数）。滑动到某个位置时，它把这 8 片分别盖在 8 个输入通道的同一处 $3 \times 3$ 小块上，各自做点积，再**全部加成一个数**（外加 1 个偏置），填进第 1 张输出特征图的这 1 个格子里。这一层共有 **16 个**这样的 8 片卷积核，各自算出 1 张特征图，所以最终输出 **16 个通道**！后面的卷积核正是靠这种“跨通道汇总”，才能识别出“竖直边缘紧挨着水平边缘”这样的组合图案。

    | `nn.Conv2d(8, 16, 3)` 里的部件 | 形状 | 含义 |
    |---|---|---|
    | 输入（单张图片） | `(8, H, W)` | 上一层输出的 8 张特征图（8 个通道） |
    | 第 1 个卷积核 | `(8, 3, 3)` + 1 个偏置 | 8 片 $3 \times 3$ 各自点积后全部加成 1 个数（共 $72 + 1 = 73$ 个旋钮） |
    | 全部 16 个卷积核（`weight`） | `(16, 8, 3, 3)` + 16 个偏置 | 16 个独立卷积核，总计 $16 \times 73 = 1{,}168$ 个旋钮 |
    | 输出（单张图片） | `(16, H', W')` | 16 张新特征图（16 个通道），每个卷积核贡献 1 张 |

    **保持尺寸不变。** `padding=1` 会在每个通道四周先补一圈宽 1 像素的 0（这叫**填充**，padding），这样 $3 \times 3$ 的卷积核滑完后高和宽与输入完全一致，只有池化才会把图缩小。

    网络末尾用 `nn.AdaptiveMaxPool2d(1)`：不管每个通道还剩多大，只保留整张图里最大的那一个数。它回答的问题是“图片里有没有出现过这个特征（不管在哪）？”。经过它之后，32 个通道就浓缩成了 32 个数字（再用 `nn.Flatten()` 整理形状），最后由一个 `nn.Linear` 算出 10 个类别分数。因为精确位置信息几乎被抹平，数字不管挪到哪里，得分都差不多。（下面要用到的 `nn.MaxPool2d(2)`，就是你刚才亲手写过的 $2 \times 2$ 最大池化。）
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一段一段地看这个网络

    下面是你要搭的网络，用来处理 $40 \times 40$ 的图片：

    | 阶段 | 做什么 |
    |---|---|
    | 1 | 卷积（1 个通道 → 8 个通道），弯折，$2 \times 2$ 最大池化 |
    | 2 | 卷积（8 个通道 → 16 个通道），弯折，$2 \times 2$ 最大池化 |
    | 3 | 卷积（16 个通道 → 32 个通道），弯折 |
    | 4 | 每个通道只留最大的那个数，展平，线性层（32 个数 → 10 个分数） |

    每次卷积都用 $3 \times 3$ 的卷积核并设 `padding=1`，所以高和宽保持不变。

    ## 动手算：跟着一张图片走

    一张图片进来时形状是 `(1, 40, 40)`：1 个通道，高 40，宽 40。算算经过阶段 1 和阶段 2 后变成多大，再仿照上面表格算出中间层 1,168 个旋钮的方法，数出第一次卷积和最后一次卷积各有几个旋钮。
    """)
    return


@app.cell(hide_code=True)
def _():
    stage_boxes = hand.boxes(['阶段 1：通道数', '阶段 1：高', '阶段 2：高', '第一次卷积的旋钮数', '最后一次卷积的旋钮数'])
    return (stage_boxes,)


@app.cell(hide_code=True)
def _(stage_boxes):
    hand.marked(stage_boxes, {'阶段 1：通道数': 8, '阶段 1：高': 20, '阶段 2：高': 10, '第一次卷积的旋钮数': 80, '最后一次卷积的旋钮数': 4640}, {
        '阶段 1：通道数': r"""阶段 1 的卷积输出 8 个通道，池化不会改变通道数""",
        '阶段 1：高': r"""`padding=1` 让高保持 40，池化再把它减半""",
        '阶段 2：高': r"""再减半一次""",
        '第一次卷积的旋钮数': r"""8 个卷积核，每个有 $3 \times 3 \times 1$ 个权重和 1 个偏置""",
        '最后一次卷积的旋钮数': r"""32 个卷积核，每个有 $3 \times 3 \times 16$ 个权重和 1 个偏置""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：make_cnn

    写出 `make_cnn()`，把表里的网络装进一个 `nn.Sequential` 并返回。它一共 11 层，每一层在上面的正文里都出现过：
    `nn.Conv2d`（输入通道数、输出通道数、卷积核大小，再加 `padding=1`）、`nn.ReLU()`、`nn.MaxPool2d(2)`、`nn.AdaptiveMaxPool2d(1)`、`nn.Flatten()`、`nn.Linear`。

    建议每个阶段写一行，每一层后面加一个逗号。
    """)
    return


@app.function
def make_cnn():
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "make_cnn", make_cnn, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""阶段 1 是 `nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),`。阶段 2、3 是同样的写法，只是通道数不同，而且阶段 3 没有池化。""",
        r"""每次卷积的输入通道数必须等于上一次的输出通道数：1 到 8，8 到 16，16 到 32。最后三层是 `nn.AdaptiveMaxPool2d(1)`、`nn.Flatten()` 和 `nn.Linear(32, 10)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    卷积比 `@` 慢，所以训练要等你点按钮（10 轮；笔记本上大约需要一分钟）。和前面一样，输出分别是旋钮总数、训练集准确率和测试集准确率。在你点按钮之前，依赖它结果的单元格会显示 `Ancestor stopped`，这是正常的。
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="训练卷积网络")
    train_button
    return (train_button,)


@app.cell
def _(X, X_test, labels, labels_test, train_button):
    mo.stop(not train_button.value, mo.md("*按下按钮开始训练。*"))
    torch.manual_seed(0)
    cnn = make_cnn()
    train(cnn, X.unsqueeze(1), labels, 10, 100, 0.003)
    with torch.no_grad():
        cnn_train = accuracy(cnn(X.unsqueeze(1)), labels)
        cnn_test = accuracy(cnn(X_test.unsqueeze(1)), labels_test)
    count_knobs(cnn), cnn_train, cnn_test
    return (cnn,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    训练完你应该会看到：总共只要 6,218 个旋钮，比扁平网络少了 66 倍，新图片上的准确率却从约 88% 飙升到了约 96%。训练和测试分数现在咬得很紧：几乎没有死记硬背，因为每个旋钮在整张图上反复复用，必须到处都管用才行。

    训练完成后，下面会画出第一层学到的 8 个卷积核（`cnn[0].weight` 的形状是 `(8, 1, 3, 3)`，代表 8 个输出通道、1 个输入通道、$3 \times 3$ 大小；`[:, 0]` 就是在取每个输出通道对应的第 0 个输入通道的 $3 \times 3$ 卷积核）。没人教过它什么是边缘，但其中好几个都自发学会了一边正、一边负的对比，和你刚才手写的一模一样：
    """)
    return


@app.cell
def _(cnn):
    kernels = cnn[0].weight.detach().numpy()[:, 0]
    maps(kernels, range(8), 70)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 走向深层：为责任修一条直达路

    既然三层这么好用，三十层会不会更强？更深的网络本该能看懂更复杂的图案。但很多年里，深网络反而训得比浅网络差，一个关键原因就是第 03 章的链式法则：责任要传回靠前的层，必须一路乘过后面每一层的斜率。如果这些乘法总体上在把数越乘越小，责任就会一路衰减直到消失——正是你在第 09 章表格里见过的现象。

    解决办法简单得让人意外。普通模块是直接用新算出的结果顶替输入：

    > `out = f(x)`

    现在改成把结果**加回**输入上：

    > `out = x + f(x)`

    这样一来，`f` 只需要学对 `x` 的*修正量*，这个修正量就叫**残差（residual）**。来看看斜率：根据第 02 章的加法求导规则，$x + f(x)$ 的斜率是 $1 + f'$（这里的 $f'$ 是数学简写，代表 $f(x)$ 的斜率）。哪怕 $f'$ 小到接近 0，整体斜率依然稳稳地停在 1 附近，责任就能几乎无损地穿过整个模块。这个 `x +` 给责任修了一条直达高速路，常被称为**跳跃连接（skip connection）**。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：走一遍残差块

    在你要写的残差块里，`f` 由第一层、一次弯折和第二层组成；算完后再把 `x` 加回来，最后对总和再做一次弯折：

    > $$\text{out} = \operatorname{relu}(x + \text{layer}_2(\operatorname{relu}(\text{layer}_1(x))))$$

    用单个数字和两个假想的层来试试：第一层做减 5，第二层做乘 3。$x = 6$ 时输出是多少？$x = 2$ 时呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    residual_boxes = hand.boxes(['x = 6 时的输出', 'x = 2 时的输出'])
    return (residual_boxes,)


@app.cell(hide_code=True)
def _(residual_boxes):
    hand.marked(residual_boxes, {'x = 6 时的输出': 9, 'x = 2 时的输出': 2}, {
        'x = 6 时的输出': r"""$6 - 5 = 1$，弯折之后还是 1，乘 3 得 3，而 $6 + 3 = 9$ 是正数""",
        'x = 2 时的输出': r"""$2 - 5 = -3$，弯折把它变成 0。那还剩什么可以加到 $x$ 上？""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $x = 2$ 时，`f` 算出 0，残差块就把 `x` 原封不动地传了下去。也就是说，一个还什么都没学会的残差块至少不会添乱，普通模块可做不到这一点。

    ## 练习 4：residual_block

    写出 `residual_block(x, conv1, conv2)`，实现上面的公式。`conv1` 和 `conv2` 是两个保持形状不变的卷积层，直接像函数一样调用即可，弯折用 `torch.relu`。
    """)
    return


@app.function
def residual_block(x, conv1, conv2):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "residual_block", residual_block, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三行：第一层和它的弯折，第二层，然后是求和以及最后的弯折。""",
        r"""`x` 要在最后一次弯折之前加上去，不是之后。加的是原来的 `x`，不是已经过了某一层的东西。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    要把这样的模块放进 `nn.Sequential`，它本身得是一个 PyTorch 网络层。自定义网络层用的正是第 04 章学过的 `class`，只多加了一小步：
    """)
    return


@app.function
def plain_block(x, conv1, conv2):
    h = torch.relu(conv1(x))
    h = conv2(h)
    return torch.relu(h)


@app.cell
def _():
    class Block(nn.Module):
        def __init__(self, channels, step):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.step = step

        def forward(self, x):
            return self.step(x, self.conv1, self.conv2)

    def deep_model(blocks, step):
        layers = [nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        for i in range(blocks):
            layers.append(Block(8, step))
        return nn.Sequential(*layers, nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(8, 10))
    return (deep_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：自己写一个网络层
    - 顺手回顾第 04 章的 `class`：`__init__` 在创建对象时自动运行，`self.xxx` 把东西挂在这个对象自己身上。
    - `class Block(nn.Module):` 表示 `Block` 是基于 `nn.Module`（PyTorch 里所有网络层的统一基类）扩展出来的新层。正因为套上了这层身份，PyTorch 才能自动找到 `Block` 里面藏着的所有旋钮。
    - `super().__init__()` 用来先完成 `nn.Module` 自身的初始化，总是写在 `__init__` 的第一行。
    - `forward` 定义这个层实际怎么算。写 `block(x)` 时，PyTorch 会自动调用 `block.forward(x)`——第 10 章里 `layer(X)` 背后也是这个机制。
    ///

    在这个 `Block` 里，`step` 是存成属性的一个函数（第 02 章讲过：函数本身也可以当值传递），所以同一个类既能跑你的 `residual_block`，也能跑不带 `x +` 的 `plain_block`。

    `deep_model(blocks, step)` 会把 `blocks` 个这样的模块首尾相连地叠起来（每个模块含两层卷积）。为了公平对比，我们用完全相同的初始权重搭出两个各含 10 个模块的深层网络，喂入同样的图片，只跑一次反向传播，看看有多少责任传回了最前面第一层（用它权重的平均斜率大小来衡量）：
    """)
    return


@app.function
def first_layer_blame(model, X, labels):
    model.zero_grad()
    F.cross_entropy(model(X), labels).backward()
    return torch.mean(torch.abs(model[0].weight.grad)).item()


@app.cell
def _(deep_model, labels, test_images, train_images):
    X_small = torch.tensor(train_images[:3000] / 255, dtype=torch.float32).unsqueeze(1)
    X_centred_test = torch.tensor(test_images / 255, dtype=torch.float32).unsqueeze(1)
    labels_small = labels[:3000]
    torch.manual_seed(0)
    plain_net = deep_model(10, plain_block)
    torch.manual_seed(0)
    skip_net = deep_model(10, residual_block)
    first_layer_blame(plain_net, X_small[:100], labels_small[:100]), first_layer_blame(skip_net, X_small[:100], labels_small[:100])
    return X_centred_test, X_small, labels_small, plain_net, skip_net


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    左边的数是普通堆叠，右边是带跳跃连接的网络。大约 0.000000000001 对大约 0.015：有了跳跃连接，传回第一层的责任多了一百亿倍以上。在普通堆叠里，第一层几乎收不到任何信号。

    下面的按钮会用 3,000 张居中数字把两个网络各训练 4 轮（大约半分钟），并显示各自的测试集准确率：先是普通堆叠，再是带跳跃连接的。后面的文字说明了你应该看到的现象。
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button_2 = mo.ui.run_button(label="训练两个深层网络")
    train_button_2
    return (train_button_2,)


@app.cell
def _(
    X_centred_test,
    X_small,
    labels_small,
    labels_test,
    plain_net,
    skip_net,
    train_button_2,
):
    mo.stop(not train_button_2.value, mo.md("*按下按钮开始训练。*"))
    train(plain_net, X_small, labels_small, 4, 50, 0.003)
    train(skip_net, X_small, labels_small, 4, 50, 0.003)
    with torch.no_grad():
        plain_test = accuracy(plain_net(X_centred_test), labels_test)
        skip_test = accuracy(skip_net(X_centred_test), labels_test)
    plain_test, skip_test
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 10% 对大约 80%（第二个数字每次运行会略有波动）。普通堆叠几乎没有责任传到第一层，只能瞎猜——10 个数字随便蒙一个，准确率正好就是 10%。完全相同的网络层，仅仅在每个模块里加上一个 `x +`，就立刻学会了。

    把这样的残差模块一路叠起来（中间通常会再多做几次缩小），就是大名鼎鼎的 **ResNet**。正是这个设计，让上百层的深度网络第一次变得可以训练。同样的技巧——把输入直接加回输出——后面还会出现在 Transformer 的每一个模块里，背后的道理完全一样。

    ## 接下来

    第一部分到此结束。你从拟合一条直线起步，现在已经亲手造出了分类器、自己的自动求导引擎和能“看图”的卷积网络。后面的一切都会建立在这些基础之上。

    第二部分转向**语言**。第一个问题是：网络只认数字，不认词语。在**第 12 章**，你会给每个词配一个向量，让意思相近的词在空间里自然靠在一起。

    ## 你现在会做的事

    - 解释为什么扁平网络对付不了位置会移动的物体
    - 手算和用代码实现卷积，并推算输出尺寸
    - 解释卷积核、特征图、通道和最大池化的作用
    - 算出一个多通道卷积层的旋钮总数
    - 用 `nn.Conv2d` 搭建并训练卷积网络
    - 解释责任为什么在深层网络里衰减，以及 `x + f(x)` 怎样解决这个问题
    - 基于 `nn.Module` 写一个类，做出自己的网络层

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 数学家会把我们算的操作叫*互相关（cross-correlation）*；严格意义上的卷积要先翻转卷积核，但卷积核本来就是训练学出来的，所以这点差别无关紧要。真实的 ResNet 还会配合*批归一化（batch normalisation）*，而彩色图片一开始有 3 个通道（红、绿、蓝）。常用来练手这类任务的标准数据集是 *CIFAR-10* 和 *ImageNet*，原始论文是 *Deep Residual Learning for Image Recognition*（He 等，2015）：这位 He 正是第 09 章初始化规则里的那位 He。
    """)
    return


if __name__ == "__main__":
    app.run()
