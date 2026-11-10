import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="11 · 用卷积看世界")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import laptop_speed, todo
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

    到现在为止，你的网络都把图片当成一长串互不相干的数字。第 300 个像素有自己的旋钮，却没有什么告诉网络第 301 个像素就挨在它旁边。这之所以行得通，是因为我们数据里的数字都规规矩矩地摆在正中间。真实的图片可不是这样：一只猫不管坐在照片哪个位置，都是猫。

    这一章你会先看到这个弱点，然后用一个想法修好它：一个在整张图片上滑动的小探测器。接着再学第二个想法，**残差**（residual），它让网络可以变得非常深。正是这两个想法，把计算机视觉从玩具问题带到了能认出几千种物体。

    下面是第 10 章的工具，可以直接用：
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

    为了让问题看得见，我们把数字打散。每张 $28 \times 28$ 的图片都被贴到一张 $40 \times 40$ 空白画布的随机位置上：
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
    两个新工具。`rng.integers(0, n)` 给出一个从 0 到 $n$（不含 $n$）的随机整数。切片也可以同时在两个方向上切：`canvas[i, r:r + 28, c:c + 28]` 是第 $i$ 张图片里左上角在第 $r$ 行、第 $c$ 列的那个 $28 \times 28$ 方块，往里面存值，就是把数字贴上去。
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
    现在用第 10 章那样的网络来训练，为了公平一点把它加大：1,600 个输入，256 个探测器。`nn.Flatten()` 这一层把每张 $40 \times 40$ 的图片变成一行 1,600 个数字，作用和 `reshape` 一样。单元格显示三个数字：旋钮数，训练图片上的准确率，测试图片上的准确率。
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
    超过 412,000 个旋钮，训练图片上大约 100%，新图片上大约 88%。这个差距就是第 09 章的**过拟合**（overfitting），现在我们能看出它从哪里来了。一个旋钮只属于画布上的一个位置。所以在左上角学到的一笔，对右下角同样的一笔毫无帮助：同一个形状要在每个位置重新学一遍。旋钮又这么多，它就干脆死记了。

    ## 会滑动的探测器

    更好的想法是这样的。让探测器保持很小，比如 $3 \times 3$ 个旋钮，不再给图片的每个部分各配一个探测器，而是让同一个探测器滑过整张图片。在每个位置，取它下面 $3 \times 3$ 的小块，算一次点积：逐个位置相乘，再加起来。这个小小的旋钮方格叫**卷积核**（kernel），滑动它的操作叫**卷积**（convolution）。

    我们手算一遍。下面是一张 $4 \times 4$ 的图片，数字在中间从 0 跳到 1，还有一个比较左右两边的卷积核：

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

    全是 0 或全是 1 的小块会得到 0，因为 −1 和 +1 互相抵消。所以结果只在数字从左到右发生变化的地方才大：这个卷积核是一个**竖直边缘探测器**。$3 \times 3$ 的卷积核在 $4 \times 4$ 的图片上，横着能放 2 个位置，竖着也是 2 个，所以输出是 $2 \times 2$。一般来说，高度为 $h$ 的图片，用大小为 $k$ 的卷积核，输出 $h - k + 1$ 行。

    ## 练习 1：convolve

    写 `convolve(image, kernel)`，卷积核是正方形的：

    1. 两行：`h, w = image.shape` 和 `k = kernel.shape[0]`
    2. `out = np.zeros((h - k + 1, w - k + 1))`
    3. `for r in range(h - k + 1):`，里面再写 `for c in range(w - k + 1):`
    4. 在两层循环里面：`patch = image[r:r + k, c:c + k]`，然后 `out[r, c] = np.sum(patch * kernel)`
    5. 循环结束后，`return out`
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
    mo.md(r"""
    现在用在真实的数字上：一个是竖直边缘探测器，另一个是把同一个卷积核转置（`.T`）后得到的，能找水平边缘。蓝色表示正的响应，红色表示负的响应：
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
    卷积的输出本身也像一张图片，是一张标出形状在哪里被找到的地图，所以叫**特征图**（feature map）。有两个性质让它成为合适的工具。

    **它跟着图片走。** 把数字向右挪 3 列（`np.roll` 平移一个数组，`axis=1` 表示沿着列的方向），特征图也只是跟着挪 3 列。“先挪再卷积”和“先卷积再挪”的最大差别正好是 0：
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
    **它只需要很少的旋钮。** 一个卷积核只有 9 个旋钮，不管图片多大，因为同样的 9 个旋钮在每个位置重复使用。在一个角落学到的东西，到每个角落都管用。

    ## 缩小：最大池化

    一个数字不只是一堆边缘，而是按某种方式排列的边缘。为了看到更大的图案，我们把特征图缩小，再做一次卷积。**最大池化**（max pooling）把特征图切成 $2 \times 2$ 的方块，每块只留最大的数。它的意思是“这个特征大概在这一带”，忘掉准确位置，并把高和宽都减半。这样下一层的 $3 \times 3$ 卷积核在原图上覆盖的范围就宽了一倍。

    ```
    1  3  0  2
    5  2  1  0      →     5  2
    0  0  4  1            1  6
    1  0  6  2
    ```

    ## 练习 2：max_pool

    写 `max_pool(M)`。`//` 是去掉小数部分的除法（`7 // 2` 是 3），所以边上多出来的一行会被丢掉。

    1. `h, w = M.shape`
    2. `out = np.zeros((h // 2, w // 2))`
    3. `for r in range(h // 2):`，里面再写 `for c in range(w // 2):`
    4. 在两层循环里面：`out[r, c] = np.max(M[2 * r:2 * r + 2, 2 * c:2 * c + 2])`
    5. 循环结束后，`return out`
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
    mo.md(r"""
    ## 让网络自己选卷积核

    刚才的边缘探测器是我们亲手挑的。网络则是自己学卷积核：那 9 个数字就是旋钮，用一贯的反向步骤和 Adam 来训练。在 PyTorch 里写成 `nn.Conv2d(1, 1, 3)`，读作“输入 1 张图，输出 1 张图，卷积核大小 3”。图片必须是四个数字组成的形状：

    > (图片数, 图数, 高, 宽)

    多出来的这个数表示每张图片有几张特征图。灰度数字只有一张。`X.unsqueeze(1)` 在位置 1 插入一个大小为 1 的新维度，于是 `(10000, 40, 40)` 变成 `(10000, 1, 40, 40)`。

    为了确认 PyTorch 做的就是你刚写的东西，我们把你的卷积核放进它的层里比较一下。（`.data` 是张量里不带账本的那些数字。）
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
    形状结尾是 $26 \times 26$，和我们的公式一致（$28 - 3 + 1$）。第二个数字是和你的特征图的最大差别：0，或者只差舍入误差。再加两样东西，就能搭网络了。

    **很多卷积核。** 一层可以同时学很多卷积核，每个给出一张自己的特征图。`nn.Conv2d(1, 8, 3)` 有 8 个卷积核，把 1 张图变成 8 张。下一层接收这 8 张：`nn.Conv2d(8, 16, 3)` 有 16 个大小为 $3 \times 3 \times 8$ 的卷积核，每个都同时看 8 张图上同一个小块。后面的卷积核就是靠这种方式，才能表达“竖直边缘的左边紧挨着水平边缘”。数一数旋钮：16 个卷积核，每个有 $3 \times 3 \times 8 = 72$ 个权重和 1 个偏置，所以是 $16 \times 73 = 1{,}168$。

    **保持大小。** `padding=1` 先在图的四周加一圈宽 1 像素的 0（这叫**填充**，padding），这样 $3 \times 3$ 的卷积核给出的输出和输入一样高一样宽，只有池化才会缩小它。

    网络的末尾用 `nn.AdaptiveMaxPool2d(1)`：不管每张图多大，只留整张图上最大的那一个数。它回答的问题是“图片里任何地方有这个特征吗？”之后 32 张图就变成了 32 个数字（`nn.Flatten()` 整理形状），最后的 `nn.Linear` 把它们变成 10 个分数。位置信息几乎没有留下，所以一个数字不管在哪里，分数都差不多。（下面列表里的 `nn.MaxPool2d(2)` 就是你刚写的 $2 \times 2$ 池化。）

    ## 练习 3：make_cnn

    写 `make_cnn()`，返回下面的网络（写成一个长长的 `return nn.Sequential(...)`，每层后面加逗号，愿意的话可以一层一行）：

    1. `nn.Conv2d(1, 8, 3, padding=1)`、`nn.ReLU()`、`nn.MaxPool2d(2)`
    2. `nn.Conv2d(8, 16, 3, padding=1)`、`nn.ReLU()`、`nn.MaxPool2d(2)`
    3. `nn.Conv2d(16, 32, 3, padding=1)`、`nn.ReLU()`
    4. `nn.AdaptiveMaxPool2d(1)`、`nn.Flatten()`、`nn.Linear(32, 10)`
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
    mo.md(r"""
    卷积比 `@` 慢，所以训练要等你按按钮（10 轮；在笔记本上最多一分钟左右）。和前面一样，显示旋钮数，以及训练和测试准确率。在你按下之前，需要它的结果的单元格会显示 `Ancestor stopped`，这是正常的。
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
    训练完你应该看到 6,218 个旋钮，比扁平网络少 66 倍，新图片上大约 96%，而扁平网络大约 88%。训练和测试分数现在很接近：几乎没有死记，因为每个旋钮在每个位置都要重复使用，必须到处都管用。

    训练完成后，下面会显示第一层的 8 个卷积核。没有人告诉它什么是边缘，但其中好几个都在比较一边和另一边，和你手工做的那个一样：
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
    ## 走向深层：责任的通道

    既然三层这么好用，三十层会不会更好？深的网络本该能看出更复杂的东西。但很多年里，深网络反而训练得比浅网络差，一个重要原因就是第 03 章的链式法则。责任要到达靠前的层，必须经过后面每一层的相乘。如果这些乘法总体上在缩小它，责任就会渐渐消失，正是你在第 09 章表格里看到的那种衰减。

    解决办法简单得让人意外。原来的块让输出取代输入：

    > `out = f(x)`

    改成把输出**加到**输入上：

    > `out = x + f(x)`

    现在 `f` 只需要学对 `x` 的*修正*，这叫**残差**。看看斜率。根据第 02 章的加法规则，`x + f(x)` 的斜率是 `1 + f'`，其中 `f'` 是 `f` 的斜率。即使 `f'` 很小，斜率也接近 1，所以责任几乎不衰减地穿过这个块。这个 `x +` 是责任的直达通道，常叫**跳跃连接**（skip connection）。

    ## 练习 4：residual_block

    写 `residual_block(x, conv1, conv2)`。`conv1` 和 `conv2` 是两个保持形状不变的卷积层：

    1. `h = torch.relu(conv1(x))`
    2. `h = conv2(h)`
    3. `return torch.relu(x + h)`
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
    mo.md(r"""
    要把这样的块放进 `nn.Sequential`，它得是一个层。自己做层用的就是第 04 章的 `class`，只多了一点：
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
    - `class Block(nn.Module):` 的意思是“`Block` 是一种 `nn.Module`”，`nn.Module` 是 PyTorch 里层的类别。它继承了层能做的一切，所以 PyTorch 能找到 `Block` 里的旋钮。
    - `super().__init__()` 先运行 `nn.Module` 自己的设置，它总是第一行。
    - `forward` 是计算本身。调用 `block(x)` 就会运行 `block.forward(x)`，这也解释了第 10 章的 `layer(X)` 是怎么工作的。
    - `step` 是存在属性里的函数（第 02 章：函数也是值），所以同一个类既能运行你的 `residual_block`，也能运行 `plain_block`，后者做同样的事，只是没有 `x +`。

    `deep_model(blocks, step)` 把这么多个块堆起来，每块两层卷积。为了公平比较，我们用同样的网络、同样的图片，走一次反向，看看有多少责任到达了最前面的一层：用它的斜率的平均大小来衡量。网络有 10 个块：
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
    左边的数是普通堆叠，右边的带跳跃连接。大约 0.000000000001 对大约 0.015：有了跳跃连接，到达第一层的责任多了一百亿倍以上。在普通堆叠里，第一层几乎什么也收不到。

    下面的按钮用 3,000 张居中的数字把两个网络各训练 4 轮（大约半分钟），显示各自的测试准确率：先是普通堆叠，然后是带跳跃连接的。后面的文字说明你应该看到什么。
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
    大约 10% 对大约 80%（第二个数字每次运行会略有不同）。普通堆叠几乎没有责任到达第一层，还在瞎猜：随便选一个数字，得到的就是 10%。同样的层，每个块里加上一个 `x +`，就能学了。

    这样一堆块（通常中间再多缩小几次）叫 **ResNet**。正是这种设计让一百层以上的网络变得可以训练。同样的技巧，把输入加回去，也出现在本课程后面 Transformer 的每一个块里，理由相同。

    ## 接下来

    第一部分到此结束。你从拟合一条直线开始，现在已经造出了分类器、自己的引擎和能看东西的网络。后面的一切都建立在这些基础上。

    第二部分转向**语言**。第一个问题是：网络需要数字，而词不是数字。在**第 12 章**，你会给每个词一个向量，让意思相近的词靠在一起。

    ## 你现在会做的事

    - 解释为什么扁平网络对付不了移动的物体
    - 手算和用代码计算卷积，并预测输出的大小
    - 解释卷积核、特征图和最大池化的作用
    - 数出一个卷积层的旋钮数
    - 用 `nn.Conv2d` 搭建并训练卷积网络
    - 解释责任为什么在深层堆叠里衰减，以及 `x + f(x)` 怎样解决它
    - 基于 `nn.Module` 写一个类，做出自己的层

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 数学家会把我们算的东西叫*互相关*（cross-correlation）；真正的卷积要先翻转卷积核，但卷积核是学出来的，所以这点差别无关紧要。真实的 ResNet 还会用*批归一化*（batch normalisation），彩色图片一开始有 3 张图（红、绿、蓝）。常用来练习这类任务的标准数据集是 *CIFAR-10* 和 *ImageNet*，原始论文是 *Deep Residual Learning for Image Recognition*（He 等，2015）：这位 He 就是第 09 章起始规则的那位 He。
    """)
    return


if __name__ == "__main__":
    app.run()
