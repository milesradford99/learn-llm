import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="19 · 让它跑得快")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from learn_llm import laptop_speed, todo
    from learn_llm.checks import check

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 19 · 让它跑得快

    训练一个模型意味着做海量的算术。让它变快，首先不在于巧妙的技巧，而在于**测量**，弄清时间到底花在哪里；然后是**估算**（数一数要做多少算术），弄清它最快可能有多快。这一章两件事都做，然后尝试两种提速办法，并发现其中一种可能反而变慢，在测试这一章的电脑上就是如此。

    ## 测量时间

    `time.perf_counter()`（第 08 章）是一个以秒计的时钟。只测一次并不可靠：任何东西第一次运行往往比较慢，因为电脑还在准备，而且每次运行都会有些波动。所以我们先运行一次预热，再计时运行几次，取平均。

    ## 练习 1：time_it

    写 `time_it(f, repeats)`，在预热调用一次之后，返回调用一次 `f` 的平均秒数：

    1. `f()`
    2. `started = time.perf_counter()`
    3. `for i in range(repeats):`，里面写 `f()`
    4. `return (time.perf_counter() - started) / repeats`
    """)
    return


@app.function
def time_it(f, repeats):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "time_it", time_it, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 数一数算术

    网络里的工作几乎全是矩阵乘法。`A @ B` 中，`A` 的形状是 `(n, k)`，`B` 的形状是 `(k, m)`，结果里的每个数都是一个长度为 `k` 的点积：`k` 次乘法，大约 `k` 次加法。这样的数共有 $n \times m$ 个，所以一次矩阵乘法大约要 $2nkm$ 次浮点运算，也就是 $2nkm$ 个**浮点运算次数**（FLOPs）。（“浮点数”就是计算机里带小数点的数，也就是我们一直在用的那种。）

    ## 练习 2：matmul_flops

    写 `matmul_flops(n, k, m)`：

    ```
    return 2 * n * k * m
    ```

    检查之后的单元格用它测量你的电脑：给两个 $1024 \times 1024$ 矩阵的乘法计时，再相除。十亿次浮点运算叫一个 **gigaFLOP**。
    """)
    return


@app.function
def matmul_flops(n, k, m):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "matmul_flops", matmul_flops, LOCALE)
    return


@app.cell
def _():
    A = torch.rand(1024, 1024)
    B = torch.rand(1024, 1024)
    def product():
        return A @ B

    seconds_32 = time_it(product, 5)
    machine_speed = matmul_flops(1024, 1024, 1024) / seconds_32
    round(machine_speed / 1e9, 1)
    return A, B, machine_speed


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这就是你电脑的最佳速度，单位是每秒 gigaFLOP。你的数字会和别人的不一样。（`product` 是一个没有输入的函数，所以 `time_it` 可以一遍遍调用它。）在测试这一章的机器上，它大约是 180；一块现代 GPU 大约快 1,000 倍，而最大规模的训练会同时用上成千上万块这样的 GPU。

    ## 训练的代价

    对一个有 `N` 个旋钮的模型，下面这条规则对做计划来说足够准确。在正向计算中，每个旋钮对每个词元大约参与一次乘法和一次加法：2 次浮点运算。反向计算大约是它的两倍：一次矩阵乘法求旋钮的斜率，另一次求传回该层输入的责任（第 07 章的 `X.T @ G` 和 `G @ W.T`）。这又是 4 次。所以：

    > $$\text{训练的浮点运算次数} \approx 6 \times N \times \text{词元数}$$

    它没有算注意力分数本身，这部分只在窗口很长时才要紧。

    ## 练习 3：training_flops

    写 `training_flops(knobs, tokens)`：

    ```
    return 6 * knobs * tokens
    ```
    """)
    return


@app.function
def training_flops(knobs, tokens):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "training_flops", training_flops, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第 18 章的训练用大约 117 万个旋钮，把大约 200 万个词元读了两遍。下面的单元格算出算术量，以及按你电脑的最佳速度要多久。单元格显示两个数字：以 teraFLOP 计的算术量，以及全速运行所需的秒数。把这个秒数和第 18 章的按钮实际用的时间比一比。一次训练达到的速度占最佳速度的比例，叫作它的**利用率**（utilisation）。
    """)
    return


@app.cell
def _(machine_speed):
    flops_ch18 = training_flops(1.17e6, 2 * 2.03e6)
    round(flops_ch18 / 1e12, 1), round(flops_ch18 / machine_speed)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 28 teraFLOP（一个 teraFLOP 是万亿次，即 $10^{12}$ 次浮点运算），全速运行只要两三分钟。在测试机器上，实际训练用了四五分钟，所以大约用到了电脑最佳速度的一半。剩下的时间花在小操作上，比如 softmax、归一化和搬运数字，它们算术很少，却要花时间。对一个 70 亿个旋钮、一万亿个词元的模型，同一个公式给出 $4.2 \times 10^{22}$ 次浮点运算，需要大约一千块 GPU，每块每秒约 200 万亿次浮点运算，以一半的最佳速度跑大约一周。

    ## 数一数内存

    用 Adam 训练时，每个旋钮需要四个数：它自己、它的斜率，以及 Adam 的两个平均值。用 32 位时，每个数 4 字节，每个旋钮 16 字节。在此之上还有**激活值**（activations）：正向计算为反向计算保留的所有值，它们随批大小和窗口长度增长。

    ## 练习 4：adam_bytes

    写 `adam_bytes(knobs)`，即 32 位时旋钮、斜率和 Adam 平均值所需的字节数：

    ```
    return 16 * knobs
    ```
    """)
    return


@app.function
def adam_bytes(knobs):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "adam_bytes", adam_bytes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    对第 18 章的模型，这不到 19 MB。对一个 70 亿个旋钮的模型，是 112 GB，大多数 GPU 都装不下。第 20 章讲的就是这时该怎么办。

    ## 提速 1：更小的数

    32 位的数能保证大约 7 位正确的数字（第 09 章）。两种 16 位格式能省一半内存，在专门为它们设计的硬件上，速度能快一倍以上：

    - **float16** 大约保证 3 位数字，但装不下大于 65,504 的数；
    - **bfloat16** 保证的位数更少，大约 2 到 3 位，但能达到和 float32 一样大的数。

    `torch.finfo` 描述一种格式：`max` 是它能表示的最大数，`eps` 是 1 和它能表示的下一个数之间的间隔，用来衡量它有多精确。`eps` 约为 0.001 就是千分之一，大约 3 位正确的数字：
    """)
    return


@app.cell
def _():
    formats = []
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        formats.append((dtype, torch.finfo(dtype).max, torch.finfo(dtype).eps))
    formats, torch.tensor(70000.0, dtype=torch.float16), (torch.tensor(1.0, dtype=torch.bfloat16) + 0.001).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    70,000 在 float16 里变成 `inf`，即无穷大：太大了。而在 bfloat16 里，1 + 0.001 还是 1：变化太小，记不下来。只用 16 位训练会丢掉小的更新，所以真实的训练用**混合精度**（mixed precision）：矩阵乘法用 16 位，而旋钮、Adam 的平均值和需要精度的求和保持 32 位。

    它能让*你的*电脑变快吗？只有处理器带有 16 位运算的硬件时才行。下面的单元格用每种格式计时同一个乘法，单位是每秒 gigaFLOP，所以越高越快。可能要几秒钟。
    """)
    return


@app.cell
def _(A, B):
    speeds = []
    for dtype_try in [torch.float32, torch.bfloat16]:
        A_try, B_try = A.to(dtype_try), B.to(dtype_try)

        def product_try():
            return A_try @ B_try

        speeds.append((dtype_try, round(matmul_flops(1024, 1024, 1024) / time_it(product_try, 3) / 1e9, 1)))
    speeds
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    在测试机器上，bfloat16 比 float32 **慢了 100 多倍**，因为它的处理器没有 16 位运算的硬件，只能一步步模拟。你的数字可能不同：一些较新的处理器和 Apple 芯片就有这种硬件。这正是要点：一种提速办法只对特定的硬件有效，想知道有没有用，只能测量。在 GPU 上，混合精度是常规做法。

    ## 提速 2：不用的东西就别造

    注意力会造出一张分数表，每对词一个分数：窗口 1,024 个词元、4 个头、一批 8 段时，大约是 3,350 万个数，134 MB，要写进内存再读出来好几次（掩码、softmax、乘法）。这时限制速度的是内存，而不是算术：处理器算得比它从主内存取数快得多。处理器旁边那一小块快速内存，也就是**缓存**（cache），速度全靠它。

    **分块注意力**（tiled attention）从不造出整张表。对每个查询，它沿着键一个分块一个分块地走，同时保留一个累计结果。窍门在 softmax 上：它需要所有分数的最大值和总和，而这要到最后才知道。但两者都可以作为累计值保存，一路修正：

    - 保留一个累计最大值 `best`，一个 $e^{\text{score} - \text{best}}$ 的累计和 `total`，以及值的累计加权和 `acc`；
    - 新的分块带来更大的最大值时，之前累计的一切都是按旧的最大值算的，所以把 `total` 和 `acc` 乘以 $e^{\text{旧 best} - \text{新 best}}$（第 06 章：$e^{a-b}$ 能正确地重新缩放）；
    - 最后，`acc / total` 正好是 softmax 的加权平均。

    这就是 **FlashAttention** 的思路。

    ## 练习 5：tiled_attention

    写 `tiled_attention(q, K, V, tile)`，输入是一个形状为 `(d,)` 的查询 `q`，以及形状为 `(n, d)` 的键 `K` 和值 `V`，不用掩码，按 $\sqrt{d}$ 缩放。它必须和 `torch.softmax(K @ q / np.sqrt(d), dim=0) @ V` 给出同样的答案。

    1. 三行：`best = -np.inf`、`total = 0.0` 和 `acc = torch.zeros(V.shape[1])`
    2. `for start in range(0, len(K), tile):`，里面写：
    3. `scores = K[start:start + tile] @ q / np.sqrt(len(q))`
    4. `new_best = max(best, torch.max(scores).item())`
    5. `fix = np.exp(best - new_best)`
    6. `weights = torch.exp(scores - new_best)`
    7. `total = total * fix + torch.sum(weights).item()`
    8. `acc = acc * fix + weights @ V[start:start + tile]`
    9. `best = new_best`
    10. 循环结束后，`return acc / total`

    （`np.inf` 是无穷大，和 `torch.inf` 一样；`np.exp(-np.inf)` 是 0，所以第一个分块也没问题。）

    下面手算一遍：每个向量 1 个数，键是 `[1, 3, 2]`，值是 `[10, 20, 30]`，每个分块 1 个（$\sqrt{d}$ 是 1，略去不写）：

    | 分块 | `best` | `fix` | `total` | `acc` |
    |---|---|---|---|---|
    | 1 | 1 | 0 | 1 | 10 |
    | 2 | 3 | $e^{-2} \approx 0.135$ | 1.135 | 21.35 |
    | 3 | 3 | 1 | 1.503 | 32.39 |

    而 $32.39 / 1.503 \approx 21.55$，和一次对三个分数做 softmax 的结果相同。
    """)
    return


@app.function
def tiled_attention(q, K, V, tile):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "tiled_attention", tiled_attention, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    用 Python 一次处理一个查询，这样很慢：重点在于*方法*。在真实的系统里，同样的步骤在处理器上的一个编译好的函数里运行，把分块保留在缓存中。PyTorch 就有一个（这叫“融合”：把这些步骤合成一个函数）：`F.scaled_dot_product_attention`。下面的单元格在两种窗口长度下，把它和第 15 章造整张掩码表的做法作比较。每一行依次是：窗口长度、造表做法的秒数、融合做法的秒数、两个答案的最大差别（类似 `5e-07`，也就是 0.0000005，只是舍入误差），以及这张表要多少 MB。
    """)
    return


@app.cell
def _():
    rows = []
    for window in [256, 1024]:
        q_all = torch.rand(8, 4, window, 32)
        k_all = torch.rand(8, 4, window, 32)
        v_all = torch.rand(8, 4, window, 32)
        mask = torch.triu(torch.ones(window, window, dtype=torch.bool), diagonal=1)

        def table_way():
            scores = (q_all @ k_all.transpose(-2, -1) / np.sqrt(32)).masked_fill(mask, -torch.inf)
            return torch.softmax(scores, dim=-1) @ v_all

        def fused_way():
            return F.scaled_dot_product_attention(q_all, k_all, v_all, is_causal=True)

        difference = torch.max(torch.abs(table_way() - fused_way())).item()
        rows.append((window, round(time_it(table_way, 3), 4), round(time_it(fused_way, 3), 4), difference, 8 * 4 * window * window * 4 / 1e6))
    rows
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    在测试机器上，融合版本在 256 时快了大约 7 倍，在 1,024 时快了大约 13 倍，答案在舍入误差内相同。窗口越长，提速越明显，因为这张表以及读写它的时间，都随窗口的*平方*增长，而融合做法从不造这张表。窗口加倍，表就变成四倍：这就是“随平方增长”的意思，也是长窗口昂贵的原因。

    ## 性能分析：找出慢在哪里

    在提速之前，先找出哪里慢。**性能分析器**（profiler）给一次运行里的每个操作计时。PyTorch 的叫 `torch.profiler`。下面的单元格对一个小 Transformer 块的一步训练做性能分析，它是 PyTorch 自己版本的第 15 章 `Block`，并列出最耗时的五个操作：
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    layer = torch.nn.TransformerEncoderLayer(128, 4, 512, dropout=0.0, batch_first=True)
    x_profile = torch.rand(32, 64, 128)
    with torch.profiler.profile() as profile:
        layer(x_profile).sum().backward()
    print(profile.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这张表很密，只看名字和 `CPU total` 这一列就行。通常排在前面的是矩阵乘法（名字里带 `mm` 的，比如 `aten::mm` 和 `AddmmBackward0`）。如果排在前面的是别的，那就是首先要查的地方。一行的总时间包括它调用的操作，所以同一段时间可能出现在好几行里。（`print` 按原样显示文本，各列对齐，所以这里用它。）

    ## 接下来

    一台电脑只能快到这个程度。**第 20 章**把训练分到几台设备上，这里是你电脑上的几个进程，并检查它们学到的和一个进程完全一样。

    ## 你现在会做的事

    - 公平地计时：先预热，再重复
    - 数出矩阵乘法和一次训练的浮点运算次数
    - 根据电脑的速度估计一次训练的时间和利用率
    - 数出用 Adam 训练需要的内存
    - 解释 float16、bfloat16 和混合精度，并测试它们在你的硬件上是否有用
    - 解释为什么注意力的分数表是瓶颈，并写出分块注意力
    - 读懂性能分析结果

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** FlashAttention 出自 Dao 等（2022）。`6N` 规则出自 *Scaling Laws for Neural Language Models*（Kaplan 等，2020），第 21 章会以它为基础。*算子融合*（kernel fusion）和 `torch.compile` 能把许多小操作合成一个，*激活检查点*（activation checkpointing）用重新计算换取激活值所占的内存。
    """)
    return


if __name__ == "__main__":
    app.run()
