import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="14 · 记忆：RNN")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.diagrams import pipeline
    from learn_llm.leaderboard import record, show
    from learn_llm.plot import chart

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 14 · 记忆：RNN

    第 13 章只靠纯计数猜下一个词，全程没用到神经网络：它最多记得前面一两个词，还把 `dog`（狗）和 `cat`（猫）当成毫不相干的词。但人读文章不是这样——你逐词读过一个句子，每读一个词就在脑子里更新一遍印象；读完一整段，你依然记得主角是谁，却并不需要把原文一字不差背下来。

    这一章我们回到 PyTorch，造一个能边读边记的神经网络：**循环神经网络**（recurrent neural network，RNN）。它维护一个数字向量作为**记忆**（memory），每读一个词就更新一次。我们会先亲手搭出一个 RNN，再看清它为什么容易忘掉开头，最后用 **LSTM** 解决这个难题——在 2010 年代的大部分时间里，语言模型都靠它领跑。

    ## 一个新的激活函数：tanh

    记忆向量要一遍又一遍地经过同一个更新步骤，所以里面的数值不能无限膨胀。前面用过的 ReLU（把负数归零、正数原样放行）没有上限，越滚越大就会失控。因此这里换用一个新的激活函数 **`tanh`**，把任意大小的数都平滑地压进 $-1$ 到 $1$ 之间：
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((-4, 4), (-1, 1), "x", "tanh(x)",
          curve=[(x / 10, float(np.tanh(x / 10))) for x in range(-40, 41)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    看它的曲线形状：在 0 附近几乎就是 $y = x$（小数字基本原样保留），越往两端越平缓地贴近 $+1$ 和 $-1$。在 PyTorch 里它写作 `torch.tanh`。下面试几个数：0.1 几乎还是 0.1，而 5 已经被压到了将近 1：
    """)
    return


@app.cell
def _():
    torch.tanh(torch.tensor([-5.0, -1.0, 0.0, 0.1, 1.0, 5.0]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 记忆的一步

    设 $x$ 是当前词的词向量（长度为 $d$，来自第 12 章的嵌入表），$h$ 是到目前为止的记忆向量（长度为 $m$）。每读一个词，新记忆就把这两部分合在一起：

    > $$\text{新的 } h = \tanh(x \times W_x + h \times W_h + b)$$

    这里的 $\times$ 表示矩阵乘法（在 PyTorch 里用 `@`，和第 12 章两个向量做点积用的 $\cdot$ 不同）：$W_x$ 把当前词投影成记忆的长度，$W_h$ 把旧记忆带到下一步，$b$ 是偏置。本质上这就是第 08 章的线性层，只是多了一个巧妙之处：它的一部分输入，正是自己上一步的输出。“循环”的意思，就是输出绕回来接着当输入。

    隔了一章重新写 PyTorch，顺手回顾两行矩阵形状：处理一整批文本时，矩阵乘法要求中间维度对齐——`(batch, d) @ (d, m)` 得到 `(batch, m)`，`(batch, m) @ (m, m)` 也得到 `(batch, m)`；再加上长度为 `(m,)` 的偏置（自动广播到每一行），整个和的形状正好是 `(batch, m)`。

    下面把这一步连走三次，读入 `the dog ran`。记忆从全零出发，每个箭头走的都是同一个步骤、共用同一组参数，只是每一步送进去的词不同：
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([('h', '全零'), ('新的 h', '读过：the'), ('新的 h', 'the dog'), ('新的 h', 'the dog ran')],
             ['一步，x = the', '一步，x = dog', '一步，x = ran'],
             '记忆 h 从全零开始。用词 the 走一步得到新的 h，用 dog 走同样的一步得到下一个，再用 ran 走同样的一步得到最后一个。')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每个词共用的都是**同一组**参数，这和第 11 章的卷积核一样：卷积核在图片的各个位置上滑动，这一步在文本的各个位置上滑动。所以不管文本是 3 个词还是 3,000 个词，网络的参数量都一样多。

    ## 动手算：记忆的两步

    看最小的情况：一个词是一个数，记忆也是一个数，参数是 $W_x = 2$、$W_h = 1$、$b = -1$。记忆从 0 开始。第一个词是 $x = 1$，第二个词是 $x = 0$。你可以直接查下面这张表：

    | $z$ | −1 | −0.24 | 0 | 0.24 | 1 | 2 |
    |---|---|---|---|---|---|---|
    | $\tanh(z)$ | −0.76 | −0.24 | 0 | 0.24 | 0.76 | 0.96 |

    对每个词，先算出送进 `tanh` 的数，再查表得到新的记忆。
    """)
    return


@app.cell(hide_code=True)
def _():
    step_boxes = hand.boxes(['第 1 个词：送进 tanh 的数', '第 1 个词之后的记忆', '第 2 个词：送进 tanh 的数', '第 2 个词之后的记忆'], step=0.01)
    return (step_boxes,)


@app.cell(hide_code=True)
def _(step_boxes):
    hand.marked(step_boxes, {'第 1 个词：送进 tanh 的数': 1, '第 1 个词之后的记忆': 0.76, '第 2 个词：送进 tanh 的数': -0.24, '第 2 个词之后的记忆': -0.24}, {
        '第 1 个词：送进 tanh 的数': r"""$x \times W_x + h \times W_h + b$，其中 $x = 1$，$h = 0$""",
        '第 1 个词之后的记忆': r"""在表里查你的第一个答案""",
        '第 2 个词：送进 tanh 的数': r"""同样的算式，这次 $x = 0$，$h$ 是第 1 个词之后的记忆""",
        '第 2 个词之后的记忆': r"""在表里查你的第三个答案""",
    }, LOCALE, tolerance=0.011)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第二个词是 0，本身没有带来任何新信息，但记忆还是变了，而且新值取决于第一个词——换一个第一个词，这里的结果就会不同。这就是记忆在起作用。

    ## 练习 1：rnn_step

    写 `rnn_step(x, h, Wx, Wh, b)`，把上面的公式写成代码，一次处理一整批文本。

    | 输入 | 形状 | 含义 |
    |---|---|---|
    | `x` | `(batch, d)` | 每段文本当前的词向量（长度 `d`） |
    | `h` | `(batch, m)` | 每段文本到目前为止的记忆（长度 `m`） |
    | `Wx` | `(d, m)` | 输入权重矩阵 |
    | `Wh` | `(m, m)` | 记忆权重矩阵 |
    | `b` | `(m,)` | 偏置向量 |

    函数返回新的记忆，形状为 `(batch, m)`。你刚才动手算的第一步就是其中一项检查：每个张量只放一个数，`x = [[1.0]]`、`h = [[0.0]]`、`Wx = [[2.0]]`、`Wh = [[1.0]]`、`b = [-1.0]`，结果应该是 `[[0.76]]`。
    """)
    return


@app.function
def rnn_step(x, h, Wx, Wh, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "rnn_step", rnn_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""这就是第 08 章的线性层，只是输入从一份变成了两份。公式里的矩阵乘法 $\times$，在代码里写成 `@`。""",
        r"""只要一行：`return`，后面用 `torch.tanh` 把整个和包起来。偏置要写在 `tanh` 里面。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 读完整段文本

    现在把上面那张流程图写成代码。整批文本以张量 `X` 送进来，形状是 `(batch, length, d)`：共有 `batch` 段文本，每段 `length` 个词，每个词是一个长度为 `d` 的向量。我们需要按顺序取出每段文本的第 $t$ 个词，最后再把每一步算出的记忆拼成一整张张量。

    /// admonition | 🧰 工具箱：拆开一个维度，拼出一个维度
    - 索引里单独一个 `:` 表示“这一维全要”。对形状为 `(batch, length, d)` 的 `X`，`X[:, 0]` 取出每段文本的第 0 个词，形状是 `(batch, d)`。
    - `torch.stack(tensors, dim=1)` 沿着一个**新建的**第 1 维，把列表里形状相同的张量叠在一起。比如三个 `(batch, m)` 的张量叠完后会变成 `(batch, 3, m)`。
    ///

    下面的单元格从一个形状为 `(2, 5, 4)` 的张量里取出两个词，再把它们叠起来，并显示两个形状。
    """)
    return


@app.cell
def _():
    words_demo = torch.rand(2, 5, 4)
    words_demo[:, 0].shape, torch.stack([words_demo[:, 0], words_demo[:, 4]], dim=1).shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：run_rnn

    写 `run_rnn(X, Wx, Wh, b)`。`X` 的形状是 `(batch, length, d)`，参数和练习 1 一样。它返回每个词之后的记忆，形状是 `(batch, length, m)`：

    1. 记忆从全零开始，形状是 `(batch, m)`。这两个大小都能从输入里读出来：`X.shape[0]` 和 `Wh.shape[0]`。
    2. 对每个位置 `t`，从第一个词到最后一个词，用一次 `rnn_step` 把第 `t` 个词和当前记忆变成新的记忆。把每一步得到的新记忆都存进一个列表。
    3. 沿第 1 维把列表叠起来。

    还是你动手算的那个例子：一段两个词的文本，`X = [[[1.0], [0.0]]]`，参数不变，结果应该是 `[[[0.76], [-0.24]]]`。
    """)
    return


@app.function
def run_rnn(X, Wx, Wh, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "run_rnn", run_rnn, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""循环之前：用 `torch.zeros` 造出 `h`，再造一个空列表。循环是 `for t in range(X.shape[1]):`。在循环里，把 `rnn_step` 的结果存回 `h`，这样下一个词才能看到它，再把 `h` 加进列表。""",
        r"""一共 6 行。每段文本的第 `t` 个词是 `X[:, t]`。最后一行返回对列表做 `torch.stack` 的结果，`dim=1`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    PyTorch 内置的 `nn.RNN` 干的正是同一件事。像第 10 章那样来核对一下：先建一个随机初始化的 `nn.RNN`，把它的参数拷贝进你的 `run_rnn`，再对比输出。（运行这个单元格前需先完成练习 2。）`nn.RNN(4, 6, batch_first=True)` 表示每个词向量长度 4、记忆长度 6；`batch_first=True` 告诉 PyTorch 把 `batch` 放在第 0 维，和我们的约定一致。和 `nn.Linear` 一样，PyTorch 内部的权重矩阵是转置存放的（所以要加 `.T`），而且拆成了两个偏置向量，相加就行。调用 `rnn(X_demo)` 会返回一对结果——每一步的记忆和最后一步的记忆，所以我们用 `[0]` 取前者。
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    rnn = nn.RNN(4, 6, batch_first=True)
    X_demo = torch.rand(2, 5, 4)
    Wx_demo = rnn.weight_ih_l0.T
    Wh_demo = rnn.weight_hh_l0.T
    b_demo = rnn.bias_ih_l0 + rnn.bias_hh_l0
    with torch.no_grad():
        difference = run_rnn(X_demo, Wx_demo, Wh_demo, b_demo) - rnn(X_demo)[0]
    torch.max(torch.abs(difference)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 `1e-07`，也就是 0.0000001（小数点向左移 7 位，第 09 章见过）。这只是浮点数的舍入误差，说明两者完全一致。

    ## 从文本到样本

    训练需要成对的输入和答案。对语言模型来说，每个位置的正确答案就是它的**下一个词**。把编号序列 `[5, 2, 9, 7, 1]` 切成长度为 2 的片段：

    ```
    X: 5 2      Y: 2 9
       9 7         7 1
    ```

    在每个位置上，`Y` 都是 `X` 往后错开一位。这样短短一串文本，就包含了很多个“猜下一个词”的训练样本。最后一个编号 `1` 只能当答案：它后面没有词了，没法作为输入。

    ## 动手算：切一段文本

    把 8 个编号 `[4, 8, 3, 6, 1, 9, 2, 5]` 切成长度为 3 的片段。因为最后一个答案还需要多占一个编号，最多能切出几行完整的片段？`X` 的最后一个编号是什么？`Y` 的第一个和最后一个编号呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    window_boxes = hand.boxes(['行数', 'X 的最后一个编号', 'Y 的第一个编号', 'Y 的最后一个编号'])
    return (window_boxes,)


@app.cell(hide_code=True)
def _(window_boxes):
    hand.marked(window_boxes, {'行数': 2, 'X 的最后一个编号': 9, 'Y 的第一个编号': 8, 'Y 的最后一个编号': 2}, {
        '行数': r"""3 行要用 9 个编号当输入，而总共只有 8 个""",
        'X 的最后一个编号': r"""`X` 装的是前 $2 \times 3 = 6$ 个编号""",
        'Y 的第一个编号': r"""第一个输入是 4，它的答案是它后面的那个编号""",
        'Y 的最后一个编号': r"""最后一个输入的答案，是它后面的那个编号""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    最后一个编号 5 完全没用上：它凑不满一行。

    ## 练习 3：windows

    写 `windows(ids, length)`。`ids` 是一个 NumPy 编号数组，`length` 是一个整数。它返回两个数组 `X, Y`，形状都是 `(n, length)`，其中 `n` 是能放下的完整行数：

    > `windows(np.array([5, 2, 9, 7, 1]), 2)` 得到 `[[5, 2], [9, 7]]` 和 `[[2, 9], [7, 1]]`

    `X` 取前 $n \times \text{length}$ 个编号，`Y` 取同样多个编号，只是从后一个位置开始。`//` 是整除（第 11 章），`reshape`（第 06 章）可以把一串数折成若干行。
    """)
    return


@app.function
def windows(ids, length):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "windows", windows, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""先算行数。要留出一个编号给最后一个答案，所以是 `len(ids) - 1` 个编号，用 `//` 除以 `length`。然后是 `ids` 的两个切片，各自 `reshape` 成 `(n, length)`。""",
        r"""一共 4 行。`X` 的切片在 `n * length` 处停下。`Y` 的切片从 1 开始，比 `X` 的切片晚一个位置停下。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 普通 RNN 为什么容易忘

    记忆只有在网络能*学会*该记什么时才有用。如果想让模型学到“开头的第一个词很重要”，训练时的**梯度**（反向传播传回来的更新信号）就必须从文本末尾一路传回第一个词。这又回到了第 03 章的链式法则和第 11 章的老问题：梯度每往回传一步，都要再乘以一个系数；连乘几十步之后，结果要么缩水到几乎为 0（梯度消失），要么越滚越大直接失控（梯度爆炸）。

    我们来实测一下。建一个刚初始化的 `nn.RNN`，喂给它随机的词向量，取出最后一步的记忆 `out[:, -1]`，把里面的数平方后求和得到一个标量，再调用 `.backward()` 反向传播，看看不同长度的文本里，最终有多大的梯度能传回到第 0 个词的向量上。（给输入 `X` 加上 `requires_grad=True`，PyTorch 就会顺便把它的梯度也算出来，存在 `X.grad` 里；`X.grad[:, 0]` 就是传到第一个词上的梯度。）
    """)
    return


@app.function
def first_word_blame(layer, length):
    torch.manual_seed(0)
    X = torch.rand(4, length, 16, requires_grad=True)
    out = layer(X)[0]
    torch.sum(out[:, -1] ** 2).backward()
    return torch.mean(torch.abs(X.grad[:, 0])).item()


@app.cell
def _():
    torch.manual_seed(0)
    plain = nn.RNN(16, 64, batch_first=True)
    plain_blame = []
    for length_try in [2, 10, 30, 60]:
        plain_blame.append((length_try, first_word_blame(plain, length_try)))
    plain_blame
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每个括号里是“(文本长度, 传到第一个词上的平均梯度)”：2 个词时大约 0.1，10 个词时降到 0.0007，30 个词时只有 0.00000002，到 60 个词时更是掉到了 `4e-15`（小数点往左移 15 位）。对 60 个词文本的最后一步记忆来说，第一个词几乎彻底“隐身”了——梯度传不回去，普通 RNN 就根本学不会跨长句记东西。

    ## LSTM：带门控的记忆

    普通 RNN 的症结在于：每读一个词，它都要把旧记忆塞进 `tanh` 里整份*重写*一遍。**LSTM**（long short-term memory，长短期记忆网络）把状态拆成了两个向量——**记忆单元**（cell）$c$ 和隐藏状态 $h$。你可以把 $c$ 想成一本**长期笔记本**（只做加减更新，不轻易整本重写），而 $h$ 是当前递出去给下一步用的**短期摘要**。

    只靠**加法**来更新 $c$，背后正是第 11 章残差连接 $x + f(x)$ 的思路：因为加法的导数是 1，梯度沿 $c$ 的加法通路往回传时不会被连乘压扁。（有趣的是，LSTM 其实比残差网络早得多，1997 年就提出来了。）

    为了精细控制这本“长期笔记本”，LSTM 引入了**门**（gate）。一个门就是一个数值全在 0 到 1 之间的向量，用第 12 章学过的 sigmoid（$\sigma$）算出来，再和另一个向量逐元素相乘：0 表示彻底关门挡住，1 表示完全放行。LSTM 一共有三个门，外加一份准备写入的新内容——**候选内容** $n$。这四样都看同一份输入：把当前词向量 $x$ 和上一步的短期摘要 $h$ 并排拼在一起（下面简称 `both`），各自经过一层独立的线性变换：

    | 名称 | 计算方式 | 回答的问题 |
    |---|---|---|
    | 遗忘门 $f$ | $\sigma(L_0(\text{both}))$ | 记忆单元里的每个数保留多少？ |
    | 输入门 $i$ | $\sigma(L_1(\text{both}))$ | 候选内容写入多少？ |
    | 输出门 $o$ | $\sigma(L_2(\text{both}))$ | 记忆单元有多少输出到 $h$ 里？ |
    | 候选内容 $n$ | $\tanh(L_3(\text{both}))$ | 这次打算写入什么新信息？ |

    接下来一步同时更新记忆单元 $c$ 和短期摘要 $h$，其中 $\times$ 表示逐元素相乘：

    > $$\text{新的 } c = f \times c + i \times n$$

    > $$\text{新的 } h = o \times \tanh(\text{新的 } c)$$

    奥妙就在第一行。只要遗忘门 $f$ 接近 1、输入门 $i$ 接近 0，记忆单元 $c$ 就能几乎原样往后传，梯度也**有可能**顺着这条通路原样往回传。“有可能”这三个字很关键，马上就会看到原因。

    ## 动手算：门在做什么

    记忆单元里的一个数 $c = 0.8$，候选内容 $n = -0.5$。算出三种门的设置下新的 $c$：$f = 1, i = 0$；然后 $f = 0, i = 1$；然后 $f = 0.5, i = 0.5$。最后，不管记忆单元里存着什么，输出门 $o = 0$ 时 $h$ 是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    gate_boxes = hand.boxes(['f = 1、i = 0 时的 c', 'f = 0、i = 1 时的 c', 'f = 0.5、i = 0.5 时的 c', 'o = 0 时的 h'], step=0.01)
    return (gate_boxes,)


@app.cell(hide_code=True)
def _(gate_boxes):
    hand.marked(gate_boxes, {'f = 1、i = 0 时的 c': 0.8, 'f = 0、i = 1 时的 c': -0.5, 'f = 0.5、i = 0.5 时的 c': 0.15, 'o = 0 时的 h': 0}, {
        'f = 1、i = 0 时的 c': r"""$1 \times 0.8 + 0 \times (-0.5)$：全部留下，什么也不写""",
        'f = 0、i = 1 时的 c': r"""$0 \times 0.8 + 1 \times (-0.5)$：全部忘掉，写入候选内容""",
        'f = 0.5、i = 0.5 时的 c': r"""0.8 的一半，加上 $-0.5$ 的一半""",
        'o = 0 时的 h': r"""在第二个公式里，$o$ 乘的是后面的全部""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    所以门可以留住记忆单元里的一个数，可以把它换掉，也可以把新旧混在一起；它们还能把记忆单元暂时藏起来不露在 $h$ 里，却不擦掉它。记忆单元的每个位置都有自己的门控数值，因此一个位置可以继续保留，另一个位置同时被替换掉。

    /// admonition | 🧰 工具箱：并排拼接，以及放在列表里的层
    - `torch.cat([a, b], dim=1)` 沿着一个**已有的**维度把张量拼起来。形状为 `(2, 3)` 和 `(2, 4)` 的两张表并排放，会拼成一张 `(2, 7)` 的表。（`torch.stack` 新建一个维度，`torch.cat` 把已有维度拉长。）
    - 在 Python 里，PyTorch 的层对象也可以像数字或字符串一样放进列表。如果 `layers` 是一个装着 `nn.Linear` 层的列表，`layers[0](a)` 就是取出第一个层来处理 `a`。
    - `torch.sigmoid(a)` 对张量里的每个数计算 $\sigma$。
    ///
    """)
    return


@app.cell
def _():
    side_demo = torch.cat([torch.zeros(2, 3), torch.ones(2, 4)], dim=1)
    layers_demo = [nn.Linear(7, 5), nn.Linear(7, 1)]
    side_demo.shape, layers_demo[0](side_demo).shape, layers_demo[1](side_demo).shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：lstm_step

    写 `lstm_step(x, h, c, L)`。`x` 的形状是 `(batch, d)`，`h` 和 `c` 的形状是 `(batch, m)`。`L` 是四个 `nn.Linear` 层组成的列表，顺序和表里一样，从 $L_0$ 到 $L_3$；每层接收 `(batch, d + m)`，输出 `(batch, m)`。函数返回一对结果：新的 $h$ 和新的 $c$。

    `both` 就是把 `x` 和 `h` 并排拼在一起，`x` 在前。接着照着表格和两个公式算就行。注意顺序：新的 $h$ 要用*新的* $c$ 来算。

    其中一项检查会用完全相同的参数，把你写的这一步和 PyTorch 内置的 LSTM 做比对。
    """)
    return


@app.function
def lstm_step(x, h, c, L):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "lstm_step", lstm_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""一共八行：`both` 一行，表格的每一行各一行，两个公式各一行，再加 `return`。三个门用 `torch.sigmoid`，候选内容用 `torch.tanh`。`*` 是逐元素相乘。""",
        r"""前两行是 `both = torch.cat([x, h], dim=1)` 和 `f = torch.sigmoid(L[0](both))`。先把新的记忆单元存回 `c`，再去算 `h`；返回时按 `h, c` 的顺序。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在对一个刚初始化的 `nn.LSTM` 做同样的测量，依然输出“(文本长度, 传到第一个词上的平均梯度)”。
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    lstm_demo = nn.LSTM(16, 64, batch_first=True)
    lstm_blame = []
    for length_lstm in [2, 10, 30, 60]:
        lstm_blame.append((length_lstm, first_word_blame(lstm_demo, length_lstm)))
    lstm_blame
    return (lstm_demo,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    结果一样糟：60 个词时大约只有 `8e-14`。这非常值得停下来想一想。刚初始化的 LSTM 参数是随机的，遗忘门算出来都在 0.5 左右——每读一个词，记忆单元就被砍掉一半，一半的一半再一半，60 步之后自然所剩无几。

    但 LSTM 和普通 RNN 的本质区别在于：LSTM 有一个专门控制遗忘程度的“开关”，普通 RNN 没有。只要把遗忘门的偏置调大，$f$ 就会接近 1。我们来手动试一下：PyTorch 内部把 `i, f, g, o`（输入门、遗忘门、候选内容、输出门）四个门的偏置首尾拼成一条长度 `4 * 64` 的向量，所以第二段 `[64:128]` 正好对应遗忘门 `f`。把它设成 3 后，$\sigma(3) \approx 0.95$。（手动改参数要写在 `torch.no_grad()` 里，告诉 PyTorch 这次改动不用记录梯度。）
    """)
    return


@app.cell
def _(lstm_demo):
    with torch.no_grad():
        lstm_demo.bias_ih_l0[64:128] = 3.0
    open_blame = []
    for length_open in [2, 10, 30, 60]:
        open_blame.append((length_open, first_word_blame(lstm_demo, length_open)))
    open_blame
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    60 个词时梯度一下子回到约 0.04：信号不再随长度衰减，比刚才强了几千亿倍！普通 RNN 没有这样简单的开关，因为它的记忆每一步都要被迫穿过一次 `tanh` 彻底改写。而在 LSTM 里，模型在训练中可以自己学会：遇到值得长期保留的线索就把遗忘门打开，无关的细节就关门丢掉。门控的参数和其他权重一样，都是通过训练自动学出来的。

    ## 训练语言模型

    整个模型的结构很清晰：词的编号进来，先经由一张词向量表转成数字向量（第 12 章的 `nn.Embedding`），再交给 `nn.LSTM` 逐词阅读，最后由 `nn.Linear` 在每个位置上为词表里的每个词打分：

    | 步骤 | 结果 | 形状 |
    |---|---|---|
    | 编号 | | `(batch, length)` |
    | `table` 查出每个编号 | 每个词一个向量 | `(batch, length, d)` |
    | `lstm` 从 `memory` 出发读它们 | 每个词之后的短期记忆，以及最终状态 | `(batch, length, m)`，以及最终状态 |
    | `out` 给词表里的每个词打分 | 每个位置上的分数 | `(batch, length, vocab)` |

    调用 `nn.LSTM` 时传入两个输入：词向量序列，以及初始记忆。它会返回一对结果：每个词之后的短期记忆 $h$，以及最终状态（它本身是一个二元组 `(h, c)`，我们直接原样传下去就行）。如果初始记忆传 `None`（Python 里表示“空”的关键字），它就会自动从全零开始。

    ## 练习 5：reader_forward

    照着表格写 `reader_forward(ids, memory, table, lstm, out)`。`table`、`lstm` 和 `out` 是那三个层，`memory` 要么是 `None`，要么是之前某次调用得到的最终状态。函数返回一对结果：分数，以及最终状态。
    """)
    return


@app.function
def reader_forward(ids, memory, table, lstm, out):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "reader_forward", reader_forward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三次调用，每一次都用上一次的结果：`table`、`lstm`、`out`。中间那个返回的是一对，所以要给两半各起一个名字，像 `a, b = ...` 这样。""",
        r"""一共 2 行。第一行是 `states, memory = lstm(table(ids), memory)`。第二行返回分数和记忆。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面把整个模型封装成一个 PyTorch 模块。它包含上面那三个层，`forward` 方法直接调用你写的 `reader_forward`。`forward` 定义里的 `memory=None` 是**默认参数**（default）：调用时如果不传 `memory`，它就自动取 `None`；而后面生成文本的 `write` 函数则会把上一步留下的状态接着传进去。数据沿用第 13 章的故事集：同样的词表，同样的编号。加载数据的单元格需要运行几秒钟。
    """)
    return


@app.cell
def _():
    class Reader(nn.Module):
        def __init__(self, vocab_size, d, m):
            super().__init__()
            self.table = nn.Embedding(vocab_size, d)
            self.lstm = nn.LSTM(d, m, batch_first=True)
            self.out = nn.Linear(m, vocab_size)

        def forward(self, ids, memory=None):
            return reader_forward(ids, memory, self.table, self.lstm, self.out)
    return (Reader,)


@app.function
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.function
def story_ids(story_list, index):
    ids = []
    for story in story_list:
        for word in to_words(story):
            ids.append(index.get(word, index["<unk>"]))
        ids.append(index["<end>"])
    return np.array(ids)


@app.cell
def _():
    train_stories, test_stories = stories()
    counts = {}
    for word in to_words(" ".join(train_stories)):
        counts[word] = counts.get(word, 0) + 1
    vocab = sorted(counts, key=counts.get, reverse=True)[:2000] + ["<unk>", "<end>"]
    index = {}
    for i in range(len(vocab)):
        index[vocab[i]] = i
    train_ids = story_ids(train_stories, index)
    test_ids = story_ids(test_stories, index)
    len(vocab), len(train_ids), len(test_ids)
    return index, test_ids, train_ids, vocab


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `train_reader` 沿用第 10 章的训练循环，只多了两处改动：摊平和梯度裁剪；`score_reader` 则用第 13 章的 `perplexity` 在测试故事上打分。

    /// admonition | 🧰 工具箱：三个小工具
    - 在 `reshape` 里，有一个维度可以写成 `-1`，意思是“让程序自动算这个大小”：一个有 24 个数的张量，`reshape` 成 `(-1, 4)` 会得到 6 行，`reshape` 成 `(-1)` 则拉成一条长向量。
    - `F.cross_entropy(S, answers, reduction="none")` 返回每个样本各自的交叉熵损失（意外程度），而不是求平均。
    - `np.concatenate(arrays)` 把一个由 NumPy 数组组成的列表首尾相接，拼成一个数组。
    ///

    **摊平。** `F.cross_entropy` 期待一张二维分数表和一条一维答案向量，但我们的模型给每段文本都输出了一张表。`S.reshape(-1, size)` 把它们首尾拼成一张大表，每个词占一行。

    **梯度裁剪。** 反向传播经过长序列时，梯度不仅可能消失，也可能突然爆炸；哪怕某一步冒出一个极大的梯度，都会把参数一下子更新得面目全非。`nn.utils.clip_grad_norm_(knobs, 1.0)` 把所有参数的梯度拼成一个大向量并算出它的总长度；如果超过 1，就按比例把所有梯度一起缩小，让总长度正好等于 1。更新方向保持不变，只是给步子大小封顶——这就是**梯度裁剪**（gradient clipping）。

    在 `score_reader` 里，加负号把意外程度变回 `perplexity` 需要的对数概率。`S, _memory = ...` 表示只留分数，忽略返回的状态。
    """)
    return


@app.function
def perplexity(log_probs):
    return np.exp(-np.mean(log_probs))


@app.function
def train_reader(model, X, Y, epochs, batch_size, rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    size = model.out.out_features
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            S, _memory = model(X[batch])
            loss = F.cross_entropy(S.reshape(-1, size), Y[batch].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return loss.item()


@app.function
def score_reader(model, X, Y):
    size = model.out.out_features
    log_probs = []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            S, _memory = model(X[start:start + 64])
            surprise = F.cross_entropy(S.reshape(-1, size), Y[start:start + 64].reshape(-1), reduction="none")
            log_probs.append(-surprise.numpy())
    return perplexity(np.concatenate(log_probs))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：送进损失的形状

    读一读 `train_reader` 里计算 `loss` 的那一行。我们训练时每批 64 段文本，每段 32 个词，词表有 2,002 项，所以 `S` 的形状是 `(64, 32, 2002)`。`S.reshape(-1, size)` 之后，分数表有多少行、多少列？`Y[batch].reshape(-1)` 里有多少个答案？
    """)
    return


@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['分数表的行数', '分数表的列数', '答案的个数'])
    return (shape_boxes,)


@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'分数表的行数': 2048, '分数表的列数': 2002, '答案的个数': 2048}, {
        '分数表的行数': r"""每个词一行：64 段文本，每段 32 个词""",
        '分数表的列数': r"""`size` 是每个词的分数个数，词表里每一项一个""",
        '答案的个数': r"""每一行分数对应一个答案""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    所以训练的每一步，都是同时回答 2,048 个“下一个词是什么”的问题，每个问题的打分方式和第 10 章识别一个手写数字一样。

    按下按钮，模型会用全部 190 万个词训练两轮（epoch），记忆向量长度设为 128。这是本课程里第一次真正需要等一会儿的训练：在普通笔记本上大约需要 2 到 5 分钟。训练完成前，依赖它结果的下游单元格会显示 `Ancestor stopped`，这是正常现象。如果之后重新运行训练单元格，需要再点一次按钮。跑完后会输出最后一轮的训练损失，以及测试集困惑度。训练时我们把文本切成 32 个词的片段，测试时用更长的 64 词片段来打分。
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="训练 LSTM（2 到 5 分钟）")
    train_button
    return (train_button,)


@app.cell
def _(Reader, test_ids, train_button, train_ids, vocab):
    mo.stop(not train_button.value, mo.md("*按下按钮开始训练。*"))
    X_np, Y_np = windows(train_ids, 32)
    X_test_np, Y_test_np = windows(test_ids, 64)
    torch.manual_seed(0)
    model = Reader(len(vocab), 64, 128)
    last_loss = train_reader(model, torch.tensor(X_np), torch.tensor(Y_np), 2, 64, 0.003)
    lstm_score = score_reader(model, torch.tensor(X_test_np), torch.tensor(Y_test_np))
    last_loss, lstm_score
    return lstm_score, model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第二个数字是测试困惑度，应该在 31 左右，明显优于三元模型的 38。（第一个数字是最后一批训练数据上的平均交叉熵损失，不是测试故事上的。）下面是排行榜，你的 LSTM 已经在上面了：
    """)
    return


@app.cell(hide_code=True)
def _(lstm_score):
    record("lstm", lstm_score)
    show(LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 用温度写作

    要让模型自己写故事，就把它刚生成的词再喂回给它，同时把记忆状态一路带下去，像第 13 章那样按概率采样。这里多加了一个可调的旋钮：在送进 softmax 之前，先把所有分数除以一个正数，叫做**温度**（temperature）：

    - 温度等于 1：原样保留模型的真实概率；
    - 温度小于 1：分数的差距被拉大，原本热门的词概率更高——写出来的句子更稳妥，但也更保守平淡；
    - 温度大于 1：分数的差距被缩小，概率分布变平——冷门词也有机会被抽到，文本更有想象力，也更容易跑偏。

    ## 动手算：拧一拧温度旋钮

    假设只有两个候选词，原始分数是 $[1, 0]$。第 12 章说过，只有两个分数时，softmax 给第一个词的概率正好是 $\sigma(\text{两数之差})$。这里提供三个查表值：$\sigma(0.5) = 0.62$，$\sigma(1) = 0.73$，$\sigma(2) = 0.88$。

    做 softmax 之前，先把两个分数都除以温度：
    - 温度为 `1` 时，分数保持 $[1, 0]$；
    - 除以温度 `0.5` 相当于分数翻倍成 $[2, 0]$；
    - 除以温度 `2` 相当于分数减半成 $[0.5, 0]$。

    算一算：温度分别为 1、0.5、2 时，第一个词分到的概率各是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    heat_boxes = hand.boxes(['温度为 1 时的份额', '温度为 0.5 时的份额', '温度为 2 时的份额'], step=0.01)
    return (heat_boxes,)


@app.cell(hide_code=True)
def _(heat_boxes):
    hand.marked(heat_boxes, {'温度为 1 时的份额': 0.73, '温度为 0.5 时的份额': 0.88, '温度为 2 时的份额': 0.62}, {
        '温度为 1 时的份额': r"""分数还是 $[1, 0]$，所以差是 1""",
        '温度为 0.5 时的份额': r"""除以 0.5，分数翻倍成 $[2, 0]$，差是 2""",
        '温度为 2 时的份额': r"""除以 2，分数减半成 $[0.5, 0]$，差是 0.5""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 6：sample_word

    写 `sample_word(scores, temperature)`。`scores` 是一个张量，每个词一个分数，形状是 `(vocab,)`；`temperature` 是一个大于 0 的数。它返回一个词编号，类型为普通的 Python 整数：先把分数除以温度，再做 softmax 得到概率，按这些概率随机抽一个。

    /// admonition | 🧰 工具箱：现成的 softmax
    `torch.softmax(a, dim=0)` 就是第 07 章的 softmax，直接用在张量上：它把一条分数向量变成加起来等于 1 的概率。`dim=0` 指明沿第 0 维计算（一维向量也只有这一维）。`torch.softmax(torch.tensor([2.0, 1.0, 0.0]), dim=0)` 大约是 `[0.67, 0.24, 0.09]`。
    ///

    其余的工具你都用过：`torch.multinomial(p, 1)` 按 `p` 里的概率抽 1 个编号（第 12 章），`.item()` 把单元素张量变成普通数字（第 10 章）。分数是 `[1.0, 0.0]`、温度是 0.1 时，几乎每次调用都应该抽到 0。
    """)
    return


@app.function
def sample_word(scores, temperature):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.function
def write(model, start, count, temperature, index, vocab):
    words = [index[start]]
    ids = torch.tensor([[index[start]]])
    memory = None
    with torch.no_grad():
        for step in range(count):
            S, memory = model(ids, memory)
            words.append(sample_word(S[0, -1], temperature))
            ids = torch.tensor([[words[-1]]])
    return " ".join([vocab[i] for i in words])


@app.cell(hide_code=True)
def _():
    check("14", "sample_word", sample_word, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""分两步：先算概率，再抽取。要在 softmax 之前除以温度，不是之后。""",
        r"""一共 2 行。第一行是 `p = torch.softmax(scores / temperature, dim=0)`。第二行从 `p` 里抽 1 个编号，末尾加 `.item()`。""",
    ], LOCALE)
    return


@app.cell
def _(index, model, vocab):
    torch.manual_seed(1)
    write(model, "once", 40, 0.8, index, vocab)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    和第 13 章二元模型生成的文本比一比：现在已经有不少片段读起来像模像样，故事里的角色也能连贯出现一阵子。把上面单元格里的 `0.8` 改成 `0.3` 或 `1.5` 再跑一次，感受一下温度带来的变化。

    留意 `write` 里传递的 `memory`：不管故事写到多长，模型对前文的全部记忆都得硬塞进这 128 个数字里。

    ## 接下来

    这种“瓶颈挤压”正是 LSTM 的极限。要求一个固定长度的向量装下整段前文，就像读完一整本书再凭记忆把它翻译出来。在**第 15 章**，你会让模型在需要时随时回头看前文的任意一个词。这个机制叫**注意力**（attention），它是所有现代语言模型的核心。

    ## 你现在会做的事

    - 解释循环神经网络如何用同一组参数一步步读完任意长度的文本
    - 手算 RNN 的一步更新，写出代码并跑通整段文本，再和 `nn.RNN` 核对结果
    - 把一串词编号切成训练用的输入 `X` 和下一个词答案 `Y`
    - 实测梯度随文本长度衰减的现象，理解普通 RNN 为什么容易忘
    - 解释 LSTM 的记忆单元 $c$、短期摘要 $h$ 和三个门如何保护记忆与梯度，并照着公式写出它的一步
    - 用梯度裁剪防止梯度爆炸
    - 训练一个 LSTM 语言模型、把它放上排行榜，并用温度控制生成文本的风格

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** LSTM 由 Hochreiter 和 Schmidhuber 于 1997 年提出。训练时给遗忘门一个正的初始偏置是常见技巧。*GRU* 是它结构更精简的近亲，只有两个门。在长序列上做反向传播称为*随时间反向传播*（backpropagation through time）。我们训练时把文本切成 32 个词的片段，梯度最多往回传 32 步，这种做法叫*截断*（truncation）。
    """)
    return


if __name__ == "__main__":
    app.run()
