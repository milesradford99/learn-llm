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

    第 13 章数数的模型只记得一两个词，还把 `dog`（狗）和 `cat`（猫）当成完全陌生的两个词。你读东西可不是这样。你逐词读过一个句子，每读一个词就更新一下脑子里的想法。读完一段话，你仍然知道在讲谁，却没有存下任何一串原文。

    **循环神经网络**（recurrent neural network，RNN）也是这么做的。它保留一个数字向量，也就是它的**记忆**（memory），每读一个词就更新一次。这一章造出一个 RNN，弄清它为什么会忘，再用 **LSTM** 解决这个问题。在 2010 年代的大部分时间里，语言模型领域都由这种设计领跑。

    ## 一个新的弯折：tanh

    记忆要一遍又一遍地经过同一个步骤，所以它不能无限增长。ReLU 没有上限。对记忆，我们用一个把每个数都压进 −1 到 1 之间的弯折，它叫 **tanh**：
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
    重要的是它的形状：在 0 附近接近 $x$ 本身，遇到大数就逐渐贴近 +1 和 −1。在 PyTorch 里写作 `torch.tanh`。下面 0.1 基本还是 0.1，而 5 被压到了将近 1：
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

    设 $x$ 是当前词的向量（第 12 章的想法），$h$ 是到目前为止的记忆，大小为 $m$。新的记忆把两者结合起来：

    > $$\text{新的 } h = \tanh(x \cdot W_x + h \cdot W_h + b)$$

    $W_x$ 把词变成记忆大小的数字，$W_h$ 把旧记忆带到下一步，$b$ 是常见的偏置。这就是第 08 章的层，只多了一个变化：它的输入有一部分是它自己上一次的输出。“循环”的意思就是绕回来。

    下面是这一步连用三次，读 `the dog ran` 的样子。记忆从全零开始，每个箭头都是同一个步骤、同一组旋钮，只是喂进去的词不同：
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
    每个词用的都是**同一组**旋钮，这和第 11 章的卷积核一样：卷积核在图片的各个位置上滑动，这一步在文本的各个位置上滑动。所以不管文本是 3 个词还是 3,000 个词，网络的旋钮一样多。

    ## 动手算：记忆的两步

    看最小的情况：一个词是一个数，记忆也是一个数，旋钮是 $W_x = 2$、$W_h = 1$、$b = -1$。记忆从 0 开始。第一个词是 $x = 1$，第二个词是 $x = 0$。你会用到这张表：

    | $z$ | −1 | −0.24 | 0 | 0.24 | 1 | 2 |
    |---|---|---|---|---|---|---|
    | $\tanh(z)$ | −0.76 | −0.24 | 0 | 0.24 | 0.76 | 0.96 |

    对每个词，先算出送进 tanh 的那个数，再算出新的记忆。
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
    第二个词是 0，它什么也没带来。但记忆还是变了，而且新的值取决于第一个词：第一个词换一个，这里的数也会不同。这就是记忆在起作用。

    ## 练习 1：rnn_step

    写 `rnn_step(x, h, Wx, Wh, b)`，也就是上面的公式，一次处理一整批文本。

    | 输入 | 形状 | 它是什么 |
    |---|---|---|
    | `x` | `(batch, d)` | 每段文本当前的词，一个有 `d` 个数字的向量 |
    | `h` | `(batch, m)` | 每段文本到目前为止的记忆 |
    | `Wx` | `(d, m)` | 旋钮 |
    | `Wh` | `(m, m)` | 旋钮 |
    | `b` | `(m,)` | 旋钮 |

    它返回新的记忆，形状是 `(batch, m)`。你动手算的第一步就是其中一项检查：每个张量都只有一个数，`x = [[1.0]]`、`h = [[0.0]]`、`Wx = [[2.0]]`、`Wh = [[1.0]]`、`b = [-1.0]`，结果是 `[[0.76]]`。
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
        r"""这就是第 08 章的层，只是输入从一个变成了两个。一批数据和旋钮矩阵之间的 $\cdot$，在代码里是 `@`。""",
        r"""只要一行：`return`，后面用 `torch.tanh` 把整个和包起来。偏置要写在 tanh 里面。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 读完整段文本

    现在把上面那张图写成代码。词以 `X` 的形式送来，形状是 `(batch, length, d)`：批里每段文本，每个词一个向量。我们要取出每段文本的第 $t$ 个词，最后再把各个记忆拼回去。

    /// admonition | 🧰 工具箱：拆开一个维度，拼出一个维度
    - 索引里单独一个 `:` 表示“这一维全要”。对形状为 `(batch, length, d)` 的 `X`，`X[:, 0]` 是每段文本的第一个词，形状是 `(batch, d)`。
    - `torch.stack(tensors, dim=1)` 沿着一个**新的**第 1 维，把一个由形状相同的张量组成的列表拼起来。三个形状为 `(batch, m)` 的张量，变成一个形状为 `(batch, 3, m)` 的张量。
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

    写 `run_rnn(X, Wx, Wh, b)`。`X` 的形状是 `(batch, length, d)`，旋钮和练习 1 一样。它返回每个词之后的记忆，形状是 `(batch, length, m)`：

    1. 记忆从零开始，形状是 `(batch, m)`。这两个大小都能从输入里读出来：`X.shape[0]` 和 `Wh.shape[0]`。
    2. 对每个位置 `t`，从第一个词到最后一个词，用一次 `rnn_step` 把第 `t` 个词和记忆变成新的记忆。把每个新的记忆都存进一个列表。
    3. 沿第 1 维把列表叠起来。

    还是你动手算的那个例子：一段两个词的文本，`X = [[[1.0], [0.0]]]`，旋钮不变，结果是 `[[[0.76], [-0.24]]]`。
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
    PyTorch 的 `nn.RNN` 做的是同样的事。我们按第 10 章的方式检查一下：取一个随机的 `nn.RNN`，把它的旋钮复制进你的函数，然后比较。（这个单元格需要先完成练习 2。）`nn.RNN(4, 6, batch_first=True)` 每个词接收 4 个数字，记忆大小是 6；`batch_first=True` 表示形状里批在最前面，和我们的一样。PyTorch 的权重是转置存放的，和 `nn.Linear` 一样，并且有两个偏置，直接相加即可。调用它会返回一对：每个词之后的记忆和最后的记忆，所以 `[0]` 取第一个。
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
    大约 `1e-07`，也就是 0.0000001：小数点往左移 7 位，和第 09 章一样。这只是舍入误差，所以两者一致。

    ## 从文本到样本

    训练需要输入和答案。对语言模型来说，每个位置的答案就是下一个词。把文本 `[5, 2, 9, 7, 1]` 切成长度为 2 的片段：

    ```
    X: 5 2      Y: 2 9
       9 7         7 1
    ```

    在每个位置上，`Y` 都是 `X` 错开一位。所以一小段文本就含有很多训练样本。最后一个编号 1 只当答案：它后面什么都没有，所以当不了输入。

    ## 动手算：切一段文本

    把 8 个编号 `[4, 8, 3, 6, 1, 9, 2, 5]` 切成长度为 3 的片段。最后一个答案需要比输入多一个编号，那么能放下几个完整的行？`X` 的最后一个编号是什么？`Y` 的第一个和最后一个编号呢？
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

    `X` 是前 $n \times \text{length}$ 个编号，`Y` 是同样多个编号，只是从后一个位置开始。`//` 是整除（第 11 章），`reshape`（第 06 章）把一串数折成若干行。
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
    ## 普通 RNN 为什么会忘

    记忆只有在网络能*学会*该记什么时才有用。要学到第一个词很重要，责任必须从文本末尾一路传回第一个词。这又是第 03 章的链式法则，也是第 11 章的问题：责任每一步都要相乘，一长串相乘的结果，要么衰减，要么爆炸。

    我们来测量一下。让一个新的、没训练过的 `nn.RNN` 读随机的词向量，取最后的记忆，走一次反向。然后看看不同长度的文本里，有多少责任到达了第一个词的向量。`out[:, -1]` 是最后的记忆。用它算出的任何一个数字都可以拿来追责；我们取平方再相加。（`X.grad[:, 0]` 是第 0 个词上的责任。给输入设 `requires_grad=True`，就连它也记账。）
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
    每一对是长度和第一个词上的责任：2 个词后大约 0.1，10 个词后 0.0007，30 个词后 0.00000002，60 个词后大约 `4e-15`，也就是小数点往左移 15 位。对 60 个词文本的最后记忆来说，第一个词简直等于不存在。所以普通 RNN 学不会把东西带过一段长文本。

    ## LSTM：带门的记忆

    问题在于，普通的那一步每次都经过 tanh *替换*记忆。**LSTM**（long short-term memory，长短期记忆）多保留一个记忆向量 $c$，叫**细胞**（cell），而且只靠**相加**来改变它。这就是第 11 章的 $x + f(x)$：相加给责任留出一条畅通的路。（LSTM 其实是更早的想法，出自 1997 年。）

    为了控制细胞，它用了**门**（gate）。门是一个 0 到 1 之间的数字向量，用第 12 章的 sigmoid 造出来，再和别的东西逐个位置相乘：0 表示挡住，1 表示放行。一共三个门，另外还有一个值：候选的新内容 $n$。这四样看的都是同一个东西，也就是并排放在一起的当前词和之前的 $h$，我们把它叫作 both。它们各有自己的一层旋钮：

    | 值 | 怎么造出来 | 回答的问题 |
    |---|---|---|
    | 遗忘门 $f$ | $\sigma(L_0(\text{both}))$ | 细胞里的每个数留下多少？ |
    | 输入门 $i$ | $\sigma(L_1(\text{both}))$ | 候选内容写入多少？ |
    | 输出门 $o$ | $\sigma(L_2(\text{both}))$ | 细胞有多少显示到 $h$ 里？ |
    | 候选内容 $n$ | $\tanh(L_3(\text{both}))$ | 可以写入什么？ |

    然后用一步改变细胞和记忆，其中 $\times$ 是逐个位置相乘：

    > $$\text{新的 } c = f \times c + i \times n$$

    > $$\text{新的 } h = o \times \tanh(\text{新的 } c)$$

    关键在第一行。当 $f$ 接近 1、$i$ 接近 0 时，细胞原样通过，责任也有可能原样通过。“有可能”这三个字很重要，后面就会看到。

    ## 动手算：门在做什么

    细胞里的一个数 $c = 0.8$，候选内容 $n = -0.5$。算出三种门的设置下新的 $c$：$f = 1, i = 0$；然后 $f = 0, i = 1$；然后 $f = 0.5, i = 0.5$。最后，不管细胞里是什么，输出门 $o = 0$ 时 $h$ 是多少？
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
    所以门可以留住细胞里的一个数，可以把它换掉，也可以把新旧混在一起；它们还能让细胞不显示在 $h$ 里，却不把它擦掉。细胞的每个位置都有自己的门数字，所以一个位置可以留着，另一个位置同时被换掉。

    /// admonition | 🧰 工具箱：并排拼接，以及放在列表里的层
    - `torch.cat([a, b], dim=1)` 沿着一个**已有的**维度把张量拼起来。形状为 `(2, 3)` 和 `(2, 4)` 的两张表并排放，成为一张形状为 `(2, 7)` 的表。（`torch.stack` 造出一个新维度，`torch.cat` 把已有的维度加长。）
    - 层和别的值一样，也是一个值，所以可以放进列表。如果 `layers` 是一个由 `nn.Linear` 层组成的列表，`layers[0](a)` 就是用第一个层处理 `a`。
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

    写 `lstm_step(x, h, c, L)`。`x` 的形状是 `(batch, d)`，`h` 和 `c` 的形状是 `(batch, m)`。`L` 是四个 `nn.Linear` 层组成的列表，顺序和表里一样，从 $L_0$ 到 $L_3$；每层接收 `(batch, d + m)`，给出 `(batch, m)`。它返回一对：新的 $h$ 和新的 $c$。

    both 就是把 `x` 和 `h` 并排放在一起，`x` 在前。然后照着表格和两个公式做。顺序很重要：新的 $h$ 是用*新的* $c$ 算出来的。

    其中一项检查会用同样的旋钮，把你的这一步和 PyTorch 自己的 LSTM 作比较。
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
        r"""一共八行：both 一行，表格的每一行各一行，两个公式各一行，再加 `return`。三个门用 `torch.sigmoid`，候选内容用 `torch.tanh`。`*` 是逐个位置相乘。""",
        r"""前两行是 `both = torch.cat([x, h], dim=1)` 和 `f = torch.sigmoid(L[0](both))`。先把新的细胞存回 `c`，再去算 `h`；返回时按 `h, c` 的顺序。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在对一个新的 `nn.LSTM` 做同样的测量，还是长度，然后是第一个词上的责任。
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
    一样糟：60 个词时大约 `8e-14`。这值得想一想。新的 LSTM 旋钮是随机的，所以遗忘门在 0.5 附近：每读一个词，细胞大约减半，一半的一半的一半，很快就没了。

    区别在于，LSTM 有一个调节这件事的旋钮，普通 RNN 没有。遗忘门的偏置很大时，$f$ 就接近 1。我们手动设置一下。PyTorch 把四个门的偏置叠在一个向量里。每个门 64 个数字，遗忘门的是位置 64 到 127。偏置取 3 时，$\sigma(3)$ 大约是 0.95。（往旋钮里存值要放在 `torch.no_grad()` 下面，这样它不会被记进账本。）
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
    60 个词时大约 0.04：责任不再衰减，比之前强了几千亿倍。普通 RNN 没有这样简单的旋钮，因为它的记忆每个词都要经过 tanh 重建一次。在 LSTM 里，训练可以为值得保留的东西打开门，其余的就关着。门也在学习，和其他旋钮一样。

    ## 训练语言模型

    整个模型的计划是这样的。词的编号进来。一张向量表把它们变成数字（第 12 章的 `nn.Embedding`），`nn.LSTM` 读它们，`nn.Linear` 在每个位置为词表里的每个词打分：

    | 步骤 | 结果 | 形状 |
    |---|---|---|
    | 编号 | | `(batch, length)` |
    | `table` 查出每个编号 | 每个词一个向量 | `(batch, length, d)` |
    | `lstm` 从 `memory` 出发读它们 | 每个词之后的记忆，以及最后的记忆 | `(batch, length, m)`，以及记忆 |
    | `out` 给词表里的每个词打分 | 每个位置上的分数 | `(batch, length, vocab)` |

    调用一个 `nn.LSTM` 要给两个输入：向量，以及出发时的记忆。它返回一对：每个词之后的记忆，和最后的记忆（它本身又是一对，$h$ 和 $c$，我们只管原样传下去）。如果出发时的记忆是 `None`，也就是 Python 里表示“什么都没有”的词，它就从零开始。

    ## 练习 5：reader_forward

    照着表格写 `reader_forward(ids, memory, table, lstm, out)`。`table`、`lstm` 和 `out` 是那三个层，`memory` 要么是 `None`，要么是之前某次调用得到的最后的记忆。它返回一对：分数，和最后的记忆。
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
    下面把模型写成一个 PyTorch 模块。它装着那三个层，它的 `forward` 就是你的函数。`forward` 第一行里的 `memory=None` 是一个**默认值**（default）：调用的人可以不给这个输入，这时它就是 `None`。后面的 `write` 会把上一次调用得到的记忆传进去。数据是第 13 章的：同样的词表，同样的编号。载入数据的那个单元格要几秒钟。
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
    `train_reader` 是第 10 章的循环，多了两处改动：摊平和裁剪。`score_reader` 用第 13 章的 `perplexity` 在测试故事上打分。

    /// admonition | 🧰 工具箱：三个小工具
    - 在 `reshape` 里，有一个大小可以写成 `-1`，意思是“能放下的那个数”：一个有 24 个数的张量，`reshape` 成 `(-1, 4)` 就有 6 行，`reshape` 成 `(-1)` 就成了一个长向量。
    - `F.cross_entropy(S, answers, reduction="none")` 给出每个样本各自的意外程度，而不是它们的均值。
    - `np.concatenate(arrays)` 把一个由 NumPy 数组组成的列表首尾相接，拼成一个数组。
    ///

    **摊平。** `F.cross_entropy` 要的是一张分数表和一个答案向量，而我们每段文本有一张表。`S.reshape(-1, size)` 把它们首尾相接，每个词一行。

    **裁剪。** 长链上的责任也可能爆炸，一个巨大的斜率就会把旋钮甩得老远。`nn.utils.clip_grad_norm_(knobs, 1.0)` 把所有斜率看成一个长向量，量出它的长度；如果超过 1，就把它们一起缩小，使这个长度刚好等于 1。方向不变，只限制大小。这叫**梯度裁剪**（gradient clipping）。

    在 `score_reader` 里，负号把意外程度变回 `perplexity` 要的对数概率。`S, _memory = ...` 留下分数，忽略记忆。
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
    所以训练的每一步，都是同时回答 2,048 个“下一个词是什么”的问题，每个问题的打分方式和第 10 章的一个数字一样。

    按下按钮，会用全部 190 万个词训练两遍，记忆大小是 128。这是这门课里第一次真正花时间的训练：在笔记本上大约 2 到 5 分钟。完成之前，需要它的结果的单元格会显示 `Ancestor stopped`，这是正常的。如果重新运行训练单元格，要再按一次按钮。单元格会显示最后的训练损失和测试困惑度。我们用 32 个词的片段训练，用 64 个词的较长片段打分。
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
    第二个数字是测试困惑度，应该在 31 左右，远低于三元模型的 38。（第一个数字是最后一批训练数据上的平均意外程度，不是测试故事上的。）下面是排行榜，你的 LSTM 已经在上面了：
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

    要写文本，就把模型自己写的上一个词喂给它，带着它的记忆往前走，像第 13 章那样采样。这里多了一个旋钮：在 softmax 之前把分数除以一个数，这个数叫**温度**（temperature）：

    - 温度为 1 时，给出模型真实的概率；
    - 小于 1 时，分数的差距拉大，本来就可能的词更可能了：文本更稳妥，也更平淡；
    - 大于 1 时，差距缩小，概率变平：文本更狂野。

    ## 动手算：拧一拧这个旋钮

    两个词的分数是 $[1, 0]$。第 12 章说过，只有两个分数时，softmax 给第一个的份额是 $\sigma$（两个分数的差）。这里有三个值：$\sigma(0.5) = 0.62$，$\sigma(1) = 0.73$，$\sigma(2) = 0.88$。温度为 1 时，第一个词的份额是多少？温度为 0.5 时呢？温度为 2 时呢？
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
        '温度为 0.5 时的份额': r"""除以 0.5，分数翻倍：$[2, 0]$""",
        '温度为 2 时的份额': r"""除以 2，分数减半：$[0.5, 0]$""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 6：sample_word

    写 `sample_word(scores, temperature)`。`scores` 是一个张量，每个词一个分数，形状是 `(vocab,)`；`temperature` 是一个大于 0 的数。它返回一个编号，是普通的 Python 数字：先把分数除以温度，再做 softmax 得到概率，按这些概率抽出来。

    /// admonition | 🧰 工具箱：现成的 softmax
    `torch.softmax(a, dim=0)` 就是第 07 章的 softmax，用在张量上：它把一个分数向量变成加起来等于 1 的概率。`dim=0` 说明沿哪一维做，而向量只有这一维。`torch.softmax(torch.tensor([2.0, 1.0, 0.0]), dim=0)` 大约是 `[0.67, 0.24, 0.09]`。
    ///

    其余的工具你都认识：`torch.multinomial(p, 1)` 按 `p` 里的概率抽 1 个编号（第 12 章），`.item()` 把只有一个数的张量变成普通数字（第 10 章）。分数是 `[1.0, 0.0]`、温度是 0.1 时，几乎每次调用都应该得到 0。
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
    和第 13 章二元模型的文本比一比。现在有些片段读起来像句子了，故事里的人物也能待上一会儿。把上面单元格里的 `0.8` 改成 `0.3` 或 `1.5`，再运行一次，看看有什么不同。

    注意 `write` 是怎么处理 `memory` 的：不管故事多长，模型对前文的全部了解都要塞进这 128 个数字。

    ## 接下来

    这种挤压正是 LSTM 的极限。要一个固定的向量装下整段文本，就像读完一本书再凭记忆把它翻译出来。在**第 15 章**，你会让模型在需要的时候回头看任何一个词。这个想法叫**注意力**（attention），它是所有现代语言模型的核心。

    ## 你现在会做的事

    - 解释循环的一步怎样在每个词上重复使用同一组旋钮
    - 亲手算出 RNN 的一步，把它写成代码并让它读完文本，再和 `nn.RNN` 核对
    - 把文本变成输入和下一个词的答案
    - 测量责任怎样随时间衰减
    - 解释 LSTM 的细胞和门怎样保护记忆和责任，并照着公式写出它的一步
    - 用裁剪限制爆炸的斜率
    - 训练语言模型，把它放上排行榜，并用温度生成文本

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** LSTM 出自 Hochreiter 和 Schmidhuber（1997）。给遗忘门一个正的起始偏置是常见做法。*GRU* 是它更简单的亲戚，有两个门。在长文本上走反向叫*随时间反向传播*（backpropagation through time）。我们的训练把文本切成 32 个词的片段，所以责任最多传 32 步，这叫*截断*（truncation）。
    """)
    return


if __name__ == "__main__":
    app.run()
