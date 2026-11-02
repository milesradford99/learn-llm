import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="13 · 记忆：RNN")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.leaderboard import record, show

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 13 · 记忆：RNN

    第 12 章数数的模型只记得一两个词，还把 `dog`（狗）和 `cat`（猫）当成完全陌生的两个词。你读东西可不是这样。你逐词读过一个句子，每读一个词就更新一下脑子里的想法。读完一段话，你仍然知道在讲谁，却没有存下任何一串原文。

    **循环神经网络**（recurrent neural network，RNN）也是这么做的。它保留一个数字向量，也就是它的**记忆**（memory），每读一个词就更新一次。这一章造出一个 RNN，弄清它为什么会忘，再用 **LSTM** 解决这个问题。在 2010 年代的大部分时间里，语言模型领域都由这种设计领跑。

    ## 一个新的弯折：tanh

    记忆要一遍又一遍地经过同一个步骤，所以它不能无限增长。ReLU 没有上限。对记忆，我们用一个把每个数都压进 −1 到 1 之间的弯折，它叫 tanh。重要的是它的形状：在 0 附近接近 `x` 本身，遇到大数就逐渐贴近 +1 和 −1。在 PyTorch 里写作 `torch.tanh`。下面 0.1 基本还是 0.1，而 5 被压到了将近 1：
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

    设 `x` 是当前词的向量（第 11 章的想法），`h` 是到目前为止的记忆，大小为 `m`。新的记忆把两者结合起来：

    > $$\text{new } h = \tanh(x \cdot W_x + h \cdot W_h + b)$$

    `Wx` 把词变成记忆大小的数字，`Wh` 把旧记忆带到下一步，`b` 是常见的偏置。这就是第 07 章的层，只多了一个变化：它的输入有一部分是它自己上一次的输出。“循环”的意思就是绕回来。每个词用的都是**同一组**旋钮，这和第 10 章的卷积核一样：卷积核在图片的各个位置上滑动，这一步在文本的各个位置上滑动。

    ## 练习 1：rnn_step

    写 `rnn_step(x, h, Wx, Wh, b)`，一次处理一整批：`x` 的形状是 `(batch, d)`，`h` 的形状是 `(batch, m)`。

    ```
    return torch.tanh(x @ Wx + h @ Wh + b)
    ```
    """)
    return


@app.function
def rnn_step(x, h, Wx, Wh, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "rnn_step", rnn_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：run_rnn

    现在读完整段文本。`X` 的形状是 `(batch, length, d)`：批里每段文本，每个词一个向量。记忆从全零开始。`X[:, t]` 取出每段文本的第 `t` 个词（`:` 表示“这一维全要”），`torch.stack(list, dim=1)` 沿着新的第 1 维把一个由张量组成的列表拼起来。

    写 `run_rnn(X, Wx, Wh, b)`，返回每个词之后的记忆，形状是 `(batch, length, m)`：

    1. 两行：`h = torch.zeros(X.shape[0], Wh.shape[0])` 和 `states = []`
    2. `for t in range(X.shape[1]):`
    3. 里面两行：`h = rnn_step(X[:, t], h, Wx, Wh, b)` 和 `states.append(h)`
    4. 循环结束后，`return torch.stack(states, dim=1)`
    """)
    return


@app.function
def run_rnn(X, Wx, Wh, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "run_rnn", run_rnn, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    PyTorch 的 `nn.RNN` 做的是同样的事。我们按第 09 章的方式检查一下：取一个随机的 `nn.RNN`，把它的旋钮复制进你的函数，然后比较。（这个单元格需要先完成练习 2。）`nn.RNN(4, 6, batch_first=True)` 每个词接收 4 个数字，记忆大小是 6；`batch_first=True` 表示形状里批在最前面，和我们的一样。PyTorch 的权重是转置存放的，和 `nn.Linear` 一样，并且有两个偏置，直接相加即可。调用它会返回一对：每个词之后的记忆和最后的记忆，所以 `[0]` 取第一个。
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
    大约 `1e-07`，也就是 0.0000001：小数点往左移 7 位，和第 08 章一样。这只是舍入误差，所以两者一致。

    ## 从文本到样本

    训练需要输入和答案。对语言模型来说，每个位置的答案就是下一个词。把文本 `[5, 2, 9, 7, 1]` 切成长度为 2 的片段：

    ```
    X: 5 2      Y: 2 9
       9 7         7 1
    ```

    在每个位置上，`Y` 都是 `X` 错开一位。所以一小段文本就含有很多训练样本。

    ## 练习 3：windows

    写 `windows(ids, length)`，把一个 NumPy 编号数组切成行。我们要比用到的多留一个编号，给最后一个答案：

    1. `n = (len(ids) - 1) // length`，这是完整的行数
    2. `X = ids[:n * length].reshape(n, length)`
    3. `Y = ids[1:n * length + 1].reshape(n, length)`
    4. `return X, Y`
    """)
    return


@app.function
def windows(ids, length):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "windows", windows, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 普通 RNN 为什么会忘

    记忆只有在网络能*学会*该记什么时才有用。要学到第一个词很重要，责任必须从文本末尾一路传回第一个词。这又是第 03 章的链式法则，也是第 10 章的问题：责任每一步都要相乘，一长串相乘的结果，要么衰减，要么爆炸。

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

    问题在于，普通的那一步每次都经过 tanh *替换*记忆。**LSTM**（long short-term memory，长短期记忆）多保留一个记忆向量 `c`，叫**细胞**（cell），而且只靠**相加**来改变它。这就是第 10 章的 `x + f(x)`：相加给责任留出一条畅通的路。（LSTM 其实是更早的想法，出自 1997 年。）

    为了控制细胞，它用了**门**（gate）。门是一个 0 到 1 之间的数字向量，用第 11 章的 sigmoid 造出来，再和别的东西逐个位置相乘：0 表示挡住，1 表示放行。一共三个门，都看当前词和之前的 `h`：

    | 门 | 回答的问题 |
    |---|---|
    | 遗忘门 `f` | 细胞里的每个数留下多少？ |
    | 输入门 `i` | 新候选内容写入多少？ |
    | 输出门 `o` | 细胞有多少显示到 `h` 里？ |

    一步的计算如下，其中 `both` 是把词和旧的 `h` 并排放在一起，`n` 是候选的新内容：

    > $$c = f \times c + i \times n$$

    > $$h = o \times \tanh(c)$$

    关键在第一行。当 `f` 接近 1、`i` 接近 0 时，细胞原样通过，责任也有可能原样通过。“有可能”这三个字很重要，后面就会看到。

    ## 练习 4：lstm_step

    写 `lstm_step(x, h, c, L)`。`L` 是四个 `nn.Linear` 层组成的列表，`[forget, input, output, new]`，每层都以 `both` 为输入。`torch.cat([a, b], dim=1)` 把两张表并排拼起来。

    1. `both = torch.cat([x, h], dim=1)`
    2. `f = torch.sigmoid(L[0](both))`
    3. `i = torch.sigmoid(L[1](both))`
    4. `o = torch.sigmoid(L[2](both))`
    5. `n = torch.tanh(L[3](both))`
    6. `c = f * c + i * n`
    7. `h = o * torch.tanh(c)`
    8. `return h, c`

    其中一项检查会用同样的旋钮，把你的这一步和 PyTorch 自己的 LSTM 作比较。
    """)
    return


@app.function
def lstm_step(x, h, c, L):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "lstm_step", lstm_step, LOCALE)
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

    下面把整个模型写成一个层：词的编号进来，一张向量表把它们变成数字（第 11 章的 `nn.Embedding`），`nn.LSTM` 读它们，`nn.Linear` 在每个位置为词表里的每个词打分。`memory=None` 让我们可以传入之前调用得到的记忆，什么也不传就从头开始；后面的 `write` 会用到。数据是第 12 章的：同样的词表，同样的编号。下面的单元格要几秒钟。
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
            states, memory = self.lstm(self.table(ids), memory)
            return self.out(states), memory
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
    `train_reader` 是第 09 章的循环，多了两处小改动。

    **摊平。** `F.cross_entropy` 要的是一张分数表和一个答案向量，而我们每段文本有一张表。`S.reshape(-1, size)` 把它们首尾相接。（`-1` 的意思是“能放下的那个数”。）

    **裁剪。** 长链上的责任也可能爆炸，一个巨大的斜率就会把旋钮甩得老远。`nn.utils.clip_grad_norm_(knobs, 1.0)` 把所有斜率看成一个长向量，量出它的长度；如果超过 1，就把它们一起缩小，使这个长度刚好等于 1。这叫**裁剪**（clipping）。方向不变，只限制大小。

    `score_reader` 用第 12 章的 `perplexity` 在测试故事上打分。`reduction="none"` 让 `F.cross_entropy` 给出每个词的意外程度，而不是均值；负号把意外程度变回 `perplexity` 要的对数概率。`S, _memory = ...` 留下分数，忽略记忆。
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
    按下按钮，会用全部 190 万个词训练两遍，记忆大小是 128。这是这门课里第一次真正花时间的训练：在笔记本上大约 2 到 5 分钟。完成之前，需要它的结果的单元格会显示 `Ancestor stopped`，这是正常的。如果重新运行训练单元格，要再按一次按钮。（`np.concatenate` 把一个由数组组成的列表拼成一个。）单元格会显示最后的训练损失和测试困惑度。我们用 32 个词的片段训练，用 64 个词的较长片段打分。
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

    要写文本，就把模型自己写的上一个词喂给它，带着它的记忆往前走，像第 12 章那样采样。这里多了一个旋钮：在 softmax 之前把分数除以一个数，这个数叫**温度**（temperature）：

    - 温度为 1 时，给出模型真实的概率；
    - 小于 1 时，分数的差距拉大，本来就可能的词更可能了：文本更稳妥，也更平淡；
    - 大于 1 时，差距缩小，概率变平：文本更狂野。

    ## 练习 5：sample_word

    写 `sample_word(scores, temperature)`。`scores` 是一个向量，每个词一个分数。`torch.multinomial(p, 1)` 按 `p` 里的概率抽 1 个编号，`.item()` 把它变成普通数字。

    1. `p = torch.softmax(scores / temperature, dim=0)`
    2. `return torch.multinomial(p, 1).item()`
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
    check("13", "sample_word", sample_word, LOCALE)
    return


@app.cell
def _(index, model, vocab):
    torch.manual_seed(1)
    write(model, "once", 40, 0.8, index, vocab)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    和第 12 章二元模型的文本比一比。现在有些片段读起来像句子了，故事里的人物也能待上一会儿。把上面单元格里的 `0.8` 改成 `0.3` 或 `1.5`，再运行一次，看看有什么不同。

    注意 `write` 是怎么处理 `memory` 的：不管故事多长，模型对前文的全部了解都要塞进这 128 个数字。

    ## 接下来

    这种挤压正是 LSTM 的极限。要一个固定的向量装下整段文本，就像读完一本书再凭记忆把它翻译出来。在**第 14 章**，你会让模型在需要的时候回头看任何一个词。这个想法叫**注意力**（attention），它是所有现代语言模型的核心。

    ## 你现在会做的事

    - 解释循环的一步怎样在每个词上重复使用同一组旋钮
    - 写出 RNN 的一步并让它读完文本，再和 `nn.RNN` 核对
    - 把文本变成输入和下一个词的答案
    - 测量责任怎样随时间衰减
    - 解释 LSTM 的细胞和门怎样保护记忆和责任
    - 用裁剪限制爆炸的斜率
    - 训练语言模型，把它放上排行榜，并用温度生成文本

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** LSTM 出自 Hochreiter 和 Schmidhuber（1997）。给遗忘门一个正的起始偏置是常见做法。*GRU* 是它更简单的亲戚，有两个门。在长文本上走反向叫*随时间反向传播*（backpropagation through time）。我们的训练把文本切成 32 个词的片段，所以责任最多传 32 步，这叫*截断*（truncation）。
    """)
    return


if __name__ == "__main__":
    app.run()
