import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="26 · 对齐")

with app.setup(hide_code=True):
    import marimo as mo
    import torch
    import torch.nn.functional as F

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.align import all_prompts, answer_scores, best_answers, judged_pairs, pretrained, right_answers, sample_answers, score_samples, share_right, to_text
    from learn_llm.checks import check
    from learn_llm.diagrams import pipeline

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 26 · 对齐

    语言模型接受的训练只有一件事：把文本接着写下去，就像训练文本接下去的样子。这和照你想要的去做不是一回事。
    拿一个只读过网页的模型，向它提一个问题，它可能回答，也可能再问你三个问题，还可能写一则广告：这些在网上的某个地方，都是问题后面会出现的内容。

    弥合“接着写文本”和“做我们想要的事”之间的差距，叫**对齐**（alignment）。它是第二个短得多的训练阶段，在预训练好的模型上进行（第 18 章）。
    告诉模型我们想要什么，有三种办法，它们对*我们*的要求各不相同：

    1. **示范。**给出一些提示的正确回答。这需要一个知道答案的人。
    2. **比较。**拿它的两个回答，说出哪个更好。这只需要一个评判者。
    3. **打分。**让它去尝试，奖励有效的做法。这只需要一种检验的办法。

    你会为每一种办法写出损失，并用这三种办法训练同一个小模型。

    ## 一个小小的世界

    对齐一个故事模型要花几个小时，还需要人来评判。所以这一章用一个小到几秒钟就能训练完的世界，在这里，“我们想要的”毫无争议：加法。
    一行文本有 6 个字符，比如 `3+4=07`。前四个是**提示**（prompt），后两个是回答，回答永远写成两位数字。

    我们的预训练模型读过几千行这样的文本，它们来自一个马虎的来源，回答是 00 到 18 之间的某个数，和那个和毫无关系。所以它清楚回答长什么样，却不知道哪个才对。
    这就是一个缩小版的预训练模型：很流畅，却没有帮助。它是第 24 章的小 transformer，有 12 个词元，对应十个数字、`+` 和 `=`。载入它就是在训练它，要几秒钟。
    """)
    return


@app.cell
def _():
    base = pretrained()
    prompts = all_prompts()
    right = right_answers(prompts)
    to_text(prompts[34:38]), to_text(best_answers(base, prompts[34:38])), share_right(base)
    return base, prompts, right


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `prompts` 存着全部 100 个提示，用词元 id 表示，从 `0+0=` 到 `9+9=`，`to_text` 把其中几个显示成文本。`best_answers` 给出模型对每个提示最可能的回答：不管是什么提示，回答都一样。
    `share_right` 是它答对的提示占 100 个提示的比例：0.01，也就是恰好某个提示的答案碰巧是它最喜欢的那个。这就是这一章要提高的数。

    ## 一个完整回答的概率

    三种办法都需要同一个量：模型觉得某个提示的某个回答有多可能。一个回答是两个词元，而模型一次只给出一个词元的概率（第 13 章）。
    整个回答的概率，是第一个数字的概率，乘以给定第一个数字之后第二个数字的概率：

    > $$p(\text{回答}) = p(\text{第一个数字}) \times p(\text{第二个数字，在第一个之后})$$

    概率相乘，所以它们的对数相加（第 13 章）。概率的对数叫**对数概率**（log-probability）。它永远不会大于 0，越接近 0 就越可能。它也就是第 07 章的意外程度加上负号：$\text{意外程度} = -\log p$。

    > $$\log p(\text{回答}) = \log p(\text{第一个数字}) + \log p(\text{第二个数字，在第一个之后})$$

    ## 动手算：一个回答

    对提示 `3+4=`，模型给第一个数字 0 的概率是 0.5。在这个 0 之后，它给 7 的概率是 0.4。回答 `07` 的概率是多少？
    已知 $\log 0.5 = -0.69$，$\log 0.4 = -0.92$，它的对数概率又是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    chance_boxes = hand.boxes(['07 的概率', '07 的对数概率'], step=0.01)
    return (chance_boxes,)


@app.cell(hide_code=True)
def _(chance_boxes):
    hand.marked(chance_boxes, {'07 的概率': 0.2, '07 的对数概率': -1.61}, {
        '07 的概率': r"""$0.5 \times 0.4$""",
        '07 的对数概率': r"""把两个对数加起来""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    而 $\log 0.2$ 的确是 $-1.61$。

    函数 `answer_scores(model, prompts, answers)` 负责这些管道活儿：它把每个提示连同它的回答喂给模型，给出模型在两个回答位置上产生的分数，形状是（回答数, 2, 12）。
    对第 `i` 个回答，`scores[i, 0]` 是第一个数字的 12 个分数，`scores[i, 1]` 是读过第一个数字之后算出的第二个数字的 12 个分数。

    ## 练习 1：answer_logprob

    写 `answer_logprob(scores, answers)`。`scores` 的形状是（回答数, 2, 12），如刚才所说。`answers` 的形状是（回答数, 2）：每个回答的两个数字，是整数。
    它返回一个张量，每个回答对应一个数：它的对数概率，按上面的公式算。

    有一个工具能完成大部分工作：`F.cross_entropy(S, labels, reduction="none")`（第 14 章），对分数 `S` 的每一行，给出它的标签的意外程度，而意外程度就是把符号反过来的对数概率。

    当所有分数都是 0 时，12 个词元每个的概率都是 $\frac{1}{12}$，所以每个回答必须得到 $2 \times \log \frac{1}{12}$，大约是 $-4.97$。
    """)
    return


@app.function
def answer_logprob(scores, answers):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "answer_logprob", answer_logprob, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""把两个数字分开处理。`scores[:, 0]` 是一张表，每个回答一行，每行 12 个分数，而 `answers[:, 0]` 装着每个回答的第一个数字：正好是 `F.cross_entropy` 要的东西。第二个数字用 1，做法一样。""",
        r"""3 行：第一个数字的意外程度，第二个数字的意外程度，然后返回它们之和的相反数。不写 `reduction="none"`，你得到的就是所有回答合在一起的一个数。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 办法 1：示范

    最直接的办法就是你已经知道的那个。收集提示和它们的正确回答，叫**示范**（demonstration），训练模型觉得这些回答很可能：损失是 $-\log p(\text{正确回答})$ 的均值。
    这就是普通的交叉熵，和预训练只有一点不同：只有回答的词元算数。我们不要求模型预测提示，因为提示是别人会交给它的。
    这叫**监督微调**（supervised fine-tuning），简称 SFT。

    示范很贵：得有一个知道答案的人，把每一条写出来。我们假装只付得起 100 个里的 30 个。下面的函数是第 10 章的循环，换上了这个损失。
    """)
    return


@app.function
def train_sft(model, prompts, answers, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for step in range(steps):
        optimizer.zero_grad()
        logprobs = answer_logprob(answer_scores(model, prompts, answers), answers)
        (-torch.mean(logprobs)).backward()
        optimizer.step()


@app.cell
def _(prompts, right):
    torch.manual_seed(1)
    shown = torch.randperm(100)[:30]
    sft_model = pretrained()
    train_sft(sft_model, prompts[shown], right[shown], 200)
    shown_right = torch.mean(torch.all(best_answers(sft_model, prompts[shown]) == right[shown], dim=1).float()).item()
    round(shown_right, 2), share_right(sft_model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    （`pretrained()` 每次都重新造出同一个起点模型，所以每种办法都从同一个地方出发。`torch.all(..., dim=1)` 在一行的两个数字都吻合时是 `True`。）

    第一个数是在它见过的那 30 个提示上答对的比例：全对。第二个数是在全部 100 个提示上的比例：大约 0.3。它把那 30 道加法背了下来，其他几乎没学到。
    这就是示范的局限。模型做的是示范所覆盖的事，而示范永远覆盖不了所有的事。

    ## 办法 2：比较

    往往没有人能写出完美的回答，但谁都能说出两个回答里哪个更好。想想一篇文章的两份摘要，或者给顾客的两条回复。
    一个**偏好对**（preference pair）是一个提示加上两个回答，再加上一个裁决：这个比那个好。

    怎么用一个裁决来训练？我们希望模型提高较好回答的概率，降低较差回答的概率。但“提高”是从哪里算起？
    从模型起步的地方。留一个冻结的模型副本，保持这次训练之前的样子，叫**参考模型**（reference model），用它来衡量每个回答移动了多远：

    > $$\text{增益}(\text{回答}) = \log p_{\text{模型}}(\text{回答}) - \log p_{\text{参考}}(\text{回答})$$

    增益为 0 表示“和以前一样可能”。一个偏好对的**差距**（margin），是较好的回答比较差的回答多增益了多少：

    > $$\text{差距} = \text{增益}(\text{较好的}) - \text{增益}(\text{较差的})$$

    我们要每个差距都大。第 12 章有一个“让这个数变大”的合适的损失：$-\log \sigma(\ldots)$，这个数为负时它很大，随着这个数变大，它逐渐减小到 0。于是：

    > $$\text{损失} = -\log \sigma(\beta \times \text{差距}) \text{ 的均值}$$

    $\beta$（beta）决定推得多用力：$\beta$ 小时，即使差距已经很大，损失还是继续往前拉。这个办法叫**直接偏好优化**（direct preference optimization），简称 DPO。
    以参考模型为基准，还有第二个作用。损失只关心每个偏好对里的那两个回答，而与参考模型比较，能防止模型偏离它原来的样子太远。

    ## 动手算：一个偏好对

    参考模型给一个偏好对的两个回答的对数概率都是 $-2$。训练了一阵之后，模型给较好的回答 $-1$，给较差的回答 $-3$。每个回答的增益是多少，差距又是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    pair_boxes = hand.boxes(['较好回答的增益', '较差回答的增益', '差距'])
    return (pair_boxes,)


@app.cell(hide_code=True)
def _(pair_boxes):
    hand.marked(pair_boxes, {'较好回答的增益': 1, '较差回答的增益': -1, '差距': 2}, {
        '较好回答的增益': r"""$-1 - (-2)$""",
        '较差回答的增益': r"""$-3 - (-2)$""",
        '差距': r"""第一个增益减去第二个：小心那两个负号""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    当 $\beta = 1$ 时，这个偏好对的损失是 $-\log \sigma(2)$，约为 0.127。还没训练之前，模型*就是*参考模型，每个增益都是 0，损失是 $-\log \sigma(0) = \log 2$，约为 0.693。

    ## 练习 2：dpo_loss

    写 `dpo_loss(better, worse, ref_better, ref_worse, beta)`。前四个输入是张量，每个偏好对一个对数概率：先是模型对较好回答和较差回答的，再是参考模型对同样两个回答的。`beta` 是一个数。
    它返回上面的损失，一个数。`F.logsigmoid` 就是第 12 章的 $\log \sigma$。

    在你手算的例子上，取 `beta = 1`，它必须得到约 0.127。
    """)
    return


@app.function
def dpo_loss(better, worse, ref_better, ref_worse, beta):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "dpo_loss", dpo_loss, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""按顺序照着三个公式来：两个增益，差距，损失。张量按位置相减，所以所有偏好对一次就处理完了。""",
        r"""2 行。把差距存进一个名字，然后返回 `beta` 乘以差距之后取 `F.logsigmoid`，再 `-torch.mean(...)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在来造偏好对。`judged_pairs(base, prompts, 16)` 让预训练模型对每个提示回答 32 次，按它自己的概率随机抽取（第 13 章），每个提示组成 16 个偏好对。
    然后由一个评判者（judge）来比较每个偏好对。我们的评判者偏爱数值更接近真实和的那个回答，如果两个一样远，就丢掉这个偏好对。
    注意评判者从来不做的事：它从不说正确回答是什么。大多数偏好对是两个错误回答，其中一个没那么错。

    训练函数先算出参考模型的对数概率，只算一次，也不记账，因为参考模型不会变。接下来就是常见的循环，用上你的损失。
    """)
    return


@app.function
def train_dpo(model, reference, pair_prompts, better, worse, steps):
    with torch.no_grad():
        ref_better = answer_logprob(answer_scores(reference, pair_prompts, better), better)
        ref_worse = answer_logprob(answer_scores(reference, pair_prompts, worse), worse)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for step in range(steps):
        optimizer.zero_grad()
        now_better = answer_logprob(answer_scores(model, pair_prompts, better), better)
        now_worse = answer_logprob(answer_scores(model, pair_prompts, worse), worse)
        dpo_loss(now_better, now_worse, ref_better, ref_worse, 1.0).backward()
        optimizer.step()


@app.cell(hide_code=True)
def _():
    dpo_button = mo.ui.run_button(label="用比较来训练（约 30 秒）")
    dpo_button
    return (dpo_button,)


@app.cell
def _(base, dpo_button, prompts):
    mo.stop(not dpo_button.value, mo.md("*按下按钮开始训练。*"))
    torch.manual_seed(0)
    pair_prompts, better, worse = judged_pairs(base, prompts, 16)
    dpo_model = pretrained()
    train_dpo(dpo_model, base, pair_prompts, better, worse, 300)
    len(pair_prompts), share_right(dpo_model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 1,500 个偏好对，答对的比例约为 0.4，从 0.01 涨了上来。没有人告诉过模型哪怕一个正确回答。它是从“这个没那么错”里学到的，跨越全部 100 个提示。
    它没能走得更远，是因为偏好对来自预训练模型自己的回答，而对许多提示来说，正确回答从来没出现在里面。比较只能教会我们那些有人想到去比较的回答。

    ## 办法 3：打分

    第三种办法根本不需要例子。对某些任务，程序可以**检查**一个回答：代码能不能通过测试，证明是不是成立，这个加法算得对不对？
    这样模型就可以练习了。它给出回答，检查器（checker）给一个**奖励**（reward），对了是 1，错了是 0，训练让得到奖励的回答变得更可能。能由程序算出来的奖励，叫**可验证的奖励**（verifiable reward）。
    从奖励中学习，叫**强化学习**（reinforcement learning）。

    /// admonition | 🧰 工具箱：把文本变成数
    - 字符串可以像列表一样取下标：`"7+2="[0]` 是 `"7"`，文本里的一个字符。
    - `int(text)` 把由数字组成的文本变成整数：`int("7")` 是 7，`int("012")` 是 12。
    ///

    ## 练习 3：reward

    写检查器 `reward(prompt, answer)`。`prompt` 是形如 `"3+4="` 的字符串，位置 0 和 2 上是个位数字，`answer` 是形如 `"07"` 的两位数字字符串。
    如果回答是这两个数字的和，它返回 `1.0`，否则返回 `0.0`。

    `reward("3+4=", "07")` 必须得到 `1.0`，`reward("3+4=", "08")` 必须得到 `0.0`。
    """)
    return


@app.function
def reward(prompt, answer):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "reward", reward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""有三段文本要先变成数，才能相加和比较：提示里的两个数字，以及作为整体的回答。""",
        r"""1 到 4 行：一个带两个 `return` 的 `if`，或者第 18 章的 `1.0 if ... else 0.0`。判断的条件是 `int(prompt[0]) + int(prompt[2]) == int(answer)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 奖励的斜率

    难处在这里。检查器是一个程序，不是带斜率的公式：没办法穿过 `reward` 往回走。而且模型的回答是按它的概率*随机抽出来*的，这也没有斜率。梯度下降能抓住什么？

    概率。假设对某一个提示，模型能给出回答 $1, 2, 3, \ldots$，概率分别是 $p_1, p_2, p_3, \ldots$，它们能得到的奖励是 $r_1, r_2, r_3, \ldots$ 它**平均**得到的奖励，就是每个奖励按它的概率加权，叫**期望奖励**（expected reward）：

    > $$\text{期望奖励} = p_1 \times r_1 + p_2 \times r_2 + p_3 \times r_3 + \ldots$$

    ## 动手算：一个期望奖励

    对 `3+4=`，模型以 0.5 的概率回答 `07`，0.3 的概率回答 `08`，0.2 的概率回答 `17`。只有 `07` 得到奖励 1。期望奖励是多少？如果训练把概率改成了 0.8、0.1 和 0.1，它又是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    expect_boxes = hand.boxes(['期望奖励', '训练之后'], step=0.1)
    return (expect_boxes,)


@app.cell(hide_code=True)
def _(expect_boxes):
    hand.marked(expect_boxes, {'期望奖励': 0.5, '训练之后': 0.8}, {
        '期望奖励': r"""$0.5 \times 1 + 0.3 \times 0 + 0.2 \times 0$""",
        '训练之后': r"""同样的和，换上新的概率""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    期望奖励是我们想要提高的，而它*确实*有斜率：奖励是固定的数，概率来自模型的旋钮。它对任何一个旋钮的斜率是

    > $$r_1 \times p_1 \text{ 的斜率} + r_2 \times p_2 \text{ 的斜率} + \ldots$$

    这个和我们算不出来：它要对每一个可能的回答求和，而对真实的模型，那就是所有可能的文本。一步代数就能解决。第 07 章发现，$\log p$ 的斜率是 $p$ 的斜率除以 $p$。把它倒过来：
    $p$ 的斜率是 $p$ 乘以 $\log p$ 的斜率。代进去：

    > $$p_1 \times \big(r_1 \times \log p_1 \text{ 的斜率}\big) + p_2 \times \big(r_2 \times \log p_2 \text{ 的斜率}\big) + \ldots$$

    把它和上面的期望奖励对比。形式相同：概率乘以某个东西。所以它也是一个平均：$r \times \log p$ 的斜率的期望值。而对概率求平均，不必走遍每个回答，也能估计出来。
    **从模型里抽几个回答，对它们求平均。**就是这样。用 PyTorch 的话说：抽出回答，查出它们的奖励，然后从下面这个损失往回走：

    > $$\text{损失} = -r \times \log p(\text{抽出的回答}) \text{ 的均值}$$

    奖励当作普通的数。加负号，是因为我们是往山下走，而想要的是奖励变高。这个结果就是**策略梯度**（policy gradient）。用话来读这个损失：
    让你尝试过的回答变得更可能，每个按它的奖励成比例地增加。

    ## 比什么好，比什么差？

    照这个样子直接用的话，这个损失学得很慢。奖励只有 0 和 1 时，错误回答的 $r = 0$，什么也教不了；而正确回答只会说“多来点这个”，不管模型在这个提示上是 16 次里才对 1 次，还是 16 次里对 15 次。

    解决办法是把每个回答和同一个提示的其他回答比较。每个提示抽一**组**（group）回答，把每个奖励换成它的**优势**（advantage）：它比自己所在的组好多少，以这一组的分散程度（spread）为单位来量。

    > $$\text{优势} = \frac{r - \text{组的均值}}{\text{组的分散程度} + 0.000001}$$

    分散程度就是第 09 章的典型大小：到均值的距离的平方的均值，再开平方根。那个很小的数是为了避免整组得到相同奖励时除以零。
    这样，错误回答的优势为负，会被往*下*推，而一堆失败里难得的一次成功，会得到很大的优势。这样按组来用的办法，是一种叫 **GRPO** 的方法的核心。

    ## 动手算：一组四个

    对一个提示的四个回答，得到的奖励是 1、0、0 和 1。这一组的均值是多少？分散程度呢？正确回答的优势和错误回答的优势又各是多少？（那个很小的数不用算。）
    """)
    return


@app.cell(hide_code=True)
def _():
    group_boxes = hand.boxes(['均值', '分散程度', '正确回答的优势', '错误回答的优势'], step=0.1)
    return (group_boxes,)


@app.cell(hide_code=True)
def _(group_boxes):
    hand.marked(group_boxes, {'均值': 0.5, '分散程度': 0.5, '正确回答的优势': 1, '错误回答的优势': -1}, {
        '均值': r"""$(1 + 0 + 0 + 1) / 4$""",
        '分散程度': r"""到均值的每个距离都是 $0.5$ 或 $-0.5$，所以每个平方都是 $0.25$：取它们均值的平方根""",
        '正确回答的优势': r"""$(1 - 0.5) / 0.5$""",
        '错误回答的优势': r"""$(0 - 0.5) / 0.5$""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    那要是四个都得了 1 呢？到均值的每个距离都是 0，每个优势也都是 0。一组回答要是全都一样，不管全对还是全错，都什么也教不了。把这一点记在心里，后面还会用到。

    ## 练习 4：advantages

    写 `advantages(rewards)`。`rewards` 是一个张量，形状是（提示数, 组大小）：每一行装着一个提示那一组的奖励。它返回同样形状的张量，每个奖励都换成它在本行里的优势。

    你需要沿着一行求均值，并且保留列的形状，这样才能广播：`torch.mean(x, dim=1, keepdim=True)`（第 17 章这样用过 `keepdim`）。

    `[[1, 0, 0, 1]]` 必须得到 `[[1, -1, -1, 1]]`，和手算一样，误差在那个很小的数之内。
    """)
    return


@app.function
def advantages(rewards):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "advantages", advantages, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""分两步。先是居中的奖励：每一行减去它自己的均值。然后是每一行的分散程度，用居中的奖励来算：平方，沿着行求均值，再用 `torch.sqrt`。""",
        r"""3 行：`centred`、`spread`，以及那个除法，分散程度要加上 `1e-6`。两个均值都需要 `dim=1, keepdim=True`。不加 `keepdim`，行数和列数相同的表照样能相减，但减的方向是错的。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：policy_loss

    写 `policy_loss(logprobs, advantage)`。两个都是张量，每个抽出的回答一项：它在模型下的对数概率，和它的优势。它返回“奖励的斜率”那一节的损失，用优势代替 $r$。

    对数概率 `[-1, -2]` 和优势 `[1, -1]`，它必须得到 $-0.5$。
    """)
    return


@app.function
def policy_loss(logprobs, advantage):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "policy_loss", policy_loss, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""把两个张量按位置相乘，求均值，再把符号反过来。""",
        r"""只要一行，以 `return -torch.mean(` 开头。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面的训练函数把各部分拼在一起。每一步，`sample_answers` 从模型现在的样子里，为每个提示抽一组 16 个回答。它返回重复了 16 遍的提示，和放在旁边的回答，各有 1,600 行。
    `score_samples` 把**你的** `reward` 对每一行当作文本运行。（一个不是两位数字的回答，比如 `=3`，直接得 0，不用问你的检查器。）然后你的 `advantages` 在每个提示的组内比较，所以奖励要先重塑成（100, 16），再变回来。
    """)
    return


@app.function
def train_rewards(model, prompts, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for step in range(steps):
        many, answers = sample_answers(model, prompts, 16)
        rewards = score_samples(reward, many, answers)
        advantage = advantages(rewards.reshape(-1, 16)).reshape(-1)
        optimizer.zero_grad()
        logprobs = answer_logprob(answer_scores(model, many, answers), answers)
        policy_loss(logprobs, advantage).backward()
        optimizer.step()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：读懂这个循环

    看 `train_rewards`，用 100 个提示和 400 步来调用它。模型每一步写多少个回答？整个运行里，你的 `reward` 被调用多少次？这个循环从我们这里得到了多少个正确回答？
    """)
    return


@app.cell(hide_code=True)
def _():
    loop_boxes = hand.boxes(['一步里的回答数', 'reward 的调用次数', '给出的正确回答数'])
    return (loop_boxes,)


@app.cell(hide_code=True)
def _(loop_boxes):
    hand.marked(loop_boxes, {'一步里的回答数': 1600, 'reward 的调用次数': 640000, '给出的正确回答数': 0}, {
        '一步里的回答数': r"""100 个提示，每个 16 个""",
        'reward 的调用次数': r"""每个回答一次，400 步里每一步都是""",
        '给出的正确回答数': r"""看函数的输入：`right` 在里面吗？""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    超过五十万次被检查的尝试，而我们没给过一个回答。这就是这个办法的交换：它不需要老师，但需要大量的练习。
    """)
    return


@app.cell(hide_code=True)
def _():
    reward_button = mo.ui.run_button(label="用奖励来训练（约 40 秒）")
    reward_button
    return (reward_button,)


@app.cell
def _(prompts, reward_button):
    mo.stop(not reward_button.value, mo.md("*按下按钮开始训练。*"))
    reward_model = pretrained()
    torch.manual_seed(0)
    train_rewards(reward_model, prompts, 400)
    to_text(prompts[34:38]), to_text(best_answers(reward_model, prompts[34:38])), share_right(reward_model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 0.9，从 0.01 涨上来：模型靠一个检查器和它自己的尝试，教会了自己做加法。（这里随机抽取的影响很大。换别的种子，我们测到的结果在 0.8 到 0.9 之间。）

    为什么不是 1？想想那个全都一样的组。对少数几个提示，模型很早就认定了一个错误回答，现在 16 次里 16 次都给出它。这一组里每个优势都是 0，所以没有什么把它推开，它永远不会去试正确的回答。
    模型只能因为它尝试过的事而得到奖励。让它不断尝试新东西，叫**探索**（exploration），是从奖励中学习一直存在的难题。

    下面是三种办法放在一起，附上我们测到的结果：
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | 办法 | 我们必须提供什么 | 答对的比例 |
    |---|---|---:|
    | 无（预训练） | | 0.01 |
    | 示范：SFT | 30 个正确回答 | 0.31 |
    | 比较：DPO | 一个评判者，约 1,500 个偏好对 | 0.40 |
    | 打分：奖励 | 一个检查器，和 640,000 次尝试 | 0.92 |

    不要把它读成排名。在这个世界里，存在完美的检查器，所以奖励轻松取胜。对“写一条体贴的回复”这样的任务，并没有检查器，比较就是能得到的最好的信号。
    真实的模型是三种办法依次用上来对齐的：先用示范把格式做对，再在上面加比较和奖励。
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("预训练", "接着写文本"), ("示范", "会回答"), ("比较", "答得好"), ("奖励", "会解题")], ["SFT", "DPO", "奖励"], "对齐模型的三种办法，通常的先后顺序")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 你最好知道的事

    每种办法教给模型的，恰好是它的信号所说的，而这并不总是你的本意。

    - 示范教的是展示出来的那些回答，包括其中的错误和风格。
    - 评判者的偏好同样会被忠实地学去。如果评判者喜欢又长又自信的回答，模型就会变得又长又自信，不管对错。
    - 检查器会被一字不差地服从。如果有办法不做任务就能拿到奖励，一个尝试足够多东西的模型会找到它。你的 `reward` 对 `07` 给 `1.0`。对 `7` 呢，对 `+7` 呢？模型会试出来的。

    检查一个对齐过的模型做的是不是本意要它做的，而不只是被打分的那些，是真实工作的大部分。

    ## 接下来

    训练到这里就完整了：分词器、现代 transformer、规范的训练运行、预算、干净的数据，还有对齐。第四部分讲模型生命的另一半：**部署**（serving）。
    **第 27 章**把一个真实的、已发布的模型的权重，载入你自己写的代码里。

    ## 你现在会做的事

    - 说出什么是对齐，以及预训练模型为什么需要它
    - 由各个词元的分数，算出一个完整回答的对数概率
    - 用示范做微调，并说出它够不到什么
    - 解释参考模型和差距，并写出 DPO 损失
    - 写一个检查器，并解释为什么奖励没有斜率，而期望奖励有
    - 在一组内把奖励变成优势，并让模型用它自己的尝试来训练
    - 说出三种信号各向你要求什么，以及各自会怎样误导

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** DPO 出自 Rafailov et al.，*Direct Preference Optimization*（2023）。策略梯度可以追溯到 Williams 的 *REINFORCE*（1992）。GRPO，即 *group relative policy optimization*，由 Shao et al. 在 *DeepSeekMath*（2024）中提出；
    完整的方法还会限制每一步能把模型改动多远，我们略去了这一点。通过一个单独的奖励模型，用人类的比较来训练，叫 *RLHF*。
    """)
    return


if __name__ == "__main__":
    app.run()
