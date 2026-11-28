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

    语言模型在预训练里只学了一件事：照着训练语料的习惯把文本接下去。这和“照我们的意思做事”完全不是一回事。
    你去问一个只读过网页的模型一个问题，它可能会老老实实回答，也可能反问你三个问题，甚至直接贴一则广告——因为在网页的某些角落里，问题后面确实就是这么接的。

    要把“接着写文本”变成“按我们的意思做事”，需要第二个短得多的训练阶段，叫**对齐**（alignment），直接在预训练好的模型上继续训（第 18 章）。
    告诉模型我们想要什么，有三种办法，它们对*我们*的要求各不相同：

    1. **示范。**直接给出一些提示的正确回答。这需要有人知道正确答案。
    2. **比较。**拿模型的两个回答，挑出更好的那个。这只需要一个评判者。
    3. **打分。**让模型自己去试，答对给奖励、答错不给。这只需要一套自动检查的办法。

    这一章里，你会亲手写出这三种办法的损失函数，并用它们分别训练同一个小模型。

    ## 一个小小的世界

    对齐一个讲故事的大模型要花上好几个小时，还得请人来评判。所以这一章我们换一个几秒钟就能训完的小世界，而且在这里“什么是好回答”毫无争议：做加法。
    每行文本只有 6 个字符，比如 `3+4=07`。前四个字符是**提示**（prompt），后两个是**回答**（answer），统一写成两位数。

    我们的预训练模型读过几千行这种文本，但数据源很不靠谱：里面的回答都是 00 到 18 之间随便填的数，跟前面的加法毫无关系。所以它非常清楚“回答应该长什么样”，却完全不知道哪个答案才对。
    这就是一个迷你版的预训练模型：出口成章，但不顶用。它用的是第 24 章的小 transformer，词表只有 12 个词元（0 到 9 十个数字，加上 `+` 和 `=`）。第一次调用 `pretrained()` 时会现场把它训出来，大概花几秒钟。
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
    `prompts` 里存着全部 100 个提示的词元编号，从 `0+0=` 到 `9+9=`，`to_text` 可以把其中几条还原成文本。`best_answers` 会给出模型觉得最可能的回答：你会发现不管问什么，它给的都是同一个数。
    `share_right` 是模型在这 100 道题里的正确率：只有 0.01——也就是唯一碰巧撞上它“口头禅”的那一道题。接下来整章的目标，就是把这个正确率拉上去。

    ## 一个完整回答的概率

    三种对齐办法都需要算同一个量：模型觉得某个回答有多可能。每个回答由两个词元组成，而模型每次只能预测一个词元的概率（第 13 章）。
    所以整个回答的概率，等于第一个数字的概率，乘以给定第一个数字后第二个数字的概率：

    > $$p(\text{回答}) = p(\text{第一个数字}) \times p(\text{第二个数字，在第一个之后})$$

    概率相乘，取对数后就变成相加（第 13 章）。概率的对数叫**对数概率**（log-probability）：

    > $$\log p(\text{回答}) = \log p(\text{第一个数字}) + \log p(\text{第二个数字，在第一个之后})$$

    记住一个规律，整章都会用到：概率在 0 到 1 之间，所以**对数概率都是负数**，越接近 0 表示越可能；而训练损失要越小越好，所以把对数概率变成损失时前面都要加负号（就像第 07 章的意外程度 $\text{意外程度} = -\log p$）。

    ## 动手算：一个回答

    面对提示 `3+4=`，模型给第一个数字 `0` 的概率是 0.5；接在这个 `0` 后面，它给 `7` 的概率是 0.4。回答 `07` 的概率是多少？
    已知 $\log 0.5 = -0.69$、$\log 0.4 = -0.92$，它的对数概率又是多少？
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
    算一下会发现，$\log 0.2$ 确实正好是 $-1.61$。

    `answer_scores(model, prompts, answers)` 已经帮我们做好了准备工作：它把每条提示连同对应的回答一起喂给模型，返回模型在两个回答位置上输出的原始分数，形状是 `(回答数, 2, 12)`。
    对第 `i` 个回答来说，`scores[i, 0]` 是第一个数字的 12 个分数，`scores[i, 1]` 是模型读完第一个数字后，给第二个数字打出的 12 个分数。

    ## 练习 1：answer_logprob

    实现 `answer_logprob(scores, answers)`。`scores` 的形状是 `(回答数, 2, 12)`，含义如上；`answers` 的形状是 `(回答数, 2)`，存着每个回答的两个数字（整数）。
    函数返回一个张量，每个回答对应一个数，也就是按上面公式算出的对数概率。

    有一个现成工具能帮你省掉大部分力气：`F.cross_entropy(S, labels, reduction="none")`（第 14 章）会针对分数矩阵 `S` 的每一行，算出对应标签的意外程度——而意外程度就是对数概率加上负号。

    当所有分数都是 0 时，12 个词元各自的概率都是 $\frac{1}{12}$，所以每个回答的对数概率应该等于 $2 \times \log \frac{1}{12}$，大约是 $-4.97$。
    """)
    return


@app.function
def answer_logprob(scores, answers):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "answer_logprob", answer_logprob, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""把两个数字分开算。`scores[:, 0]` 是一张表，每行对应一个回答的 12 个分数，而 `answers[:, 0]` 装着每个回答的第一个数字——正好直接传给 `F.cross_entropy`。第二个数字把下标换成 1，做法一样。""",
        r"""一共 3 行：算出第一个数字的意外程度，算出第二个数字的意外程度，然后返回两者之和的相反数。注意要写 `reduction="none"`，否则它会自动把所有回答合成一个平均数。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 办法 1：示范

    最直接的办法你其实已经会了。收集一批“提示 + 正确回答”作为**示范**（demonstration），训练模型提高这些正确回答的概率：损失就是 $-\log p(\text{正确回答})$ 的均值。
    这本质上还是普通的交叉熵训练，跟预训练只有一个区别：只算回答那几个词元的损失，不要求模型去预测提示——因为提示是用户直接给它的。
    这种做法叫**监督微调**（supervised fine-tuning），简称 **SFT**。

    示范数据很贵，每一条都得靠懂答案的人写出来。我们假装预算只够准备 100 道题里的 30 道。下面的函数还是第 10 章熟悉的训练循环，只是换上了刚才的损失。
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
    （每次调用 `pretrained()` 都会重新造出完全一样的初始模型，保证三种办法站在同一条起跑线上。`torch.all(..., dim=1)` 在某一行两个数字全都对上时返回 `True`。）

    第一个数是模型在见过的那 30 道题上的正确率：全对（1.0）。第二个数是在全部 100 道题上的正确率：大约只有 0.3。它只是把那 30 道题死记硬背了下来，没见过的基本还是不会。
    这就是示范的局限：示范覆盖到哪，模型就学会哪；而示范永远不可能把所有情况都覆盖完。

    ## 办法 2：比较

    很多时候，没人能写出绝对完美的答案，但人人都能一眼看出两个回答里哪个更好——比如一篇文章的两份摘要，或者给用户的两条回复。
    一条**偏好对**（preference pair）就是“一个提示 + 两个回答 + 一个评判结论（这个比那个好）”。

    怎么拿这种对比结果来训练？我们希望模型提高较好回答的概率、压低较差回答的概率。但“提高”是跟谁比呢？
    跟训练前的自己比。我们留一份冻结的初始模型副本，叫**参考模型**（reference model），用它来量每个回答的对数概率涨了还是跌了：

    > $$\text{增益}(\text{回答}) = \log p_{\text{模型}}(\text{回答}) - \log p_{\text{参考}}(\text{回答})$$

    增益是 0，说明和训练前一样可能；增益为正，说明变更有可能了。一个偏好对的**差距**（margin），就是较好回答比较差回答多涨了多少：

    > $$\text{差距} = \text{增益}(\text{较好的}) - \text{增益}(\text{较差的})$$

    我们希望每对回答的差距都尽量大。第 12 章用过的 $-\log \sigma(\ldots)$ 正适合“把一个数推大”：记住三个锚点就有手感了（取 $\beta = 1$）——差距为 $-2$（学反了）时损失约 `2.13`，狠狠往回推；差距为 $0$（还没拉开）时损失约 `0.69`；差距拉到 $+2$ 时损失降到约 `0.13`，基本达标。于是损失函数就是：

    > $$\text{损失} = -\log \sigma(\beta \times \text{差距}) \text{ 的均值}$$

    $\beta$（beta）控制推得有多狠：$\beta$ 越小，哪怕差距已经不小，损失依然会继续往前拉。这套方法叫**直接偏好优化**（direct preference optimization），简称 **DPO**。
    用参考模型当锚点还有第二个好处：损失只盯着每对里的两个回答，如果不拉住缰绳，模型可能会跑偏，参考模型能让它不至于离原本的语言能力太远。

    ## 动手算：一个偏好对

    参考模型给一对回答的对数概率都是 $-2$。训练一阵后，模型给较好回答的对数概率涨到了 $-1$，给较差回答的降到了 $-3$。两个回答的增益分别是多少？差距又是多少？（注意两个负号相减别算错！）
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
    取 $\beta = 1$ 时，这一对的损失是 $-\log \sigma(2) \approx 0.127$。训练刚开始时，模型和参考模型一模一样，所有增益都是 0，损失就是 $-\log \sigma(0) = \log 2 \approx 0.693$。

    ## 练习 2：dpo_loss

    实现 `dpo_loss(better, worse, ref_better, ref_worse, beta)`。前四个输入都是一维张量，每个偏好对对应一个对数概率：前两个来自当前模型（较好回答、较差回答），后两个来自参考模型（同样两个回答）；`beta` 是一个数。
    函数返回上面的标量损失。`F.logsigmoid` 就是第 12 章的 $\log \sigma$。

    代入你刚才手算的例子、取 `beta = 1`，结果应该约等于 0.127。
    """)
    return


@app.function
def dpo_loss(better, worse, ref_better, ref_worse, beta):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "dpo_loss", dpo_loss, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""顺着三个公式一步步写就行：先算两个增益，再算差距，最后算损失。张量相减是逐位进行的，所有偏好对一次就算完了。""",
        r"""一共 2 行：先把差距算出来存进一个变量，再用 `F.logsigmoid` 算 `beta` 乘以差距的对数 sigmoid，最后取 `-torch.mean(...)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    接下来准备偏好对。`judged_pairs(base, prompts, 16)` 会让预训练模型按自己的概率分布，给每个提示随机抽 32 个回答（第 13 章），两两配成 16 个偏好对。
    然后由一个评判者（judge）逐对打分：哪个回答的数值离真实和更近，就判哪个更好；一样近就丢掉这一对。
    注意评判者从来不做的事：它从头到尾都没告诉过模型正确答案是什么。绝大多数偏好对里，两个回答其实都是错的，只是其中一个“没那么错”。

    下面的训练函数会先用 `torch.no_grad()` 算一遍参考模型的对数概率——因为参考模型冻结不动，算一次就够了，不用记录梯度。后面就是熟悉的训练循环，换上你写的 `dpo_loss`。
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
    大约 1,500 个偏好对，正确率从 0.01 涨到了约 0.4。全程没人教过模型哪怕一个标准答案，它单靠“这个比那个没那么错”，就在全部 100 道题上学出了不少门道。
    但它也很难再往上涨了：因为这些偏好对全是预训练模型自己随手答出来的，很多题的正确答案压根没在里面出现过。比较法只能从“已经摆上台面的回答”里挑好坏，没出现过的答案它教不了。

    ## 办法 3：打分

    第三种办法干脆连例子都不要。对不少任务来说，写段程序就能自动**检查**回答对不对：代码能不能跑过测试？数学证明有没有漏洞？加法算得对不对？
    有了检查器，模型就能自己练题了。它给出一个回答，检查器（checker）就发一个**奖励**（reward）——答对给 1，答错给 0；训练时把拿到奖励的回答概率调高。这种能靠程序自动算出来的奖励，叫**可验证奖励**（verifiable reward）。
    靠奖励反馈来学习，就是**强化学习**（reinforcement learning）。

    /// admonition | 🧰 工具箱：把文本变成数
    - 字符串可以像列表一样按位置取字符：`"7+2="[0]` 是 `"7"`，也就是第 0 个字符。
    - `int(text)` 可以把纯数字字符串转成整数：`int("7")` 是 7，`int("012")` 是 12。
    ///

    ## 练习 3：reward

    实现检查器 `reward(prompt, answer)`。`prompt` 是形如 `"3+4="` 的字符串，下标 0 和 2 的位置各是一个个位数；`answer` 是形如 `"07"` 的两位数字字符串。
    如果回答正好等于这两个数字的和，返回 `1.0`，否则返回 `0.0`。

    `reward("3+4=", "07")` 应返回 `1.0`，`reward("3+4=", "08")` 应返回 `0.0`。
    """)
    return


@app.function
def reward(prompt, answer):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "reward", reward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""有三段文本要先转成整数，才能相加和比较：提示里的两个数字，以及整个回答。""",
        r"""1 到 4 行就能写完：用 `if` 配两个 `return`，或者用第 18 章的 `1.0 if ... else 0.0`。判断条件是 `int(prompt[0]) + int(prompt[2]) == int(answer)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 奖励的斜率从哪来

    这里有个棘手的问题：检查器 `reward` 只是一段普通程序，不是平滑的数学公式，反向传播根本穿不过去；而且模型的回答是按概率*随机抽出来*的，抽样这一步同样没法求斜率。那梯度下降还能抓住什么来调参数？

    答案是**概率本身**。假设面对同一个提示，模型可能给出回答 $1, 2, 3, \ldots$，概率分别是 $p_1, p_2, p_3, \ldots$，对应的奖励分别是 $r_1, r_2, r_3, \ldots$。它**平均**能拿到的奖励，就是把每个奖励按概率加权求和，叫**期望奖励**（expected reward）：

    > $$\text{期望奖励} = p_1 \times r_1 + p_2 \times r_2 + p_3 \times r_3 + \ldots$$

    ## 动手算：一个期望奖励

    面对 `3+4=`，模型以 0.5 的概率回答 `07`，0.3 的概率回答 `08`，0.2 的概率回答 `17`。只有 `07` 能拿到奖励 1。它的期望奖励是多少？如果训练后这三个概率变成了 0.8、0.1 和 0.1，期望奖励又变成多少？
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
    期望奖励正是我们想拉高的目标，而且它**有斜率**：奖励 $r_i$ 只是固定常数，而概率 $p_i$ 是由模型参数平滑算出来的。它对任意一个参数的斜率就是：

    > $$r_1 \times p_1 \text{ 的斜率} + r_2 \times p_2 \text{ 的斜率} + \ldots$$

    这个和没法直接算——它要穷举*所有*可能的回答，对真实的语言模型来说，那几乎是无穷无尽的文本。但只要一步代数变形就能化解。第 07 章推过：$\log p$ 的斜率等于 $p$ 的斜率除以 $p$。把 $p$ 乘到左边，就得到 $p$ 的斜率等于 $p$ 乘以 $\log p$ 的斜率。（拿个具体数验算一下：当 $p = 0.5$ 时，$\log p$ 涨 `0.02`，对应的 $p$ 恰好涨 $0.5 \times 0.02 = 0.01$。）把这个关系代进上面的式子：

    > $$p_1 \times \big(r_1 \times \log p_1 \text{ 的斜率}\big) + p_2 \times \big(r_2 \times \log p_2 \text{ 的斜率}\big) + \ldots$$

    对照上面期望奖励的式子看看，结构一模一样：都是“概率 $p_i$ 乘以某一项”再求和。也就是说，它也是一个按模型概率分布加权的平均数——而这种平均数根本不用穷举所有回答就能估算出来。道理很直白：**按模型当前的概率抽 100 次，概率为 `0.5` 的回答自然会被抽中大约 50 次，所以直接对抽到的样本求平均，就自动按 $p_i$ 加了权！**

    落到 PyTorch 里，整个做法非常干脆：先从模型里抽一批回答，算出各自的奖励，然后对下面这个损失做反向传播：

    > $$\text{损失} = -r \times \log p(\text{抽到的回答}) \text{ 的均值}$$

    注意：这里的奖励 $r$（以及马上要用的优势）只是一个普通数值，不参与反向求导，所以梯度只沿着 $\log p$ 往回传。前面加负号，是因为梯度下降在找损失的最低点，而我们要让奖励越高越好。这套推导出的更新法则叫**策略梯度**（policy gradient）。说白了就是一句话：把模型刚试过的回答概率调高，奖励越高的回答，调得越用力。

    ## 比谁好，比谁差？

    直接用上面这个损失，学起来会很慢。奖励只有 0 和 1 时，答错了 $r = 0$，这一步白试、什么也没学到；答对了也只会千篇一律地“加把劲”，不管这道题模型是 16 次里好不容易蒙对 1 次，还是早就稳稳对 15 次。

    改进办法很直观：拿同一道题的多个回答互相比较。对每个提示一次抽一**组**（group）回答，把每个回答的奖励换成它的**优势**（advantage）——也就是看它比同组平均分高出多少，再除以这组奖励的分散程度（spread）来做归一化：

    > $$\text{优势} = \frac{r - \text{同组均值}}{\text{同组分散程度} + 0.000001}$$

    这里的分散程度就是第 09 章算过的典型大小（标准差）：每个数离均值的距离先平方、再求平均、最后开平方根。分母上加的那个极小数，是为了防止整组奖励全都一样时出现除以 0。
    这样一来，答错的回答优势变成负数，会被往下压；在一堆错答案里难得冒出来的正确答案，则会拿到很大的正优势，被狠狠往上推。这种“按组对比算优势”的思路，正是 **GRPO** 方法的核心。

    ## 动手算：一组四个

    同一道题的四个回答，拿到的奖励是 1、0、0、1。这一组的均值是多少？分散程度是多少？正确回答和错误回答的优势又各是多少？（分母上那个极小数忽略不计。）
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
    那要是四个回答全都得了 1 呢？每个数离均值的距离都是 0，算出来的优势也全是 0。只要一组回答全对或全错、毫无差别，这一组就教不了模型任何东西。记住这一点，后面马上会撞见它。

    ## 练习 4：advantages

    实现 `advantages(rewards)`。`rewards` 是一个二维张量，形状为 `(提示数, 组大小)`，每一行装着同一道提示那一组的奖励。函数返回同样形状的张量，把每个奖励换成它在本行里的优势。

    你需要在行内求均值，同时保留列这一维，方便广播相减：`torch.mean(x, dim=1, keepdim=True)`（第 17 章用过 `keepdim`）。

    输入 `[[1, 0, 0, 1]]` 时，输出应该约等于 `[[1, -1, -1, 1]]`，和手算结果一致（误差在那个极小数之内）。
    """)
    return


@app.function
def advantages(rewards):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "advantages", advantages, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""分两步：先算每行减去各自均值后的差值；再用这些差值算每行的分散程度——先平方、沿行求均值、再开平方根（`torch.sqrt`）。""",
        r"""一共 3 行：差值 `centred`、分散程度 `spread`，最后相除（分母记得加上 `1e-6`）。两次求均值都要带 `dim=1, keepdim=True`；如果漏掉 `keepdim`，遇到行数和列数相等的方阵时虽然不会报错，但减的方向全错了。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：policy_loss

    实现 `policy_loss(logprobs, advantage)`。两个输入都是一维张量，每个抽到的回答占一项：分别是当前模型给它的对数概率，以及它的优势。函数返回“奖励的斜率从哪来”一节的损失，只是把 $r$ 换成优势。

    当对数概率为 `[-1, -2]`、优势为 `[1, -1]` 时，结果必须等于 $-0.5$。
    """)
    return


@app.function
def policy_loss(logprobs, advantage):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "policy_loss", policy_loss, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""把两个张量逐位乘起来，求均值，再取相反数。""",
        r"""一行就能搞定，以 `return -torch.mean(` 开头。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面的训练函数把所有零件拼了起来。每一步，`sample_answers` 会从当前模型里为每个提示抽一组 16 个回答，返回重复了 16 遍的提示以及对应的回答，各有 1,600 行。
    `score_samples` 会把每一行转成文本，交给**你写的** `reward` 打分（如果回答根本不是两位数字，比如 `=3`，就直接记 0 分，不麻烦你的检查器）。接着你的 `advantages` 需要在每道题自己的 16 个回答里做比较，所以代码用 `reshape(-1, 16)` 把 1,600 个奖励排成 `(100, 16)`，算完优势后再用 `.reshape(-1)` 展平回去——这里的 `-1` 表示由 PyTorch 自动推算这一维的长度（先推算出 `1600 / 16 = 100`，展平时再变回 `1600`）。
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

    看看 `train_rewards`，如果我们传入全部 100 个提示、训练 400 步：模型每一步要写出多少个回答？整个训练过程里，你的 `reward` 会被调用多少次？这期间我们一共给了模型多少个标准答案？
    """)
    return


@app.cell(hide_code=True)
def _():
    loop_boxes = hand.boxes(['一步里的回答数', 'reward 的调用次数', '给出的正确回答数'])
    return (loop_boxes,)


@app.cell(hide_code=True)
def _(loop_boxes):
    hand.marked(loop_boxes, {'一步里的回答数': 1600, 'reward 的调用次数': 640000, '给出的正确回答数': 0}, {
        '一步里的回答数': r"""100 个提示，每个抽 16 个""",
        'reward 的调用次数': r"""每个回答查一次，一共跑 400 步""",
        '给出的正确回答数': r"""看看函数的参数列表：里面有 `right` 吗？""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    六十多万次自动检查的练习，而我们连一个标准答案都没给过。这就是强化学习的取舍：不用请老师写答案，但得让模型做海量练习。
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
    正确率从 0.01 一路涨到了约 0.9：只靠一个检查器和自己的不断试错，模型居然自学学会了加法。（这里随机抽样的运气成分不小，换几个随机种子，结果大致在 0.8 到 0.9 之间波动。）

    为什么没到 1.0？还记得前面说的“全一样的组”吗？对少数几道题，模型在早期就死死认准了某个错答案，现在 16 次抽样里 16 次都写它。整组奖励全是 0，优势也全是 0，没有任何力量能把它推开，它也就再也不会去尝试正确答案了。
    模型只能靠“试过的做法”拿到奖励。如何让它持续尝试新可能，叫做**探索**（exploration），是强化学习里一个经久不衰的难题。

    把三种办法放在一起对比，附上我们实测的结果：
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | 办法 | 我们需要提供什么 | 正确率 |
    |---|---|---:|
    | 无（仅预训练） | — | 0.01 |
    | 示范：SFT | 30 个正确回答 | 0.31 |
    | 比较：DPO | 一个评判者，约 1,500 个偏好对 | 0.40 |
    | 打分：奖励 | 一个检查器，加上 640,000 次试错 | 0.92 |

    别把这张表当成高低排名。在这个加法小世界里，正好有完美的自动检查器，所以打分法轻松胜出；但如果任务是“写一条体贴的回复”，根本写不出自动检查器，比较法就是能拿到的最好信号。
    现实中的大模型往往是三种办法接力上阵：先用示范把回答格式立住，再叠加上比较和奖励训练。
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("预训练", "接着写文本"), ("示范", "会回答"), ("比较", "答得好"), ("奖励", "会解题")], ["SFT", "DPO", "奖励"], "对齐模型的三种办法，通常的先后顺序")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 你最好心里有数

    每种办法教给模型的，都是“信号字面上要求的东西”，而这未必等于你心里真正想要的东西。

    - **示范**会照单全收，连同示范里的笔误、偏见和行文习惯一起学进去。
    - **比较**也会忠实迎合评判者的口味。如果评判者偏爱又长又自信的回答，模型就会学会写得又长又笃定——不管说得对不对。
    - **检查器**最容易被钻空子。只要存在“不真正完成任务也能骗到高分”的捷径，试错次数够多的模型迟早会把它挖出来。你的 `reward` 给 `07` 打 `1.0`，那 `7` 呢？`+7` 呢？模型真会去试。

    确认一个对齐后的模型确实在做我们**本意**想让它做的事，而不只是在应付打分规则，才是对齐工作里最花心思的大头。

    ## 接下来

    到这里，模型训练的全流程就拼齐了：分词器、现代 transformer、完整的训练循环、算力预算、清洗数据，以及对齐。第四部分将进入模型生命的下半场：**部署服务**（serving）。
    **第 27 章**里，你会把一个真实公开发布的模型权重，直接载入自己亲手写的代码里跑起来。

    ## 你现在会做的事

    - 说清什么是对齐，以及预训练模型为什么还需要对齐
    - 根据各个词元的原始分数，算出整个回答的对数概率
    - 用示范数据做监督微调（SFT），并说明它的局限在哪里
    - 解释参考模型和差距（margin）的作用，并写出 DPO 损失
    - 写出自动检查器，并讲清为什么单个奖励没有斜率，而期望奖励有斜率
    - 在同组内把奖励转成优势，并让模型靠自己的试错来训练
    - 说出三种对齐信号各自需要我们提供什么，以及各自可能把模型带偏到哪里

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** DPO 出自 Rafailov 等，*Direct Preference Optimization*（2023）。策略梯度可以追溯到 Williams 的 *REINFORCE*（1992）。GRPO（*group relative policy optimization*）由 Shao 等在 *DeepSeekMath*（2024）中提出；
    完整版方法还会限制每一步能把模型改动多远，我们这里略去了。如果先用人类比较数据训练一个单独的奖励模型，再用强化学习去优化，那就是常说的 *RLHF*。
    """)
    return


if __name__ == "__main__":
    app.run()
