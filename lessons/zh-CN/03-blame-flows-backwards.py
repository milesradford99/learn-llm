import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="03 · 责任往回传")

with app.setup(hide_code=True):
    import time

    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.diagrams import one_day
    from learn_llm.plot import chart

    LOCALE = "zh-CN"
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 03 · 责任往回传

    第 02 章最后留下了一个小缺口。米娅的损失里装着 $(w \times 14 + b - 63)^2$ 这样的式子（$^2$ 表示“自己乘自己”）：外面套着一层平方，括号里面又藏着旋钮 $w$。
    前面学过的斜率规则还应付不了这种嵌套式子。这一章，你会学会求任何由小步骤拼出来的算式的精确斜率，
    并见识到训练几乎所有神经网络——包括这门课最后的语言模型——的核心思想。

    下面是前两章已经写好、本章会直接复用的几个函数：计算直线损失的 `predict`、`squared_error`、`mse`（第 01 章），以及通过微小改动估算斜率的 `nudge_slope`（第 02 章）。
    """)
    return


@app.function
def predict(w, b, x):
    return w * x + b


@app.function
def squared_error(guess, truth):
    return (guess - truth) ** 2


@app.function
def mse(w, b, xs, ys):
    running = 0
    for x, y in zip(xs, ys):
        running = running + squared_error(predict(w, b, x), y)
    return running / len(xs)


@app.function
def nudge_slope(f, x):
    h = 0.0001
    return (f(x + h) - f(x)) / h


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一天的数据，拆成四小步

    先看米娅第一天的数据：气温 `t = 14` 度，卖出 `s = 63` 个冰淇淋。假设我们把直线的两个旋钮先调到 `w = 10`、`b = -60`。
    这一天的平方误差可以拆成四个小步骤，每一步只做**一件**简单的事，并给中间结果起个名字。
    （代码里名字末尾带 `1` 表示“第 1 天”；正文和图里我们就简写成 `p`、`g`、`e`、`L`。）
    """)
    return


@app.cell
def _():
    p1 = 10 * 14      # p = w × t：斜率乘以气温
    g1 = p1 + (-60)   # g = p + b：预测销量
    e1 = g1 - 63      # e = g − s：预测误差
    L1 = e1 * e1      # L = e × e：平方误差
    p1, g1, e1, L1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    （最后一行用逗号把四个变量排在一起，一次输出四个结果。页面会把它们显示成从 0 开始编号的列表：
    第 0 项 `p` = 140，第 1 项 `g` = 80，第 2 项 `e` = 17，第 3 项 `L` = 289。）

    把这四步画成流程图，看起来就直观多了：每个方框装着一个数值，箭头表示“用左边的值算出右边的值”，箭头旁的符号标明了运算方式。
    左边固定不动的是观测数据（气温 `t` 和下方的实际销量 `s`），而旋钮 `w` 和 `b` 则是我们要调整的参数。
    """)
    return


@app.cell(hide_code=True)
def _():
    one_day({"w": 10, "t": 14, "p": 140, "b": -60, "g": 80, "s": 63, "e": 17, "L": 289})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这种展示“每个值怎样一步步算出来”的流程图，叫作**计算图**（computation graph）。

    ## 每一步单独看都很简单

    一次只盯住一条箭头，问一个第 02 章就很熟悉的问题：保持其他数值不动，如果把箭头起点的值稍微调大一点点，终点的值会跟着变化多少倍？
    这个变化倍数就是这条箭头的**局部斜率**（local slope）。这里的每一条，用第 02 章学过的规则都能直接算出来：

    | 这一步 | 微调哪个值 | 终点变化倍数（局部斜率） | 依据（第 02 章规则） | 本例数值 |
    |---|---|---|---|---|
    | $L = e \times e$ | $e$ | $2 \times e$ | $x \times x$ 的斜率是 $2 \times x$ | $2 \times 17 = \boldsymbol{34}$ |
    | $e = g - s$ | $g$ | `1` | $g$ 增加 $h$，$e$ 也增加 $h$ | **1** |
    | $g = p + b$ | $p$ | `1` | $p$ 增加 $h$，$g$ 也增加 $h$ | **1** |
    | $g = p + b$ | $b$ | `1` | 同上，换成 $b$ | **1** |
    | $p = w \times t$ | $w$ | $t$ | $c \times x$ 的斜率是 $c$（这里 $c$ 是固定不变的 $t$） | **14** |

    ## 汇率相乘

    /// admonition | 🧰 工具箱：连续换汇
    假设 1 欧元能换 8 元人民币，1 元人民币能换 20 日元。那 1 欧元能换多少日元？先把 1 欧元换成 8 元，
    再把这 8 元每一元都换成 20 日元：$8 \times 20 = \boldsymbol{160}$ 日元。沿着一条链连续换算时，汇率直接**相乘**。
    ///

    局部斜率就像汇率一样，告诉我们“前一个数变动一点，后一个数跟着变动几倍”。现在把 `w` 微调一个很小的量 $h$，顺着箭头一路追踪到最终损失 $L$：

    | 数值 | 变化量 |
    |---|---|
    | $w$ | $h$ |
    | $p$ | $14 \times h$（局部斜率 14） |
    | $g$ | $1 \times 14 \times h = 14 \times h$（局部斜率 1） |
    | $e$ | $1 \times 14 \times h = 14 \times h$（局部斜率 1） |
    | $L$ | 约 $34 \times 14 \times h = 476 \times h$（局部斜率 34） |

    也就是说，$L$ 的变化量是 $w$ 的 476 倍——$L$ 对 $w$ 的斜率正好是 **$14 \times 1 \times 1 \times 34 = 476$**。
    把整条路径上的局部斜率一路相乘，这就是微积分里大名鼎鼎的**链式法则**（chain rule）。

    我们用第 02 章的 `nudge_slope` 实际测一下，看看结果是不是 476：
    """)
    return


@app.cell
def _():
    def loss_day1(w):
        return (w * 14 + (-60) - 63) ** 2

    nudge_slope(loss_day1, 10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    步长 $h = 0.0001$ 时，测出的斜率大约是 `476.0196`（后面的尾数是计算机小数舍入带来的微小误差），和链式法则算出的 **476** 非常接近。
    把步长 $h$ 缩得更小，测量值会越来越贴近 **476**：

    | 步长 $h$ | 0.1 | 0.01 | 0.001 | 0.0001 |
    |---|---|---|---|---|
    | 测出的斜率 | 495.6 | 477.96 | 476.196 | 476.0196 |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion({
        "想看为什么链式法则完全精确？（选读）": mo.md(r"""
        刚才用 $h = 0.0001$ 测出来的是 `476.0196`，比 476 多了 `0.0196`。这个小零头是从哪儿来的？

        前三个步骤（$\times 14$、$+ b$、$- s$）都是严格的按比例缩放或平移，不会产生任何多余项，所以 $e$ 精确地增加了 $14 \times h$。
        只有最后一步平方（$L = e \times e$）会留下一个小零头。回想第 02 章的正方形面积图：原本边长是 $e = 17$，两条边各增加 $14 \times h$ 后，
        面积多出两条长边（$2 \times 17 \times 14 \times h = 476 \times h$），外加角落里一个小正方形 $(14 \times h) \times (14 \times h) = 196 \times h \times h$。
        把总变化量除以 $h$，得到：

        > $$\text{测出的斜率} = 476 + 196 \times h$$

        代入 $h = 0.0001$，正好是 $476 + 0.0196 = 476.0196$。和第 02 章的 $2 \times x + h$ 一样，只要 $h$ 不断缩小，$196 \times h$ 就会趋近于 0，精确斜率不多不少正是 **476**。

        如果平方出现在中间、后面还跟着乘以 3 呢？比如 $3 \times (x \times x)$。平方多出的小角 $h \times h$ 乘以 3 变成 $3 \times h \times h$，除以 $h$ 后是 $3 \times h$，当 $h$ 缩小时照样消失。所以不管步骤怎么排列，链式法则都是完全精确的。
        """)
    })
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 倒着算：责任

    顺着算时，我们要从 `w` 出发推一遍，再从 `b` 出发推一遍；如果模型有上百万个旋钮，就得推上百万遍。
    换个方向会聪明得多：**从终点 `L` 出发，倒着往回算**。

    我们给计算图里的每个方框记一个**责任**（blame），表示这个方框每微小变动 1 单位，最终损失 `L` 会跟着变动多少倍。
    （注意：这里的“每变动 1 单位”指的依然是极小变动下的变化率，不是真的把数值直接加 1——比如把 $e$ 从 17 直接加到 18，$L$ 会增加 35，比 34 多了 1，那是步子太大带来的平方余项。）

    倒着算每个框的责任，全程只需要记住一句口诀：

    > **每个框的责任 = 它对下一个框的影响倍数（局部斜率） $\times$ 下一个框的责任**

    从终点 `L` 出发，顺着箭头一步步往回套用这句口诀：

    - **`L`（终点）**：它自己就是最终损失，变动 1 单位，损失就变动 1 单位，所以责任 = **1**。
    - **$e \to L$**：局部斜率是 $2 \times 17 = 34$，下一个框 `L` 的责任是 1 $\to$ $e$ 的责任 = $34 \times 1 = \boldsymbol{34}$。
    - **$g \to e$**：局部斜率是 1，下一个框 $e$ 的责任是 34 $\to$ $g$ 的责任 = $1 \times 34 = \boldsymbol{34}$。
    - **$p \to g$**：局部斜率是 1，下一个框 $g$ 的责任是 34 $\to$ $p$ 的责任 = $1 \times 34 = \boldsymbol{34}$。
    - **$b \to g$**：局部斜率也是 1，下一个框 $g$ 的责任是 34 $\to$ $b$ 的责任 = $1 \times 34 = \boldsymbol{34}$。
    - **$w \to p$**：局部斜率是 14，下一个框 $p$ 的责任是 34 $\to$ $w$ 的责任 = $14 \times 34 = \boldsymbol{476}$。

    （责任也可以是负数。责任为负，说明把这个框调大反而会让损失*减小*；代入梯度下降的更新公式时，减号会负负得正，顺理成章地把旋钮往大调。）

    下面每个方框下方的红字，就是它分担到的责任：
    """)
    return


@app.cell(hide_code=True)
def _():
    one_day({"w": 10, "t": 14, "p": 140, "b": -60, "g": 80, "s": 63, "e": 17, "L": 289},
            {"w": "责任 476", "p": "责任 34", "b": "责任 34", "g": "责任 34", "e": "责任 34", "L": "责任 1"})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $w$ 的责任就是它的斜率 476，和前面用链式法则算出来的一模一样。这当然不是巧合：往前算的是 $14 \times 1 \times 1 \times 34$，往回算的是
    $34 \times 1 \times 1 \times 14$，还是那几个数，只是倒过来乘，结果完全一样。$b$ 的责任也就是它的斜率 34。
    这种从终点倒推回起点的过程，叫作**反向传播**（backpropagation）：最终损失的责任沿着计算图往回流动，一直传到每一个旋钮。

    为什么一定要倒着走？注意 `w` 和 `b` 共用路径的后半段（`g` → `e` → `L`）。倒着走时，这段公共路径只算**一次**，
    所有旋钮都能直接复用。整个反向过程每个框只经过一次、每条箭头只做一次乘法，计算量和正向算损失差不多。
    因此哪怕模型有十亿个旋钮，也只要再多花大约算一次损失的工夫，就能一口气拿到全部十亿个斜率。对比一下数值轻推：**每个**旋钮都要单独算两次损失。

    把所有旋钮的损失斜率收集到一起，合称**梯度**（gradient）。第 02 章那种沿着下坡方向一步步调参的方法叫“梯度下降”，名字正是从这儿来的。

    ## 练习 1：手工填表

    轮到你亲手算一遍了。取米娅第三天的数据：`t = 20`，`s = 137`，旋钮依然是 `w = 10`、`b = -60`。
    先**往前**算出各个方框的值，再套用口诀**往回**算出各个责任。把答案填进下面的输入框，每填一个都会即时核对。
    """)
    return


@app.cell(hide_code=True)
def _():
    hand = mo.ui.dictionary({
        name: mo.ui.number(label=f"{name[6:]} 的责任" if name.startswith("blame") else name, step=1)
        for name in ("p", "g", "e", "L", "blame e", "blame g", "blame p", "blame b", "blame w")
    })
    return (hand,)


@app.cell(hide_code=True)
def _(hand):
    _answers = {"p": 200, "g": 140, "e": 3, "L": 9,
                "blame e": 6, "blame g": 6, "blame p": 6, "blame b": 6, "blame w": 120}
    _names = {"p": "p", "g": "g", "e": "e", "L": "L", "blame e": "e 的责任", "blame g": "g 的责任",
              "blame p": "p 的责任", "blame b": "b 的责任", "blame w": "w 的责任"}
    _hints = {
        "p": r"$p = w \times t$", "g": "$g = p + b$（$b$ 是负数）", "e": "$e = g - s$", "L": r"$L = e \times e$",
        "blame e": r"$L = e \times e$ 的局部斜率是 $2 \times e$；乘以 $L$ 的责任（1）",
        "blame g": "局部斜率 1，乘以 $e$ 的责任",
        "blame p": "局部斜率 1，乘以 $g$ 的责任",
        "blame b": "局部斜率 1，乘以 $g$ 的责任",
        "blame w": "局部斜率 $t$（这里是 20），乘以 $p$ 的责任",
    }
    _lines = []
    for _name, _right in _answers.items():
        _v = hand.value[_name]
        if _v is None:
            _lines.append(f"- ⬜ **{_names[_name]}**：{_hints[_name]}")
        elif abs(_v - _right) < 1e-9:
            _lines.append(f"- ✅ **{_names[_name]}** = {_v:g}")
        else:
            _lines.append(f"- ❌ **{_names[_name]}** = {_v:g}，还不对。提示：{_hints[_name]}")
    _done = all(hand.value[n] is not None and abs(hand.value[n] - r) < 1e-9 for n, r in _answers.items())
    mo.vstack([
        mo.hstack([hand["p"], hand["g"], hand["e"], hand["L"]], justify="start"),
        mo.hstack([hand["blame e"], hand["blame g"], hand["blame p"], hand["blame b"], hand["blame w"]],
                  justify="start", wrap=True),
        mo.callout(mo.md("\n".join(_lines)), kind="success" if _done else "neutral"),
        one_day({"w": 10, "t": 20, "b": -60, "s": 137,
                 **({k: _answers[k] for k in ("p", "g", "e", "L")} if _done else {})},
                {k.split()[1]: f"责任 {_answers[k]}" for k in _answers if k.startswith("blame")} if _done else {}),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一个值被用了两次，责任相加

    还有一种常见情况。看 $a \times c$ 这一步，乘号**两边**都是可以变动的数。如果只微调 $a$（$c$ 固定），结果的变化倍数是 $c$；
    如果只微调 $c$（$a$ 固定），结果的变化倍数是 $a$。这就是第 02 章的 $c \times x$ 规则，对左右两边各用一次。

    那如果 $a$ 和 $c$ 同时增加 $h$ 呢？想象一个宽 $a$、长 $c$ 的长方形，两条边各加长 $h$，面积就变成：

    > $$(a + h) \times (c + h) = a \times c + \boldsymbol{c \times h} + \boldsymbol{a \times h} + h \times h$$

    代入具体数字一眼就明白：设 $a = 3$、$c = 5$、$h = 0.01$。原本面积是 $3 \times 5 = 15$，两条边各加 $0.01$ 后变成 $3.01 \times 5.01 = 15.0801$。
    多出来的 $0.0801$ 正好分成三块：

    - 来自 $a$ 变长的那一条：$5 \times 0.01 = 0.05$
    - 来自 $c$ 变长的那一条：$3 \times 0.01 = 0.03$
    - 角落里的极小方块：$0.01 \times 0.01 = 0.0001$（和前两块相比微乎其微，$h$ 越小时越可以忽略不计）

    可以看到，经过第一条箭头（来自 $a$）的影响 $c \times h$，和经过第二条箭头（来自 $c$）的影响 $a \times h$，并排加在总变化量里。

    现在把它用到 $x \times x$ 上：**同一个**方框 $x$ 同时连到乘号两边，相当于 $a$ 和 $c$ 都是 $x$，而且一起变动：

    > $x$ 的责任 =（经过第一条箭头）+（经过第二条箭头）= $x + x$ = **$2 \times x$**

    这和第 02 章推出的 $x \times x$ 斜率完全吻合：当 $a = c = x$ 时，这个长方形正好就是正方形。
    由此得到反向传播的第二条核心规则：**一个值如果分叉连向多条箭头，每条箭头传回来的责任直接相加**。

    米娅 8 天的平均损失也遵循同样的道理。每一天都要用到同一个旋钮 `w`，所以 `w` 会从 8 天各自收到一份责任；把这 8 份责任加起来，
    就是 8 天*总损失*对 `w` 的斜率。接着求平均要把总损失除以 8——**除以 8 就是乘以 $\frac{1}{8}$**。
    根据第 02 章的常数倍规则（函数乘以常数 $c$，斜率也乘以 $c$），总斜率同样要乘以 $\frac{1}{8}$（即除以 8）。一句话总结：

    > $$\text{平均损失的斜率} = \text{每天斜率的平均}$$

    ## 练习 2：用代码算一天的斜率

    写出 `day_slopes(w, b, t, s)`，把你刚才手工算的过程写成代码：先往前算出中间值，再往回算出责任。它返回**两个**值：`w` 的斜率和 `b` 的斜率。

    要同时返回两个值，用逗号隔开即可：`return blame_w, blame_b`。调用这个函数的地方也用同样的方式接住两个返回值：
    `dw, db = day_slopes(10, -60, 14, 63)` 会把 476 存进 `dw`，把 34 存进 `db`。（和 `for t, s in zip(...)` 一样：两个变量名接住两个值。）

    下面的代码格里已经用注释写好了每一步：先是往前的三步，再是往回算的各个责任，最后是 `return`。
    把注释变成代码的方法是删掉每行开头的 `#` **以及紧跟在后面的那个空格**，让这一行和 `todo()` 左侧对齐；全部取消注释后，再删掉 `todo()`。
    （如果报错 `IndentationError: unexpected indent`，说明某一行 `#` 后面的空格没删干净。）注意切换为英文输入法。

    `L` 自己不需要单独写一行：它的责任是 1，所以 `blame_e = 2 * e * 1`，直接写成 `2 * e` 就行。
    """)
    return


@app.function
def day_slopes(w, b, t, s):
    # p = w * t
    # g = p + b
    # e = g - s
    # blame_e = 2 * e
    # blame_g = blame_e
    # blame_p = blame_g
    # blame_b = blame_g
    # blame_w = t * blame_p
    # return blame_w, blame_b
    todo()


@app.cell(hide_code=True)
def _():
    check("03", "day_slopes", day_slopes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：计算全部数据的梯度

    写出 `gradient(w, b, xs, ys)`，返回**平均**损失对 `w` 和对 `b` 的斜率。按刚才推出的结论，它就是每天斜率的平均值：

    1. 准备两个从 0 开始的累计变量：`total_w = 0` 和 `total_b = 0`。
    2. 用 `for t, s in zip(xs, ys):` 遍历每一天。调用 `dw, db = day_slopes(w, b, t, s)` 拿到当天的斜率，分别累加到 `total_w` 和 `total_b` 上。
    3. 循环结束后（按 Backspace，让 `return` 和 `for` 对齐）：`return total_w / len(xs), total_b / len(xs)`。

    （记得保持英文输入法：逗号、冒号、括号都要用半角。）
    """)
    return


@app.function
def gradient(w, b, xs, ys):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("03", "gradient", gradient, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 用数值轻推做梯度检查

    手写的梯度很容易藏着不易察觉的小错误，算出来的数看着还挺像那么回事。工程实践中，大家总会拿手写梯度和数值轻推的结果对比一遍，
    这个习惯叫作**梯度检查**（gradient check）。下面用几组不同的旋钮取值来检验你的 `gradient`：
    """)
    return


@app.cell(hide_code=True)
def _():
    try:
        _rows = []
        for _w, _b in ((10, -60), (12, -108), (6, 0), (15, -150)):
            _dw, _db = gradient(_w, _b, temps, sales)
            _nw = nudge_slope(lambda v: mse(v, _b, temps, sales), _w)
            _nb = nudge_slope(lambda v: mse(_w, v, temps, sales), _b)
            _rows.append(f"| {_w}, {_b} | {_dw:,.3f} | {_nw:,.3f} | {_db:,.3f} | {_nb:,.3f} |")
        _out = mo.md("| w, b | 你的 w 斜率 | 轻推测量 | 你的 b 斜率 | 轻推测量 |\n"
                     "|---|---|---|---|---|\n" + "\n".join(_rows))
    except Exception:
        _out = mo.md("*完成练习 2 和 3 后就能运行梯度检查了。*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    练习通过后，每一对数字都会高度吻合，只差步长 $h = 0.0001$ 带来的极小误差。`w` 那一列的轻推结果多出大约 0.06，
    这正是前面见过的平方余项——等于米娅 8 天 $t \times t$ 的平均值（约 603.5）乘以 $h = 0.0001$。（如果你的两列差得很远，上面的检查反馈会提示哪道练习出了问题。）

    ## 练习 4：用精确梯度做梯度下降

    写出 `fit(w, b, xs, ys, rate, steps)`，用你刚写好的精确梯度代替轻推，同时优化两个旋钮。用 `for i in range(steps):` 循环 `steps` 次，每次循环里：

    - `dw, db = gradient(w, b, xs, ys)`
    - `w = w - rate * dw` 和 `b = b - rate * db`（沿用第 02 章的更新规则，两个旋钮各更新一次）

    循环结束后 `return w, b`。（同样保持英文输入法。）
    """)
    return


@app.function
def fit(w, b, xs, ys, rate, steps):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("03", "fit", fit, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 速度对比：数值轻推 vs 反向传播

    下面两种方法都从 `w = 6`、`b = 0` 出发拟合米娅的数据，都跑 20,000 步、用相同的学习率。
    左边用第 02 章的数值轻推测斜率，右边用你写的 `fit`。点一下按钮，看看谁跑得更快。
    """)
    return


@app.cell(hide_code=True)
def _():
    race = mo.ui.run_button(label="开始比赛")
    race
    return (race,)


@app.cell(hide_code=True)
def _(race):
    mo.stop(not race.value, mo.md("*点击**开始比赛**查看结果。*"))
    try:
        _start = time.perf_counter()
        _w, _b = 6, 0
        for _i in range(20000):
            _sw = nudge_slope(lambda v: mse(v, _b, temps, sales), _w)
            _sb = nudge_slope(lambda v: mse(_w, v, temps, sales), _b)
            _w, _b = _w - 0.001 * _sw, _b - 0.001 * _sb
        _t_nudge = time.perf_counter() - _start
        _start = time.perf_counter()
        _fw, _fb = fit(6, 0, temps, sales, 0.001, 20000)
        _t_back = time.perf_counter() - _start
        _out = mo.md(f"""
| | 数值轻推 | 反向传播 |
|---|---|---|
| 每步要算几次损失 | 4 次（每个旋钮 2 次） | 大约 1 次（一次正向 + 一次反向） |
| 用时 | {_t_nudge:.2f} 秒 | {_t_back:.2f} 秒 |
| 最终的 w、b | {_w:.2f}, {_b:.1f} | {_fw:.2f}, {_fb:.1f} |
| 最终损失 | {mse(_w, _b, temps, sales):.1f} | {mse(_fw, _fb, temps, sales):.1f} |
""")
    except Exception:
        _out = mo.md("*完成练习 2 到 4 后才能运行对比。*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    两者的结果一模一样，而且哪怕只有两个旋钮，反向传播也已经更快了。旋钮少时差距还不算夸张；
    一旦模型有十亿个旋钮，数值轻推每走一步就要多算十亿次损失，而反向传播依然只需要倒着走一遍。

    ## 下一章预告

    这一章的反向步骤是你手工推导出来的。可一旦算式换了样子（比如直线换成曲线、输入变多、或者搭起真正的神经网络），每次都手工推一遍就太麻烦了。
    在**第 04 章**里，你会亲手写一个小引擎：它在正向计算时自动记下每一步，算完后自己倒着走回来——把记账的苦活彻底交给计算机。

    ## 这一章你掌握了什么

    - 把复杂算式拆成由简单步骤组成的计算图
    - 用第 02 章的规则写出每条箭头的局部斜率
    - 用链式法则把局部斜率相乘，并在一个值被多次使用时把责任相加
    - 手工和用代码实现反向传播，并用梯度检查验证结果
    - 用精确梯度运行梯度下降拟合模型

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**在微积分教材里，一个值的责任叫作它的*偏导数*，写作 $\frac{\partial L}{\partial w}$（意思是“其他变量保持不变时，$L$ 对 $w$ 的斜率”）。
    """)
    return


if __name__ == "__main__":
    app.run()
