import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="03 · 责任往回传")

with app.setup(hide_code=True):
    import time

    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.nav import nav
    from learn_llm.diagrams import one_day
    from learn_llm.plot import chart

    LOCALE = "zh-CN"
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 03 · 责任往回传

    第 02 章最后留了一个缺口。米娅的损失由 `(w × 14 + b − 63)²`（`²` 表示“自己乘自己”）这样的部分组成：是平方，但平方的东西本身又依赖 `w`。
    我们的斜率规则还管不了这种情况。这一章，你会学会求任何由小步骤组成的算式的精确斜率，
    并看到训练所有神经网络（包括这门课最后的语言模型）的那个核心思想。

    下面是前几章写过、这一章会再用到的函数：直线的损失（第 01 章）和用轻推测斜率（第 02 章）。
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
    ## 一天，一步一步算

    取米娅的第一天：`t = 14` 度，卖了 `s = 63` 个。取旋钮为 `w = 10`、`b = -60` 的直线。
    这一天的平方误差分四个小步骤算出来。每一步只做**一件**事，并给结果起个名字。
    （代码里的名字末尾带 `1`，表示“第 1 天”；在文字和图里我们就简单地叫 `p`、`g`、`e`、`L`。）
    """)
    return


@app.cell
def _():
    p1 = 10 * 14      # p = w × t：猜测里和陡度有关的部分
    g1 = p1 + (-60)   # g = p + b：猜测
    e1 = g1 - 63      # e = g − s：误差
    L1 = e1 * e1      # L = e × e：平方误差
    p1, g1, e1, L1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    （最后一行用逗号隔开，一次要四个值。页面会把它们显示成一个带编号的列表，编号从 0 开始：
    第 0 项是 `p` = 140，第 1 项是 `g` = 80，第 2 项是 `e` = 17，第 3 项是 `L` = 289。）

    下面把同样的步骤画成图。每个框装一个值。箭头表示“这个值被用来算出那个值”，箭头上的符号说明怎么算。
    不能改的数据在左边（`t`，以及靠下的 `s`）；旋钮 `w` 和 `b` 是我们想调的值。
    """)
    return


@app.cell(hide_code=True)
def _():
    one_day({"w": 10, "t": 14, "p": 140, "b": -60, "g": 80, "s": 63, "e": 17, "L": 289})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这种表示“值怎样由别的值算出来”的图，叫作**计算图**（computation graph）。

    ## 每一步单独看都很简单

    一次只看一个箭头，问第 02 章的那个问题：如果箭头起点的值被轻推一点点，终点的值**每单位**会动多少？
    （其他一切保持不动。）这就是这个箭头的**局部斜率**（local slope）。每一个都是你已经会的规则：

    | 步骤 | 轻推谁 | 结果每单位移动 | 为什么（第 02 章的规则） | 这里 |
    |---|---|---|---|---|
    | `L = e × e` | `e` | `2 × e` | x × x 的斜率是 2 × x | 2 × 17 = **34** |
    | `e = g − s` | `g` | `1` | `g` 增加 h，`e` 也增加 h | **1** |
    | `g = p + b` | `p` | `1` | `p` 增加 h，`g` 也增加 h | **1** |
    | `g = p + b` | `b` | `1` | 同上，换成 `b` | **1** |
    | `p = w × t` | `w` | `t` | c × x 的斜率是 c（这里 c 是 `t`，它固定不变） | **14** |

    ## 汇率相乘

    /// admonition | 🧰 工具箱：连续换汇
    假设 1 欧元能换 8 元人民币，1 元人民币能换 20 日元。1 欧元能换多少日元？先把 1 欧元换成 8 元，
    再把这 8 元每一元都换成 20 日元：8 × 20 = **160** 日元。沿着一条链换算时，汇率要**相乘**。
    ///

    局部斜率正是这样的汇率：“每 1 单位这个，换成多少单位那个”。跟着 `w` 的一个微小轻推 `h`，一路走到 `L`：

    | 值 | 移动了 |
    |---|---|
    | `w` | `h` |
    | `p` | `14 × h`（汇率 14） |
    | `g` | `1 × 14 × h` = `14 × h`（汇率 1） |
    | `e` | `1 × 14 × h` = `14 × h`（汇率 1） |
    | `L` | 大约 `34 × 14 × h` = `476 × h`（汇率 34；那一点点多余部分下面解释） |

    所以 `L` 的移动量是 `w` 的 476 倍：`L` 对 `w` 的斜率是 **14 × 1 × 1 × 34 = 476**。
    沿着路径把局部斜率相乘，这叫作**链式法则**（chain rule）。

    这真的精确吗？看看多余的部分可能从哪里来。前三步都是“直线型”的步骤（`× 14`、`+ b`、`− s`）：
    对它们来说，变化量**恰好**等于汇率乘以轻推量，没有任何多余（就像第 02 章的 `7 × (x + h) = 7 × x + 7 × h`）。
    所以 `e` 恰好移动 `14 × h`。只有最后一步平方会多出东西。用第 02 章的正方形图，边长 `e = 17`，
    轻推量不是 `h` 而是 `14 × h`：它多出两条 `17 × 14 × h`（合起来 `2 × 17 × 14 × h` = `476 × h`），
    再加上一个小角 `(14 × h) × (14 × h)` = `196 × h × h`。除以 `h`：

    > 测出的斜率 = 476 + 196 × h

    和第 02 章的 `2 × x + h` 是同一个规律：轻推量越小，测量值越逼近 **476**。（如果“弯曲”步骤在中间，后面还跟着直线型步骤呢？
    看 `3 × (x × x)`。平方留下一个小角 `h × h`；后面的 `× 3` 把它变成 `3 × h × h`；除以 `h` 得到 `3 × h`，
    `h` 变小时它照样消失。直线型步骤只会按比例放大缩小多余部分，不会妨碍它消失。）
    用轻推验证一下：
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
    用 `h = 0.0001` 轻推得到大约 `476.0196`（后面那一长串数字是计算机的舍入误差），也就是 `476 + 196 × 0.0001`：正是链式法则的 476，加上我们刚才预测的多余部分。
    轻推量更小，就更接近：

    | 轻推量 h | 0.1 | 0.01 | 0.001 | 0.0001 |
    |---|---|---|---|---|
    | 测出的斜率 | 495.6 | 477.96 | 476.196 | 476.0196 |

    ## 倒着算：责任

    我们可以从 `w` 出发把轻推往前传一遍，再从 `b` 出发传一遍，每个旋钮都这样来一遍。但有一个更聪明的顺序。
    从终点开始，给每个框一个**责任**（blame）：这个框每移动 1 单位，`L` 移动多少。然后沿着箭头一个一个往回走。（这里的“每 1 单位”指的是微小轻推时的比率，和第 02 章一样，
    不是真的整整移动 1：把 `e` 从 17 改成 18，`L` 会移动 35，也就是 34 加上平方留下的多余部分。）

    - `L` 自己：`L` 移动 1，`L` 就移动 1。责任 **1**。
    - `e`：`e` 每移动 1 单位，`L` 移动 34（局部斜率），而 `L` 每动 1 单位就是动 1。责任 34 × 1 = **34**。
    - `g`：`g` 每移动 1 单位，`e` 移动 1，而 `e` 每动 1 单位，`L` 就动 34。责任 1 × 34 = **34**。
    - `p`：`p` 每移动 1 单位，`g` 移动 1。责任 1 × 34 = **34**。
    - `b`：`b` 每移动 1 单位，也让 `g` 移动 1。责任 1 × 34 = **34**。
    - `w`：`w` 每移动 1 单位，`p` 移动 14，而 `p` 每动 1 单位，`L` 就动 34。责任 14 × 34 = **476**。

    （责任也可以是负数。负的责任表示把这个框调高反而会让损失*变小*；这时步进规则里的减号会把旋钮往上推，也就是往下坡走。）

    全部只用了一条规则：

    > 一个框的责任 = 它那条箭头的局部斜率 × 箭头所指那个框的责任

    下面每个框下方的红字就是它的责任：
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
    `w` 的责任就是它的斜率 476，和链式法则的答案一样。这不是巧合：往前我们算的是 14 × 1 × 1 × 34，往回算的是
    34 × 1 × 1 × 14，同样几个数，只是顺序反过来，而乘法的顺序不影响结果。`b` 的责任就是它的斜率 34。
    这种倒着走的过程叫作**反向传播**（backpropagation）：对损失的责任沿着计算图往回传，传到每一个旋钮。

    为什么要倒着走？注意 `w` 和 `b` 共用路径的后半段（`g` → `e` → `L`）。倒着走时，这段共用的部分只算**一次**，
    所有旋钮都能重复使用它。往回走时每个框只经过一次，每条箭头只做一次乘法，和往前走一样。
    所以一个有十亿个旋钮的模型，只要再多花大约算一次损失的工夫，就能得到全部十亿个斜率。对比一下轻推：**每个**旋钮都要算两次损失。

    所有旋钮的损失斜率合在一起，叫作**梯度**（gradient）。第 02 章那种一步步往下坡走的方法叫“梯度下降”，名字就是这么来的。

    ## 练习 1：手工填表

    现在你来做一遍。取米娅的第三天：`t = 20`，`s = 137`，旋钮还是 `w = 10`、`b = -60`。
    先**往前**算（各个值），再**往回**算（各个责任）。把每个答案填进对应的框里；每填一个都会马上检查。
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
        "p": "p = w × t", "g": "g = p + b（b 是负数）", "e": "e = g − s", "L": "L = e × e",
        "blame e": "L = e × e 的局部斜率是 2 × e；乘以 L 的责任（1）",
        "blame g": "局部斜率 1，乘以 e 的责任",
        "blame p": "局部斜率 1，乘以 g 的责任",
        "blame b": "局部斜率 1，乘以 g 的责任",
        "blame w": "局部斜率 t（这里是 20），乘以 p 的责任",
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

    还有一种情况。看 `a × c` 这样的步骤，其中**两边**都是会动的值。只轻推 `a`（`c` 不动），结果每单位移动 `c`；
    轻推 `c`（`a` 不动），结果每单位移动 `a`。这就是第 02 章的 `c × x` 规则，对每一边各用一次。

    如果两边同时动，各动 `h` 呢？想象一个边长为 `a` 和 `c` 的长方形，就像第 02 章的正方形。两条边都加长 `h`，
    会沿一条边多出一条 `c × h`，沿另一条边多出一条 `a × h`，再加一个小角 `h × h`：

    > (a + h) × (c + h) = a × c + **c × h** + **a × h** + h × h

    `c × h` 这一条是经过第一条箭头（来自 `a`）的影响，`a × h` 是经过第二条箭头（来自 `c`）的影响。
    它们并排出现在总和里，所以**相加**。小角 `h × h` 是老样子的多余部分，`h` 变小时就消失了。

    现在看 `x × x`：**同一个**框同时连到两边，所以 `a` 和 `c` 都是 `x`，而且一起动。按这条规则：

    > x 的责任 =（经过第一条箭头）+（经过第二条箭头）= x + x = **2 × x**

    这正是第 02 章 `x × x` 的斜率，本来也应该如此：当 `a = c = x` 时，这个长方形*就是*第 02 章的正方形。
    一般的规则是：一个值连到好几条箭头时，经过每条箭头传回来的责任要加起来。

    米娅的完整损失也一样。损失是 8 天的**平均**，而每一天都用到了 `w`。所以 `w` 要从 8 天各收一份责任，加起来，
    再除以 8（求平均要除以 8，而“c 乘以一个函数”的斜率是 c 乘以它的斜率）。一句话：

    > 平均损失的斜率 = 每天斜率的平均

    ## 练习 2：用代码算一天的斜率

    写出 `day_slopes(w, b, t, s)`，做你刚才手工做的事：先往前算，再往回算。它返回**两个**值：`w` 的斜率和 `b` 的斜率。

    要返回两个值，用逗号隔开：`return blame_w, blame_b`。调用这个函数的地方也用同样的方式接住两个值：
    `dw, db = day_slopes(10, -60, 14, 63)` 会把 476 存进 `dw`，把 34 存进 `db`。（就像 `for t, s in zip(...)`：两个名字，两个值。）

    单元格里用注释列出了步骤，每行一个框：先是往前的三步，再是往回的各个责任，最后是 `return`。
    要把注释变成代码，就删掉 `#` **和它后面的那个空格**，让这一行和 `todo()` 从同一列开始。每一行都这样做，最后删掉 `todo()`。
    （如果看到 `IndentationError: unexpected indent`，说明 `#` 后面的空格还在。）记得用英文输入法。

    `L` 自己不需要单独一行：它的责任是 1，所以 `blame_e = 2 * e * 1`，也就是 `2 * e`。
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
    ## 练习 3：所有天的梯度

    写出 `gradient(w, b, xs, ys)`：**平均**损失对 `w` 和对 `b` 的斜率。按刚才推出的结论，它就是每天斜率的平均：

    1. 两个累计值从 0 开始：`total_w = 0` 和 `total_b = 0`。
    2. 用 `for t, s in zip(xs, ys):` 逐天循环。用 `dw, db = day_slopes(w, b, t, s)` 得到当天的斜率，把它们分别加到各自的累计值上。
    3. 循环结束后（按 Backspace，让 `return` 和 `for` 对齐）：`return total_w / len(xs), total_b / len(xs)`。

    （记得用英文输入法：逗号、冒号、括号都要用半角。）
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
    ## 用轻推检查梯度

    手写的梯度可能藏着错误，而且算出来的数看着还挺像样。专业人士总会拿它和轻推结果对比，这个习惯叫作**梯度检查**
    （gradient check）。下面是在几组不同旋钮取值下对你的梯度做的检查：
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
        _out = mo.md("*完成练习 2 和 3 后就能做梯度检查了。*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    练习完成后，每一对都吻合，只差轻推留下的那一点点。`w` 那一列的轻推结果差了大约 0.06：这又是那个多余部分，
    这里等于米娅各天 `t × t` 的平均（约 603.5）乘以 `h = 0.0001`。（如果你的两列差得比这多得多，上面的检查会告诉你该看哪道练习。）

    ## 练习 4：用精确斜率做梯度下降

    写出 `fit(w, b, xs, ys, rate, steps)`：对两个旋钮做梯度下降，用你的精确梯度代替轻推。用 `for i in range(steps):` 重复 `steps` 次，循环里：

    - `dw, db = gradient(w, b, xs, ys)`
    - `w = w - rate * dw` 以及 `b = b - rate * db`（第 02 章的步进规则，每个旋钮各用一次）

    循环结束后 `return w, b`。（同样用英文输入法。）
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
    ## 轻推和反向传播比一比

    下面两种方法都从 `w = 6`、`b = 0` 出发拟合米娅的直线，都走 20,000 步、用同样的学习率。
    左边用轻推测斜率（像第 02 章那样）；右边用你的 `fit`。按下按钮，让它们比一比。
    """)
    return


@app.cell(hide_code=True)
def _():
    race = mo.ui.run_button(label="开始比赛")
    race
    return (race,)


@app.cell(hide_code=True)
def _(race):
    mo.stop(not race.value, mo.md("*按**开始比赛**。*"))
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
| | 轻推 | 反向传播 |
|---|---|---|
| 每步要算几次损失 | 4 次（每个旋钮 2 次） | 大约 1 次（一次往前 + 往回） |
| 用时 | {_t_nudge:.2f} 秒 | {_t_back:.2f} 秒 |
| 最后的 w、b | {_w:.2f}, {_b:.1f} | {_fw:.2f}, {_fb:.1f} |
| 最后的损失 | {mse(_w, _b, temps, sales):.1f} | {mse(_fw, _fb, temps, sales):.1f} |
""")
    except Exception:
        _out = mo.md("*完成练习 2 到 4 后才能比赛。*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    答案一样，而且即使只有两个旋钮，反向传播也更快。两个旋钮时差距不大；有十亿个旋钮时，轻推每一步要多算十亿次损失，
    而反向传播仍然只需要往回走一遍。

    ## 下一步

    这个算式的往回步骤是你手工推出来的。每当算式变了（曲线代替直线、更多输入、神经网络），往回的步骤都得重新手工推一遍。
    在**第 04 章**里，你会做一个小引擎：它在计算时记下每一步，然后自己往回走，让计算机来记账。

    ## 你现在会做的事

    - 把一个计算拆成由小步骤组成的计算图
    - 用第 02 章的规则求出每一步的局部斜率
    - 用乘法把局部斜率串起来，一个值被用了多次时把责任相加
    - 手工和用代码做反向传播，并用轻推检查梯度
    - 用精确梯度拟合模型

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**在教材里，一个值的责任叫作它的*偏导数*，写作 ∂L/∂w（“其他一切保持不动时，L 对 w 的斜率”）。
    """)
    return


@app.cell(hide_code=True)
def _():
    nav("03", LOCALE)
    return


if __name__ == "__main__":
    app.run()
