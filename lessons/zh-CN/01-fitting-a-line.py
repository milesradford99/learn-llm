import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="01 · 用直线拟合数据")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.nav import nav
    from learn_llm.plot import chart

    LOCALE = "zh-CN"
    BEST_B = -108  # 最后那张“只留一个旋钮”的图里固定的起始高度


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 01 · 用直线拟合数据

    这一章里，程序会第一次从数据中学出规律。每一块都由你用第 00 章学过的工具亲手搭出来。

    ## 问题

    米娅在街边卖冰淇淋。她连续八天记下了中午的气温和当天卖出的冰淇淋数量。
    天气预报说明天 30 °C，她该准备多少个？

    下面是她的记录，写成了两个列表。位置是对应的：第一个气温配第一个销量（那天 14 °C，卖了 63 个），
    第二个配第二个，依此类推。
    """)
    return


@app.cell
def _():
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]
    return sales, temps


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 并排走过两个列表

    想一天一天地看，就要把同一天的气温*和*销量放在一起。`zip` 就是干这个的：它像拉链的两边一样，
    把两个列表一对一对地扣在一起。

    `for t, s in zip(temps, sales):` 每轮循环取出一对：`t` 是那天的气温，`s` 是那天的销量。
    （`in` 前面写了两个名字，用逗号隔开，因为每轮会同时拿到两个值。）

    `print` 会把值作为一行文字显示在单元格下方。在循环里需要用它：平时单元格只显示最后一行的结果，
    而 `print` 每运行一次就显示一次。
    """)
    return


@app.cell
def _(sales, temps):
    for t, s in zip(temps, sales):
        print(t, s)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 画出来看看

    /// admonition | 🧰 工具箱：怎么看图
    图表把一对对数字变成一个个点。底部那条线（**横轴**，也叫 x 轴）表示第一个数，这里是气温：越往右越热。
    左边那条竖线（**纵轴**，也叫 y 轴）表示第二个数，这里是销量：越往上卖得越多。
    每个点代表一天：先往右找到它的气温，再往上找到它的销量。
    ///

    找找 14 °C 那天的点：它在底部 14 的正上方，高度是 63。
    """)
    return


@app.cell(hide_code=True)
def _(sales, temps):
    chart((10, 36), (0, 320), "气温（°C）", "卖出的冰淇淋（个）", dots=list(zip(temps, sales)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这些点从左往右一路上升：天越热，卖得越多。而且上升得挺均匀，几乎排成一条直线。
    如果能用数字描述这条线，就能读出任何气温下的销量，包括 30 °C。

    ## 用数字描述一条直线

    下面这个算式把气温变成对销量的猜测：

    > 猜测 = w × 气温 + b

    `w` 和 `b` 是由我们来选的两个数。看看它们各管什么：

    - **`w`** 是**陡度**：每升高一度，多卖多少个。`w = 12` 表示每热一度，多卖 12 个。`w` 越大，线越陡。
    - **`b`** 是**起始高度**：直线在 0 °C 处的高度。因为气温是 0 时，算式变成 `w × 0 + b`，而 `w × 0` 等于 0，
      只剩下 `b`。图左边缘的大圆点标出了它。米娅不会在 0 °C 卖冰淇淋，所以 `b` 是负数也没关系；
      只有点所在的那一段线才要紧。

    拖动两个滑块，看那条红线。试着让它贴着这些点。注意 `w` 的作用：线会绕着 0 °C 处的大圆点转动，
    就像门绕着门轴转。所以改了 `w` 之后，通常还得再调一调 `b`。
    """)
    return


@app.cell(hide_code=True)
def _():
    w_slider = mo.ui.slider(0, 20, step=0.5, value=6, label="w（陡度）", show_value=True)
    b_slider = mo.ui.slider(-200, 100, step=1, value=-100, label="b（起始高度）", show_value=True)
    return b_slider, w_slider


@app.cell(hide_code=True)
def _(b_slider, sales, temps, w_slider):
    mo.vstack([
        mo.hstack([w_slider, b_slider], justify="start", gap=2),
        chart((0, 36), (-200, 320), "气温（°C）", "卖出的冰淇淋（个）",
              dots=list(zip(temps, sales)), lines=[(w_slider.value, b_slider.value)],
              marker=(0, b_slider.value)),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    像这样把输入变成猜测的算式，叫作**模型**（model）。它里面可以调的数，这里是 `w` 和 `b`，
    叫作**参数**（parameter）：可以把它们想成模型上的旋钮。在整门课里，“学习”只有一个意思：
    **为旋钮找到好的取值**。今天你用手转；下一章，让计算机自己转。

    ## 练习 1：模型的猜测

    （提醒：写代码时切换到英文输入法，括号、逗号、冒号都要用英文符号。）

    写出 `predict(w, b, x)`：它返回对输入 `x` 的猜测，用 Python 写就是 `w * x + b`。
    （这个算式对任何输入都适用，所以输入就叫 `x`，而不叫“气温”。）
    """)
    return


@app.function
def predict(w, b, x):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("01", "predict", predict, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一条线错得有多离谱？

    用眼睛看，有些线明显比别的好。要让计算机来比较，我们需要一个**数**，表示一条线错了多少。

    先看一天。**误差**（error）就是猜测减去真实值。（和第 00 章的红色报错框不是一回事：这里它是一个数，表示猜偏了多少。）

    - 猜 150，真实 137：误差 = 150 − 137 = **13**（多猜了 13 个）
    - 猜 120，真实 137：误差 = 120 − 137 = **−17**（少猜了 17 个）

    在图上，每个误差就是点和线之间那段虚线。下表列出了你调出的这条线每天的误差。
    （这里的两个滑块和上面是同一组：动哪一个，所有副本都会跟着变。）
    """)
    return


@app.cell(hide_code=True)
def _(b_slider, sales, temps, w_slider):
    _w, _b = w_slider.value, b_slider.value
    _rows = "\n".join(
        f"| {_t} | {_s} | {_w * _t + _b:g} | {_w * _t + _b - _s:g} |" for _t, _s in zip(temps, sales)
    )
    mo.vstack([
        mo.hstack([w_slider, b_slider], justify="start", gap=2),
        mo.hstack([
            chart((0, 36), (-200, 320), "气温（°C）", "卖出的冰淇淋（个）",
                  dots=list(zip(temps, sales)), lines=[(_w, _b)], marker=(0, _b),
                  segments=[(_t, _s, _t, _w * _t + _b) for _t, _s in zip(temps, sales)]),
            mo.md("| 气温 | 卖出 | 猜测 | 误差 |\n|---|---|---|---|\n" + _rows),
        ], align="center"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在我们想给整条线一个数。最直接的想法是把误差加起来。可是看看误差 13 和 −13：
    它们加起来是 0，好像这条线完美无缺，其实两天都猜错了。猜多和猜少互相抵消了。

    解决办法是先把每个误差**平方**（自己乘自己），再加起来。

    /// admonition | 🧰 工具箱：平方
    一个数的平方就是它乘以它自己：3² = 3 × 3 = 9。在 Python 里写作 `3 ** 2`。
    `**` 比 `+`、`-` 先算，所以“差的平方”要加括号：`(10 - 7) ** 2` 等于 9，而 `10 - 7 ** 2` 是 10 − 49。

    负数的平方是正数。为什么 (−3) × (−3) = 9？看第一个数每次减 1 时的规律：2 × (−3) = −6，1 × (−3) = −3，
    0 × (−3) = 0。结果每次都**增加** 3。再往下走一步：(−1) × (−3) = 3，(−2) × (−3) = 6，(−3) × (−3) = 9。
    所以每个平方误差都大于或等于零，谁也抵消不了谁。

    平方还会让大的误差比小的误差重要得多：误差 2 变成 4，误差 10 却变成 100。
    某一天错得很离谱的线，会被狠狠扣分。
    ///

    ## 练习 2：平方误差

    写出 `squared_error(guess, truth)`：先算出误差（猜测减真实值），再返回它的平方。
    """)
    return


@app.function
def squared_error(guess, truth):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("01", "squared_error", squared_error, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 整条线的一个数：损失

    把每一天的平方误差都算出来，再求它们的**均值**（第 00 章的平均数：加起来，再除以个数）。
    这就是**均方误差**（mean squared error，MSE）。越小越好；等于 0 说明这条线恰好穿过每一个点。

    衡量一个模型错了多少的数，有个统称：**损失**（loss）。这一章用的损失就是均方误差。
    感受一下它的大小：如果一条线每天都差大约 5 个，每个平方误差就是 25 左右，它们的平均也是 25 左右。

    ## 练习 3：一条线的损失

    写出 `mse(w, b, xs, ys)`：旋钮为 `w` 和 `b` 的直线，在输入 `xs`、真实答案 `ys` 上的均方误差。
    你需要的都已经有了：

    1. 累计值从 0 开始：`running = 0`。
    2. 用 `for x, y in zip(xs, ys):` 逐对循环。每一对里，用你写的 `predict` 得到猜测，
       再用你写的 `squared_error` 和 `y` 比较。像第 00 章用 `big = big + 1` 计数那样，把结果加到累计值上。
       分两行写最简单：先 `guess = predict(w, b, x)`，再把 `squared_error(guess, y)` 加到累计值上：
       `running = running + ...`。Python 先用旧的 `running` 算出右边，再把结果存回 `running`。
    3. 循环结束后，返回累计值除以数据一共有几对（`len(xs)`）。

    先完成练习 1 和 2：这道题会调用它们。
    """)
    return


@app.function
def mse(w, b, xs, ys):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("01", "mse", mse, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 寻宝

    现在把你的损失当成记分牌。移动滑块，让损失尽可能小。
    把它降到 **100 以下**，米娅就可以相信你的直线了。

    一个管用的办法：

    1. 先从数据估算陡度：点从 63 升到 288（多卖 225 个），同时气温从 14 升到 33（热了 19 度）。
       大约每度多卖 225 ÷ 19 ≈ 12 个，所以把 `w` 调到 12 附近。
    2. 一边看损失的数字（不只看图），一边调 `b`，停在损失最小的地方。
    3. 把 `w` 往上或往下调一格，再重做第 2 步。如果损失变小了，就继续往这个方向调；如果没有，就退回去。
       调 `w` 不再有帮助时就停下。

    想精细调节的话，先点一下滑块，再用键盘的**方向键**：每按一下移动一格。
    """)
    return


@app.cell(hide_code=True)
def _(b_slider, sales, temps, w_slider):
    try:
        _loss = mse(w_slider.value, b_slider.value, temps, sales)
        _verdict = (
            mo.callout(mo.md(f"**损失：{_loss:,.1f}。** 低于 100：好直线！它预测 30 °C 时能卖 "
                             f"**{predict(w_slider.value, b_slider.value, 30):g}** 个。"), kind="success")
            if _loss < 100 else mo.callout(mo.md(f"**损失：{_loss:,.1f}。** 继续调：降到 100 以下。"), kind="warn")
        )
    except Exception:
        _verdict = mo.callout(mo.md("*完成练习 3 后，这里会显示损失。*"), kind="neutral")
    mo.vstack([
        mo.hstack([w_slider, b_slider], justify="start", gap=2),
        chart((0, 36), (-200, 320), "气温（°C）", "卖出的冰淇淋（个）",
              dots=list(zip(temps, sales)), lines=[(w_slider.value, b_slider.value)],
              marker=(0, b_slider.value)),
        _verdict,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## 损失的形状

    你是怎么找到好直线的？多半是：稍微动一下一个滑块，看损失有没有变小，再动一下。我们把这个寻找过程画出来。

    为了能画成图，只留一个旋钮：把起始高度固定在 `b = {BEST_B}`（接近最好的取值），只改变 `w`。
    这张图的坐标轴换了：横轴是 `w`，不是气温；纵轴是损失。所以曲线上的每个点代表一整条直线：
    它的陡度，以及它错了多少。这些都是用**你写的** `mse` 算出来的。拖动滑块，大圆点会沿着曲线移动。
    """)
    return


@app.cell(hide_code=True)
def _():
    valley_w = mo.ui.slider(10, 14, step=0.1, value=10.5, label="w", show_value=True)
    return (valley_w,)


@app.cell(hide_code=True)
def _(sales, temps, valley_w):
    try:
        _curve = [(10 + i * 0.05, mse(10 + i * 0.05, BEST_B, temps, sales)) for i in range(81)]
        _w = valley_w.value
        _out = mo.vstack([
            valley_w,
            chart((10, 14), (0, 2700), "w（陡度），b 固定为 -108", "损失",
                  curve=_curve, marker=(_w, mse(_w, BEST_B, temps, sales))),
        ])
    except Exception:
        _out = mo.md("*完成练习 3 后才能看到这张图。*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    损失形成了一个**山谷**。太平的线（左边）和太陡的线（右边）损失都很高；最好的 `w` 就在谷底。
    找好的旋钮取值，就是找山谷的最低点。

    只有一两个旋钮时，用眼睛就能找到。而这门课最后的语言模型有*几十亿*个旋钮，没人能拖动几十亿个滑块。
    在**第 02 章**里，计算机会自己找到谷底：办法是每一步都往下坡走。

    ## 你现在会做的事

    - 看懂由点组成的图，用 `zip` 并排走过两个列表
    - 用两个旋钮描述一条直线：陡度 `w` 和起始高度 `b`
    - 用损失衡量模型错了多少：均方误差
    - 通过让损失变小来找到好的旋钮取值

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**这种方法叫*线性回归*（linear regression），是统计学里最古老的工具之一。
    """)
    return


@app.cell(hide_code=True)
def _():
    nav("01", LOCALE)
    return


if __name__ == "__main__":
    app.run()
