import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="05 · 一次处理很多数")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.nav import nav

    LOCALE = "zh-CN"
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    rain = [0, 1, 0, 1, 0, 0, 1, 0]
    weekend = [0, 0, 1, 0, 0, 1, 0, 0]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 05 · 一次处理很多数

    第 04 章的引擎一次只处理一个数。真正的模型要同时处理成千上万、甚至几百万个数。这一章，你会学着一次处理整串的数，
    并且只用几行代码就算出很多个旋钮的梯度。

    ## 米娅的本子上还有别的列

    米娅记下的不只是气温。她还记了那天有没有**下雨**（下雨记 1，没下记 0），以及是不是**周末**（是记 1，不是记 0）。
    下面再列一遍她这 8 天的记录，并列出最好的那条直线 `w × t + b`（损失最低，大约 27）的猜测，以及它差了多少。（编号从 0 开始，这是 Python 的数法。）
    """)
    return


@app.cell(hide_code=True)
def _():
    _w, _b = 11.9445, -107.4326  # 这 8 天最好的直线
    _lines = ["| 编号 | 气温 | 下雨 | 周末 | 卖出 | 直线的猜测 | 差多少 |", "|---|---|---|---|---|---|---|"]
    for _i, (_t, _r, _k, _s) in enumerate(zip(temps, rain, weekend, sales)):
        _g = _w * _t + _b
        _lines.append(f"| {_i} | {_t} °C | {_r} | {_k} | {_s} | {_g:.0f} | {_g - _s:+.0f} |")
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    看“差多少”这一列。在三个下雨天（第 1、3、6 天），直线猜**多了**：下雨时大家买得少。在两个周末（第 2、5 天），它猜**少了**。
    直线只知道气温，所以它做不到更好。我们把另外两列也交给模型。

    ## 每个输入一个旋钮

    给每个输入配一个自己的旋钮：

    > 猜测 = w_t × 气温 + w_r × 下雨 + w_k × 周末 + b

    `w_t` 是每升高 1 度多卖几个冰淇淋，`w_r` 是下雨天多卖（或少卖）几个，`w_k` 是周末多卖几个（k 取自英文 wee**k**end“周末”，因为 w 已经被旋钮用了），`b` 和以前一样是起始高度。
    在第 1 天（17 °C，下雨，不是周末）试试旋钮 `w_t = 12`、`w_r = -8`、`w_k = 4`、`b = -105`：

    > 12 × 17 + (-8) × 1 + 4 × 0 + (-105) = 204 − 8 + 0 − 105 = 91

    她卖了 90 个：比直线猜的 96 更接近。

    看看这个算式的样子：旋钮 `[12, -8, 4]` 和这一天的输入 `[17, 1, 0]` 是**按位置一一配对**相乘的，第一个配第一个，第二个配第二个，
    第三个配第三个，再把乘积加起来。最后再加上 `b`。

    像 `[12, -8, 4]` 这样的一串数叫作**向量**（vector）。把两个向量按位置一一配对相乘、再把乘积加起来，叫作它们的**点积**（dot product）：

    > dot([12, -8, 4], [17, 1, 0]) = 12 × 17 + (-8) × 1 + 4 × 0 = 196

    其实你买东西时就在算点积。苹果 2 元、面包 3 元、牛奶 1 元，你买了 4 个苹果、1 个面包、2 盒牛奶，账单就是价格和数量的点积：
    2 × 4 + 3 × 1 + 1 × 2 = 13。模型的猜测也是这样：每个旋钮就像它那个输入的“单价”。

    ## 练习 1：点积

    写出 `dot(ws, xs)`：两个一样长的向量的点积。它很像第 01 章算损失的循环：先让 `total` 从 0 开始，
    用 `for w, x in zip(ws, xs):` 按位置一一配对地遍历两个列表，每次把 `w * x` 加到总和上。最后返回总和。
    （`ws` 和 `xs` 末尾的 s 是英文的复数，表示“一串 w”“一串 x”。记得用英文输入法。）
    """)
    return


@app.function
def dot(ws, xs):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("05", "dot", dot, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一次看所有的天：一张表

    每一天都是一个有三个输入的向量。把 8 天叠起来，就得到一张表：每天一**行**，每个输入一**列**。
    像这样的一张数表叫作**矩阵**（matrix）。这一张有 8 行 3 列，所以说它的**形状**（shape）是 8 乘 3，写作 `(8, 3)`：先行，后列。

    在普通的 Python 里，矩阵就是一个列表的列表：每一行是里面的一个列表。（`rows` 的意思是“各行”。）
    """)
    return


@app.cell
def _():
    rows = [
        [14, 0, 0],
        [17, 1, 0],
        [20, 0, 1],
        [22, 1, 0],
        [25, 0, 0],
        [28, 0, 1],
        [31, 1, 0],
        [33, 0, 0],
    ]
    len(rows)
    return (rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `len(rows)` 是 8：外面的列表装着 8 个值，每个值本身又是一个装着 3 个数的列表。用 `for row in rows:` 循环，每次拿到一天的输入。

    ## 练习 2：预测所有的天

    写出 `predict_all(rows, ws, b)`：一个猜测的列表，每行一个。先做一个空列表 `guesses = []`。对 `rows` 里的每一个 `row`，
    猜测是 `dot(ws, row) + b`；用 `guesses.append(...)`（第 04 章）把它加到列表末尾。最后返回 `guesses`。
    （`predict_all` 是“全部预测”，`guesses` 是“各个猜测”。）
    """)
    return


@app.function
def predict_all(rows, ws, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("05", "predict_all", predict_all, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    用上面那组旋钮，`predict_all(rows, [12, -8, 4], -105)` 猜的是 63、91、139、151、195、235、259 和 291。
    她实际卖了 63、90、137、148、193、234、257 和 288。

    ## NumPy：同样的循环，写一次就够

    这样的循环太常见了，所以有现成的工具：**NumPy**，一个用来处理整个向量和矩阵的 Python 库（library，一套写好的代码）。
    这一页已经用下面这一行把它载入了：

    ```python
    import numpy as np
    ```

    意思是“载入 NumPy，并且让我用简称 `np` 叫它”。之后，NumPy 提供的所有东西都要在前面加上 `np.` 来用，比如 `np.array`。

    `np.array(...)` 把一个列表（或者列表的列表）变成一个 NumPy **数组**（array）：一个能一次对里面所有数做计算的向量或矩阵。
    数组有一个 `shape` 属性，说明它有几行几列。（和前几章一样，最后一行用逗号隔开的几个值会显示成编号的列表；
    形状 `(8, 3)` 本身也会显示成一个列表，里面是它的两个数 8 和 3；`y` 的形状 `(8,)` 则显示成只有一个 8 的列表。）
    """)
    return


@app.cell
def _(rows):
    X = np.array(rows)
    y = np.array(sales)
    X.shape, y.shape
    return X, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `X` 的形状是 `(8, 3)`：8 行 3 列。`y` 是销量，形状是 `(8,)`：一个有 8 个数的向量。（只有一个数、后面带逗号的形状，就是向量。）
    按习惯，矩阵用大写字母命名。

    ### 一次对每个数做计算

    对数组来说，`+`、`-`、`*`、`/` 都是**逐个位置**计算的。这和列表不一样，对比一下：
    """)
    return


@app.cell
def _():
    list_sum = [1, 2, 3] + [10, 20, 30]
    array_sum = np.array([1, 2, 3]) + np.array([10, 20, 30])
    array_times = np.array([1, 2, 3]) * np.array([10, 20, 30])
    list_sum, array_sum, array_times
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    对两个列表用 `+`，是把它们**接**成一个更长的列表。对两个数组用 `+`，是逐个位置相加：`[11, 22, 33]`。
    对数组用 `*`，也是逐个位置相乘：`[10, 40, 90]`。（像这样在编号列表里显示时，NumPy 的数组不带逗号，写作 `[11 22 33]`。）

    单独一个数会作用到每个位置上：
    """)
    return


@app.cell
def _():
    np.array([1, 2, 3]) * 10 + 5
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### `@`：点积

    在 NumPy 里，点积有自己的符号：`@`。两个向量之间的 `@`，正好就是你的 `dot`：
    """)
    return


@app.cell
def _():
    np.array([12, -8, 4]) @ np.array([17, 1, 0])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    196，和前面一样。（NumPy 显示单个数时可能写成 `np.int64(196)` 或 `np.float64(...)`：只看括号里的那个数就行。）

    在矩阵和向量之间，`矩阵 @ 向量` 是让**每一行**都和这个向量做点积，每行得到一个数。这正好就是去掉 `+ b` 的 `predict_all`：
    """)
    return


@app.cell
def _(X):
    X @ np.array([12, -8, 4]) - 105
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    同样的 8 个猜测，一行就算完了：`X @ w` 每天给出一个点积，`- 105` 会从每一个里减掉。

    ### 形状必须对得上

    做 `矩阵 @ 向量` 时，每一行都得和向量一样长：这里是 3 个输入配 3 个旋钮。用形状来说：`(8, 3) @ (3,)` 得到 `(8,)`。
    **中间挨着**的两个数（3 和 3）必须相同，算完就消掉了；剩下的是两边的数（8）。如果对不上，NumPy 会报错停下。
    下一个单元格故意这样做：3 个输入只配了 2 个旋钮，让你看看这个报错长什么样：
    """)
    return


@app.cell
def _(X):
    X @ np.array([12, -8])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    报错里提到了 `matmul`（*matrix multiplication*“矩阵乘法”的缩写，也就是 NumPy 对 `@` 的叫法），还有对不上的大小：
    中间那段（`gufunc signature ...`）不用管，有用的是最后一句：`size 2 is different from 3`（“大小 2 和 3 不一样”）。看到它时，检查一下形状。

    ## 练习 3：用 NumPy 预测

    写出 `predict_np(X, w, b)`：用 `@` 算出 `X` 每一行的猜测。这里 `X` 是一个矩阵，`w` 是一个向量，都是 NumPy 数组。一行就够：`return X @ w + b`。
    """)
    return


@app.function
def predict_np(X, w, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("05", "predict_np", predict_np, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：一次算出损失

    损失就是第 01 章的均方误差：每天的误差 `e = 猜测 − 卖出`，再求 `e × e` 的均值。用数组，每一步都同时作用在 8 天上：

    1. `e = X @ w + b - y`：8 个误差，组成一个向量。
    2. `e * e`：8 个平方误差（逐个位置相乘）。
    3. `np.mean(...)`：NumPy 求数组里所有数的均值。

    写出 `mse_np(X, y, w, b)`，返回 `np.mean(e * e)`。（`mse` 是 mean squared error“均方误差”的缩写。）
    """)
    return


@app.function
def mse_np(X, y, w, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("05", "mse_np", mse_np, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    旋钮取 `w = [10, 0, 0]`、`b = -60` 时，下雨和周末都被关掉了，模型就是第 01 章的直线 `10 × t − 60`。
    它的损失是 177.5，和你在第 01 章算出的一样。旋钮取 `[12, -8, 4]` 和 `-105` 时，损失是 4。

    ## 很多个旋钮的梯度

    现在来算斜率。第 03 章对直线的某一天算过：`e = w × t + b − s`，这一天的损失是 `e × e`，`e` 的责任是 `2 × e`，于是

    - `w` 的斜率是 `2 × e × t`（从 `e` 到 `w` 的局部斜率是 `t`，也就是 `w` 乘的那个输入）；
    - `b` 的斜率是 `2 × e`（从 `e` 到 `b` 的局部斜率是 1）。

    平均损失的梯度，就是这些斜率在所有天上的**均值**。

    有三个输入时，某一天的误差是 `e = w_t × t + w_r × r + w_k × k + b − s`。道理和以前一样。只轻推 `w_r`、其他旋钮不动时，算式里只有 `w_r × r` 这一项会变，`w_r` 每变 1，它变 `r`，
    所以从 `e` 到 `w_r` 的局部斜率是 `r`。每个旋钮都是这样：它只和自己的那个输入相乘，所以按同样的规则：

    - `w_t` 的斜率 = `2 × e × t`，`w_r` 的斜率 = `2 × e × r`，`w_k` 的斜率 = `2 × e × k`；
    - `b` 的斜率 = `2 × e`。

    所以，**每个旋钮的斜率就是 2 × e 乘以这个旋钮自己的输入**，再对所有天求均值。以 `w_r` 为例：拿出 8 个 `2 × e` 组成的向量，
    和“下雨”那一列按位置一一配对相乘，加起来，再除以 8。“一一配对相乘再加起来”就是点积：`2 × e` 和**下雨那一列**的点积。

    ### `向量 @ 矩阵`：每一列做一次点积

    `@` 也可以反过来用。`向量 @ 矩阵` 是让这个向量和矩阵的**每一列**做点积，每列得到一个数：
    """)
    return


@app.cell
def _():
    np.array([1, 1]) @ np.array([[1, 2, 3], [4, 5, 6]])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `[[1, 2, 3], [4, 5, 6]]` 的三列是 `[1, 4]`、`[2, 5]` 和 `[3, 6]`。它们分别和 `[1, 1]` 做点积，得到 5、7、9。
    用形状来说：`(2,) @ (2, 3)` 得到 `(3,)`；同样是中间挨着的两个数相同，算完就消掉。

    所以 `(2 * e) @ X` 一次就给出我们要的三个点积：`2 × e` 分别和气温列、下雨列、周末列的点积。
    形状：`(8,) @ (8, 3)` 得到 `(3,)`，每个旋钮一个数。再除以天数，就是均值。

    顺序很重要。`X @ (2 * e)` 是 `(8, 3) @ (8,)`：中间挨着的 3 和 8 对不上，会得到 matmul 报错。向量要放在前面。

    ## 练习 5：用 NumPy 算梯度

    写出 `gradient_np(X, y, w, b)`，返回 `w` 里所有旋钮的斜率，以及 `b` 的斜率：

    1. `e = X @ w + b - y`
    2. `grad_w = (2 * e) @ X / len(y)`：一个向量，`w` 里的每个旋钮一个斜率。（`len(y)` 是 8，也就是天数：`len` 对数组也能用。）
    3. `grad_b = np.mean(2 * e)`：`b` 的斜率。
    4. `return grad_w, grad_b`

    （`grad` 是 gradient“梯度”的缩写。记得用英文输入法。）
    """)
    return


@app.function
def gradient_np(X, y, w, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("05", "gradient_np", gradient_np, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    旋钮取直线的值（`[10, 0, 0]`、`-60`）时，`w_t` 和 `b` 的斜率是 -94 和 2.5：和第 03 章的梯度一模一样，因为多出来的旋钮都关掉了。
    两个新斜率告诉我们新旋钮该往哪边走：`w_r` 的斜率是 6.25（正的，所以步进规则 `旋钮 − 学习率 × 斜率` 让 `w_r` **变小**：
    下雨会让销量变少）；`w_k` 的斜率是 -2.75（负的，所以 `w_k` **变大**：周末会让销量变多）。

    ## 用轻推来检查

    这是个新公式，所以用第 02 章的办法检查一下：把一个旋钮轻推一个很小的 `h`，看损失变了多少，再除以 `h`。
    给 `w` 加上 `[h, 0, 0]`，就只轻推了 `w_t`（逐个位置相加）；加上 `[0, h, 0]` 只轻推 `w_r`；加上 `[0, 0, h]` 只轻推 `w_k`。
    这个单元格用到了你的 `mse_np`，所以在练习 4 通过之前会报错：
    """)
    return


@app.cell
def _(X, y):
    h = 0.0001
    w_line = np.array([10, 0, 0])
    base = mse_np(X, y, w_line, -60)
    nudge_t = (mse_np(X, y, w_line + np.array([h, 0, 0]), -60) - base) / h
    nudge_r = (mse_np(X, y, w_line + np.array([0, h, 0]), -60) - base) / h
    nudge_k = (mse_np(X, y, w_line + np.array([0, 0, h]), -60) - base) / h
    nudge_t, nudge_r, nudge_k
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约是 -94、6.25 和 -2.75，和 `gradient_np` 一样（多出来的那一点点来自 `h`，和第 02 章一样）。

    ## 快了多少？

    这里有 20 万个编出来的日子（气温、下雨、周末都由计算机随机生成，销量大致跟着它们走）。
    下面的单元格在它们上面分别给你的普通 Python 版 `predict_all` 和 NumPy 版 `predict_np` 计时。（需要先完成练习 1 到 3。）
    """)
    return


@app.cell(hide_code=True)
def _():
    _rng = np.random.default_rng(5)
    _many = np.column_stack([_rng.uniform(10, 38, 200_000), _rng.integers(0, 2, 200_000), _rng.integers(0, 2, 200_000)])
    _w = np.array([12.0, -8.0, 4.0])
    try:
        _start = time.perf_counter()
        predict_all(_many.tolist(), _w.tolist(), -105)
        _loops = time.perf_counter() - _start
        _start = time.perf_counter()
        predict_np(_many, _w, -105)
        _numpy = time.perf_counter() - _start
        _out = mo.md(
            f"普通循环：**{_loops:.3f} 秒**。NumPy：**{_numpy:.4f} 秒**，"
            f"大约**快 {_loops / _numpy:,.0f} 倍**。"
        )
    except Exception:
        _out = mo.md("*完成练习 1 到 3 后才能看到计时。*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    同样的猜测，同样的数学。NumPy 在它自己的高速机制里跑这些循环，而不是让 Python 一步一步地走。
    有了这样的速度，才能在大量数据上训练。

    ## 收获：训练全部四个旋钮

    现在做梯度下降，和第 02、03 章一样，只是所有旋钮一起动，找出它们的好值。先说一个细节，你在第 04 章见过：气温（14 到 33）比下雨和周末的值（0 或 1）
    大得多，所以一个学习率没法同时适合所有旋钮。我们再次把气温改成从 25 °C 算起、以 10 度为单位，另外两列不动：

    > `Xs = (X - np.array([25, 0, 0])) / np.array([10, 1, 1])`

    这里又用到 NumPy 的一个规则。`X` 的形状是 `(8, 3)`，而 `np.array([25, 0, 0])` 的形状是 `(3,)`：只有一行。
    NumPy 会把这一行用到 `X` 的**每一行**上：每一天的气温减去 25，其余两列减去 0。接着 `/ np.array([10, 1, 1])`
    把每一天的气温除以 10，其余两列除以 1。像这样把一个较小的数组用到每一行上（或者把单独一个数用到每个位置上），叫作**广播**（broadcasting）。

    下面的单元格让所有旋钮从 0 开始，按步进规则走 500 步。`w - 0.1 * grad_w` 一次更新 `w` 里的三个旋钮（逐个位置相减）。
    （它用到了你的 `gradient_np` 和 `mse_np`。）
    """)
    return


@app.cell
def _(X, y):
    Xs = (X - np.array([25, 0, 0])) / np.array([10, 1, 1])
    fit_w = np.array([0.0, 0.0, 0.0])
    fit_b = 0.0
    for step in range(500):
        grad_w, grad_b = gradient_np(Xs, y, fit_w, fit_b)
        fit_w = fit_w - 0.1 * grad_w
        fit_b = fit_b - 0.1 * grad_b
    fit_w, fit_b, mse_np(Xs, y, fit_w, fit_b)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    损失降到大约 **0.54**。最好的直线的损失大约是 **27**。把旋钮换回米娅的单位来读：

    - `w_t` ≈ 每 10 度 119：大约**每升高 1 度多卖 11.9 个**；
    - `w_r` ≈ -8.4：下雨天大约**少卖 8 个**；
    - `w_k` ≈ 4.2：周末大约**多卖 4 个**；
    - `b` ≈ 193：不下雨的工作日、25 °C 时的猜测（这时换算后的气温是 0）。

    你在表格里用眼睛看出来的下雨和周末的影响，模型自己找到了，还量出了大小。

    ## 下一步

    现在，对很多输入、很多天，你都能用几行代码算出猜测、损失和梯度。在**第 06 章**里，你会教模型认出手写的数字：
    每张图片是一个有几百个数的向量，模型也不再只给一个猜测，而是给十个，每个数字一个。

    ## 你现在会做的事

    - 说清楚什么是向量、点积、矩阵和它的形状
    - 用普通循环写出点积，以及一整张表的猜测
    - 使用 NumPy 数组：逐个位置的计算、用 `@` 做点积、`np.mean`
    - 检查形状对不对得上：`(8, 3) @ (3,)` 得到 `(8,)`，`(8,) @ (8, 3)` 得到 `(3,)`
    - 一次算出很多个旋钮的梯度，并用轻推检查它
    - 用广播把各列换算到合适的大小，训练一个有多个输入的模型

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**这样的模型叫作*多元线性回归*。`向量 @ 矩阵` 和 `矩阵.T @ 向量` 是一样的，其中 `.T` 是*转置*
    （行和列互换）；书里常写作 `Xᵀ e`。
    """)
    return


@app.cell(hide_code=True)
def _():
    nav("05", LOCALE)
    return


if __name__ == "__main__":
    app.run()
