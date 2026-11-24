import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="05 · 一次处理很多数")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np

    from learn_llm import hand, todo
    from learn_llm.checks import check
    from learn_llm.diagrams import grids

    LOCALE = "zh-CN"
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    rain = [0, 1, 0, 1, 0, 0, 1, 0]
    weekend = [0, 0, 1, 0, 0, 1, 0, 0]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 05 · 一次处理很多数

    第 04 章的引擎一次只处理一个数，但真正的模型往往要同时处理成千上万、甚至上百万个数。这一章，你会学着把一整串数打包在一起算，
    并且只用几行代码，就能一次算出很多个旋钮的梯度。

    ## 米娅的本子上还有别的列

    米娅记下的不只是气温。她还记了那天有没有**下雨**（下雨记 1，没下记 0），以及是不是**周末**（是记 1，不是记 0）。
    下面再看看她这 8 天的记录，并附上只看气温时最好的那条直线 $w \times t + b$（它的损失最低，大约是 27）猜了多少，以及和实际销量差了多少。（编号从 0 开始，和 Python 的习惯一致。）
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
    看“差多少”这一列。在三个下雨天（第 1、3、6 天），直线猜**多了**——下雨时大家买得少；在两个周末（第 2、5 天），它又猜**少了**。
    直线只看得见气温，再怎么调也没法兼顾这些情况。现在我们把另外两列也交给模型。

    ## 每个输入一个旋钮

    给每个输入配一个自己的旋钮：

    > $$\text{猜测} = w_t \times \text{气温} + w_r \times \text{下雨} + w_k \times \text{周末} + b$$

    $w_t$ 表示气温每升高 1 度多卖几个冰淇淋，$w_r$ 表示下雨天多卖（或少卖）几个，$w_k$ 表示周末多卖几个（$k$ 取自英文 wee**k**end“周末”，因为 $w$ 已经被旋钮占用了），$b$ 和以前一样是起始高度。
    拿第 1 天（17 °C，下雨，不是周末）试试旋钮 $w_t = 12$、$w_r = -8$、$w_k = 4$、$b = -105$：

    > $$12 \times 17 + (-8) \times 1 + 4 \times 0 + (-105) = 204 - 8 + 0 - 105 = 91$$

    她实际卖了 90 个，比直线猜的 96 准得多。

    看看这个算式的结构：旋钮 `[12, -8, 4]` 和这一天的输入 `[17, 1, 0]` 是**按位置一一配对**相乘的——第一个配第一个，第二个配第二个，
    第三个配第三个——再把乘积加起来，最后加上 `b`。

    像 `[12, -8, 4]` 这样的一串数叫作**向量**（vector）。把两个向量按位置一一配对相乘、再把乘积加起来，叫作它们的**点积**（dot product）：

    > $$\operatorname{dot}([12, -8, 4], [17, 1, 0]) = 12 \times 17 + (-8) \times 1 + 4 \times 0 = 196$$

    其实你买东西结账时就在算点积。苹果 2 元、面包 3 元、牛奶 1 元，买了 4 个苹果、1 个面包、2 盒牛奶，总账单就是价格和数量的点积：
    $2 \times 4 + 3 \times 1 + 1 \times 2 = 13$。模型做猜测也是一回事：每个旋钮就像对应输入的“单价”。

    ### 手算：再算两天

    旋钮不变（$w_t = 12$、$w_r = -8$、$w_k = 4$、$b = -105$），算出另外两天的猜测：第 3 天（22 °C，下雨，不是周末）和第 5 天（28 °C，没下雨，是周末）。
    先把每一天写成三个输入组成的向量，和 `[12, -8, 4]` 做点积，再加上 $b$。
    """)
    return


@app.cell(hide_code=True)
def _():
    day_guesses = hand.boxes(["第 3 天", "第 5 天"])
    return (day_guesses,)


@app.cell(hide_code=True)
def _(day_guesses):
    hand.marked(day_guesses, {"第 3 天": 151, "第 5 天": 235}, {
        "第 3 天": r"""输入是 `[22, 1, 0]`：$12 \times 22 + (-8) \times 1 + 4 \times 0$，再 $- 105$""",
        "第 5 天": r"""输入是 `[28, 0, 1]`：$12 \times 28 + (-8) \times 0 + 4 \times 1$，再 $- 105$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    她实际卖了 148 和 234。留意一下 0 和 1 的作用：不下雨的日子，$w_r$ 乘以 0 就直接消掉了；下雨的日子乘以 1，才会生效。
    “下雨”这个输入就像一个**开关**，控制着下雨旋钮的开和关。

    ## 练习 1：点积

    写出 `dot(ws, xs)`，计算两个等长向量的点积。思路和第 01 章算损失的循环很像：先让 `total` 从 0 开始，
    用 `for w, x in zip(ws, xs):` 按位置一一配对地遍历两个列表，每次把 `w * x` 加到 `total` 上，最后返回 `total`。
    （`ws` 和 `xs` 末尾的 s 是英文复数，表示“一串 w”“一串 x”。记得用英文输入法。）
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
    ## 把所有天数放进一张表

    每一天都是一个包含三个输入的向量。把 8 天叠在一起，就得到一张表：每天一**行**，每个输入一**列**。
    像这样的一张数表叫作**矩阵**（matrix）。这张表有 8 行 3 列，所以我们说它的**形状**（shape）是 8 乘 3，写作 `(8, 3)`——先写行数，再写列数。

    在普通的 Python 里，矩阵就是一个“列表里的列表”：外层列表装各行，里面每个小列表就是一行。（`rows` 的意思是“各行”。）
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
    `len(rows)` 是 8：外层列表装着 8 个元素，每个元素本身又是一个装着 3 个数的列表。用 `for row in rows:` 循环，每次就能拿到一天的输入。

    ## 练习 2：预测所有的天

    写出 `predict_all(rows, ws, b)`，返回一个包含所有猜测的列表，每一行对应一个猜测。先建一个空列表 `guesses = []`。对 `rows` 里的每一行 `row`，
    算出猜测 `dot(ws, row) + b`，再用 `guesses.append(...)`（第 04 章）把它加到列表末尾，最后返回 `guesses`。
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

    意思是“导入 NumPy，并给它起个简称 `np`”。之后，调用 NumPy 里的功能时，都在前面加上 `np.` 就行，比如 `np.array`。

    `np.array(...)` 可以把列表（或者列表里的列表）转换成 NumPy **数组**（array）——一种能一次性对所有数做计算的向量或矩阵。
    每个数组都有一个 `shape` 属性，用来查看它有几行几列。（最后一行用逗号隔开的几个值会并排显示成带编号的列表；
    输出里，形状 `(8, 3)` 会显示成包含 8 和 3 的小列表，而 `y` 的形状 `(8,)` 则显示成只有一个 8。）
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
    `X` 的形状是 `(8, 3)`：8 行 3 列。`y` 是销量，形状是 `(8,)`：一个有 8 个数的向量。（括号里的逗号不是印刷错误——Python 里只有一个元素的元组必须带逗号，表示这是一个长度为 8 的一维数组。）
    按习惯，矩阵通常用大写字母命名。

    ### 一次对每个数做计算

    对数组来说，`+`、`-`、`*`、`/` 都是**逐个位置**计算的。这和普通列表不一样，对比一下：
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
    对两个列表用 `+`，是把它们**拼接**成一个更长的列表；而对两个数组用 `+`，则是逐个位置相加：`[11, 22, 33]`。
    对数组用 `*`，同样是逐个位置相乘：`[10, 40, 90]`。（显示在输出列表里时，NumPy 数组的元素之间不带逗号，写作 `[11 22 33]`。）

    如果拿单独一个数和数组运算，它会同时作用到数组里的每个位置：
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

    在 NumPy 里，点积有自己的专属符号：`@`。两个向量之间的 `@`，正好就是你刚才写的 `dot`：
    """)
    return


@app.cell
def _():
    np.array([12, -8, 4]) @ np.array([17, 1, 0])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    196，和前面一样。（NumPy 显示单个数时可能写成 `np.int64(196)` 或 `np.float64(...)`，只看括号里的那个数就行。）

    在矩阵和向量之间，`矩阵 @ 向量` 是让**每一行**都和这个向量做点积，每行得到一个数。这正好就是去掉 $+ b$ 的 `predict_all`：
    """)
    return


@app.cell
def _(X):
    X @ np.array([12, -8, 4]) - 105
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    同样的 8 个猜测，一行代码就搞定了：`X @ w` 算出每天的点积，`- 105` 再从每个结果里各减掉 105。下面把 `@` 做的事画出来：
    `X` 的每一行和旋钮做点积，得到一个数；图里标出的是第 1 行（第 1 天），它的点积是 $17 \times 12 + 1 \times (-8) + 0 \times 4 = 196$：
    """)
    return


@app.cell(hide_code=True)
def _():
    _rows = [[14, 0, 0], [17, 1, 0], [20, 0, 1], [22, 1, 0], [25, 0, 0], [28, 0, 1], [31, 1, 0], [33, 0, 0]]
    _w = [12, -8, 4]
    grids([
        {"name": "X  (8, 3)", "values": _rows, "rows": [f"第 {i} 天" for i in range(8)],
         "cols": ["气温", "下雨", "周末"], "marks": {(1, c): "blue" for c in range(3)}},
        "@",
        {"name": "w  (3,)", "values": [[v] for v in _w], "marks": {(r, 0): "red" for r in range(3)}},
        "=",
        {"name": "X @ w  (8,)", "values": [[sum(a * b for a, b in zip(r, _w))] for r in _rows],
         "marks": {(1, 0): "box"}},
    ], "X 有 8 行，每行 3 个输入；w 有 3 个旋钮；X @ w 有 8 个数，每行一个点积。第 1 行 17、1、0 和 12、-8、4 做点积得 196")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这里把旋钮画成竖着的一列，方便看出它依次去配每一行：第一个配第一个，第二个配第二个，第三个配第三个。算出的 8 个结果再各自减去 105，就是 8 天的猜测。

    ### 形状必须对得上

    做 `矩阵 @ 向量` 时，每一行都得和向量一样长：这里是 3 个输入配 3 个旋钮。用形状来说：`(8, 3) @ (3,)` 得到 `(8,)`。
    **中间挨着**的两个数（3 和 3）必须相同，算完就消掉了；剩下的是两边的数（8）。

    可以把形状想成两张首尾相接的多米诺骨牌：挨着的两头必须是同一个数，接上之后，只看得见外侧的两头：

    | 左边 | @ | 右边 | 挨着的两头 | 结果 |
    |---|---|---|---|---|
    | `(8, 3)` | @ | `(3,)` | 3 和 3：对得上 | `(8,)`，剩下外侧的 8 |
    | `(2, 3)` | @ | `(3,)` | 3 和 3：对得上 | `(2,)` |
    | `(8, 3)` | @ | `(2,)` | 3 和 2：对不上 | 报错 |

    如果对不上，NumPy 会报错停下。
    下一个单元格故意演示一下：3 个输入只配了 2 个旋钮，让你看看这个报错长什么样：
    """)
    return


@app.cell
def _(X):
    X @ np.array([12, -8])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    报错里提到了 `matmul`（*matrix multiplication*“矩阵乘法”的缩写，也就是 NumPy 对 `@` 的内部叫法），以及对不上的维度大小。
    中间那串（`gufunc signature ...`）不用管，真正有用的是最后一句：`size 2 is different from 3`（“长度 2 和 3 不匹配”）。以后看到这个报错，第一时间检查形状就行。

    ## 练习 3：用 NumPy 预测

    写出 `predict_np(X, w, b)`，用 `@` 算出 `X` 每一行的猜测。这里 `X` 是一个矩阵，`w` 是一个向量，都是 NumPy 数组。一行就够：`return X @ w + b`。
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

    损失就是第 01 章的均方误差：每天的误差 $e = \text{猜测} - \text{卖出}$，再求 $e \times e$ 的均值。用数组，每一步都同时作用在 8 天上：

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
    当旋钮取 `w = [10, 0, 0]`、$b = -60$ 时，下雨和周末的权重都是 0（相当于关掉了），模型就退回成了只看气温的直线 $10 \times t - 60$，
    算出来的损失是 177.5。而换上 `[12, -8, 4]` 和 `-105`、把下雨和周末都用起来后，损失一下子降到了 4。

    ## 很多个旋钮的梯度

    现在来算斜率。先回忆单输入直线某一天的算法：误差 $e = w \times t + b - s$，这一天的损失是 $e \times e$，$e$ 的责任是 $2 \times e$，于是

    - $w$ 的斜率是 $2 \times e \times t$（从 $e$ 到 $w$ 的局部斜率是 $t$，也就是 $w$ 乘的那个输入）；
    - $b$ 的斜率是 $2 \times e$（从 $e$ 到 $b$ 的局部斜率是 1）。

    平均损失的梯度，就是这些斜率在所有天上的**均值**。

    有三个输入时，某一天的误差是 $e = w_t \times t + w_r \times r + w_k \times k + b - s$，道理完全一样。只轻推 $w_r$、其他旋钮不动时，算式里只有 $w_r \times r$ 这一项会变——$w_r$ 每增加 1，误差就增加 $r$，
    所以从 $e$ 到 $w_r$ 的局部斜率就是 $r$。每个旋钮都只和自己的输入相乘，按同样的规则：

    - $w_t$ 的斜率 = $2 \times e \times t$，$w_r$ 的斜率 = $2 \times e \times r$，$w_k$ 的斜率 = $2 \times e \times k$；
    - $b$ 的斜率 = $2 \times e$。

    所以，**每个旋钮的斜率就是 $2 \times e$ 乘以它自己的输入**，再对所有天求均值。以 $w_r$ 为例，把 8 天的式子展开来写就是：

    > $$(2 e_0 r_0 + 2 e_1 r_1 + \dots + 2 e_7 r_7) \div 8$$

    一眼就能看出，括号里正是把 8 个 $2 \times e$ 和“下雨”那一列按位置一一配对相乘再相加——也就是它们的**点积**——最后再除以 8。

    下面用真实的数逐天算一遍，旋钮还是取刚才那条只看气温的直线（$w_t = 10$、$w_r = 0$、$w_k = 0$、$b = -60$）：
    """)
    return


@app.cell(hide_code=True)
def _():
    _lines = ["| 天 | 气温 | 下雨 | 周末 | 误差 $e$ | $2 \\times e$ | $2 \\times e \\times \\text{下雨}$ | $2 \\times e \\times \\text{周末}$ |",
              "|---|---|---|---|---|---|---|---|"]
    _sum_r = _sum_k = 0
    for _i, (_t, _r, _k, _s) in enumerate(zip(temps, rain, weekend, sales)):
        _e = 10 * _t - 60 - _s
        _sum_r, _sum_k = _sum_r + 2 * _e * _r, _sum_k + 2 * _e * _k
        _lines.append(f"| {_i} | {_t} | {_r} | {_k} | {_e} | {2 * _e} | {2 * _e * _r} | {2 * _e * _k} |")
    _lines.append(f"| **总和** | | | | | | **{_sum_r}** | **{_sum_k}** |")
    _lines.append(f"| **均值**（$\\div 8$） | | | | | | **{_sum_r / 8:g}** | **{_sum_k / 8:g}** |")
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    又是开关的作用。在“下雨”那一列里，只有真正下雨的日子（第 1、3、6 天）是 1，所以只有这三天的误差会算进总和：$40 + 24 + (-14) = 50$，$50 / 8 = 6.25$。
    没下雨的日子乘以 0 就直接归零了，不会影响 $w_r$。这很合情理：下雨旋钮该怎么调，本来就只能看下过雨的日子。
    同理，“周末”那一列里只有第 2、5 天起作用：$6 + (-28) = -22$，$-22 / 8 = -2.75$。

    ### `向量 @ 矩阵`：每一列做一次点积

    前面用 `X @ w` 时，是拿向量去配矩阵的每一**行**；现在反过来了，我们要拿向量去配矩阵的每一**列**。好消息是 `@` 两头都能用。可以记住这样一个统一的画面——**NumPy 的一维数组本身不分横竖**：

    - 放在 `@` **右边**（`X @ w`，`(8, 3) @ (3,)` → `(8,)`）时，它**竖着**去配 `X` 的每一**行**（算 8 天的预测值）；
    - 放在 `@` **左边**（`(2 * e) @ X`，`(8,) @ (8, 3)` → `(3,)`）时，它**横着**去配 `X` 的每一**列**（把 8 天的误差汇总给 3 个旋钮）。

    先拿一个小例子看看 `向量 @ 矩阵` 怎么算：
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

    这一次向量是横着躺下的一行，依次顺着每一列往下走。图里标出的是第 1 列：$1 \times 2 + 1 \times 5 = 7$。
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "v  (2,)", "values": [[1, 1]], "marks": {(0, 0): "blue", (0, 1): "blue"}},
        "@",
        {"name": "M  (2, 3)", "values": [[1, 2, 3], [4, 5, 6]], "marks": {(0, 1): "red", (1, 1): "red"}},
        "=",
        {"name": "v @ M  (3,)", "values": [[5, 7, 9]], "marks": {(0, 1): "box"}},
    ], "向量 1、1 和矩阵的每一列做点积：第 1 列是 2 和 5，得到 7")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 手算：`向量 @ 矩阵`

    算出 `[2, -1] @ [[1, 0, 3], [2, 1, 0]]`。矩阵的形状是 `(2, 3)`，所以答案有 3 个数，每列一个。
    """)
    return


@app.cell(hide_code=True)
def _():
    by_column = hand.boxes(["第 0 列", "第 1 列", "第 2 列"])
    return (by_column,)


@app.cell(hide_code=True)
def _(by_column):
    hand.marked(by_column, {"第 0 列": 0, "第 1 列": -1, "第 2 列": 6}, {
        "第 0 列": r"""第 0 列是 `[1, 2]`：$2 \times 1 + (-1) \times 2$""",
        "第 1 列": r"""第 1 列是 `[0, 1]`：$2 \times 0 + (-1) \times 1$""",
        "第 2 列": r"""第 2 列是 `[3, 0]`：$2 \times 3 + (-1) \times 0$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    回到冰淇淋的例子。`X` 的每一列对应一个输入（气温、下雨、周末），而向量 $2 \times e$ 则是每天一个数。

    所以 `(2 * e) @ X` 一次就能给出我们要的三个点积：$2 \times e$ 分别和气温列、下雨列、周末列的点积。
    形状：`(8,) @ (8, 3)` 得到 `(3,)`，正好每个旋钮一个数。再除以天数，就是均值。

    顺序很重要。`X @ (2 * e)` 是 `(8, 3) @ (8,)`：中间挨着的 3 和 8 对不上，会触发 matmul 报错。向量必须放在前面。

    ## 练习 5：用 NumPy 算梯度

    写出 `gradient_np(X, y, w, b)`，返回 `w` 里所有旋钮的斜率，以及 `b` 的斜率：

    1. `e = X @ w + b - y`
    2. `grad_w = (2 * e) @ X / len(y)`：一个向量，`w` 里的每个旋钮一个斜率。（`len(y)` 是 8，也就是天数：`len` 对数组也能用。）
    3. `grad_b = np.mean(2 * e)`：`b` 的斜率。
    4. `return grad_w, grad_b`

    （为什么一个用 `/ len(y)`，另一个用 `np.mean`？因为 `@` 已经做完了求和，只差除以 `len(y)`；而 `2 * e` 还没求和，所以直接 `np.mean` 一步到位。`grad` 是 gradient“梯度”的缩写，记得用英文输入法。）
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
    把这组只看气温的旋钮（`[10, 0, 0]`、`-60`）代入 `gradient_np`，算出来 $w_t$ 和 $b$ 的斜率分别是 -94 和 2.5——由于下雨和周末的旋钮现在都是 0，这两个斜率就是单输入直线原本的梯度。
    另外两个新斜率则指出了新旋钮该往哪边调：$w_r$ 的斜率是 6.25（正数，按步进规则 $\text{旋钮} - \text{学习率} \times \text{斜率}$ 会把 $w_r$ 往**小**里调，
    说明下雨会让销量减少）；$w_k$ 的斜率是 -2.75（负数，会把 $w_k$ 往**大**里调，说明周末会让销量增加）。

    ## 用轻推来检查

    这是个新公式，所以用“轻推”的办法验算一下：把一个旋钮轻推一个很小的 `h`，看损失变了多少，再除以 `h`。
    给 `w` 加上 `[h, 0, 0]`，就只轻推了 $w_t$（逐个位置相加）；加上 `[0, h, 0]` 只轻推 $w_r$；加上 `[0, 0, h]` 只轻推 $w_k$。
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
    结果大约是 -94、6.25 和 -2.75，和 `gradient_np` 算出来的一致（微小的零头来自步长 `h` 本身不是无限小）。

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
    算出来的结果和数学原理完全一样，只是 NumPy 把循环放到底层编译好的高速代码里一口气算完，省去了 Python 逐条解释执行的开销。
    正是靠这种速度，我们才能在海量数据上训练模型。

    ## 收获：训练全部四个旋钮

    现在就可以用梯度下降同时调整所有旋钮，找出最合适的一组值了。动手前先处理一个小细节：气温的数值范围（14 到 33）比下雨和周末（0 或 1）
    大得多，如果直接训练，同一个学习率很难同时照顾到它们。所以我们把气温换算一下——以 25 °C 为起点、每 10 度记为 1 格，另外两列保持不变：

    > `Xs = (X - np.array([25, 0, 0])) / np.array([10, 1, 1])`

    这里又用到了 NumPy 的一个顺手特性。`X` 的形状是 `(8, 3)`，而 `np.array([25, 0, 0])` 的形状是 `(3,)`，相当于只有一行。
    做减法时，NumPy 会自动把这一行**逐行配对**到 `X` 的每一行上：每天的气温减 25，其余两列减 0。接着的 `/ np.array([10, 1, 1])`
    也是同理，把每天的气温除以 10，其余两列除以 1。像这样自动把较小的数组按行对齐并套用到整个矩阵（或者把单个数套用到每个位置），叫作**广播**（broadcasting）。
    以前三天为例：
    """)
    return


@app.cell(hide_code=True)
def _():
    _dots = ["⋮", "⋮", "⋮"]
    _blue = {(r, c): "blue" for r in range(3) for c in range(3)}
    _red = {(r, c): "red" for r in range(3) for c in range(3)}
    mo.vstack([
        grids([
            {"name": "X", "values": [[14, 0, 0], [17, 1, 0], [20, 0, 1], _dots], "rows": ["第 0 天", "第 1 天", "第 2 天", ""]},
            "−",
            {"name": "每一行都用 [25, 0, 0]", "values": [[25, 0, 0]] * 3 + [_dots], "marks": _blue},
            "=",
            {"name": "X − [25, 0, 0]", "values": [[-11, 0, 0], [-8, 1, 0], [-5, 0, 1], _dots]},
        ], "先从 X 的每一行减去 25、0、0：第 0 天的 14、0、0 变成 -11、0、0"),
        grids([
            {"name": "X − [25, 0, 0]", "values": [[-11, 0, 0], [-8, 1, 0], [-5, 0, 1], _dots],
             "rows": ["第 0 天", "第 1 天", "第 2 天", ""]},
            "÷",
            {"name": "每一行都用 [10, 1, 1]", "values": [[10, 1, 1]] * 3 + [_dots], "marks": _red},
            "=",
            {"name": "Xs", "values": [[-1.1, 0, 0], [-0.8, 1, 0], [-0.5, 0, 1], _dots]},
        ], "再把每一行除以 10、1、1：第 0 天变成 -1.1、0、0"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    NumPy 底层并不会真的把这一行复制 8 份，而是对每一天重复读取同一行，效果和复制出来完全一样。下雨和周末两列算完保持不变，因为减 0、除以 1 不会改变任何数。

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
    损失降到了大约 **0.54**，而之前只看气温的最佳直线损失大约是 **27**。把这组旋钮换回米娅原来的单位来看：

    - $w_t$ $\approx$ 每 10 度 119：大约**每升高 1 度多卖 11.9 个**；
    - $w_r$ $\approx$ -8.4：下雨天大约**少卖 8 个**；
    - $w_k$ $\approx$ 4.2：周末大约**多卖 4 个**；
    - $b$ $\approx$ 193：不下雨的工作日、25 °C 时的猜测（这时换算后的气温是 0）。

    你在表格里用眼睛看出来的下雨和周末的影响，模型自己找到了，还量出了具体大小。

    ## 下一步

    现在，对很多输入、很多天，你都能用几行代码算出猜测、损失和梯度。在**第 06、07 章**里，你会教模型认出手写的数字：
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


if __name__ == "__main__":
    app.run()
