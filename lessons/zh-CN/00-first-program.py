import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="00 · 你的第一个程序")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.nav import nav

    LOCALE = "zh-CN"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 00 · 你的第一个程序

    欢迎！学完这门课，你会亲手从零搭建一个自己的语言模型，也就是聊天机器人背后的那种程序。
    而这一切，都从这里的几行 Python 代码开始。

    你现在什么都不需要会。每当需要一个新概念，页面就会在用到它的地方当场讲清楚。

    ## 这个页面怎么用

    这个页面是一个**笔记本**（notebook），由一个个**单元格**（cell）组成。有的单元格是文字，
    比如这一个；有的单元格是 Python 代码，代码的运行结果显示在代码正下方。

    - **运行**代码单元格：点进单元格，按 **Shift + Enter**。运行后光标会跳到下一个单元格，想继续修改就再点回来。
      （鼠标移到单元格上时，旁边也会出现一个 ▶ 运行按钮。）
    - **写代码前先切换到英文输入法**（中文输入法下一般按 Shift 切换）。括号、冒号、逗号、空格都必须是英文的；
      `（）：，` 这些中文符号 Python 不认识。报错里出现 `invalid character`，多半就是混进了中文符号。
    - 写代码时，在以 `:` 结尾的一行后按 **Enter**，下一行会自动往右缩进四个空格；按 **Backspace** 可以退回四格。
    - 你的修改会自动保存在你自己的那份课程副本里。课程不会被你弄坏，放心大胆地试。
    - 迟早你会在某个单元格下面看到红色的**报错**框。这很正常，高手也天天遇到：它只是说明
      Python 没能执行某条指令。看它的最后一行，改好代码，再运行一次就行。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 把 Python 当计算器

    下面的单元格里是一个很小的 Python 程序：`2 + 3`，结果显示在它下方。
    改改数字，再运行一次试试。
    """)
    return


@app.cell
def _():
    2 + 3
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Python 能做常见的算术，只是有些符号和纸上写法不同：

    | 纸上写法 | Python 写法 | 含义 | 结果 |
    |---|---|---|---|
    | 7 × 6 | `7 * 6` | 乘 | `42` |
    | 10 ÷ 4 | `10 / 4` | 除 | `2.5` |
    | 2¹⁰ | `2 ** 10` | 乘方：10 个 2 相乘 | `1024` |

    在下面的单元格里多试几个，一次试一个：单元格只显示**最后一行**的结果。
    Python 不认识 × 和 ÷，要写成 `*` 和 `/`。和纸上一样，先算 `*`、`/`，再算 `+`、`-`：
    `1 + 2 * 3` 等于 `7`。括号里的先算：`(1 + 2) * 3` 等于 `9`。
    """)
    return


@app.cell
def _():
    7 * 6
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 给值起个名字

    同一个数反复输入很累。你可以给一个值起个**名字**，之后用名字代替它。
    这种存着值的名字叫**变量**（variable）。

    `price = 12` 的意思是“从现在起，`price` 就代表 12”。这里的 `=` 意思是*存进去*，
    而不是*等于*。下一个单元格把 12 存进 `price`，把 3 存进 `count`。存值的单元格下面不显示结果，这是正常的。

    名字由你来起，可以用英文字母、数字和 `_`，但不能有空格：`new_price` 可以，`new price` 不行。
    """)
    return


@app.cell
def _():
    price = 12
    count = 3
    return count, price


@app.cell
def _(count, price):
    price * count
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    在写着 `price = 12` 的那个单元格里，把它改成 `price = 15`，再按 **Shift + Enter**。
    下面的结果会自己从 `36` 变成 `45`：只要 `price` 变了，用到它的单元格就会重新运行。

    /// note | 这个笔记本的一条规则
    每个名字只能在**一个**单元格里用 `=` 存值。如果你在另一个单元格里再写 `price = 20`，
    笔记本会报错 *“The variable 'price' was defined by another cell”*（意思是这个名字已经在别的单元格里存过值了）。换个新名字（比如 `new_price`），或者直接改原来的单元格。
    （函数*内部*用到的名字不受这条限制，函数马上就会讲到。）
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 函数：有名字的菜谱

    **函数**（function）就像一份小菜谱：你给它一些输入，它按步骤做完，交回一个答案。
    下面是一个把数翻倍的函数：

    ```python
    def double(x):
        return x * 2
    ```

    逐行来看：

    - `def double(x):` 表示“我要定义一份叫 `double` 的菜谱，它的输入叫 `x`”。
      行末的冒号 `:` 表示“菜谱的步骤从下一行开始”。
    - 下一行往右缩进了四个空格，这叫**缩进**（indent）。Python 靠缩进知道这一行属于这份菜谱。
    - `return x * 2` 表示“答案是 `x * 2`”。`return`（返回）会把答案**交回**给调用它的地方。

    使用函数叫作**调用**（call）它：写出函数名，后面的括号里放上输入。
    `double(5)` 就是让菜谱在 `x` 代表 `5` 时运行一遍。
    """)
    return


@app.function
def double(x):
    return x * 2


@app.cell
def _():
    double(5)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：摄氏度转华氏度

    轮到你了。世界上大多数地方用摄氏度（°C）表示温度，美国用华氏度（°F）。换算规则是：

    > 摄氏温度乘以 9，再除以 5，最后加上 32。

    下面单元格的第一行把输入命名为 `celsius`（这次不是 `x`），所以规则里说“摄氏温度”的地方，
    都要写 `celsius`。

    单元格里你会看到两样东西：

    - 以 `#` 开头的一行是**注释**：写给人看的说明，Python 会忽略它。
    - `todo()` 是个占位符，意思是“还没写”。只选中 `todo()` 这几个字，把它换成 `return`
      加上换算算式。行首的四个空格不要动。

    然后按 **Shift + Enter**。每运行一次，单元格下方的检查都会告诉你结果。如果出现红色报错、检查结果不见了，
    说明单元格里有输入错误：看看是不是漏了 `:`、括号不成对、空格没对齐（报错里的 `IndentationError`
    就是这个意思），或者混进了中文符号。改好后检查结果就会回来。
    """)
    return


@app.function
def to_fahrenheit(celsius):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("00", "to_fahrenheit", to_fahrenheit, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    temperature = mo.ui.slider(-40, 100, value=20, label="摄氏度", show_value=True)
    return (temperature,)


@app.cell(hide_code=True)
def _(temperature):
    try:
        _answer = f"**{temperature.value} °C = {to_fahrenheit(temperature.value):g} °F**"
    except Exception:
        _answer = "*先完成练习 1，再拖动滑块试试你的函数。*"
    mo.hstack([temperature, mo.md(_answer)], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一次处理很多值：列表

    **列表**（list）按顺序装着好几个值，写在方括号里，用逗号隔开：`[3, 1, 4, 1, 5]`。
    图片、文字甚至整本书，在计算机里都是用一长串数字来表示的，所以这门课处处都会用到列表。

    `len`（*length* 的缩写，意思是“长度”）是 Python 自带的函数，它会告诉你列表里有几个值：
    """)
    return


@app.cell
def _():
    scores = [3, 1, 4, 1, 5]
    len(scores)
    return (scores,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 对每个值做同一件事：循环

    **循环**（loop）会对列表里的每一个值各执行一遍同样的步骤。
    下一个单元格数一数有几个分数大于 2。先逐行读一遍：

    - `big = 0` 把 `0` 存进 `big`，表示目前数到的个数。
    - `for s in scores:` 表示“依次取出 `scores` 里的每个值，叫它 `s`，对它运行下面缩进的那几行”。
      有 5 个分数，那几行就运行 5 遍。（`s` 只是我们随便起的名字，换成别的也行。）
    - `if s > 2:` 表示“只有当 `s` 大于 2 时，才运行我下面缩进的那一行”。
      （`>` 表示*大于*，`<` 表示*小于*。）
    - `big = big + 1` 表示“先算出 `big + 1`，再把结果存回 `big`”。当成数学式子看很奇怪，
      但别忘了 `=` 的意思是*存进去*。所以 `big` 每次加一。

    注意每行前面的空格：`if` 那一行在循环里面（4 个空格），计数那一行在 `if` 里面（8 个空格）。
    每当一行以 `:` 结尾，属于它的那些行就要再往右缩进四个空格。
    """)
    return


@app.cell
def _(scores):
    big = 0
    for s in scores:
        if s > 2:
            big = big + 1
    big
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：把列表加起来

    写出 `total(numbers)`，它返回列表里所有数的和。单元格里已经用注释写好了思路：

    1. 用一个名字存下累计值，从 `0` 开始，比如 `running = 0`。
    2. 用循环逐个取出 `numbers` 里的数（比如 `for n in numbers:`），把**这个数**（不是 1）加到累计值上。
    3. 循环结束后，`return` 这个累计值。

    注意空格：函数里的行从 4 个空格开始；循环里面的那一行从 8 个空格开始。
    写完循环里那一行后，按 **Enter**，再按一次 **Backspace** 退回到 4 个空格，在那里写 `return`，
    这样它只在循环**全部结束之后**运行一次。如果函数只返回了第一个数，说明 `return` 还在循环里面。
    """)
    return


@app.function
def total(numbers):
    # 思路：累计值从 0 开始；对每个数，把它加到累计值上；
    # 循环结束后，返回累计值。
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("00", "total", total, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：平均数

    /// admonition | 🧰 工具箱：平均数（均值）
    一组数的**平均数**，也叫**均值**（mean），就是它们的总和除以个数。
    2、4、9 的平均数是 (2 + 4 + 9) ÷ 3 = 15 ÷ 3 = 5。
    ///

    写出 `average(numbers)`。先完成练习 2，因为这里可以直接用上它：函数可以调用别的函数。
    在 `average` 里，`total(numbers)` 得到总和，`len(numbers)` 得到个数。
    别忘了括号：单写 `total` 只是那份菜谱本身，`total(numbers)` 才是它算出的答案。
    """)
    return


@app.function
def average(numbers):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("00", "average", average, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 你现在会做的事

    - 运行 Python 代码并看懂结果
    - 把值存进变量
    - 写出接收输入、用 `return` 返回答案的函数
    - 用列表装很多值，并用循环逐个处理

    这些已经够你开始教计算机学习了。在**第 01 章**里，你会用的正是这些工具：
    让程序从数据里学出规律，再去预测它从没见过的数。

    ---

    <!-- go-deeper -->
    **深入阅读（可选，学这门课用不到）：**官方
    [Python 教程](https://docs.python.org/zh-cn/3/tutorial/) 对这些概念讲得更详细。
    """)
    return


@app.cell(hide_code=True)
def _():
    nav("00", LOCALE)
    return


if __name__ == "__main__":
    app.run()
