import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="04 · 记账")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import hand, todo
    from learn_llm.checks import check
    from learn_llm.diagrams import cards, one_day
    from learn_llm.plot import chart

    LOCALE = "zh-CN"
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 04 · 记账

    第 03 章里，你手工推导出米娅那条直线的反向步骤，再把它们写成 `day_slopes` 里的代码。这套做法行得通，但只管这一个算式。
    一旦算式变了——比如把直线换成曲线，或者换成神经网络——所有反向步骤都得从头手工推一遍。

    这一章，我们要搭一个小**引擎**（engine）来替你干这件苦差事。正向计算时，引擎把每一步都记进一本账，就像店主记流水账一样；
    算完之后，它把账本倒着翻一遍，按照第 03 章的规则，自动把责任分给每个框。

    ## 引擎要做什么

    回顾一下第 03 章的规则：每个框都有一份责任，衡量的是“轻轻推一下这个框，它每变化 1，损失会跟着变多少”。倒着往回算时：

    > 一个框的责任 = 把从它出发的每条箭头的
    > （局部斜率 $\times$ 箭头指向的框的责任）全部加起来

    因此，对每个框，引擎都要记住三样东西：它存的**数值**、它是由哪些框**算出来的**，以及通往每个来源的**局部斜率**。
    普通的数字记不下这么多信息，我们需要在 Python 里自定义一种新的数据类型。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一种新的值：类

    到目前为止，你用过数字、列表和函数。Python 允许你自己定义一种新的数据类型，在里面划分出带名字的格子，想记什么就存什么。
    定义这种新类型的模板叫作**类**（class）。

    想想医院里的挂号登记表。空白表格只印一种模板，上面印着带标签的格子：姓名、年龄、电话。每位病人领到**自己的一份**，填上各自的信息。
    **类**就是那张空白模板；用它创建出来的每一个对象，就是一份填好的表。我们这张表有四个格子：数值本身、它的责任、它由哪些框算出来，以及通往这些来源的局部斜率。
    代码如下：
    """)
    return


@app.class_definition
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0
        self.parents = []
        self.local_slopes = []


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    逐行来看：

    - `class Value:` 意思是“下面定义一种叫 `Value` 的新类型模板”。
    - `def __init__(self, data):` 是每次创建一个新的 `Value` 时，Python 会自动运行的初始化函数。`__init__` 这个名字（两边各两个下划线，是 *initialise*“初始化”的缩写）是 Python 规定死的。它有两个输入：`self` 代表刚建好、正在填写的这个新对象本身（`self` 的意思是“自己”），`data` 是你传给它的数。
    - `self.data = data` 把这个数存进对象内部一个叫 `data` 的格子里。对象里的这种格子叫作**属性**（attribute）。这里 `data` 出现了两次，意思不同：左边的 `self.data` 是对象里面的格子，右边的 `data` 是 `def` 那一行传进来的输入。
    - `self.grad = 0` 让它的责任从 0 开始。（`grad` 是 gradient 的缩写。第 03 章把所有旋钮的斜率合称“梯度”，而在代码里，大家习惯也给每个单独的框起名叫 `.grad`，用来存它自己的斜率/责任。从 0 开始，是因为责任要在后面一点点累加。）
    - `self.parents = []` 和 `self.local_slopes = []` 先设为空列表，等后面再填入：这个框由哪些框算出来（`parents` 原意是“父母”，这里指“来源”），以及通往每个来源的局部斜率（`local_slopes`）。

    创建一个 `Value` 的写法和调用函数一样，但只需要传**一个**参数：`Value(3)`，不需要自己传 `self`。写下 `v = Value(3)` 时，Python 会在背后做三件事：

    1. 先建出一个空白的新 `Value` 对象。
    2. 自动运行 `__init__`，把这个新对象当作 `self` 传进去，把 `3` 当作 `data` 传进去。于是 `self.data = data` 就把它的 `data` 格子填成了 3。
    3. 把填好的对象返回，赋给变量名 `v`。

    用点号 `.` 可以读取里面的格子：`v.data` 就是“`v` 里面的 `data` 格子”。每个对象都有自己独立的一套格子，所以下面的 `v` 和 `u` 互不干扰，各自装着不同的数。
    （和第 03 章一样，最后一行用逗号隔开的几个值，会显示成一个从 0 开始编号的列表。）
    """)
    return


@app.cell
def _():
    v = Value(3)
    u = Value(10)
    v.data, u.data, v.grad, v.parents
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    画成卡片，`v` 和 `u` 就是同一张登记表的两份副本，各自填了不同的数。它们的 `parents` 都是空的：它们不是由别的框算出来的，而是你直接输入的原始数值。
    """)
    return


@app.cell(hide_code=True)
def _():
    cards([{"title": "v", "lines": ["data = 3", "grad = 0", "parents = []", "local_slopes = []"]},
           {"title": "u", "lines": ["data = 10", "grad = 0", "parents = []", "local_slopes = []"]}],
          "两张有同样四个格子的卡片：v 装着 3，u 装着 10；两者的责任都是 0，都没有来源")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    同样可以用点号修改格子里的内容：`v.grad = 5` 会把它的责任改成 5。

    ## 账本

    账本就是一个普通的列表。引擎每算完一步，就把结果记到账本末尾。列表自带一个现成的功能：`book.append(x)` 会把 `x` 追加到 `book` 的最后（`append` 的意思是“追加”）。
    又是点号——`append` 是每个列表都自带的函数。下面用 `shelf`（架子）随便举个例子：
    """)
    return


@app.cell
def _():
    shelf = []
    shelf.append(5)
    shelf.append(8)
    shelf
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一步：乘法

    引擎是这样计算两个值相乘的：先像平常一样算出乘积，再把需要记账的信息填进结果框里，最后把结果记进账本：
    """)
    return


@app.function
def mul(a, b, book):
    out = Value(a.data * b.data)
    out.parents = [a, b]
    out.local_slopes = [b.data, a.data]
    book.append(out)
    return out


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - `out = Value(a.data * b.data)`：新建一个框，存下两个数的乘积。
    - `out.parents = [a, b]`：记下它是由 `a` 和 `b` 算出来的。
    - `out.local_slopes = [b.data, a.data]`：记下对应的局部斜率。在 $a \times b$ 里，轻轻推一下 $a$，结果变化 $b$ 倍；
      轻轻推一下 $b$，结果变化 $a$ 倍。顺序要和 `parents` 一一对应：第一个斜率配第一个来源。
    - `book.append(out)`：把这一步记进账本。
    - `return out`：把新框返回，方便后面的计算接着用。

    试试看：
    """)
    return


@app.cell
def _():
    shop_book = []
    x = Value(3)
    y = Value(4)
    z = mul(x, y, shop_book)
    z.data, z.local_slopes, len(shop_book)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `z` 装着 12；它的局部斜率是 4（通往 `x`）和 3（通往 `y`）；账本里有一条记录。画成卡片：
    """)
    return


@app.cell(hide_code=True)
def _():
    cards([{"title": "x", "lines": ["data = 3", "parents = []", "local_slopes = []"]},
           {"title": "y", "lines": ["data = 4", "parents = []", "local_slopes = []"]},
           {"title": "z（账本第 0 条）", "lines": ["data = 12", "parents = [x, y]", "local_slopes = [4, 3]"],
            "mark": "now"}],
          "卡片 x 装着 3，y 装着 4；卡片 z 装着 12，来源是 x 和 y，局部斜率是 4 和 3")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    把 `z` 的卡片读成一句话就是：“我是 12，由 `x` 和 `y` 算出来。`x` 动一点，我跟着动 4 倍；`y` 动一点，我跟着动 3 倍。”
    反向计算需要知道的全部信息，都在这张卡片里了。注意只有 `z` 进了账本：`x` 和 `y` 不是算出来的，而是你直接输入的。

    这里有一个很关键的细节：`z.parents = [x, y]` 存的**不是**数字 3 和 4 的副本，而是直接指向 `x` 和 `y` 这两个对象本身！
    所以后面 `backward` 修改 `parent.grad` 时，外面原来的 `x.grad` 和 `y.grad` 会跟着一起变——这也是为什么 `backward` 不需要 `return`。

    ## 练习 1：加法

    照同样的方式写出 `add(a, b, book)`。先想一想：在 $a + b$ 里，$a$ 增加一点点，$a + b$ 也增加同样多；对 $b$ 也一样，所以两个局部斜率都是 1。
    写代码时记得切换成英文输入法。（`out` 表示输出，`add` 表示相加。）
    """)
    return


@app.function
def add(a, b, book):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("04", "add", add, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：减法

    写出 `sub(a, b, book)`，对应 `a - b`。注意第二个局部斜率：如果 `b` 增加一个很小的 `h`，结果 `a - b` 就**减少** `h`。
    所以 `b` 每增加 1，结果就变化 `-1`，也就是减少 1。（`sub` 是 subtract“减”的缩写。）
    """)
    return


@app.function
def sub(a, b, book):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("04", "sub", sub, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：平方

    写出 `square(a, book)`，对应 $a \times a$。它只有**一个**来源 $a$（所以 `out.parents = [a]`），只有一个局部斜率：
    第 02 章 $x \times x$ 的斜率，也就是 $2 \times x$。这里的 $x$ 就是 `a.data`。（`square` 是“平方”。键盘上用 `*` 表示乘号，Python 不认识 $\times$。）
    """)
    return


@app.function
def square(a, book):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("04", "square", square, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 倒着读账本

    现在是引擎的核心。计算结束后，账本按发生的顺序列出了每一步，最后一条就是损失。分配责任的方法是：

    1. 损失对自己的责任是 1：`loss.grad = 1`。
    2. **从最后一条到第一条**过一遍账本。对每一条记录，把责任传给它的每个来源：把（局部斜率 $\times$ 这条记录的责任）加到来源的责任上。

    之所以是把责任**加**上去，是因为一个框连到好几条箭头时，责任要加起来（第 03 章）：从这个框出发的每条箭头都会带回自己那一份。

    为什么要从后往前？一个框总是在它的来源**之后**才被记进账本：来源没算出来，它就没法算。所以倒着读时，
    我们总是先遇到所有*用到*某个值的框，然后才遇到这个值本身。等走到一个框时，所有用到它的框都已经把自己那份责任传给了它：
    它的责任已经齐了，可以放心往下传。

    拿第 03 章米娅的第 1 天来看。依次记下 $p = w \times t$、$g = p + b$、$e = g - s$、$L = e \times e$，账本的顺序就是：第 0 条是 $p$，然后是 $g$、$e$、$L$。
    用第 1 天的数，$p = 140$，$g = 80$，$e = 17$，$L = 289$，所以 $e \times e$ 的局部斜率是 $2 \times 17 = 34$。

    拖动滑块，从末尾开始一次读一条记录。蓝色框出的是正在读的那一条；责任到达时，会出现在各个框的下方。
    """)
    return


@app.cell(hide_code=True)
def _():
    read_step = mo.ui.slider(0, 4, value=0, label="已读的记录数", show_value=True)
    read_step
    return (read_step,)


@app.cell(hide_code=True)
def _(read_step):
    _says = [
        r"""**开始。** 损失对自己的责任是 1：$L$ 得到责任 1。还没有读任何记录。""",
        r"""**读第 3 条，$L$。** 它的责任是 1。它唯一的来源是 $e$，局部斜率是 34：$e$ 得到 $34 \times 1 = 34$。""",
        r"""**读第 2 条，$e$。** 唯一用到它的 $L$ 已经读过了，所以它的责任 34 已经齐了。$g$ 得到 $1 \times 34 = 34$，$s$ 得到 $-1 \times 34 = -34$。""",
        r"""**读第 1 条，$g$。** 唯一用到它的 $e$ 已经处理完：责任 34。$p$ 得到 $1 \times 34$，$b$ 得到 $1 \times 34$。""",
        r"""**读第 0 条，$p$。** 唯一用到它的 $g$ 已经处理完：责任 34。$w$ 得到 $14 \times 34 = 476$，$t$ 得到 $10 \times 34 = 340$。
        账本读完了：每个框都拿到了自己的责任。""",
    ]
    _arrive = [{"L": 1}, {"e": 34}, {"g": 34, "s": -34}, {"p": 34, "b": 34}, {"w": 476, "t": 340}]
    _k = read_step.value
    _blames = {name: blame for step in _arrive[:_k + 1] for name, blame in step.items()}
    _book = [("p", 140, "[w, t]", "[14, 10]"), ("g", 80, "[p, b]", "[1, 1]"),
             ("e", 17, "[g, s]", "[1, -1]"), ("L", 289, "[e]", "[34]")]
    _now = 4 - _k  # 正在读的记录：第 1 到 4 步依次是 3、2、1、0
    mo.vstack([
        mo.md(_says[_k]),
        one_day({"w": 10, "t": 14, "p": 140, "b": -60, "g": 80, "s": 63, "e": 17, "L": 289},
                {name: f"责任 {blame:g}" for name, blame in _blames.items()},
                highlight=_book[_now][0] if _k else ""),
        cards([{"title": f"第 {i} 条：{name}",
                "lines": [f"data = {data}", f"parents = {parents}", f"local_slopes = {slopes}",
                          f"grad = {_blames.get(name, 0):g}"],
                "mark": "now" if _k and i == _now else "done" if _k and i > _now else ""}
               for i, (name, data, parents, slopes) in enumerate(_book)],
              "账本：p、g、e、L 四条记录，每张卡片写着它的值、来源、局部斜率和目前的责任",
              arrows="→"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每一步里，正在读的那个框都已经拿到了**完整**的责任：唯一用到它的框在账本里排在更后面，已经读过了。
    最后 $w$ 是 476，$b$ 是 34：和第 03 章的数一样，而我们一个反向步骤都没有单独去推。

    ### 手算：一个框被用了两次

    米娅的算式里，每个框只被用一次。现在换一个有框被用了**两次**的。取 $a = 2$、$b = 5$，算式是先 $u = a + b$，再 $v = u \times a$。
    记下来，账本里有两条记录：
    """)
    return


@app.cell(hide_code=True)
def _():
    cards([{"title": "a", "lines": ["data = 2"]}, {"title": "b", "lines": ["data = 5"]},
           {"title": "第 0 条：u", "lines": ["data = 7", "parents = [a, b]", "local_slopes = [1, 1]"]},
           {"title": "第 1 条：v", "lines": ["data = 14", "parents = [u, a]", "local_slopes = [2, 7]"]}],
          "a 装着 2，b 装着 5；第 0 条是 u，等于 7，由 a 和 b 算出，局部斜率 1 和 1；"
          "第 1 条是 v，等于 14，由 u 和 a 算出，局部斜率 2 和 7")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    （$v = u \times a$，所以 $u$ 每变 1，$v$ 变 $a = 2$；$a$ 每变 1，$v$ 变 $u = 7$。）从 $v$ 开始，责任为 1，
    手工倒着读账本，填出各个责任：先填读完第 1 条之后的，再填读完第 0 条之后的。记住：责任是**加**到一个框已有的责任上的。
    """)
    return


@app.cell(hide_code=True)
def _():
    two_users = hand.boxes(["u.grad", "读完第 1 条后的 a.grad", "最后的 a.grad", "b.grad"])
    return (two_users,)


@app.cell(hide_code=True)
def _(two_users):
    hand.marked(two_users, {"u.grad": 2, "读完第 1 条后的 a.grad": 7, "最后的 a.grad": 9, "b.grad": 2}, {
        "u.grad": r"""第 1 条 $v$ 的责任是 1，它把（通往 $u$ 的局部斜率）$\times 1$ 传给 $u$""",
        "读完第 1 条后的 a.grad": r"""$v$ 把（通往 $a$ 的局部斜率）$\times 1$ 传给 $a$""",
        "最后的 a.grad": r"""第 0 条 $u$ 现在把 $1 \times$（它的责任）传给 $a$，**加在** $a$ 已有的责任上""",
        "b.grad": r"""第 0 条 $u$ 把 $1 \times$（它的责任）传给 $b$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $a$ 的责任来自**两条**路：直接从 $v$ 来的 7，以及经过 $u$ 来的 2。直接验算一下：$v = (a + b) \times a = a \times a + b \times a$。
    把 $b$ 当成固定常数，$a \times a$ 对 $a$ 的斜率是 $2 \times a$，$b \times a$ 对 $a$ 的斜率是 $b$，合起来就是 $2 \times a + b = 2 \times 2 + 5 = 9$。
    这就是为什么引擎要把责任**加**到来源上，而不是直接覆盖：覆盖会把其中一条路给丢掉。

    ### 顺序反了

    看看顺序反了会出什么错。取 $x \times x + x$，$x = 3$。它的斜率是 $2 \times 3 + 1 = 7$（$x \times x$ 的斜率加上 $x$ 的斜率）。
    下面的单元格把它记下来，然后**从前往后**读账本。（它用到了你的 `add`，所以在练习 1 通过之前会报错。`entry` 表示“一条记录”。）
    """)
    return


@app.cell
def _():
    wrong_book = []
    x3 = Value(3)
    product = mul(x3, x3, wrong_book)
    total3 = add(product, x3, wrong_book)

    total3.grad = 1
    for entry in wrong_book:  # 从前往后：错误的顺序
        for parent, slope in zip(entry.parents, entry.local_slopes):
            parent.grad = parent.grad + slope * entry.grad
    x3.grad
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    结果是 `1`，不是 `7`。（下面说的 $x$ 就是单元格里的 $x_3$，乘积是 `product`，总和是 `total3`。）账本里有两条记录：先是乘积（`mul(x3, x3)`，它的 `parents` 是 `[x3, x3]`：**同一个**框出现两次，$\times$ 的两边各一次），然后是总和。
    从前往后：

    1. 先处理乘积。它的责任还是 0，因为总和还没把责任传给它。所以 $x$ 从第一边得到 $3 \times 0$，从第二边也得到 $3 \times 0$：什么也没得到。
    2. 然后处理总和（责任 1）：传给乘积 $1 \times 1$，传给 $x$ $1 \times 1$。现在 $x$ 的责任是 1。乘积终于拿到了自己的责任 1，可它那条记录已经读过了，这份责任再也传不下去。

    倒着读时，先处理总和：乘积得到 1，$x$ 得到 1。再处理乘积，它已经拿齐了责任 1，传给 $x$ $3 \times 1 + 3 \times 1 = 6$。合计：$1 + 6 = 7$。

    要从最后一条倒着遍历一个列表，Python 提供了 `reversed`（倒序）：`for entry in reversed(book):`。

    ## 练习 4：反向传播

    写出 `backward(loss, book)`。你写的四行要替换掉 `todo()`，所以它们都在 `def` 里面。前两行对齐，都从原来 `todo()` 的位置开始；
    后两行各比上一行多缩进一级，因为它们各自在上一行的循环里面。下面是整个函数，可以看清每一行放在哪里：

    ```python
    def backward(loss, book):
        loss.grad = 1
        for entry in reversed(book):
            for parent, slope in zip(entry.parents, entry.local_slopes):
                parent.grad = parent.grad + slope * entry.grad
    ```

    只输入 `def` 下面的四行：`def` 那一行你的单元格里已经有了。（`parent` 表示“来源”，`slope` 表示“斜率”。记得切换成英文输入法。）

    它不需要 `return` 任何东西：正如前面说的，`entry.parents` 直接指向原本的那些对象，修改 `parent.grad` 就已经把责任写回原来的框里了。
    """)
    return


@app.function
def backward(loss, book):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("04", "backward", backward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 用引擎算米娅的第 1 天

    下面再算一次第 03 章里的第 1 天（`t = 14`，`s = 63`，旋钮 `w = 10`，`b = -60`），不过这次全交给你的引擎，不用手写任何反向步骤。（练习 1 到 4 全部通过之前，这个单元格和下面的图都会报错。）
    """)
    return


@app.cell
def _():
    day_book = []
    w = Value(10)
    b = Value(-60)
    t = Value(14)
    s = Value(63)
    p = mul(w, t, day_book)
    g = add(p, b, day_book)
    e = sub(g, s, day_book)
    L = square(e, day_book)
    backward(L, day_book)
    w.grad, b.grad
    return L, b, e, g, p, s, t, w


@app.cell(hide_code=True)
def _(L, b, e, g, p, s, t, w):
    one_day({"w": w.data, "t": t.data, "p": p.data, "b": b.data, "g": g.data, "s": s.data, "e": e.data, "L": L.data},
            {n: f"责任 {v.grad:g}" for n, v in (("w", w), ("p", p), ("b", b), ("g", g), ("e", e), ("L", L))})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `w` 是 476，`b` 是 34：和你在第 03 章手工算出的一样，图里每个框都拿到了自己的责任。
    数据框 `t` 和 `s` 也是用 `Value` 做的，所以它们也会收到责任；只是这些责任从来用不上，因为数据我们改不了。

    ## 练习 5：整个损失的梯度

    写出 `engine_gradient(w_num, b_num, xs, ys)`：米娅**平均**损失对两个旋钮的斜率，全部由引擎算出。它就是第 03 章的 `gradient`，
    只不过你只需要写正向计算（`total` 表示总和，`guess` 表示猜测，`mean` 表示均值）。步骤如下：

    1. `book = []`，然后把旋钮包装成 `Value`：`w = Value(w_num)` 和 `b = Value(b_num)`（`w_num` 是普通数字，`w` 是包在框里的 `Value`）。
       从这里开始，框和框之间只用 `add`、`sub`、`mul`、`square` 连起来，不要用 `+`、`-`、`*`：只有这些函数才会把这一步记进账本。
    2. 一个累计总和，它本身也是一个 `Value`：`total = Value(0)`。
    3. 用 `for x, y in zip(xs, ys):` 循环。在里面：`guess = add(mul(w, Value(x), book), b, book)`，
       然后 `e = sub(guess, Value(y), book)`，然后 `total = add(total, square(e, book), book)`。
    4. 循环结束后（这几行和 `for` 对齐，不在循环里面），均值就是总和乘以 `1 / len(xs)`：
       `mean = mul(total, Value(1 / len(xs)), book)`。
    5. `backward(mean, book)`，然后 `return w.grad, b.grad`。

    为了看清这个循环往账本里写了什么，只取两天。每一天往账本里加五条记录，最后求平均再加一条：

    | 记录 | 由谁写下 | 内容 |
    |---|---|---|
    | 第 0 到 4 条 | 第 1 天 | $w \times x_1$，然后 $+ b$，然后 $- y_1$，然后平方，然后加到总和上 |
    | 第 5 到 9 条 | 第 2 天 | 第 2 天同样的五步 |
    | 第 10 条 | 循环之后 | 总和 $\times \frac{1}{2}$：平均值，也就是最后一条 |

    `total = add(total, ..., book)` 这一行并不会改动旧的总和框：它会算出一个**新**框（旧总和加上今天的误差平方），再让变量名 `total` 改指向这个新框，
    就像 `n = n + 1` 让 `n` 指向一个新数一样。旧框依然安稳地留在账本里，作为新框的来源。
    """)
    return


@app.function
def engine_gradient(w_num, b_num, xs, ys):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("04", "engine_gradient", engine_gradient, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    在 `w = 10`、`b = -60` 处，它给出 `(-94, 2.5)`，和第 03 章手工搭出来的梯度一模一样。

    ## 每次都用新框

    有一个陷阱。因为 `backward` 是把责任**加**上去的，所以对同一批框运行两次，责任就会加两遍。
    下面的单元格把第 1 天记一次账，然后倒着读两遍。你可能以为会翻倍，变成 952。（和上面的单元格一样，它需要练习 1 到 4。）
    """)
    return


@app.cell
def _():
    twice_book = []
    w2 = Value(10)
    p2 = mul(w2, Value(14), twice_book)
    g2 = add(p2, Value(-60), twice_book)
    e2 = sub(g2, Value(63), twice_book)
    loss2 = square(e2, twice_book)
    backward(loss2, twice_book)
    first = w2.grad
    backward(loss2, twice_book)
    first, w2.grad
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第二次，`w` 的责任变成了 2380，是 476 的五倍。第二遍时，每个框里还留着第一遍的责任，新的那一份又加在上面，
    一个框接一个框地沿着账本传下去：

    - `L` 又被设成 1（那一行是直接赋值，不是累加）。
    - $e$ 原来是 34，再加 $34 \times 1$：68。
    - $g$ 原来是 34，再加 $1 \times 68$：102。
    - $p$ 原来是 34，再加 $1 \times 102$：136。
    - $w$ 原来是 476，再加 $14 \times 136 = 1904$：2380。

    每个框都把掺着旧账的错误责任继续往回传，结果误差像滚雪球一样越滚越大。这就是为什么 `engine_gradient` 每次调用都要新建一批 `Value` 框、
    换一本新账本，让所有责任重新归零。（真正的深度学习框架也有这条铁律，后面你还会再遇到。）

    ## 收获：换个算式，不用再推反向步骤

    热浪来袭，米娅照常出摊。但气温超过大约 33 °C 后，销量反而开始**下降**——天太热了，大家都不愿出门。下面是全部 11 天的数据：
    """)
    return


@app.cell(hide_code=True)
def _():
    hot_temps = temps + [35, 37, 39]
    hot_sales = sales + [280, 262, 236]
    mo.md(f"`temps` = {hot_temps}  \n`sales` = {hot_sales}")
    return hot_sales, hot_temps


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    直线没法先上升再下降。**曲线**可以：

    > $$\text{猜测} = a \times x \times x + w \times x + b$$

    有三个旋钮 `a`、`w` 和 `b`。`a` 为负数时，曲线向下弯，像一座小山。

    这里还有个小细节：这里的 $x$ 是**从 25 °C 算起、以 10 度为单位**的气温：`x = (t - 25) / 10`，所以 14 °C 变成 -1.1，39 °C 变成 1.4。
    你可以把它当成一把重新标过刻度的温度计：把舒服的 25 °C 当作 0 度，每 10 °C 算作 1 格——这样 `b` 就是 25 °C 时的预测销量，`w` 就是气温每升高 10 °C 大概多卖多少个。
    为什么要换算？第 02 章里你看到过 $b$ 挪得很慢：一个旋钮乘上的数越大，损失对它的反应就越强烈，所以对 $w$ 安全的学习率，对 $b$ 来说又太小。
    这里 $a$ 乘的是 $x \times x$：

    | $x$ 取 | $x \times x$ 最小 | 最大 |
    |---|---|---|
    | 原始气温 $t$ | $14 \times 14 = 196$ | $39 \times 39 = 1521$ |
    | `t / 10` | $1.4 \times 1.4 = 1.96$ | $3.9 \times 3.9 = 15.21$ |
    | `(t - 25) / 10` | 0 | $1.4 \times 1.4 = 1.96$ |

    用原始气温的话，$a$ 乘的数最大到 1521，$b$ 乘的是 1：对 $a$ 足够小的学习率，会让 $b$ 几乎一动不动。只除以 10 有帮助，
    但 $x \times x$ 仍能到 15。先减去 25，气温的中点就移到了 0，于是 $x$ 大约在 -1 到 1.5 之间，$x \times x$ 在 0 到 2 之间。
    这样三个旋钮乘的数（$x \times x$、$x$ 和 1）大小差不多，一个学习率就能适合所有旋钮。

    换成曲线的话，按第 03 章的做法就得手工推一套新的反向步骤。有了你的引擎，你只需要写正向计算。
    按下按钮，用 500 步梯度下降拟合这条曲线，每一步的梯度都由你的引擎算出：
    """)
    return


@app.cell(hide_code=True)
def _():
    fit_curve = mo.ui.run_button(label="拟合曲线")
    fit_curve
    return (fit_curve,)


@app.cell(hide_code=True)
def _(fit_curve, hot_sales, hot_temps):
    mo.stop(not fit_curve.value, mo.md("*按**拟合曲线**开始。*"))
    try:
        _xs = [(t - 25) / 10 for t in hot_temps]
        _knobs = [0.0, 0.0, 0.0]
        for _i in range(500):
            _book = []
            _a, _w, _b = Value(_knobs[0]), Value(_knobs[1]), Value(_knobs[2])
            _total = Value(0)
            for _x, _y in zip(_xs, hot_sales):
                _xv = Value(_x)
                _guess = add(add(mul(_a, square(_xv, _book), _book), mul(_w, _xv, _book), _book), _b, _book)
                _total = add(_total, square(sub(_guess, Value(_y), _book), _book), _book)
            _mean = mul(_total, Value(1 / len(_xs)), _book)
            backward(_mean, _book)
            _knobs = [k - 0.1 * v.grad for k, v in zip(_knobs, (_a, _w, _b))]
        _ka, _kw, _kb = _knobs
        _curve = [(t, _ka * ((t - 25) / 10) ** 2 + _kw * (t - 25) / 10 + _kb) for t in range(12, 42)]
        _out = mo.vstack([
            chart((10, 42), (0, 320), "气温（°C）", "卖出的冰淇淋（个）",
                  dots=list(zip(hot_temps, hot_sales)), curve=_curve),
            mo.md(f"a = **{_ka:.1f}**，w = **{_kw:.1f}**，b = **{_kb:.1f}**，损失 = **{_mean.data:,.1f}**。"
                  f"（在这 11 天上，最好的直线损失大约是 810。）"),
        ])
    except Exception:
        _out = mo.md("*完成练习 1 到 4 后才能拟合曲线。*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    曲线随着气温上升，到了热浪里又往下拐，损失也比任何直线都低得多。因为 `x` 是以 25 °C 为零点、每 10 °C 为 1 格来算的，
    三个旋钮的含义和之前直线的 `w = 10`、`b = -60` 不太一样：`b` 就是 25 °C（`x = 0`）时的预测销量，大约 205 个冰淇淋；`w` 表示在 25 °C 附近，
    气温每升高 10 °C 销量大约增加 101 个；而负的 `a` 让曲线在热浪来袭时往下弯。全程你没有手推过一步反向公式：引擎默默记完账，自动替你算好了。

    ## 下一步

    你的引擎一次处理一个数：每次乘法都是账本里的一条记录。真正的模型要同时处理成千上万甚至几百万个数，每个数记一笔账就太慢了。
    在**第 05 章**里，你会学着一次处理整串的数（向量和矩阵），再把同样的反向规则教给它们。

    ## 你现在会做的事

    - 定义一个带 `__init__` 的类，用点号读取和修改属性
    - 用 `append` 往列表末尾追加元素，用 `reversed` 倒序遍历
    - 搭出一个能记录计算过程、并自动完成反向传播的小引擎
    - 说清楚为什么账本必须倒着读，以及为什么每次计算都要用新框
    - 只写正向计算，就能拟合一个新模型

    ---

    <!-- go-deeper -->
    **深入阅读（可选）：**这种引擎叫作*自动微分*（反向模式），也叫 *autograd*。从账本里得到的这种顺序叫作*拓扑序*。
    """)
    return


if __name__ == "__main__":
    app.run()
