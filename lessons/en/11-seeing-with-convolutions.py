import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="11 · Seeing with convolutions")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.images import digits as show_digits
    from learn_llm.images import maps

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 11 · Seeing with convolutions

    Your networks so far treat a picture as a long list of unrelated numbers. Pixel 300 has a knob
    of its own, and nothing tells the network that pixel 301 sits right beside it. That worked
    because every digit in our data is neatly centred. Real pictures aren't like that: a cat is a
    cat wherever it sits in the photo.

    In this chapter you'll see that weakness, fix it with one idea (a small detector that slides
    over the whole picture), and then learn a second idea, the **residual**, which lets
    networks grow very deep. These two ideas took computer vision from toy problems to
    recognising thousands of kinds of objects.

    Here are chapter 10's tools, ready to use:
    """)
    return


@app.function
def train_step(model, optimizer, X, labels):
    optimizer.zero_grad()
    S = model(X)
    loss = F.cross_entropy(S, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


@app.function
def train(model, X, labels, epochs, batch_size, rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            loss = train_step(model, optimizer, X[batch], labels[batch])
    return loss


@app.function
def accuracy(S, labels):
    return torch.mean((torch.argmax(S, dim=1) == labels).float()).item()


@app.function
def count_knobs(model):
    total = 0
    for p in model.parameters():
        total = total + p.numel()
    return total


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Moving digits

    To make the problem visible, we'll scatter the digits. Each $28 \times 28$ picture gets pasted at a random
    spot on an empty $40 \times 40$ canvas:
    """)
    return


@app.function
def shift_digits(images, size, seed):
    rng = np.random.default_rng(seed)
    canvas = np.zeros((len(images), size, size))
    for i in range(len(images)):
        r = rng.integers(0, size - 27)
        c = rng.integers(0, size - 27)
        canvas[i, r:r + 28, c:c + 28] = images[i] / 255
    return canvas


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: a random whole number, and a square slice
    - `rng.integers(0, n)` gives a random whole number from 0 up to, but not including, $n$.
    - A slice works in two directions at once: `M[1:3, 0:2]` is the $2 \times 2$ square of `M` in rows
      1 and 2, columns 0 and 1. In the cell above, `canvas[i, r:r + 28, c:c + 28]` is the
      $28 \times 28$ square of picture $i$ whose top-left corner is at row $r$, column $c$, and storing
      into it pastes the digit there.
    ///
    """)
    return


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    X_np = shift_digits(train_images, 40, 0)
    X_test_np = shift_digits(test_images, 40, 1)
    X = torch.tensor(X_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    labels = torch.tensor(train_labels, dtype=torch.long)
    labels_test = torch.tensor(test_labels, dtype=torch.long)
    show_digits(X_np[:6] * 255, train_labels[:6])
    return X, X_test, labels, labels_test, test_images, train_images


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now train chapter 10's kind of network on them, made bigger to give it a fair chance: 1,600
    inputs and 256 detectors. `nn.Flatten()` is a layer that turns each $40 \times 40$ picture into a row of
    1,600 numbers, the same job as `reshape`. The cell shows three numbers: the knobs, then the accuracy on
    the training pictures, then on the test pictures.
    """)
    return


@app.cell
def _(X, X_test, labels, labels_test):
    torch.manual_seed(0)
    flat_model = nn.Sequential(nn.Flatten(), nn.Linear(1600, 256), nn.ReLU(), nn.Linear(256, 10))
    train(flat_model, X, labels, 10, 100, 0.002)
    with torch.no_grad():
        flat_train = accuracy(flat_model(X), labels)
        flat_test = accuracy(flat_model(X_test), labels_test)
    count_knobs(flat_model), flat_train, flat_test
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Over 412,000 knobs, about 100% on its training pictures, about 88% on new ones. That gap is chapter 09's
    **overfitting**, and now we can see where it comes from. A knob belongs to one spot on the
    canvas. So a stroke learned near the top left teaches the network nothing about the same
    stroke near the bottom right: it has to learn every shape again at every position, and with
    so many knobs, it memorises instead.

    ## A detector that slides

    Here's the better idea. Keep the detector small, say $3 \times 3$ knobs, and instead of giving each
    part of the picture its own detector, slide one detector over all of it. At each position, take
    the $3 \times 3$ patch underneath it, and compute a dot product: multiply position by position, add
    up. The small grid of knobs is called a **kernel**, and sliding it is a **convolution**.

    Let's do it by hand. Here's a $4 \times 4$ picture whose numbers jump from 0 to 1 halfway across, and a kernel
    that compares left with right:

    ```
    picture          kernel
    0  0  1  1       -1  0  1
    0  0  1  1       -1  0  1
    0  0  1  1       -1  0  1
    0  0  1  1
    ```

    Put the kernel on the top-left patch (rows 0–2, columns 0–2):

    $(-1 \times 0) + (0 \times 0) + (1 \times 1)$ in each of the 3 rows = $1 + 1 + 1 = \boldsymbol{3}$.

    Slide it one step right (columns 1–3): $(-1 \times 0) + (0 \times 1) + (1 \times 1) = 1$ in each row, again **3**.

    A patch of only 0s, or only 1s, would give 0, since the $-1$s and $+1$s cancel. So
    the result is big exactly where the numbers change from left to right: this kernel is a
    **vertical edge detector**. A $3 \times 3$ kernel fits in 2 positions across and 2 down on a $4 \times 4$ picture,
    so the output is $2 \times 2$. Why 2? On a row of width 4, a kernel of width 3 can only place its left edge
    at column 0 (covering columns 0–2) or column 1 (covering columns 1–3) — counting from 0 up to
    $4 - 3 = 1$ gives $1 + 1 = 2$ positions. In general, a picture of height $h$ gives $h - k + 1$ rows
    for a kernel of size $k$ (and width $w$ gives $w - k + 1$ columns).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: slide the kernel

    Your turn, with the same kernel and a new picture:

    ```
    picture          kernel
    0  0  0  1       -1  0  1
    0  0  1  1       -1  0  1
    0  1  1  1       -1  0  1
    1  1  1  1
    ```

    In each row of a patch, this kernel gives the right-hand number minus the left-hand one. Work out the four numbers of the output.
    Then one more: how many rows does the output have for a $28 \times 28$ picture and a $5 \times 5$ kernel?
    """)
    return


@app.cell(hide_code=True)
def _():
    slide_boxes = hand.boxes(['top left', 'top right', 'bottom left', 'bottom right', 'rows for 28 and 5'])
    return (slide_boxes,)


@app.cell(hide_code=True)
def _(slide_boxes):
    hand.marked(slide_boxes, {'top left': 2, 'top right': 2, 'bottom left': 2, 'bottom right': 1, 'rows for 28 and 5': 24}, {
        'top left': r"""rows 0 to 2, columns 0 to 2: the three rows give 0, 1 and 1""",
        'top right': r"""rows 0 to 2, columns 1 to 3: the three rows give 1, 1 and 0""",
        'bottom left': r"""rows 1 to 3, columns 0 to 2: the three rows give 1, 1 and 0""",
        'bottom right': r"""rows 1 to 3, columns 1 to 3: the three rows give 1, 0 and 0""",
        'rows for 28 and 5': r"""$h - k + 1$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: convolve

    Write `convolve(image, kernel)` for a matrix `image` and a square matrix `kernel`:

    1. Get the sizes. `image.shape` holds the height and the width, and `kernel.shape[0]` is the kernel's size $k$.
    2. Start the output as zeros, with $h - k + 1$ rows and $w - k + 1$ columns.
    3. Visit every output position with two loops, one inside the other: `r` over the output's rows, `c` over its columns.
    4. At each position, take the patch: the $k \times k$ square of the picture whose top-left corner is at row `r`, column `c`. It's a slice in two directions, as in `canvas[i, r:r + 28, c:c + 28]` above.
       Multiply the patch by the kernel, position by position, add everything up with `np.sum`, and store the result at `out[r, c]`.
    5. Give back the output.

    On the worked example above, it must give `[[3, 3], [3, 3]]`.
    """)
    return


@app.function
def convolve(image, kernel):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "convolve", convolve, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`h, w = image.shape` names the two sizes. The loops are `for r in range(...)` with `for c in range(...)` inside it, and the work happens inside both.""",
        r"""The patch is `image[r:r + k, c:c + k]`, with a comma between the rows part and the columns part. The `return` comes after both loops.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now on a real digit, with the vertical edge detector and the same kernel turned on its side
    (`.T`), which finds horizontal edges. Blue means a positive response, red a negative one:
    """)
    return


@app.cell
def _(train_images):
    digit = train_images[0] / 255
    edge_v = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
    edge_h = edge_v.T
    response_v = convolve(digit, edge_v)
    response_h = convolve(digit, edge_h)
    maps([digit, response_v, response_h], ["picture", "vertical edges", "horizontal edges"], 140)
    return digit, edge_v, response_v


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The output of a convolution is itself a kind of picture, a map of where the shape was found.
    That's why it's called a **feature map**. Two properties make this the right tool.

    /// admonition | 🧰 Toolbox: shifting an array
    `np.roll(a, 1)` moves every item one place along, and the last one comes round to the front:
    `np.roll(np.array([1, 2, 3]), 1)` is `[3, 1, 2]`. For a matrix, `axis=1` shifts along the columns.
    ///

    **It moves with the picture.** Move the digit 3 columns to the right with `np.roll`, and the map just moves 3 columns too. The largest
    difference between "move, then convolve" and "convolve, then move" is exactly 0:
    """)
    return


@app.cell
def _(digit, edge_v, response_v):
    moved = np.roll(digit, 3, axis=1)
    np.max(np.abs(convolve(moved, edge_v) - np.roll(response_v, 3, axis=1)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **It needs few knobs.** One kernel has 9 knobs, however big the picture is, because the same 9
    are reused at every position. What it learns in one corner works in every corner.

    ## Shrinking: max pooling

    A digit is more than edges: it's edges arranged in a certain way. To see bigger patterns, we
    shrink the feature map and convolve again. **Max pooling** cuts the map into $2 \times 2$ squares
    and keeps the largest number of each. It says "the feature was somewhere around here",
    forgets the exact position, and halves the height and width. So a $3 \times 3$ kernel in the next layer covers a patch
    twice as wide in the original picture.

    ```
    1  3  0  2
    5  2  1  0      →     5  2
    0  0  4  1            1  6
    1  0  6  2
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: pool a map

    Pool this map: write the largest number of each $2 \times 2$ square.

    ```
    2  0  1  1
    0  3  0  1
    7  1  0  0
    2  2  0  5
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    pool_boxes = hand.boxes(['top left', 'top right', 'bottom left', 'bottom right'])
    return (pool_boxes,)


@app.cell(hide_code=True)
def _(pool_boxes):
    hand.marked(pool_boxes, {'top left': 3, 'top right': 1, 'bottom left': 7, 'bottom right': 5}, {
        'top left': r"""the largest of 2, 0, 0 and 3""",
        'top right': r"""the largest of 1, 1, 0 and 1""",
        'bottom left': r"""the largest of 7, 1, 2 and 2""",
        'bottom right': r"""the largest of 0, 0, 0 and 5""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: division that drops the rest
    `//` is division that drops what's after the point: `7 // 2` is 3, and `8 // 2` is 4.
    ///

    ## Exercise 2: max_pool

    Write `max_pool(M)` for a matrix `M`. The output has half the rows and half the columns, counted with `//`, so with an odd number of rows, the last one gets left out.

    Use the frame of `convolve`: zeros of the right shape, two loops over the output's positions, one number stored at each. The square for output position `r`, `c` starts at row $2 \times r$ and column $2 \times c$ of `M`, and is 2 rows high and 2 columns wide. `np.max` gives its largest number.

    On the example in the text above, it must give `[[5, 2], [1, 6]]`. On your hand example, it must give your four numbers.
    """)
    return


@app.function
def max_pool(M):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "max_pool", max_pool, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""The output has `h // 2` rows and `w // 2` columns, and the loops run over those.""",
        r"""The square is the slice from `2 * r` up to `2 * r + 2` in the rows, and the same with `c` in the columns.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Letting the network choose its kernels

    We picked our edge detector by hand. A network learns its kernels instead: the 9 numbers are just
    knobs, trained by the same backward pass and Adam as always.

    /// admonition | 🧰 Toolbox: convolution, ready-made
    In PyTorch, a convolution with a kernel of knobs is the layer `nn.Conv2d(1, 1, 3)`, read as
    "1 input channel, 1 output channel, kernel size 3". Pictures must arrive with a 4D shape:
    **(pictures, channels, height, width)**. A **channel** is one 2D grid (one feature map) per
    picture: a greyscale digit has 1 channel, and when a convolution layer produces 8 feature maps,
    the next layer receives 8 channels. `X.unsqueeze(1)` inserts a new dimension of size 1 at
    position 1, turning `(10000, 40, 40)` into `(10000, 1, 40, 40)`.
    ///

    To be sure PyTorch does what you just wrote, let's load your kernel into its layer and
    compare. (`.data` is the plain numbers inside a tensor, without the books.)
    """)
    return


@app.cell
def _(digit, edge_v, response_v):
    conv = nn.Conv2d(1, 1, 3, bias=False)
    conv.weight.data = torch.tensor(edge_v).reshape(1, 1, 3, 3)
    digit_4d = torch.tensor(digit).reshape(1, 1, 28, 28)
    with torch.no_grad():
        response_t = conv(digit_4d)
    response_t.shape, np.max(np.abs(response_t.numpy()[0, 0] - response_v))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The shape ends in $26 \times 26$, as our formula says ($28 - 3 + 1$), and the second number is the largest
    difference from your map: 0, or as near as rounding allows. Two more pieces, and we can build a network.

    **Many channels, many kernels.** One layer learns many kernels at once, each producing its own
    output channel (feature map). `nn.Conv2d(1, 8, 3)` has 8 kernels and turns 1 input channel into
    8 output channels.

    What happens when the next layer, `nn.Conv2d(8, 16, 3)`, takes those 8 channels in? Imagine the
    8 channels as 8 transparent sheets stacked directly on top of one another. The 1st kernel is not
    a single flat $3 \times 3$ grid — it is a **stack of 8 slices**, each $3 \times 3$ (so
    $8 \times 3 \times 3 = 72$ weights in all). At any single position, it lays its 8 slices over
    the same $3 \times 3$ patch across all 8 input channels, computes the dot product on each slice,
    and **adds all 8 results together into a single number** (plus 1 bias). That one number goes
    into 1 cell of the 1st output feature map. Because the layer has **16** such 8-slice kernels,
    it produces **16** output channels — and that's how a later kernel can say "a vertical edge just
    left of a horizontal edge".

    | piece in `nn.Conv2d(8, 16, 3)` | shape | what it means |
    |---|---|---|
    | input (one picture) | `(8, H, W)` | 8 stacked feature maps (8 channels) from the previous layer |
    | 1 kernel | `(8, 3, 3)` + 1 bias | 8 slices of $3 \times 3$, summed into 1 output cell ($72 + 1 = 73$ knobs) |
    | all 16 kernels (`weight`) | `(16, 8, 3, 3)` + 16 biases | 16 independent kernels ($16 \times 73 = 1{,}168$ knobs total) |
    | output (one picture) | `(16, H', W')` | 16 new feature maps (16 channels), one per kernel |

    **Keeping the size.** `padding=1` adds a one-pixel margin of zeros around each channel first, so a
    $3 \times 3$ kernel gives an output of the same height and width as its input, and only pooling shrinks it.

    And for the end of the network, `nn.AdaptiveMaxPool2d(1)` keeps just the single largest number of
    each whole channel, whatever its size. It answers "was this feature anywhere in the picture?"
    After it, the 32 channels have become 32 numbers (`nn.Flatten()` tidies the shape), which a final
    `nn.Linear` turns into 10 scores. Since almost nothing about the position survives, a digit
    scores nearly the same wherever it is. (`nn.MaxPool2d(2)`, in the list below, is the $2 \times 2$ pooling you just wrote.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The network, stage by stage

    Here is the network you'll build, for the $40 \times 40$ pictures:

    | stage | what it does |
    |---|---|
    | 1 | a convolution from 1 channel to 8 channels, a bend, $2 \times 2$ max pooling |
    | 2 | a convolution from 8 channels to 16, a bend, $2 \times 2$ max pooling |
    | 3 | a convolution from 16 channels to 32, a bend |
    | 4 | the largest number of each whole channel, flatten, a linear layer from 32 numbers to 10 scores |

    Every convolution has $3 \times 3$ kernels and `padding=1`, so it keeps the height and width.

    ## By hand: follow one picture

    One picture goes in with shape `(1, 40, 40)`: 1 channel, 40 high, 40 wide. Fill in what comes out of stage 1 and stage 2. Then count the knobs of the first and the last convolution, the way the table counted 1,168 for the middle one.
    """)
    return


@app.cell(hide_code=True)
def _():
    stage_boxes = hand.boxes(['stage 1: channels', 'stage 1: height', 'stage 2: height', 'knobs, first convolution', 'knobs, last convolution'])
    return (stage_boxes,)


@app.cell(hide_code=True)
def _(stage_boxes):
    hand.marked(stage_boxes, {'stage 1: channels': 8, 'stage 1: height': 20, 'stage 2: height': 10, 'knobs, first convolution': 80, 'knobs, last convolution': 4640}, {
        'stage 1: channels': r"""the convolution of stage 1 gives 8 channels, and pooling doesn't change the number of channels""",
        'stage 1: height': r"""padding keeps 40, then pooling halves it""",
        'stage 2: height': r"""halved once more""",
        'knobs, first convolution': r"""8 kernels, each with $3 \times 3 \times 1$ weights and 1 bias""",
        'knobs, last convolution': r"""32 kernels, each with $3 \times 3 \times 16$ weights and 1 bias""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: make_cnn

    Write `make_cnn()`, giving back the network of the table as one `nn.Sequential`. It has 11 layers, and all of them appeared in the text above:
    `nn.Conv2d` with its input and output channel counts, the kernel size and `padding=1`; `nn.ReLU()`; `nn.MaxPool2d(2)`; `nn.AdaptiveMaxPool2d(1)`; `nn.Flatten()`; `nn.Linear`.

    Put one stage on each line, with a comma after every layer.
    """)
    return


@app.function
def make_cnn():
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "make_cnn", make_cnn, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Stage 1 is `nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),`. Stages 2 and 3 follow the same pattern with different channel counts, and stage 3 has no pooling.""",
        r"""Each convolution's input channels must equal the previous one's output channels: 1 to 8, 8 to 16, 16 to 32. The last three layers are `nn.AdaptiveMaxPool2d(1)`, `nn.Flatten()` and `nn.Linear(32, 10)`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Convolutions take longer than `@`, so training waits for the button (10 epochs; up to a
    minute or so on a laptop). As before, it shows the knobs, then training and test accuracy. Until
    you press it, the cells that need its result say `Ancestor stopped`. That's normal.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Train the convolutional network")
    train_button
    return (train_button,)


@app.cell
def _(X, X_test, labels, labels_test, train_button):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    torch.manual_seed(0)
    cnn = make_cnn()
    train(cnn, X.unsqueeze(1), labels, 10, 100, 0.003)
    with torch.no_grad():
        cnn_train = accuracy(cnn(X.unsqueeze(1)), labels)
        cnn_test = accuracy(cnn(X_test.unsqueeze(1)), labels_test)
    count_knobs(cnn), cnn_train, cnn_test
    return (cnn,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After training you should see 6,218 knobs, 66 times fewer than the flat network, and about 96% on new
    pictures, against about 88%.
    The training and test scores are now close together: there's little memorising going on,
    because each knob is reused at every position and has to be useful everywhere.

    Below, once training is done, are the 8 kernels of the first layer (`cnn[0].weight` has shape
    `(8, 1, 3, 3)` — 8 output channels, 1 input channel, $3 \times 3$; `[:, 0]` selects the
    $3 \times 3$ kernel for input channel 0 of each output channel). Nobody told it about edges,
    but several of them compare one side with the other, as your hand-made one does:
    """)
    return


@app.cell
def _(cnn):
    kernels = cnn[0].weight.detach().numpy()[:, 0]
    maps(kernels, range(8), 70)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Going deep: the road for blame

    If three layers work so well, would thirty be better? Deep networks should be able to see more complicated
    things. But for years, they trained worse than shallow ones, and one big reason is the chain rule
    from chapter 03. Blame reaches an early layer by being multiplied through every layer after it.
    If those multiplications tend to shrink it, then the blame fades away, the same fading you saw
    in chapter 09's table.

    The fix is almost too simple. A plain block replaces its input with a new output:

    > `out = f(x)`

    Instead, use a block that **adds** to its input:

    > `out = x + f(x)`

    `f` now only has to learn a *correction* to `x`, called a **residual**. Look at the slope. By
    chapter 02's sum rule, the slope of $x + f(x)$ is $1 + f'$ (here $f'$ is mathematical shorthand
    for the slope of $f(x)$). Even when $f'$ is tiny, the slope stays near 1, so blame passes
    through the block nearly undimmed. The `x +` is a direct road for the blame, which is often
    called a **skip connection**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: once through a residual block

    In the block you'll write, `f` is a first layer, a bend, and a second layer. Then `x` is added, and one last bend is applied to the sum:

    > $$\text{out} = \operatorname{relu}(x + \text{layer}_2(\operatorname{relu}(\text{layer}_1(x))))$$

    Try it on single numbers, with made-up layers: the first layer subtracts 5, and the second multiplies by 3. What comes out for $x = 6$? And for $x = 2$?
    """)
    return


@app.cell(hide_code=True)
def _():
    residual_boxes = hand.boxes(['out for x = 6', 'out for x = 2'])
    return (residual_boxes,)


@app.cell(hide_code=True)
def _(residual_boxes):
    hand.marked(residual_boxes, {'out for x = 6': 9, 'out for x = 2': 2}, {
        'out for x = 6': r"""$6 - 5 = 1$, the bend leaves 1, times 3 is 3, and $6 + 3 = 9$ is positive""",
        'out for x = 2': r"""$2 - 5 = -3$, and the bend turns that into 0. What is left to add to $x$?""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For $x = 2$, `f` gave 0 and the block passed `x` on unchanged. A residual block that has learned nothing yet does no harm, which a plain block can't promise.

    ## Exercise 4: residual_block

    Write `residual_block(x, conv1, conv2)`, doing the formula above. `conv1` and `conv2` are two convolution layers that keep the shape, called like functions, and the bend is `torch.relu`.
    """)
    return


@app.function
def residual_block(x, conv1, conv2):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("11", "residual_block", residual_block, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three lines: the first layer and its bend, the second layer, then the sum with its bend.""",
        r"""Add `x` before the last bend, not after it. And add the plain `x`, not something that has already been through a layer.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To use a block like this in `nn.Sequential`, it must be a layer. Making your own layer is
    chapter 04's `class`, with one addition:
    """)
    return


@app.function
def plain_block(x, conv1, conv2):
    h = torch.relu(conv1(x))
    h = conv2(h)
    return torch.relu(h)


@app.cell
def _():
    class Block(nn.Module):
        def __init__(self, channels, step):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.step = step

        def forward(self, x):
            return self.step(x, self.conv1, self.conv2)

    def deep_model(blocks, step):
        layers = [nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        for i in range(blocks):
            layers.append(Block(8, step))
        return nn.Sequential(*layers, nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(8, 10))
    return (deep_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: a layer of your own
    - Quick recap of `class` (chapter 04): `__init__` runs automatically when an object is created, and `self.xxx` attaches data to that object itself.
    - `class Block(nn.Module):` says that `Block` is a kind of `nn.Module`, PyTorch's base type for layers. Building on `nn.Module` is how PyTorch automatically finds all the knobs inside a `Block`.
    - `super().__init__()` runs `nn.Module`'s own setup first. It always goes on the first line.
    - `forward` is the calculation. Calling `block(x)` runs `block.forward(x)`, which explains
      how `layer(X)` worked in chapter 10.
    ///

    In this `Block`, `step` is a function stored on the object (chapter 02: functions are values), so the same class can
    run your `residual_block`, or `plain_block`, which does the same work without the `x +`.

    `deep_model(blocks, step)` stacks `blocks` copies of `Block` in a row, two convolutions each. To compare fairly,
    we use the same network, the same pictures and one backward pass, and look at how much blame reaches the
    very first layer: the mean size of its slopes, with 10 blocks:
    """)
    return


@app.function
def first_layer_blame(model, X, labels):
    model.zero_grad()
    F.cross_entropy(model(X), labels).backward()
    return torch.mean(torch.abs(model[0].weight.grad)).item()


@app.cell
def _(deep_model, labels, test_images, train_images):
    X_small = torch.tensor(train_images[:3000] / 255, dtype=torch.float32).unsqueeze(1)
    X_centred_test = torch.tensor(test_images / 255, dtype=torch.float32).unsqueeze(1)
    labels_small = labels[:3000]
    torch.manual_seed(0)
    plain_net = deep_model(10, plain_block)
    torch.manual_seed(0)
    skip_net = deep_model(10, residual_block)
    first_layer_blame(plain_net, X_small[:100], labels_small[:100]), first_layer_blame(skip_net, X_small[:100], labels_small[:100])
    return X_centred_test, X_small, labels_small, plain_net, skip_net


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The left number is the plain stack, the right one has skip connections. About 0.000000000001 against
    about 0.015: over ten billion times more blame reaches the first layer with the skips. In the plain
    stack, the first layer gets practically none.

    The button below trains both for 4 epochs on 3,000 centred digits (it takes about half a
    minute) and shows the test accuracy of each: plain first, then with skips. The text after it describes what you should see.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button_2 = mo.ui.run_button(label="Train both deep networks")
    train_button_2
    return (train_button_2,)


@app.cell
def _(
    X_centred_test,
    X_small,
    labels_small,
    labels_test,
    plain_net,
    skip_net,
    train_button_2,
):
    mo.stop(not train_button_2.value, mo.md("*Press the button to train.*"))
    train(plain_net, X_small, labels_small, 4, 50, 0.003)
    train(skip_net, X_small, labels_small, 4, 50, 0.003)
    with torch.no_grad():
        plain_test = accuracy(plain_net(X_centred_test), labels_test)
        skip_test = accuracy(skip_net(X_centred_test), labels_test)
    plain_test, skip_test
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 10% against roughly 80% (the second number changes a little from run to run). The plain stack, with almost no blame reaching its first layer, is still
    guessing: 10% is what you'd get by picking a digit at random. The same layers with an `x +` in each block learn.

    A stack of blocks like this, usually with some more shrinking in between, is called a **ResNet**. It's the
    design that made networks of over 100 layers trainable. The same trick, adding the input back,
    is in every block of the transformers later in this course, for the same reason.

    ## What's next

    That completes Part 1. You started by fitting a line; you've now built classifiers, your own
    engine, and networks that see. Everything from here on rests on these foundations.

    Part 2 turns to **language**. The first problem is that a network needs numbers, and words
    aren't numbers. In **Chapter 12** you'll give every word a vector, in such a way that words with
    similar meanings end up close together.

    ## What you can do now

    - explain why a flat network struggles with moving objects
    - compute a convolution by hand and in code, and predict the size of its output
    - explain what kernels, feature maps and max pooling do
    - count the knobs of a convolution layer
    - build and train a convolutional network with `nn.Conv2d`
    - explain why blame fades in deep stacks and how `x + f(x)` fixes it
    - write your own layer as a class based on `nn.Module`

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** mathematicians would call what we computed a *cross-correlation*; a true
    convolution flips the kernel first, a difference that doesn't matter when kernels are learned.
    Real ResNets also use *batch normalisation*, and colour pictures start with 3 channels (red, green,
    blue). The standard practice sets for this chapter's ideas are *CIFAR-10* and *ImageNet*, and
    the original paper is *Deep Residual Learning for Image Recognition* (He et al., 2015): the
    same He as chapter 09's start.
    """)
    return


if __name__ == "__main__":
    app.run()
