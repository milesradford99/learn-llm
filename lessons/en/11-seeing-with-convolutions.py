import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="11 · Seeing with convolutions")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import laptop_speed, todo
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
    Two new tools. `rng.integers(0, n)` gives a random whole number from 0 up to, but not
    including, $n$. And a slice works in two directions at once: `canvas[i, r:r + 28, c:c + 28]` is
    the $28 \times 28$ square of picture $i$ whose top-left corner is at row $r$, column $c$, and storing into
    it pastes the digit there.
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

    A patch of only 0s, or only 1s, would give 0, since the −1s and +1s cancel. So
    the result is big exactly where the numbers change from left to right: this kernel is a
    **vertical edge detector**. A $3 \times 3$ kernel fits in 2 positions across and 2 down on a $4 \times 4$ picture,
    so the output is $2 \times 2$. In general, a picture of height $h$ gives $h - k + 1$ rows for a kernel of size $k$.

    ## Exercise 1: convolve

    Write `convolve(image, kernel)`, for a square kernel:

    1. Two lines: `h, w = image.shape` and `k = kernel.shape[0]`
    2. `out = np.zeros((h - k + 1, w - k + 1))`
    3. `for r in range(h - k + 1):`, and inside it `for c in range(w - k + 1):`
    4. Inside both: `patch = image[r:r + k, c:c + k]`, then
       `out[r, c] = np.sum(patch * kernel)`
    5. After the loops, `return out`
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

    **It moves with the picture.** Move the digit 3 columns to the right (`np.roll` shifts an array,
    and `axis=1` means along the columns), and the map just moves 3 columns too. The largest
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

    ## Exercise 2: max_pool

    Write `max_pool(M)`. `//` is division that drops what's after the point (`7 // 2` is 3), so an
    odd row at the edge gets left out.

    1. `h, w = M.shape`
    2. `out = np.zeros((h // 2, w // 2))`
    3. `for r in range(h // 2):`, and inside it `for c in range(w // 2):`
    4. Inside both: `out[r, c] = np.max(M[2 * r:2 * r + 2, 2 * c:2 * c + 2])`
    5. After the loops, `return out`
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
    mo.md(r"""
    ## Letting the network choose its kernels

    We picked our edge detector by hand. A network learns its kernels instead: the 9 numbers are just
    knobs, trained by the same backward pass and Adam as always. In PyTorch that's
    `nn.Conv2d(1, 1, 3)`, read as "1 map in, 1 map out, kernel size 3". Pictures must arrive with a shape of four numbers:

    > (pictures, maps, height, width)

    The extra number is how many maps each picture has. A grey digit has one map. `X.unsqueeze(1)`
    inserts a new dimension of size 1 at position 1, so `(10000, 40, 40)` becomes `(10000, 1, 40, 40)`.

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

    **Many kernels.** One layer learns many kernels at once, each giving its own map.
    `nn.Conv2d(1, 8, 3)` has 8 kernels and turns 1 map into 8. The next layer takes those 8 maps
    in: `nn.Conv2d(8, 16, 3)` has 16 kernels of size $3 \times 3 \times 8$, each looking at the same small patch across all 8 maps. That's how a later kernel can
    say "a vertical edge just left of a horizontal edge". Counting knobs: each of the 16 kernels has
    $3 \times 3 \times 8 = 72$ weights and 1 bias, so $16 \times 73 = 1{,}168$.

    **Keeping the size.** `padding=1` adds a one-pixel margin of zeros around the map first, so a
    $3 \times 3$ kernel gives an output of the same height and width as its input, and only pooling shrinks it.

    And for the end of the network, `nn.AdaptiveMaxPool2d(1)` keeps just the single largest number of
    each whole map, whatever its size. It answers "was this feature anywhere in the picture?"
    After it, the 32 maps have become 32 numbers (`nn.Flatten()` tidies the shape), which a final
    `nn.Linear` turns into 10 scores. Since almost nothing about the position survives, a digit
    scores nearly the same wherever it is. (`nn.MaxPool2d(2)`, in the list below, is the $2 \times 2$ pooling you just wrote.)

    ## Exercise 3: make_cnn

    Write `make_cnn()`, giving back this network (type it as one long
    `return nn.Sequential(...)`, with a comma after each layer, one layer per line if you like):

    1. `nn.Conv2d(1, 8, 3, padding=1)`, `nn.ReLU()`, `nn.MaxPool2d(2)`
    2. `nn.Conv2d(8, 16, 3, padding=1)`, `nn.ReLU()`, `nn.MaxPool2d(2)`
    3. `nn.Conv2d(16, 32, 3, padding=1)`, `nn.ReLU()`
    4. `nn.AdaptiveMaxPool2d(1)`, `nn.Flatten()`, `nn.Linear(32, 10)`
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

    Below, once training is done, are the 8 kernels of the first layer. Nobody told it about edges, but several of
    them compare one side with the other, as your hand-made one does:
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

    The fix is almost too simple. Instead of a block that replaces its input,

    > `out = f(x)`

    use a block that **adds** to it:

    > `out = x + f(x)`

    `f` now only has to learn a *correction* to `x`, called a **residual**. Look at the slope. By
    chapter 02's sum rule, the slope of `x + f(x)` is `1 + f'`, where `f'` is the slope of `f`. Even when
    `f'` is tiny, the slope stays near 1, so blame passes through the block nearly undimmed. The
    `x +` is a direct road for the blame, which is often called a **skip connection**.

    ## Exercise 4: residual_block

    Write `residual_block(x, conv1, conv2)`. `conv1` and `conv2` are two convolution layers that keep the shape:

    1. `h = torch.relu(conv1(x))`
    2. `h = conv2(h)`
    3. `return torch.relu(x + h)`
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
    - `class Block(nn.Module):` means "a `Block` is a kind of `nn.Module`", PyTorch's kind of layer. It
      inherits everything a layer can do. That's why PyTorch can find the knobs of a `Block`.
    - `super().__init__()` runs `nn.Module`'s own setup first. It's always the first line.
    - `forward` is the calculation. Calling `block(x)` runs `block.forward(x)`, which explains
      how `layer(X)` worked in chapter 10.
    - `step` is a function kept in a slot (chapter 02: functions are values), so the same class can
      run your `residual_block`, or `plain_block`, which does the same work without the `x +`.

    `deep_model(blocks, step)` stacks that many blocks, two convolutions each. To compare fairly,
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
    Real ResNets also use *batch normalisation*, and colour pictures start with 3 maps (red, green,
    blue). The standard practice sets for this chapter's ideas are *CIFAR-10* and *ImageNet*, and
    the original paper is *Deep Residual Learning for Image Recognition* (He et al., 2015): the
    same He as chapter 09's start.
    """)
    return


if __name__ == "__main__":
    app.run()
