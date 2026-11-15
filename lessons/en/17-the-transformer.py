import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="17 · The Transformer")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.diagrams import block_road
    from learn_llm.images import maps
    from learn_llm.leaderboard import record, show

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 17 · The Transformer

    In chapter 16 you built a self-attention layer: every word asks the words before it, in several heads at once. It ended with a gap. To attention,
    `the dog bit the man` and `the man bit the dog` are the same bag of words.

    This chapter closes that gap, and then builds the model. By the end you'll have a tiny **GPT** with
    about a million knobs, and it will take first place on your leaderboard. Three small pieces are still missing: positions, a steady scale, and the block.
    Everything else you've met: `x + f(x)` (chapter 11), layers and Adam (chapter 10), and the training loop of chapter 14.

    ## From chapter 16

    Here are your five functions from chapter 16, finished, and the layer built from them.
    """)
    return


@app.function
def future_mask(n):
    return torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)


@app.function
def attend(Q, K, V, mask):
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    scores = scores.masked_fill(mask, -torch.inf)
    A = torch.softmax(scores, dim=-1)
    return A @ V, A


@app.function
def split_heads(x, heads):
    batch, words, d = x.shape
    return x.reshape(batch, words, heads, d // heads).transpose(1, 2)


@app.function
def join_heads(x):
    batch, heads, words, piece = x.shape
    return x.transpose(1, 2).reshape(batch, words, heads * piece)


@app.function
def self_attention_step(x, to_query, to_key, to_value, mix, heads):
    Q = split_heads(to_query(x), heads)
    K = split_heads(to_key(x), heads)
    V = split_heads(to_value(x), heads)
    blended, A = attend(Q, K, V, future_mask(x.shape[1]))
    return mix(join_heads(blended)), A


@app.cell
def _():
    class SelfAttention(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.heads = heads
            self.query = nn.Linear(d, d, bias=False)
            self.key = nn.Linear(d, d, bias=False)
            self.value = nn.Linear(d, d, bias=False)
            self.mix = nn.Linear(d, d)

        def forward(self, x):
            out, self.shares = self_attention_step(x, self.query, self.key, self.value, self.mix, self.heads)
            return out
    return (SelfAttention,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Piece 1: positions

    To give attention a sense of order, we tell each word where it stands, in the simplest way there is:
    a second `nn.Embedding`, with one learned vector for each **position**. Position 0 has a vector,
    position 1 has another, and so on. Add it to the word's vector, and the same word at two different places no longer has the same vector.

    It also sets a limit. A table with 32 positions has nothing to say about position 33, so our model can see only the latest 32 words.
    (Chapter 20 shows a smarter way.)

    ## Exercise 1: embed

    /// admonition | 🧰 Toolbox: a row of positions
    `torch.arange(4)` is the tensor of the whole numbers 0, 1, 2, 3. Handing a tensor of ids to an `nn.Embedding` gives one vector for each id (chapter 12),
    so `table(torch.arange(4))` gives the vectors of positions 0 to 3, with shape `(4, d)`.

    Adding shapes `(batch, 4, d)` and `(4, d)` broadcasts (chapter 05): the same four position vectors are added to every text in the batch.
    ///

    Write `embed(ids, words, positions)`. `ids` has shape `(batch, n)`, and `words` and `positions` are two `nn.Embedding` tables.
    Give back, for each word, its word vector plus the vector of the position it stands at, with shape `(batch, n, d)`. The number of words `n` is `ids.shape[1]`.
    """)
    return


@app.function
def embed(ids, words, positions):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("17", "embed", embed, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two lookups, added: the word table gets the ids, and the position table gets the positions 0 to `n - 1`.""",
        r"""One line. The positions don't depend on which words stand there: they are `torch.arange(ids.shape[1])`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Does it work? Here are two texts of three made-up word ids, `4 7 1` and `7 4 1`: the same words, with the first two swapped. The last word is the same
    in both, and it sees all three words in both. The cell runs a fresh attention layer on each text and compares what the last word gets: first with word vectors alone, then with your `embed`.
    It shows the biggest difference in each case. (It uses your `embed`, so it shows an error until Exercise 1 passes. That's normal, and the same goes for later cells that use your functions.)
    """)
    return


@app.cell
def _(SelfAttention):
    torch.manual_seed(0)
    words_demo = nn.Embedding(10, 8)
    positions_demo = nn.Embedding(3, 8)
    layer_demo = SelfAttention(8, 2)
    one_order = torch.tensor([[4, 7, 1]])
    other_order = torch.tensor([[7, 4, 1]])
    with torch.no_grad():
        bare = layer_demo(words_demo(one_order))[0, -1] - layer_demo(words_demo(other_order))[0, -1]
        placed = layer_demo(embed(one_order, words_demo, positions_demo))[0, -1] - layer_demo(embed(other_order, words_demo, positions_demo))[0, -1]
    torch.max(torch.abs(bare)).item(), torch.max(torch.abs(placed)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The first number is 0, or a rounding error such as `3e-08`: with word vectors alone, the last word gets exactly the same blend, whichever order the others came in.
    The second is clearly not 0: about 0.25. With positions added, the layer can tell the two texts apart.

    ## Piece 2: a steady scale

    Stacking many layers lets values fade or explode, the trouble from chapter 09's depth table. He's start fixed that for a plain stack of layers at the beginning of training.
    Here we use a tool that keeps working throughout: **layer normalisation**. It rescales each word's vector, on its own, to mean 0 and typical size 1:
    subtract the vector's mean, then divide by its typical size, the square root of chapter 09's variance.

    Take the vector `[10, 10, 30, 30]`. Its mean is 20. Subtracting gives `[-10, -10, 10, 10]`, the variance is the mean of the squares, 100, and
    the typical size is $\sqrt{100} = 10$. Dividing gives `[-1, -1, 1, 1]`.

    Now try `[1000, 1000, 3000, 3000]`: the result is the same `[-1, -1, 1, 1]`. However large or small the numbers coming in, what goes on to the next layer is in scale.

    ## By hand: normalise a vector

    Your turn, with the vector `[3, 9, 11, 17]`. Work out its mean and its variance, then the first and the last number of the normalised vector.
    """)
    return


@app.cell(hide_code=True)
def _():
    norm_boxes = hand.boxes(['mean', 'variance', 'first', 'last'], step=0.1)
    return (norm_boxes,)


@app.cell(hide_code=True)
def _(norm_boxes):
    hand.marked(norm_boxes, {'mean': 10, 'variance': 25, 'first': -1.4, 'last': 1.4}, {
        'mean': r"""add the four numbers and divide by 4""",
        'variance': r"""subtract the mean from each number, square, and take the mean: $(49 + 1 + 1 + 49) / 4$""",
        'first': r"""$(3 - \text{mean}) / \sqrt{\text{variance}}$, which is negative""",
        'last': r"""$(17 - \text{mean}) / \sqrt{\text{variance}}$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: layer_norm

    /// admonition | 🧰 Toolbox: mean and variance along the last dimension
    For a tensor `T` of any shape, `torch.mean(T, dim=-1, keepdim=True)` is the mean along the last dimension. `keepdim=True` is PyTorch's spelling of chapter 07's `keepdims=True`:
    the result keeps a last dimension of size 1, so it can be subtracted from `T` row by row.

    `torch.var(T, dim=-1, keepdim=True, unbiased=False)` is the variance, the same way. `unbiased=False` asks for chapter 09's plain variance. (Left to itself, PyTorch divides by one less
    than the count, a refinement from statistics that we don't need.)
    ///

    Write `layer_norm(x)`, normalising along the last dimension, for `x` of any shape:

    > $$\frac{x - \text{mean}}{\sqrt{\text{variance} + 0.00001}}$$

    The tiny number guards against dividing by 0 when all the numbers of a vector are equal. In Python you can write it `1e-5`. `torch.sqrt` takes the square root of a tensor.

    One of the checks compares your function with PyTorch's own `nn.LayerNorm`. The real layer adds two learned knobs for each number, a scale and a shift,
    so that training can undo the normalising where that helps.
    """)
    return


@app.function
def layer_norm(x):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("17", "layer_norm", layer_norm, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three lines: the mean, the variance, then the formula. Both need `dim=-1` and `keepdim=True`.""",
        r"""The `1e-5` goes inside the square root, added to the variance. If the check against `nn.LayerNorm` fails by a little, look at `unbiased=False`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Piece 3: the block

    Attention lets words talk to each other. Then each word needs time to think about what it heard. So a **block** has two
    parts: a self-attention layer, and after it a small network from chapter 08, applied to each word separately, with a hidden layer four times as wide.

    Both parts sit on chapter 11's road for blame, `x + f(x)`, with the normalising done just before each part:

    > $$x \leftarrow x + \text{attention}(\text{norm}_1(x))$$

    > $$x \leftarrow x + \text{think}(\text{norm}_2(x))$$

    The arrow means "`x` becomes". Here is the same thing as a picture. The blue roads carry `x` past each part, untouched, and each part only adds its contribution.
    """)
    return


@app.cell(hide_code=True)
def _():
    block_road('x', 'norm', 'attention', 'think',
               'x goes through norm and attention and is added back to itself, then through norm and think and is added back again')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: once through a block

    Follow one number through, with made-up layers that are easy to compute. `x` starts at 3. Both norms subtract 1, `attention` doubles, and `think` multiplies by 3.
    What is `x` after the first line, and after the second?
    """)
    return


@app.cell(hide_code=True)
def _():
    road_boxes = hand.boxes(['after attention', 'after think'])
    return (road_boxes,)


@app.cell(hide_code=True)
def _(road_boxes):
    hand.marked(road_boxes, {'after attention': 7, 'after think': 25}, {
        'after attention': r"""the norm gives $3 - 1 = 2$, attention doubles it to 4, and that is added to the 3 you started with""",
        'after think': r"""start from your first answer: the norm takes 1 off, think multiplies by 3, and the result is added to your first answer""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: block_step

    Write `block_step(x, norm1, attention, norm2, think)`, doing the two lines above. The four layers are called like functions.
    """)
    return


@app.function
def block_step(x, norm1, attention, norm2, think):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("17", "block_step", block_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Each line adds to `x`. It doesn't replace it.""",
        r"""The normalising happens inside, on the way into each part: `attention` gets `norm1(x)`, but what is added to is the plain `x`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The whole model

    Now the pieces go together. Follow the shapes for a batch of texts of `n` words, with `d` numbers per word:

    | step | shape |
    |---|---|
    | the word ids | `(batch, n)` |
    | `embed`: word vectors plus position vectors | `(batch, n, d)` |
    | the blocks, one after another | `(batch, n, d)` |
    | one last normalisation | `(batch, n, d)` |
    | `out`, an `nn.Linear` giving one score for every word of the vocabulary | `(batch, n, vocab)` |

    At each position, the scores are the model's prediction for the *next* word. Thanks to the mask, position 5 has seen only words 0 to 5, so every position of every text is a training example at once.

    ## Exercise 4: gpt_forward

    Write `gpt_forward(ids, words, positions, blocks, norm, out)`, following the table. `blocks` is one layer that already holds the whole stack
    (an `nn.Sequential`, chapter 10), so you call it once. Use your `embed`.
    """)
    return


@app.function
def gpt_forward(ids, words, positions, blocks, norm, out):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("17", "gpt_forward", gpt_forward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Four calls, each on the result of the one before: `embed`, `blocks`, `norm`, `out`.""",
        r"""The order matters: normalise after the blocks, and score last. `embed` needs the ids and both tables.""",
    ], LOCALE)
    return


@app.cell
def _(SelfAttention):
    class Block(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.attention = SelfAttention(d, heads)
            self.norm2 = nn.LayerNorm(d)
            self.think = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

        def forward(self, x):
            return block_step(x, self.norm1, self.attention, self.norm2, self.think)

    class TinyGPT(nn.Module):
        def __init__(self, vocab_size, longest, d, heads, layers):
            super().__init__()
            self.words = nn.Embedding(vocab_size, d)
            self.positions = nn.Embedding(longest, d)
            self.blocks = nn.Sequential(*[Block(d, heads) for i in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, vocab_size)

        def forward(self, ids):
            return gpt_forward(ids, self.words, self.positions, self.blocks, self.norm, self.out)
    return (TinyGPT,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here are the block and the model as PyTorch modules. Each holds its layers, and each `forward` is one of your functions.
    `[Block(d, heads) for i in range(layers)]` builds the blocks, and the `*` in front hands them to `nn.Sequential` one by one (chapter 09).

    ## By hand: where the knobs are

    Before training a big one, count a small one: `TinyGPT(10, 4, 8, 2, 1)` has a vocabulary of 10 words, 4 positions, 8 numbers per word, 2 heads and 1 block.
    An `nn.Embedding` with `r` rows of `d` numbers has $r \times d$ knobs. An `nn.Linear(a, b)` has $a \times b$ weights and `b` biases.
    Read the two classes above, and count four of the parts.
    """)
    return


@app.cell(hide_code=True)
def _():
    count_boxes = hand.boxes(['word table', 'position table', 'think', 'out'])
    return (count_boxes,)


@app.cell(hide_code=True)
def _(count_boxes):
    hand.marked(count_boxes, {'word table': 80, 'position table': 32, 'think': 552, 'out': 90}, {
        'word table': r"""10 words, 8 numbers each""",
        'position table': r"""4 positions, 8 numbers each""",
        'think': r"""two layers: `nn.Linear(8, 32)` has $8 \times 32 + 32$ knobs, and `nn.Linear(32, 8)` has $32 \times 8 + 8$""",
        'out': r"""`nn.Linear(8, 10)`: $8 \times 10$ weights and 10 biases""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here is PyTorch's count for the whole small model:
    """)
    return


@app.cell
def _(TinyGPT):
    small = TinyGPT(10, 4, 8, 2, 1)
    sum([p.numel() for p in small.parameters()])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    1,066. Your four parts make 754 of them. The rest are the attention layer (three matrices of $8 \times 8$, and `mix` with 72) and three normalisations with 16 knobs each.

    Keep an eye on the two tables with one row per vocabulary word, the word table and `out`. In the small model they hold 170 of 1,066 knobs.
    With a real vocabulary they become the biggest parts of the model, as you'll see in a moment.

    ## The data and the training loop

    The data is chapter 13's, with the same vocabulary and the same test stories, so the score goes on the same leaderboard.
    The functions below are from chapters 12 to 14: cutting text into words, turning stories into ids, cutting ids into windows, perplexity, sampling a word with a temperature,
    the training loop with clipping, and scoring. The cell after them shows the vocabulary size, the training words and the test words.
    """)
    return


@app.function
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.function
def story_ids(story_list, index):
    ids = []
    for story in story_list:
        for word in to_words(story):
            ids.append(index.get(word, index["<unk>"]))
        ids.append(index["<end>"])
    return np.array(ids)


@app.function
def windows(ids, length):
    n = (len(ids) - 1) // length
    return ids[:n * length].reshape(n, length), ids[1:n * length + 1].reshape(n, length)


@app.function
def perplexity(log_probs):
    return np.exp(-np.mean(log_probs))


@app.function
def sample_word(scores, temperature):
    p = torch.softmax(scores / temperature, dim=0)
    return torch.multinomial(p, 1).item()


@app.function
def train_gpt(model, X, Y, epochs, batch_size, rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    size = model.out.out_features
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X[batch]).reshape(-1, size), Y[batch].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return loss.item()


@app.function
def score_gpt(model, X, Y):
    size = model.out.out_features
    log_probs = []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            S = model(X[start:start + 64])
            log_probs.append(-F.cross_entropy(S.reshape(-1, size), Y[start:start + 64].reshape(-1), reduction="none").numpy())
    return perplexity(np.concatenate(log_probs))


@app.cell
def _():
    train_stories, test_stories = stories()
    counts = {}
    for word in to_words(" ".join(train_stories)):
        counts[word] = counts.get(word, 0) + 1
    vocab = sorted(counts, key=counts.get, reverse=True)[:2000] + ["<unk>", "<end>"]
    index = {}
    for i in range(len(vocab)):
        index[vocab[i]] = i
    train_ids = story_ids(train_stories, index)
    test_ids = story_ids(test_stories, index)
    len(vocab), len(train_ids), len(test_ids)
    return index, test_ids, train_ids, vocab


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: what reaches the loss

    Read `train_gpt`. We'll train on windows of 32 words, in batches of 64, with a vocabulary of 2,002 words. `F.cross_entropy` wants a table with one row per prediction and one column per
    vocabulary word, so `reshape(-1, size)` flattens batch and position into one dimension. How many rows and columns does that table have for one batch?
    """)
    return


@app.cell(hide_code=True)
def _():
    loss_boxes = hand.boxes(['rows', 'columns'])
    return (loss_boxes,)


@app.cell(hide_code=True)
def _(loss_boxes):
    hand.marked(loss_boxes, {'rows': 2048, 'columns': 2002}, {
        'rows': r"""every position of every window is a prediction: 64 windows of 32 words""",
        'columns': r"""one score for each word of the vocabulary""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    One step of training learns from 2,048 predictions at once. An LSTM gave as many, but had to walk through the 32 words one after another. Here all positions are computed together.

    The button builds a model with 128 numbers per word, 4 heads and 3 blocks, and trains on all 1.9 million words,
    twice. It takes about 3 to 7 minutes on a laptop. Until it finishes, the cells that need its result say `Ancestor stopped`. That's
    normal. If you re-run the training cell, press the button again. The cell shows three numbers: the knobs, the last training loss, and the test perplexity.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Train the transformer (3 to 7 minutes)")
    train_button
    return (train_button,)


@app.cell
def _(TinyGPT, test_ids, train_button, train_ids, vocab):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    X_np, Y_np = windows(train_ids, 32)
    X_test_np, Y_test_np = windows(test_ids, 32)
    torch.manual_seed(0)
    model = TinyGPT(len(vocab), 32, 128, 4, 3)
    knobs = sum([p.numel() for p in model.parameters()])
    last_loss = train_gpt(model, torch.tensor(X_np), torch.tensor(Y_np), 2, 64, 0.002)
    gpt_score = score_gpt(model, torch.tensor(X_test_np), torch.tensor(Y_test_np))
    knobs, last_loss, gpt_score.item()
    return gpt_score, model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 1.1 million knobs, and a test perplexity of about 23. Of those knobs, the word table and `out` hold about 257,000 each, nearly half of the model between them,
    and the three blocks share about 594,000. Here's the leaderboard:
    """)
    return


@app.cell(hide_code=True)
def _(gpt_score):
    record("transformer", gpt_score)
    show(LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First place, if you've run chapter 14: 23 against the LSTM's 31, in a similar training time. Chapter 13's counting models are far behind.

    ## Exercise 5: write_ids

    Chapter 14's `write` carried a memory from word to word. A transformer has no memory. At each step it reads the text so far, again, and predicts the next word.
    But its position table has a limited number of rows, so it can be handed only the latest words.

    /// admonition | 🧰 Toolbox: the end of a list
    `items[-3:]` is the last 3 items of a list. If the list has fewer than 3, it's the whole list.
    ///

    Write `write_ids(model, first, count, temperature, longest)`. It starts from the word id `first`, adds `count` more ids one at a time, and gives back the whole list of ids.
    For each new id:

    1. Take the latest `longest` ids, and make a tensor of shape `(1, n)` from them: a batch holding one text. `torch.tensor([some_list])` does that.
    2. Run the model. Its scores have shape `(1, n, vocab)`.
    3. The prediction for the next word sits at the last position of the only text: `S[0, -1]`. Pick an id from those scores with chapter 14's `sample_word(scores, temperature)`, and append it.

    Nothing is being trained, so put the loop inside `torch.no_grad()` (chapter 10).
    """)
    return


@app.function
def write_ids(model, first, count, temperature, longest):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("17", "write_ids", write_ids, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""The frame is chapter 14's: a list that starts with one id, a `for` loop that runs `count` times, and `append` at the end of each turn.""",
        r"""Slice the list before making the tensor, with `-longest:`. The model must never get more than `longest` ids, or its position table runs out.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now let it write 50 words from `once`, at temperature 0.8. Expect fluent sentences that wander: that's what a 23 gets you.
    """)
    return


@app.cell
def _(index, model, vocab):
    torch.manual_seed(1)
    " ".join([vocab[i] for i in write_ids(model, index["once"], 50, 0.8, 32)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And here are the shares of one head in the last block, as in chapter 15: one row for each word, one column for each earlier word it looks at. The upper right is blank, because that's the future.
    """)
    return


@app.cell
def _(index, model):
    sentence = "once upon a time there was a little girl named lily she loved her dog".split()
    with torch.no_grad():
        model(torch.tensor([[index[word] for word in sentence]]))
    looks = model.blocks[-1].attention.shares[0, 0].numpy()
    mo.vstack([maps([looks], ["word (down) looks at word (across)"], 320), " ".join(sentence)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    (This is the first of the 4 heads.) Each row adds up to 1. Don't expect each head to mean something tidy: patterns like these are often hard to read, and that's normal.

    ## What's next

    You've built the core of a modern language model, and it's small enough to understand completely. The difference between this and a famous model is mostly
    size and data. Big models have billions of knobs, and were trained on far more text.

    One thing you've seen but not yet questioned: how do we know a model is any good at a task, beyond one score? **Chapter 18** looks at how
    models are reused on new tasks, and at the many ways to fool yourself when you measure them.

    ## What you can do now

    - give attention a sense of order with a position table
    - normalise a layer by hand and in code, and say what it's for
    - build a block from two `x + f(x)` steps
    - follow the shapes through a GPT, from ids to scores
    - count a model's knobs, and say where most of them are
    - train a tiny GPT, put it on the leaderboard, and make it write

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** the Transformer is from *Attention Is All You Need* (Vaswani et al., 2017), and GPT from OpenAI's papers from 2018 on. The original used fixed
    sine waves for positions. The order "norm, then attention" is called *pre-norm*. Real models also use *dropout*, and many share one table between the word vectors and `out`, which is called *weight tying*.
    """)
    return


if __name__ == "__main__":
    app.run()
