import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="13 · Memory: RNNs")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.leaderboard import record, show

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 13 · Memory: RNNs

    Chapter 12's counting models remember one or two words, and treat `dog` and `cat` as total
    strangers. You read differently. You go through a sentence word by word, and at each word you
    update what you have in mind. By the end of a paragraph you still know who it's about, without
    having stored any exact word sequence.

    A **recurrent neural network** (RNN) does the same. It keeps a vector of numbers, its **memory**, and
    updates it with each word it reads. This chapter builds one, finds out why it forgets, and fixes that with the
    **LSTM**, the design that led language modelling through most of the 2010s.

    ## A new bend: tanh

    Memory passes through the same step again and again, so it must not grow without limit. ReLU has no ceiling. For
    memory we use a bend that squashes every number into the range −1 to 1:

    What matters is its shape: close to `x` near 0, and flattening out towards
    +1 and −1 for large numbers. In PyTorch it's `torch.tanh`. Below, 0.1 stays at about 0.1, while 5 is squashed to almost 1:
    """)
    return


@app.cell
def _():
    torch.tanh(torch.tensor([-5.0, -1.0, 0.0, 0.1, 1.0, 5.0]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One step of memory

    Say `x` is the vector of the current word (chapter 11's idea), and `h` is the memory so far, of size `m`. The new memory combines the two:

    > new h = tanh(x @ Wx + h @ Wh + b)

    `Wx` turns a word into memory-sized numbers, `Wh` carries the old memory forward, and `b` is the usual
    bias. It's chapter 07's layer, with one twist: part of its input is its own previous output. "Recurrent" just means it comes back round.
    The **same** knobs are used at every word, like chapter 10's kernels: a kernel slides over positions in a picture,
    and this step slides over positions in a text.

    ## Exercise 1: rnn_step

    Write `rnn_step(x, h, Wx, Wh, b)`, for a whole batch at once: `x` has shape `(batch, d)` and `h` has shape `(batch, m)`.

    ```
    return torch.tanh(x @ Wx + h @ Wh + b)
    ```
    """)
    return


@app.function
def rnn_step(x, h, Wx, Wh, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "rnn_step", rnn_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: run_rnn

    Now read a whole text. `X` has shape `(batch, length, d)`: for each text in the batch, one vector per word.
    Memory starts at all zeros. `X[:, t]` picks word number `t` of every text (`:` means "all of this dimension"), and
    `torch.stack(list, dim=1)` joins a list of tensors along a new dimension 1.

    Write `run_rnn(X, Wx, Wh, b)`, giving back the memory after every word, with shape `(batch, length, m)`:

    1. `h = torch.zeros(X.shape[0], Wh.shape[0])`, and `states = []`, on two lines
    2. `for t in range(X.shape[1]):`
    3. Inside it, two lines: `h = rnn_step(X[:, t], h, Wx, Wh, b)` and `states.append(h)`
    4. After the loop, `return torch.stack(states, dim=1)`
    """)
    return


@app.function
def run_rnn(X, Wx, Wh, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "run_rnn", run_rnn, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    PyTorch's `nn.RNN` is the same thing. Let's check, chapter 09 style: take a random `nn.RNN`, copy its
    knobs into your function, and compare. (This cell needs exercise 2.) `nn.RNN(4, 6, batch_first=True)` takes 4 numbers per word and keeps a memory of 6;
    `batch_first=True` says the batch comes first in the shape, as in ours. PyTorch stores its weights on their side, as `nn.Linear` does, and
    keeps two biases, which simply add. Calling it gives back a pair, the memories after every word and the last memory, so `[0]` takes the first.
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    rnn = nn.RNN(4, 6, batch_first=True)
    X_demo = torch.rand(2, 5, 4)
    Wx_demo = rnn.weight_ih_l0.T
    Wh_demo = rnn.weight_hh_l0.T
    b_demo = rnn.bias_ih_l0 + rnn.bias_hh_l0
    with torch.no_grad():
        difference = run_rnn(X_demo, Wx_demo, Wh_demo, b_demo) - rnn(X_demo)[0]
    torch.max(torch.abs(difference)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About `1e-07`, which is 0.0000001: the point moves 7 places left, as in chapter 08. That's only rounding, so
    the two agree.

    ## From text to examples

    To train, we need inputs and answers. For a language model, the answer at each position is
    simply the next word. For the text `[5, 2, 9, 7, 1]`, cut into pieces of length 2:

    ```
    X: 5 2      Y: 2 9
       9 7         7 1
    ```

    At every position, `Y` is `X` moved along by one. So one short piece holds many training examples.

    ## Exercise 3: windows

    Write `windows(ids, length)`, cutting a NumPy array of ids into rows. We need one id more than we use, for the last answer:

    1. `n = (len(ids) - 1) // length`, the number of whole rows
    2. `X = ids[:n * length].reshape(n, length)`
    3. `Y = ids[1:n * length + 1].reshape(n, length)`
    4. `return X, Y`
    """)
    return


@app.function
def windows(ids, length):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "windows", windows, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why plain RNNs forget

    A memory is only useful if the network can *learn* what to keep. To learn that the first word
    mattered, blame from the end of the text must travel all the way back to the first word.
    That's chapter 03's chain rule again, and chapter 10's problem again: blame gets multiplied at each
    step, and a long chain of multiplications either fades or explodes.

    Let's measure it. We run a fresh, untrained `nn.RNN` over random word vectors, take the
    last memory, and go backward. Then we look at how much blame reached the very first
    word's vector, for texts of different lengths. `out[:, -1]` is the last memory. Any single number made from it
    would do as something to blame; we square and add. (`X.grad[:, 0]` is the blame on word 0. Setting
    `requires_grad=True` on the input keeps the books on it too.)
    """)
    return


@app.function
def first_word_blame(layer, length):
    torch.manual_seed(0)
    X = torch.rand(4, length, 16, requires_grad=True)
    out = layer(X)[0]
    torch.sum(out[:, -1] ** 2).backward()
    return torch.mean(torch.abs(X.grad[:, 0])).item()


@app.cell
def _():
    torch.manual_seed(0)
    plain = nn.RNN(16, 64, batch_first=True)
    plain_blame = []
    for length_try in [2, 10, 30, 60]:
        plain_blame.append((length_try, first_word_blame(plain, length_try)))
    plain_blame
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each pair is a length and the blame on its first word: about 0.1 after 2 words, 0.0007 after 10,
    0.00000002 after 30, and about `4e-15` after 60, which means the point moves 15 places left. To the last memory of a 60-word text, the first word might as well
    not exist. So a plain RNN can't learn to carry something across a long text.

    ## The LSTM: a memory with gates

    The trouble is that the plain step *replaces* the memory every time, through a tanh. The **LSTM** ("long short-term memory")
    keeps a second memory vector `c`, the **cell**, and changes it only by **adding**. That's chapter 10's `x + f(x)`:
    addition keeps a clear road for blame. (The LSTM is the older idea: it dates from 1997.)

    To control the cell it uses **gates**. A gate is a vector of numbers between 0 and 1, made with chapter 11's sigmoid,
    multiplied position by position with something else: 0 blocks, 1 lets through. There are three, and they all look at the current word and the previous `h`:

    | gate | question it answers |
    |---|---|
    | forget `f` | how much of each cell number should be kept? |
    | input `i` | how much of the new candidate should be written? |
    | output `o` | how much of the cell should show in `h`? |

    One step, where `both` is the word and the old `h` side by side, and `n` is the candidate new content:

    > c = f × c + i × n

    > h = o × tanh(c)

    The first line is the key. With `f` near 1 and `i` near 0, the cell passes through untouched, and so could the blame.
    The word "could" matters, as we'll see.

    ## Exercise 4: lstm_step

    Write `lstm_step(x, h, c, L)`. `L` is a list of four `nn.Linear` layers, `[forget, input, output, new]`, each taking `both` as its input.
    `torch.cat([a, b], dim=1)` joins two tables side by side.

    1. `both = torch.cat([x, h], dim=1)`
    2. `f = torch.sigmoid(L[0](both))`
    3. `i = torch.sigmoid(L[1](both))`
    4. `o = torch.sigmoid(L[2](both))`
    5. `n = torch.tanh(L[3](both))`
    6. `c = f * c + i * n`
    7. `h = o * torch.tanh(c)`
    8. `return h, c`

    One of the checks compares your step with PyTorch's own LSTM, using the same knobs.
    """)
    return


@app.function
def lstm_step(x, h, c, L):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "lstm_step", lstm_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now the same measurement on a fresh `nn.LSTM`, as before: length, then blame on the first word.
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    lstm_demo = nn.LSTM(16, 64, batch_first=True)
    lstm_blame = []
    for length_lstm in [2, 10, 30, 60]:
        lstm_blame.append((length_lstm, first_word_blame(lstm_demo, length_lstm)))
    lstm_blame
    return (lstm_demo,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Just as bad: about `8e-14` at 60 words. That's worth a moment's thought. A fresh LSTM has random knobs, so
    its forget gates sit near 0.5: at every word, the cell is cut roughly in half, and half of a half of a half soon fades to nothing.

    The difference is that an LSTM has a dial for this, and a plain RNN doesn't. A big bias in the forget gate makes `f` close to 1. Let's set it by hand.
    PyTorch stacks the four gates' biases in one vector. With 64 numbers per gate, the forget gate's are positions 64 to 127.
    A bias of 3 gives σ(3), about 0.95. (Storing into a knob must happen under `torch.no_grad()`, so it isn't written in the book.)
    """)
    return


@app.cell
def _(lstm_demo):
    with torch.no_grad():
        lstm_demo.bias_ih_l0[64:128] = 3.0
    open_blame = []
    for length_open in [2, 10, 30, 60]:
        open_blame.append((length_open, first_word_blame(lstm_demo, length_open)))
    open_blame
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 0.04 at 60 words: the blame no longer fades, and it's hundreds of billions of times stronger than before. A plain RNN has no simple dial like this, because its
    memory is rebuilt through a tanh at every word. In an LSTM, training can open the gate for whatever is worth keeping, and leave it closed for the rest.
    Gates learn, like any other knobs.

    ## Training a language model

    Here's the whole model as a layer: word ids go in, a table of vectors turns them into
    numbers (chapter 11's `nn.Embedding`), an `nn.LSTM` reads them, and an `nn.Linear` gives a score for every
    word in the vocabulary, at every position. `memory=None` lets us hand in the memory from an earlier call, or nothing to start fresh; `write`, further down, uses this.
    The data is chapter 12's: same vocabulary, same ids. The cell below takes a few seconds.
    """)
    return


@app.cell
def _():
    class Reader(nn.Module):
        def __init__(self, vocab_size, d, m):
            super().__init__()
            self.table = nn.Embedding(vocab_size, d)
            self.lstm = nn.LSTM(d, m, batch_first=True)
            self.out = nn.Linear(m, vocab_size)

        def forward(self, ids, memory=None):
            states, memory = self.lstm(self.table(ids), memory)
            return self.out(states), memory
    return (Reader,)


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
    `train_reader` is chapter 09's loop with two small additions.

    **Flattening.** `F.cross_entropy` wants a table of scores and a vector of answers, but we have one
    table per text. `S.reshape(-1, size)` lays them end to end. (`-1` means "whatever number fits".)

    **Clipping.** Blame in a long chain can also explode, and one huge slope would throw the knobs far away. `nn.utils.clip_grad_norm_(knobs, 1.0)`
    measures the length of all the slopes together, as one long vector, and if it's over 1, scales them all down
    so that it's 1. The direction is kept, only the size is limited.

    `score_reader` measures on the test stories with chapter 12's `perplexity`.
    `reduction="none"` asks `F.cross_entropy` for each word's surprise instead of the mean, and the minus
    sign turns surprises into the log chances that `perplexity` expects. `S, _memory = ...` keeps the scores and ignores the memory.
    """)
    return


@app.function
def perplexity(log_probs):
    return np.exp(-np.mean(log_probs))


@app.function
def train_reader(model, X, Y, epochs, batch_size, rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    size = model.out.out_features
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            S, _memory = model(X[batch])
            loss = F.cross_entropy(S.reshape(-1, size), Y[batch].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return loss.item()


@app.function
def score_reader(model, X, Y):
    size = model.out.out_features
    log_probs = []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            S, _memory = model(X[start:start + 64])
            surprise = F.cross_entropy(S.reshape(-1, size), Y[start:start + 64].reshape(-1), reduction="none")
            log_probs.append(-surprise.numpy())
    return perplexity(np.concatenate(log_probs))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The button trains on all 1.9 million words, twice, with a memory of 128 numbers. This is the first
    training in this course that takes real time: about 2 to 5 minutes on a laptop. Until it
    finishes, the cells that need its result say `Ancestor stopped`. That's normal. If you re-run the training cell, press the button again.
    (`np.concatenate` joins a list of arrays into one.) The cell then shows the last training loss, and the test perplexity.
    We train on pieces of 32 words, and score on longer pieces of 64.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Train the LSTM (2 to 5 minutes)")
    train_button
    return (train_button,)


@app.cell
def _(Reader, test_ids, train_button, train_ids, vocab):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    X_np, Y_np = windows(train_ids, 32)
    X_test_np, Y_test_np = windows(test_ids, 64)
    torch.manual_seed(0)
    model = Reader(len(vocab), 64, 128)
    last_loss = train_reader(model, torch.tensor(X_np), torch.tensor(Y_np), 2, 64, 0.003)
    lstm_score = score_reader(model, torch.tensor(X_test_np), torch.tensor(Y_test_np))
    last_loss, lstm_score
    return lstm_score, model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The second number, the test perplexity, should come out at about 31, well below the trigram's 38.
    (The first is the mean surprise on the last training batch, not on the test stories.) Here's the leaderboard, with your LSTM on it:
    """)
    return


@app.cell(hide_code=True)
def _(lstm_score):
    record("lstm", lstm_score)
    show(LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Writing with temperature

    To write text, we feed the model its own last word, carry its memory along, and sample as in chapter 12. There's
    one new dial. Divide the scores by a number called the **temperature** before softmax:

    - temperature 1 gives the model's real chances;
    - below 1, the gaps between scores grow, and the likely words get even likelier: safer, duller text;
    - above 1, the gaps shrink, and the chances flatten out: wilder text.

    ## Exercise 5: sample_word

    Write `sample_word(scores, temperature)`. `scores` is a vector of one score per word. `torch.multinomial(p, 1)` draws 1 id using
    the chances in `p`, and `.item()` makes it a plain number.

    1. `p = torch.softmax(scores / temperature, dim=0)`
    2. `return torch.multinomial(p, 1).item()`
    """)
    return


@app.function
def sample_word(scores, temperature):
    # ← replace todo() below with your code
    todo()


@app.function
def write(model, start, count, temperature, index, vocab):
    words = [index[start]]
    ids = torch.tensor([[index[start]]])
    memory = None
    with torch.no_grad():
        for step in range(count):
            S, memory = model(ids, memory)
            words.append(sample_word(S[0, -1], temperature))
            ids = torch.tensor([[words[-1]]])
    return " ".join([vocab[i] for i in words])


@app.cell(hide_code=True)
def _():
    check("13", "sample_word", sample_word, LOCALE)
    return


@app.cell
def _(index, model, vocab):
    torch.manual_seed(1)
    write(model, "once", 40, 0.8, index, vocab)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare that with chapter 12's bigram text. Stretches of it now read like sentences, and the people in the story stay around
    for a while. Change `0.8` to `0.3` or `1.5` in the cell above and run it again to see the difference.

    Notice what `write` does with `memory`: everything the model knows about the story so far has to fit in
    those 128 numbers, however long the story gets.

    ## What's next

    That squeeze is the LSTM's limit. Asking one fixed vector to hold a whole text is like reading a book,
    then trying to translate it from memory. In **Chapter 14** you'll let a model look back at any word it wants, when it needs it.
    This idea, **attention**, is the heart of every modern language model.

    ## What you can do now

    - explain how a recurrent step reuses the same knobs at every word
    - write an RNN step and run it over a text, and check it against `nn.RNN`
    - turn text into inputs and next-word answers
    - measure how blame fades through time
    - explain how an LSTM's cell and gates protect memory and blame
    - use clipping to limit exploding slopes
    - train a language model, put it on the leaderboard, and generate text with a temperature

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** the LSTM is from Hochreiter and Schmidhuber (1997). Starting forget gates with a positive bias is common practice. The *GRU* is a
    simpler relative with two gates. Going backward through a long text is called *backpropagation through time*. Our training cuts the text into
    pieces of 32 words, so blame travels at most 32 steps; this is called *truncation*.
    """)
    return


if __name__ == "__main__":
    app.run()
