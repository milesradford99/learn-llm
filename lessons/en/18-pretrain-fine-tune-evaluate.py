import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="18 · Pretrain, fine-tune, evaluate")

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
    from learn_llm.diagrams import pipeline

    LOCALE = "en"
    laptop_speed()
    ANIMALS = ["dog", "cat", "bird", "fish"]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 18 · Pretrain, fine-tune, evaluate

    Training on 1.9 million words needed no labels, because the text was its own answer. Most useful tasks aren't like that. To sort emails or
    spot a disease, someone must label examples by hand, and labels are expensive.

    The standard trick is **pretraining**: train a language model on lots of plain text, then reuse it on a task with few labels.
    That's what this chapter tries. But it's really about something else: **how to tell whether a result is real.** You'll fool a model on
    purpose, meet a do-nothing model that scores 40%, and see a popular claim fail to show up in the numbers.

    ## A task from the stories

    We have no hand-labelled data, so we'll make a task from the stories. Take every story that mentions exactly one of these
    four animals: `dog`, `cat`, `bird`, `fish`. The label is that animal. Now **blank out** the word wherever it appears, and ask the
    model which animal the story is about. It must work it out from clues like `barked` or `swam`. (Only those exact words are blanked, so a `dogs` can still slip through.)

    /// admonition | 🧰 Toolbox: choosing between two values
    `a if test else b` is an expression: it gives `a` when the test holds, and `b` otherwise. It fits inside a list-building
    expression, to choose each item: `[0 if x < 0 else x for x in [3, -1, 2]]` gives `[3, 0, 2]`.
    ///

    ## Exercise 1: blank_out

    Write `blank_out(ids, hidden, blank)`. `ids` is a list of word ids, `hidden` is a list of the ids to hide, and `blank` is the id to put in their place.
    It gives back a new list of the same length:

    > `blank_out([5, 7, 5, 9], [5], 0)` gives `[0, 7, 0, 9]`
    """)
    return


@app.function
def blank_out(ids, hidden, blank):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "blank_out", blank_out, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""One list-building expression over `ids`. Each item is either `blank` or the id itself, and the test is whether the id is `in hidden`.""",
        r"""It's one line, shaped like the Toolbox example: `return [... if ... else ... for i in ids]`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The model, and a new last layer

    **From earlier chapters.** The two cells below hold chapter 17's model, smaller. Their code is folded away, because you wrote all of it there.
    Two things are new. The model has a switch: `.features(ids)` gives the vectors from the last block, before the scores,
    and `forward` is just `self.out(self.features(ids))`. And the three attention layers sit in an `nn.ModuleList`, a list that tells PyTorch its layers have knobs to train.
    """)
    return


@app.function(hide_code=True)
def attend(Q, K, V, mask):
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    A = torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1)
    return A @ V, A


@app.cell(hide_code=True)
def _():
    class Block(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.heads = heads
            self.norm1 = nn.LayerNorm(d)
            self.attention = nn.ModuleList([nn.Linear(d, d, bias=False) for i in range(3)])
            self.mix = nn.Linear(d, d)
            self.norm2 = nn.LayerNorm(d)
            self.think = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

        def look(self, x):
            batch, words, d = x.shape
            Q, K, V = [layer(x).reshape(batch, words, self.heads, d // self.heads).transpose(1, 2) for layer in self.attention]
            mask = torch.triu(torch.ones(words, words, dtype=torch.bool), diagonal=1)
            return self.mix(attend(Q, K, V, mask)[0].transpose(1, 2).reshape(batch, words, d))

        def forward(self, x):
            x = x + self.look(self.norm1(x))
            return x + self.think(self.norm2(x))

    class TinyGPT(nn.Module):
        def __init__(self, vocab_size, longest, d, heads, layers):
            super().__init__()
            self.words = nn.Embedding(vocab_size, d)
            self.positions = nn.Embedding(longest, d)
            self.blocks = nn.Sequential(*[Block(d, heads) for i in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, vocab_size)

        def features(self, ids):
            x = self.words(ids) + self.positions(torch.arange(ids.shape[1]))
            return self.norm(self.blocks(x))

        def forward(self, ids):
            return self.out(self.features(ids))
    return (TinyGPT,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To classify a story, we put a new, small layer on top of those features. Each example will be the first 64 words of a story, and the
    features have 96 numbers per word. The plan:
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([('ids', '(batch, 64)'), ('features', '(batch, 64, 96)'), ('one vector', '(batch, 96)'), ('scores', '(batch, 4)')],
             ['the GPT body', 'mean over the words', 'new layer: the head'],
             'The ids go through the GPT body to give features, the mean over the words gives one vector per story, and a new layer called the head gives four scores.')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The GPT's blocks are called the **body**, and the new `nn.Linear(96, 4)` the **head**. Averaging over the words turns a story of any length into one vector, so the head
    can score it: one score for each animal.

    ## Exercise 2: classify

    Write `classify(features, head)`, the last two arrows of the picture. `features` has shape `(batch, words, d)` and `head` is an `nn.Linear(d, 4)`.
    It gives back the scores, of shape `(batch, 4)`. `torch.mean(x, dim=1)` takes the mean along dimension 1, which then disappears from the shape.
    """)
    return


@app.function
def classify(features, head):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "classify", classify, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two steps, one inside the other: the mean over the words, then the head called on the result.""",
        r"""It's one line: `return head(...)`, with `torch.mean` of `features` along `dim=1` inside the brackets.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `Classifier` holds a body and a head, and its `forward` is your function. The cells after it prepare the data. Until Exercises 1 and 2 pass, cells that
    need them show an error. That's normal, and the same goes for the later exercises.
    """)
    return


@app.cell
def _():
    class Classifier(nn.Module):
        def __init__(self, body, d):
            super().__init__()
            self.body = body
            self.head = nn.Linear(d, len(ANIMALS))

        def forward(self, ids):
            return classify(self.body.features(ids), self.head)
    return (Classifier,)


@app.function(hide_code=True)
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.function(hide_code=True)
def story_ids(story_list, index):
    ids = []
    for story in story_list:
        for word in to_words(story):
            ids.append(index.get(word, index["<unk>"]))
        ids.append(index["<end>"])
    return np.array(ids)


@app.function
def labelled(story_list, index):
    animal_ids = [index[a] for a in ANIMALS]
    examples = []
    for story in story_list:
        ids = [index.get(word, index["<unk>"]) for word in to_words(story)]
        found = [a for a in range(4) if animal_ids[a] in ids]
        if len(found) == 1:
            examples.append((ids, found[0]))
    return examples


@app.function
def to_tensors(examples, hidden, blank, end):
    rows = []
    for ids, label in examples:
        row = blank_out(ids, hidden, blank)[:64]
        rows.append(row + [end] * (64 - len(row)))
    return torch.tensor(rows), torch.tensor([label for ids, label in examples])


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
    hidden = [index[a] for a in ANIMALS]
    train_examples = labelled(train_stories, index)
    test_examples = labelled(test_stories, index)
    X_test, Y_test = to_tensors(test_examples, hidden, index["<unk>"], index["<end>"])
    len(train_examples), len(test_examples), torch.bincount(Y_test).tolist()
    return (
        X_test,
        Y_test,
        hidden,
        index,
        test_examples,
        train_examples,
        train_stories,
        vocab,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 3,300 labelled stories for training, and 265 for testing: 107 dog, 55 cat, 79 bird, 24 fish.

    /// admonition | 🧰 Toolbox: counting labels
    `np.bincount(a)` counts how often each whole number appears in an array: position 0 of the result is the number of 0s, position 1 the number of 1s,
    and so on. `np.bincount(np.array([0, 2, 2, 1, 2]))` is `[1, 1, 3]`. `torch.bincount`, used above, is its PyTorch twin.
    ///

    ## First, a baseline

    Before any model, ask what a score means. If 40% of test stories are about dogs, a "model" that always says `dog` scores 40% without reading a word.
    That's the **majority baseline**, and a model is only interesting if it beats it. It works in the other direction too: on a task
    where 99% of emails are fine, a filter that blocks nothing scores 99%.

    ## Exercise 3: majority_share

    Write `majority_share(labels)`. `labels` is a NumPy array of whole numbers, 0 or above. It gives back the share of the labels
    taken by the commonest one: the accuracy of always giving that answer.

    > `majority_share(np.array([1, 1, 1, 0, 2, 2]))` gives `0.5`
    """)
    return


@app.function
def majority_share(labels):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "majority_share", majority_share, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Count each label with `np.bincount`, take the biggest count with `np.max`, and divide by how many labels there are.""",
        r"""It's one line. In the example the counts are `[1, 3, 2]`, the biggest is 3, and there are 6 labels.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Better than accuracy

    Accuracy hides which mistakes a model makes. For one class, say `fish`, sort the predictions into three counts:

    - **true positives** ($tp$): said fish, and it was fish;
    - **false positives** ($fp$): said fish, and it wasn't;
    - **false negatives** ($fn$): it was fish, and the model said something else.

    From those come two measures:

    > $$\text{precision} = \frac{tp}{tp + fp} \qquad \text{recall} = \frac{tp}{tp + fn}$$

    **Precision** asks: when it says fish, how often is it right? **Recall** asks: of the real fish, how many did it find?

    A filter that never says "spam" finds no spam, so its recall is 0, and it makes no guesses to be precise about, so we score that 0 too. A filter that says "spam" for everything has recall 1 and poor precision. These two numbers keep each other honest.

    ## By hand: count the three kinds

    Ten stories, with the model's guess and the true label for each. F is fish, D is dog.

    | story | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | guess | F | F | F | D | F | D | D | F | D | D |
    | label | F | F | D | D | F | F | D | F | F | D |

    For the class fish, count $tp$, $fp$ and $fn$. Then work out the precision and the recall, to two decimals.
    """)
    return


@app.cell(hide_code=True)
def _():
    fish_boxes = hand.boxes(['tp', 'fp', 'fn', 'precision', 'recall'], step=0.01)
    return (fish_boxes,)


@app.cell(hide_code=True)
def _(fish_boxes):
    hand.marked(fish_boxes, {'tp': 4, 'fp': 1, 'fn': 2, 'precision': 0.8, 'recall': 0.67}, {
        'tp': r"""the columns with F in both rows""",
        'fp': r"""the columns with guess F and label D""",
        'fn': r"""the columns with guess D and label F""",
        'precision': r"""$tp$ divided by the number of F guesses, $tp + fp$""",
        'recall': r"""$tp$ divided by the number of F labels, $tp + fn$""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The model is right 7 times out of 10, and that one number can't tell you that it misses a third of the fish.

    /// admonition | 🧰 Toolbox: counting with comparisons
    - Comparing an array with a number gives an array of `True` and `False`: `np.array([1, 0, 1]) == 1` is `[True, False, True]`.
    - `&` means "and", position by position: `True` only where both sides are `True`. Each comparison needs its own brackets:
      `(a == 1) & (b != 1)`.
    - `np.sum` of such an array counts its `True`s.
    ///

    ## Exercise 4: precision_recall

    Write `precision_recall(guesses, labels, c)`. `guesses` and `labels` are NumPy arrays of class numbers, of the same length, and `c` is the class to measure.
    It gives back the pair: precision, then recall. If the bottom of a fraction is 0, that measure is 0: `max(n, 1)` is a short way to keep a bottom from being 0, because the top is 0 then too.

    With fish as class 1 and dog as class 0, the table above is one of the checks, and gives about `(0.8, 0.67)`.
    """)
    return


@app.function
def precision_recall(guesses, labels, c):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "precision_recall", precision_recall, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three counts, each an `np.sum` of two comparisons joined with `&`. For `tp` both `guesses` and `labels` equal `c`. For `fp` the guess equals `c` and the label doesn't (`!=`). For `fn` it's the other way round.""",
        r"""It's 4 lines. The first is `tp = np.sum((guesses == c) & (labels == c))`. The last gives back two fractions with a comma between them, precision first, each with `max(..., 1)` as its bottom.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Is a difference real?

    With only 265 test stories, scores are noisy. To see how noisy, imagine a model that gets each story right with a chance of exactly 0.6.
    Its true score is 60%. Now test it on 200 different test sets of 265 stories each. The cell below does that with random numbers
    (`rng.random(shape)` gives numbers between 0 and 1, so each is below 0.6 with chance 0.6), and shows the lowest and the highest of the 200 scores.
    """)
    return


@app.cell
def _():
    rng = np.random.default_rng(0)
    right = rng.random((200, 265)) < 0.6
    test_scores = np.mean(right, axis=1)
    round(np.min(test_scores).item(), 3), round(np.max(test_scores).item(), 3)
    return (test_scores,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The same model scored anywhere from about 52% to about 69%, depending only on which stories happened to be in the test. So a single score
    is the truth plus some luck, and we need to know how big the luck can be.

    Picture each story as a coin that lands "right" (1) with chance $p$, and otherwise "wrong" (0). One such result has variance
    $p \times (1 - p)$, which you can take on trust. Chapter 09 gave two rules for the rest: variances add, so $n$ stories add up to variance $n \times p \times (1 - p)$; and dividing by $n$ to get the mean divides the variance by $n^2$. That leaves $p \times (1 - p) / n$, and its square root tells how far a measured score tends to stray from the truth.

    Twice that size covers about 95% of cases, another fact to take on trust, so this is the usual **margin of error**:

    > $$\text{margin} = 2 \times \sqrt{\frac{p \times (1 - p)}{n}}$$

    ## By hand: two margins

    A model scores $p = 0.5$ on a test of $n = 100$ examples. What's the margin? And what is it with four times the examples, $n = 400$?
    """)
    return


@app.cell(hide_code=True)
def _():
    margin_boxes = hand.boxes(['margin for n = 100', 'margin for n = 400'], step=0.01)
    return (margin_boxes,)


@app.cell(hide_code=True)
def _(margin_boxes):
    hand.marked(margin_boxes, {'margin for n = 100': 0.1, 'margin for n = 400': 0.05}, {
        'margin for n = 100': r"""$0.5 \times 0.5 / 100 = 0.0025$, and its square root is 0.05""",
        'margin for n = 400': r"""$0.25 / 400 = 0.000625$, and its square root is 0.025""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Four times the data only halves the margin. That's why small test sets are so hard to escape from.

    ## Exercise 5: margin

    Write `margin(p, n)`, the formula above. `p` is a score between 0 and 1 and `n` the number of test examples. `np.sqrt` is the square root.

    > `margin(0.5, 100)` gives `0.1`

    The cell after the check counts how many of the 200 simulated scores landed within one margin of the true 0.6.
    """)
    return


@app.function
def margin(p, n):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "margin", margin, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""One line. The fraction goes inside `np.sqrt`, and the 2 multiplies the result.""",
        r"""If `margin(0.5, 100)` gives 0.005, the square root is missing. If it gives 0.05, the 2 is missing.""",
    ], LOCALE)
    return


@app.cell
def _(test_scores):
    np.mean(np.abs(test_scores - 0.6) < margin(0.6, 265)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 0.95, as promised: 19 scores in 20 fall within the margin, and 1 in 20 falls outside it.

    For $p = 0.6$ and $n = 265$ the margin is about 0.06, or 6 points. When comparing two scores, each has its own noise, and by adding variances the margin for the difference is about 1.4 times bigger: about 8.5 points here. So two models at 58% and 61% on this test can't be told apart.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Pretraining

    The button below trains the smaller GPT on all the training stories, labels ignored, for about 2 to 4 minutes. Until then, cells that need it say `Ancestor stopped`. That's normal.
    (The model has seen the test stories' *kind* of text, but not the test stories themselves: only `train_stories` go in.)
    """)
    return


@app.function
def pretrain(model, ids, epochs):
    n = (len(ids) - 1) // 64
    X = torch.tensor(ids[:n * 64].reshape(n, 64))
    Y = torch.tensor(ids[1:n * 64 + 1].reshape(n, 64))
    size = model.out.out_features
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for epoch in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, 64):
            batch = order[start:start + 64]
            optimizer.zero_grad()
            F.cross_entropy(model(X[batch]).reshape(-1, size), Y[batch].reshape(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return {name: value.clone() for name, value in model.state_dict().items()}


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Pretrain (2 to 4 minutes)")
    train_button
    return (train_button,)


@app.cell
def _(TinyGPT, index, train_button, train_stories, vocab):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    torch.manual_seed(0)
    pretrained_model = TinyGPT(len(vocab), 64, 96, 4, 2)
    saved = pretrain(pretrained_model, story_ids(train_stories, index), 2)
    len(saved)
    return pretrained_model, saved


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: saving the knobs
    `model.state_dict()` gives a dictionary of all of a model's knobs, by name, and `model.load_state_dict(saved)` puts a saved set back into a model of the same design.
    ///

    So `saved` lets us start several models from the same pretrained point. The number shown is how many named tensors of knobs there are.

    ## With no labels at all

    A language model can try this task without a single label. At each blank, ask for the chance of each of the four animals as the next word, add up
    the logs across the story's blanks, and take the best. The cell shows the share correct, with its margin.
    """)
    return


@app.function
def guess_without_labels(model, examples, hidden, blank):
    guesses = []
    for ids, label in examples:
        row = torch.tensor([blank_out(ids, hidden, blank)[:64]])
        with torch.no_grad():
            chances = torch.log_softmax(model(row)[0], dim=1)
        totals = np.zeros(4)
        for t in range(1, row.shape[1]):
            if ids[t] in hidden:
                totals = totals + chances[t - 1, hidden].numpy()
        guesses.append(np.argmax(totals).item())
    return np.array(guesses)


@app.cell
def _(Y_test, hidden, index, pretrained_model, test_examples):
    no_label_guesses = guess_without_labels(pretrained_model, test_examples, hidden, index["<unk>"])
    labels_np = Y_test.numpy()
    no_label_score = np.mean(no_label_guesses == labels_np)
    majority_share(labels_np).item(), no_label_score.item(), margin(no_label_score, len(labels_np)).item()
    return (labels_np,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The majority baseline is 40%. The pretrained model gets about 51%, with no labels, and the margin of 6 points clears the baseline. Pretraining has done something real.

    ## Fine-tuning

    Now use labels. **Fine-tuning** means starting from pretrained knobs, adding a new last layer, and training gently on labelled examples.
    Is that better than starting from random knobs? There's a middle way too: **freezing** the body, so that only the new layer learns. Here are the three, on the picture's two parts:

    | | body | head |
    |---|---|---|
    | **from scratch** | random start, trained | random start, trained |
    | **frozen body** | pretrained, not trained | random start, trained |
    | **fine-tuned** | pretrained, trained gently | random start, trained |

    To freeze a knob, set its `requires_grad` to `False` (chapter 10), so no blame reaches it and no optimizer moves it.

    ## Exercise 6: freeze

    Write `freeze(model)`. It sets `requires_grad` to `False` on every knob of the model, and gives nothing back: it changes the model it's given, so it needs no `return`.
    `model.parameters()` goes through a model's knobs (chapter 10).
    """)
    return


@app.function
def freeze(model):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "freeze", freeze, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""A loop over `model.parameters()`, with one line inside it that stores `False`.""",
        r"""It's 2 lines. Storing uses one `=`, as in `p.requires_grad = False`. Two, `==`, would only compare.""",
    ], LOCALE)
    return


@app.function
def fit(model, X, Y, epochs, rate):
    knobs = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(knobs, lr=rate)
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), 32):
            batch = order[start:start + 32]
            optimizer.zero_grad()
            F.cross_entropy(model(X[batch]), Y[batch]).backward()
            optimizer.step()


@app.function
def share_right(model, X, Y):
    with torch.no_grad():
        return torch.mean((torch.argmax(model(X), dim=1) == Y).float()).item()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The next cell tries all three on 100, 300 and 1,000 labelled stories, and takes a minute or so. Each gets settings that suit it: frozen bodies train only one layer, so they get more epochs, and fine-tuning uses a small rate so as not to disturb the pretrained knobs. Before it runs, make a guess: which column will win?
    """)
    return


@app.cell(hide_code=True)
def _(
    Classifier,
    TinyGPT,
    X_test,
    Y_test,
    hidden,
    index,
    saved,
    train_examples,
    vocab,
):
    rows = ["| labels | from scratch | frozen body | fine-tuned |", "|---|---|---|---|"]
    best_for_table = {}
    for n in [100, 300, 1000]:
        chosen = [train_examples[i] for i in np.random.default_rng(0).permutation(len(train_examples))[:n].tolist()]
        X_n, Y_n = to_tensors(chosen, hidden, index["<unk>"], index["<end>"])
        cells = []
        for kind, epochs, rate in (("scratch", 15, 0.001), ("frozen", 40, 0.003), ("tuned", 10, 0.0005)):
            torch.manual_seed(1)
            body = TinyGPT(len(vocab), 64, 96, 4, 2)
            if kind != "scratch":
                body.load_state_dict(saved)
            if kind == "frozen":
                freeze(body)
            classifier = Classifier(body, 96)
            fit(classifier, X_n, Y_n, epochs, rate)
            cells.append(f"{share_right(classifier, X_test, Y_test):.1%}")
            best_for_table[(n, kind)] = classifier
        rows.append(f"| {n} | " + " | ".join(cells) + " |")
    mo.md("\n".join(rows))
    return (best_for_table,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now read that table with Exercise 5 in hand. (Two cells showing the same number is a coincidence.) At 100 and 300 labels the gaps are a few points, far inside the margin of about 8.5, so the columns can't be told apart. At 1,000 labels, from scratch leads by 7 or 8 points, right at the edge of that margin: probably real, but not certain.

    **So pretraining gave no clear gain here.** Writing this chapter, we expected the usual story, where it wins clearly, and tried several settings. It didn't happen. That's the honest result of a small model pretrained on a small amount of text, and spotting a task where it would shine is part of the craft.
    Pretraining pays off when the pretrained model is far bigger than anything you could train on your labels, and when your labels are very few. What's striking about the earlier 51% isn't its size, but that it needed no labels at all.

    Here's the fine-tuned model at 300 labels, with each row as (animal, precision, recall). Notice `fish`, the rarest animal, where accuracy alone would hide how the model does. With only 24 fish stories, those two numbers are noisy too.
    """)
    return


@app.cell
def _(X_test, best_for_table, labels_np):
    with torch.no_grad():
        tuned_guesses = torch.argmax(best_for_table[(300, "tuned")](X_test), dim=1).numpy()
    report = []
    for c in range(4):
        p, r = precision_recall(tuned_guesses, labels_np, c)
        report.append((ANIMALS[c], round(p.item(), 2), round(r.item(), 2)))
    report
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Leakage: fooling yourself on purpose

    **Leakage** means information about the answer sneaks into what the model sees. Let's cause it. Below, we forget to blank the animal words, in both the training and the test stories, and get a fine score. Then we measure the same model on the blanked stories, which is the real task.
    """)
    return


@app.cell
def _(Classifier, TinyGPT, X_test, Y_test, index, train_examples, vocab):
    chosen_leak = [train_examples[i] for i in np.random.default_rng(0).permutation(len(train_examples))[:300].tolist()]
    X_leak, Y_leak = to_tensors(chosen_leak, [], index["<unk>"], index["<end>"])
    X_test_leak = torch.tensor([(ids[:64] + [index["<end>"]] * 64)[:64] for ids, label in labelled(stories()[1], index)])
    torch.manual_seed(1)
    leaky = Classifier(TinyGPT(len(vocab), 64, 96, 4, 2), 96)
    fit(leaky, X_leak, Y_leak, 15, 0.001)
    share_right(leaky, X_test_leak, Y_test), share_right(leaky, X_test, Y_test)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 81%, then about 45%. The first number looks like a breakthrough, and is worthless: the model just reads the answer. On the real task it's barely above always saying `dog`. A test set protects you only if it's clean.

    ## A quieter leak: the same text twice

    Leaks are usually subtler than that one. A common one: collections of text contain copies, and when the collection is cut into training and test parts,
    a copy can land on each side. The model then gets tested on a text it has trained on, with its answer. Here's a split with that leak:
    """)
    return


@app.function
def leaky_split(texts, n_test):
    cut = len(texts) - n_test
    return texts[:cut], texts[cut:]


@app.cell
def _():
    leaky_split(["a", "b", "a", "c", "b"], 2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: find the leak

    Look at the cell's result: the training texts, then the test texts. How many of the 3 training texts also appear in the test part?
    And how many training texts are left once those are taken out?
    """)
    return


@app.cell(hide_code=True)
def _():
    leak_boxes = hand.boxes(['training texts that are also in the test', 'training texts left'])
    return (leak_boxes,)


@app.cell(hide_code=True)
def _(leak_boxes):
    hand.marked(leak_boxes, {'training texts that are also in the test': 1, 'training texts left': 2}, {
        'training texts that are also in the test': r"""the test part is `["c", "b"]`. Which of `"a"`, `"b"`, `"a"` is in it?""",
        'training texts left': r"""3 minus the ones you just counted""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 7: clean_split

    Write `clean_split(texts, n_test)`, a split without that leak. `texts` is a list of strings and `n_test` a whole number. Like `leaky_split`, it gives back
    the pair: training texts, then test texts, and the test part is the last `n_test` texts. But the training part leaves out every text that also appears in the test part.

    > `clean_split(["a", "b", "a", "c", "b"], 2)` gives `(["a", "a"], ["c", "b"])`

    We take from the training side, not the test side, so the test set stays the same whatever the training set is.
    """)
    return


@app.function
def clean_split(texts, n_test):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "clean_split", clean_split, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Start as `leaky_split` does, so you have the two parts. Then build the training list again, keeping only the texts that are `not in` the test part: chapter 12's list-building expression with an `if` at the end.""",
        r"""It can be 4 lines: the cut, the test part, the filtered training part, and the `return`. The filter is `[t for t in ... if t not in test]`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Other leaks are harder to see: a date that gives away the answer, or a number
    tuned on the test set again and again (chapter 13's $k$). Repeating a test until it looks good is itself a leak, and it's the same reason shared **benchmarks**, standard test sets used by everyone, slowly wear out.

    ## Looking both ways: BERT

    GPT reads left to right, because its job is to write. For understanding a text that already exists, why not look both ways? **BERT** is a transformer with no future mask, trained on a different job: hide 15% of the words, and predict them from both sides.
    For the blank in `the ▢ barked loudly`:

    | model | words it can use to fill the blank |
    |---|---|
    | GPT | `the` |
    | BERT | `the`, `barked`, `loudly` |

    We tried that here too, expecting it to beat the 51% from before, since it can see the `barked` that comes after a blank. On this data it scored about 32%, below the baseline. It takes more data than a small model on a small text can offer. So keep that as a lesson rather than a recipe: an idea that wins at huge scale isn't guaranteed to win at small scale.

    ## What's next

    That completes Part 2. You can represent words, score language models, and build a transformer. Part 3 rebuilds your tiny GPT the way real labs do, starting from its very first step.
    Our word list breaks on any word it hasn't seen, and on any other language. **Chapter 19** replaces it with a proper **tokenizer**.

    ## What you can do now

    - explain pretraining, fine-tuning and freezing, and when each makes sense
    - put a new head on a pretrained body
    - compare any score with a baseline
    - count true and false positives by hand, compute precision and recall, and say what each one protects against
    - estimate a margin of error, and decide whether two scores really differ
    - recognise leakage, and split data so that no text sits on both sides
    - explain how BERT differs from GPT

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** BERT is from Devlin et al. (2018). Precision and recall combine into the *F1 score*. Comparing two models on the same test is done properly with a *paired* test such as McNemar's. Benchmarks include *SST-2* and *SQuAD*, and using an existing model on a new task is called *transfer learning*.
    """)
    return


if __name__ == "__main__":
    app.run()
