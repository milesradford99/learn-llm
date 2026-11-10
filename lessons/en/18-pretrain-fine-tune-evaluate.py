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

    from learn_llm import laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories

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

    ## Exercise 1: blank_out

    Write `blank_out(ids, hidden, blank)`. `ids` is a list of word ids, `hidden` is a list of ids to hide, and `blank` is the id to put in their place:

    ```
    return [blank if i in hidden else i for i in ids]
    ```

    The expression `a if test else b` gives `a` when the test holds, and `b` otherwise.
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
    mo.md(r"""
    Here's chapter 17's model, smaller, and with a switch: `.features(ids)` gives the vectors from the last block, before the scores. A new layer on
    top of those will do the classifying: `Classifier` averages the vectors over the 64 words, then applies one new `nn.Linear`. You've seen the rest in chapter 17, so you don't need to study this code. The next cells prepare the data: each example is the first 64 words of a story.
    """)
    return


@app.function
def attend(Q, K, V, mask):
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    A = torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1)
    return A @ V, A


@app.cell
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

    class Classifier(nn.Module):
        def __init__(self, body, d):
            super().__init__()
            self.body = body
            self.head = nn.Linear(d, len(ANIMALS))

        def forward(self, ids):
            return self.head(torch.mean(self.body.features(ids), dim=1))
    return Classifier, TinyGPT


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
    About 3,300 labelled stories for training, and 265 for testing: 107 dog, 55 cat, 79 bird, 24 fish. `torch.bincount` counts how often each number appears.

    ## First, a baseline

    Before any model, ask what a score means. If 40% of test stories are about dogs, a "model" that always says `dog` scores 40% without reading a word.
    That's the **majority baseline**, and a model is only interesting if it beats it. It works in the other direction too: on a task
    where 99% of emails are fine, a filter that blocks nothing scores 99%.

    ## Exercise 2: majority_share

    Write `majority_share(labels)` for a NumPy array of whole numbers. `np.bincount` is the NumPy twin of `torch.bincount`:

    ```
    return np.max(np.bincount(labels)) / len(labels)
    ```

    ## Better than accuracy

    Accuracy hides which mistakes a model makes. For one class, say `fish`, sort the predictions into three counts:

    - **true positives** (`tp`): said fish, and it was fish;
    - **false positives** (`fp`): said fish, and it wasn't;
    - **false negatives** (`fn`): it was fish, and the model said something else.

    From those come two measures:

    > **precision** = tp / (tp + fp): when it says fish, how often is it right?

    > **recall** = tp / (tp + fn): of the real fish, how many did it find?

    A filter that never says "spam" finds no spam, so its recall is 0, and it makes no guesses to be precise about, so we score that 0 too. A filter that says "spam" for everything has recall 1 and poor precision. These two numbers keep each other honest.

    ## Exercise 3: precision_recall

    Write `precision_recall(guesses, labels, c)` for NumPy arrays, giving back both for class `c`. Comparing gives `True`/`False`, `&` means "and" position by position, and `np.sum` counts the `True`s. To avoid dividing by zero, `max(n, 1)` gives at least 1.

    1. `tp = np.sum((guesses == c) & (labels == c))`
    2. `fp = np.sum((guesses == c) & (labels != c))`
    3. `fn = np.sum((guesses != c) & (labels == c))`
    4. `return tp / max(tp + fp, 1), tp / max(tp + fn, 1)`
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
    mo.md(r"""
    ## Is a difference real?

    With only 265 test stories, scores are noisy. Picture each story as a coin that lands "right" (1) with chance $p$, and otherwise "wrong" (0). One such result has variance
    $p \times (1 - p)$, which you can take on trust. Chapter 09 gave two rules for the rest: variances add, so $n$ stories add up to variance $n \times p \times (1 - p)$; and dividing by $n$ to get the mean divides the variance by $n^2$. That leaves $p \times (1 - p) / n$, and its square root tells how far a measured score tends to stray from the truth.

    Twice that size covers about 95% of cases, another fact to take on trust, so this is the usual **margin of error**:

    > $$\text{margin} = 2 \times \sqrt{\frac{p \times (1 - p)}{n}}$$

    ## Exercise 4: margin

    Write `margin(p, n)`:

    ```
    return 2 * np.sqrt(p * (1 - p) / n)
    ```

    For `p = 0.6` and `n = 265` that's about 0.06, or 6 points. When comparing two scores, each has its own noise, and by adding variances the margin for the difference is about 1.4 times bigger: about 8.5 points here. So two models at 58% and 61% on this test can't be told apart.
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
    (`model.state_dict()` holds all the knobs by name, and `load_state_dict` puts them back. So `saved` lets us start several models from the same point.
    The number shown is how many named tensors of knobs there are.)

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
    Is that better than starting from random knobs? There's a middle way too: **freezing** the body, so that only the new layer learns. To freeze, set `requires_grad` to `False` on every knob of the body, so no blame reaches it.

    ## Exercise 5: freeze

    Write `freeze(model)`:

    1. `for p in model.parameters():`
    2. Inside it: `p.requires_grad = False`

    It needs no `return`: it changes the model it's given.
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
    Now read that table with Exercise 4 in hand. (Two cells showing the same number is a coincidence.) At 100 and 300 labels the gaps are a few points, far inside the margin of about 8.5, so the columns can't be told apart. At 1,000 labels, from scratch leads by 7 or 8 points, right at the edge of that margin: probably real, but not certain.

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
    Leaks are usually subtler than this: the same text in both train and test, a date that gives away the answer, or a number
    tuned on the test set again and again (chapter 13's `k`). Repeating a test until it looks good is itself a leak, and it's the same reason shared **benchmarks**, standard test sets used by everyone, slowly wear out.

    ## Looking both ways: BERT

    GPT reads left to right, because its job is to write. For understanding a text that already exists, why not look both ways? **BERT** is a transformer with no future mask, trained on a different job: hide 15% of the words, and predict them from both sides.

    We tried that here too, expecting it to beat the 51% from before, since it can see the `barked` that comes after a blank. On this data it scored about 32%, below the baseline. It takes more data than a small model on a small text can offer. So keep that as a lesson rather than a recipe: an idea that wins at huge scale isn't guaranteed to win at small scale.

    ## What's next

    That completes Part 2. You can represent words, score language models, and build a transformer. Part 3 rebuilds your tiny GPT the way real labs do, starting from its very first step.
    Our word list breaks on any word it hasn't seen, and on any other language. **Chapter 19** replaces it with a proper **tokenizer**.

    ## What you can do now

    - explain pretraining, fine-tuning and freezing, and when each makes sense
    - compare any score with a baseline
    - compute precision and recall, and say what each one protects against
    - estimate a margin of error, and decide whether two scores really differ
    - recognise leakage, and explain why a test set can wear out
    - explain how BERT differs from GPT

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** BERT is from Devlin et al. (2018). Precision and recall combine into the *F1 score*. Comparing two models on the same test is done properly with a *paired* test such as McNemar's. Benchmarks include *SST-2* and *SQuAD*, and using an existing model on a new task is called *transfer learning*.
    """)
    return


if __name__ == "__main__":
    app.run()
