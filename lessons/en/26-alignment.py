import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="26 · Alignment")

with app.setup(hide_code=True):
    import marimo as mo
    import torch
    import torch.nn.functional as F

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.align import all_prompts, answer_scores, best_answers, judged_pairs, pretrained, right_answers, sample_answers, score_samples, share_right, to_text
    from learn_llm.checks import check
    from learn_llm.diagrams import pipeline

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 26 · Alignment

    A language model is trained to do one thing: continue text the way its training text continues. That's not the same as doing what you want.
    Ask a model trained only on web pages a question, and it may answer, or ask three more questions, or write an advertisement: all of those follow questions somewhere on the web.

    Closing the gap between "continues text" and "does what we want" is called **alignment**. It's a second, much shorter phase of training, done on a pretrained model (chapter 18).
    There are three ways to tell a model what we want, and they differ in what they ask of *us*:

    1. **Show it.** Give the right answer for some prompts. This needs someone who knows the answers.
    2. **Compare.** Take two of its answers and say which is better. This needs only a judge.
    3. **Score it.** Let it try, and reward what works. This needs only a way to check.

    You'll write the loss for each, and train the same small model in all three ways.

    ## A tiny world

    Aligning a story model would take hours and a human judge. So this chapter uses a world small enough to train in seconds, where "what we want" is beyond argument: sums.
    A line of text has 6 characters, like `3+4=07`. The first four are the **prompt** and the last two the **answer**, always written with two digits.

    Our pretrained model has read thousands of such lines from a careless source, where the answer was some number from 00 to 18 with no relation to the sum. So it knows exactly what an answer looks like, and has no idea which one is right.
    That's a pretrained model in miniature: fluent and unhelpful. It's chapter 24's small transformer, with 12 tokens for the ten digits, `+` and `=`. Loading it trains it, which takes a few seconds.
    """)
    return


@app.cell
def _():
    base = pretrained()
    prompts = all_prompts()
    right = right_answers(prompts)
    to_text(prompts[34:38]), to_text(best_answers(base, prompts[34:38])), share_right(base)
    return base, prompts, right


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `prompts` holds all 100 prompts as token ids, from `0+0=` to `9+9=`, and `to_text` shows some of them as text. `best_answers` gives the model's most likely answer to each: the same answer whatever the prompt.
    `share_right` is the share of the 100 prompts it answers correctly: 0.01, the one prompt whose answer happens to be its favourite. That's the number this chapter will raise.

    ## The chance of a whole answer

    All three methods need one quantity: how likely the model finds a given answer to a prompt. An answer is two tokens, and the model produces chances for one token at a time (chapter 13).
    The chance of the whole answer is the chance of its first digit, times the chance of its second digit given the first:

    > $$p(\text{answer}) = p(\text{first digit}) \times p(\text{second digit, after the first})$$

    Chances multiply, so their logs add (chapter 13). The log of a chance is called a **log-probability**:

    > $$\log p(\text{answer}) = \log p(\text{first digit}) + \log p(\text{second digit, after the first})$$

    Keep one sign rule in mind for the whole chapter: probabilities lie between 0 and 1, so **log-probabilities are always negative**, and the closer to 0 the more likely. Meanwhile, training tries to make the loss as *small* as possible, so turning a log-probability into a loss always puts a minus sign in front (just like chapter 07's $\text{surprise} = -\log p$).

    ## By hand: one answer

    For the prompt `3+4=`, a model gives the digit 0 a chance of 0.5 as the first digit. After that 0, it gives 7 a chance of 0.4. What's the chance of the answer `07`?
    And what's its log-probability, given that $\log 0.5 = -0.69$ and $\log 0.4 = -0.92$?
    """)
    return


@app.cell(hide_code=True)
def _():
    chance_boxes = hand.boxes(['chance of 07', 'log-probability of 07'], step=0.01)
    return (chance_boxes,)


@app.cell(hide_code=True)
def _(chance_boxes):
    hand.marked(chance_boxes, {'chance of 07': 0.2, 'log-probability of 07': -1.61}, {
        'chance of 07': r"""$0.5 \times 0.4$""",
        'log-probability of 07': r"""add the two logs""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And $\log 0.2$ is indeed $-1.61$.

    The function `answer_scores(model, prompts, answers)` does the plumbing: it feeds the model each prompt followed by its answer and gives back the scores it produced for the two answer positions, with the shape (answers, 2, 12).
    For answer `i`, `scores[i, 0]` are the 12 scores for the first digit, and `scores[i, 1]` the 12 scores for the second digit, computed after reading the first.

    ## Exercise 1: answer_logprob

    Write `answer_logprob(scores, answers)`. `scores` has the shape (answers, 2, 12), as just described. `answers` has the shape (answers, 2): the two digits of each answer, as whole numbers.
    It gives back a tensor with one number per answer: its log-probability, from the formula above.

    You have a tool that does most of it: `F.cross_entropy(S, labels, reduction="none")` (chapter 14) gives, for each row of scores `S`, the surprise of its label, and a surprise is a log-probability with the sign turned.

    When every score is 0, each of the 12 tokens has a chance of $\frac{1}{12}$, so every answer must get $2 \times \log \frac{1}{12}$, about $-4.97$.
    """)
    return


@app.function
def answer_logprob(scores, answers):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "answer_logprob", answer_logprob, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Treat the two digits separately. `scores[:, 0]` is a table with a row of 12 scores for each answer, and `answers[:, 0]` holds each answer's first digit: exactly what `F.cross_entropy` takes. The same with 1 for the second digit.""",
        r"""It's 3 lines: the first digits' surprises, the second digits' surprises, then give back minus their sum. Without `reduction="none"` you'd get one number for all the answers together.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Way 1: show it

    The most direct way is the one you know. Collect prompts with their right answers, called **demonstrations**, and train the model to find those answers likely: the loss is the mean of $-\log p(\text{right answer})$.
    It's plain cross-entropy, with one difference from pretraining: only the answer's tokens count. The model isn't asked to predict the prompt, since the prompt is what it will be given.
    This is **supervised fine-tuning**, SFT for short.

    Demonstrations are expensive: somebody who knows the answers has to write each one. We'll pretend we could afford 30 of the 100. The function below is chapter 10's loop with that loss.
    """)
    return


@app.function
def train_sft(model, prompts, answers, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for step in range(steps):
        optimizer.zero_grad()
        logprobs = answer_logprob(answer_scores(model, prompts, answers), answers)
        (-torch.mean(logprobs)).backward()
        optimizer.step()


@app.cell
def _(prompts, right):
    torch.manual_seed(1)
    shown = torch.randperm(100)[:30]
    sft_model = pretrained()
    train_sft(sft_model, prompts[shown], right[shown], 200)
    shown_right = torch.mean(torch.all(best_answers(sft_model, prompts[shown]) == right[shown], dim=1).float()).item()
    round(shown_right, 2), share_right(sft_model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    (`pretrained()` builds the same starting model again each time, so every method starts from the same place. `torch.all(..., dim=1)` is `True` for a row when both digits match.)

    The first number is the share right on the 30 prompts it was shown: all of them. The second is the share over all 100: about 0.3. It learned the 30 sums by heart and little else.
    That's the limit of showing. The model does what the demonstrations cover, and demonstrations never cover everything.

    ## Way 2: compare

    Often nobody can write the perfect answer, but anyone can say which of two answers is better. Think of two summaries of an article, or two replies to a customer.
    A **preference pair** is a prompt with two answers and a verdict: this one is better than that one.

    How do we train on a verdict? We want the model to raise the chance of the better answer and lower the chance of the worse one. But "raise" from where?
    From where the model started. Keep a frozen copy of the model as it was before this training, called the **reference model**, and measure how far each answer has moved:

    > $$\text{gain}(\text{answer}) = \log p_{\text{model}}(\text{answer}) - \log p_{\text{reference}}(\text{answer})$$

    A gain of 0 means "as likely as before". The **margin** of a pair is how much more the better answer gained than the worse one:

    > $$\text{margin} = \text{gain}(\text{better}) - \text{gain}(\text{worse})$$

    We want every margin to be large and positive. Chapter 12 had the exact loss for making a number large: $-\log \sigma(\ldots)$, which is big when the number is negative and fades to 0 as it grows. Three anchor values give you a feel for it (with $\beta = 1$): at $\text{margin} = -2$ the loss is about `2.13` (pushing hard), at $\text{margin} = 0$ it is about `0.69` (even), and at $\text{margin} = +2$ it drops to about `0.13` (nearly done). So:

    > $$\text{loss} = \text{mean of } -\log \sigma(\beta \times \text{margin})$$

    $\beta$ (beta) sets how hard to push: with a small $\beta$ the loss keeps pulling even at large margins. This method is called **direct preference optimization**, DPO.
    Measuring from the reference does a second job. The loss only cares about the two answers in each pair, and comparing with the reference keeps the model from wandering far from what it was.

    ## By hand: one pair

    The reference gives both answers of a pair a log-probability of $-2$. After some training, the model gives the better one $-1$ and the worse one $-3$. What's each answer's gain, and what's the margin?
    """)
    return


@app.cell(hide_code=True)
def _():
    pair_boxes = hand.boxes(['gain of the better', 'gain of the worse', 'margin'])
    return (pair_boxes,)


@app.cell(hide_code=True)
def _(pair_boxes):
    hand.marked(pair_boxes, {'gain of the better': 1, 'gain of the worse': -1, 'margin': 2}, {
        'gain of the better': r"""$-1 - (-2)$""",
        'gain of the worse': r"""$-3 - (-2)$""",
        'margin': r"""the first gain minus the second: mind the two minus signs""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With $\beta = 1$ the loss of this pair is $-\log \sigma(2)$, about 0.127. Before any training the model *is* the reference, every gain is 0, and the loss is $-\log \sigma(0) = \log 2$, about 0.693.

    ## Exercise 2: dpo_loss

    Write `dpo_loss(better, worse, ref_better, ref_worse, beta)`. The first four inputs are tensors with one log-probability per pair: the model's for the better and the worse answer, then the reference's for the same two. `beta` is a number.
    It gives back the loss above, one number. `F.logsigmoid` is chapter 12's $\log \sigma$.

    On your hand exercise, with `beta = 1`, it must give about 0.127.
    """)
    return


@app.function
def dpo_loss(better, worse, ref_better, ref_worse, beta):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "dpo_loss", dpo_loss, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Follow the three formulas in order: the two gains, the margin, the loss. Tensors subtract position by position, so all the pairs are handled at once.""",
        r"""It's 2 lines. Keep the margin in a name, then give back `-torch.mean(F.logsigmoid(...))` of `beta` times the margin.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now the pairs. `judged_pairs(base, prompts, 16)` lets the pretrained model answer every prompt 32 times, at random with its own chances (chapter 13), and forms 16 pairs per prompt.
    A judge then compares each pair. Our judge prefers the answer whose number is nearer the true sum, and drops the pair if both are equally far.
    Notice what the judge never does: it never says what the right answer is. Most pairs are two wrong answers, one of them less wrong.

    The training function computes the reference's log-probabilities once, with no books kept, since the reference never changes. Then it's the usual loop with your loss.
    """)
    return


@app.function
def train_dpo(model, reference, pair_prompts, better, worse, steps):
    with torch.no_grad():
        ref_better = answer_logprob(answer_scores(reference, pair_prompts, better), better)
        ref_worse = answer_logprob(answer_scores(reference, pair_prompts, worse), worse)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for step in range(steps):
        optimizer.zero_grad()
        now_better = answer_logprob(answer_scores(model, pair_prompts, better), better)
        now_worse = answer_logprob(answer_scores(model, pair_prompts, worse), worse)
        dpo_loss(now_better, now_worse, ref_better, ref_worse, 1.0).backward()
        optimizer.step()


@app.cell(hide_code=True)
def _():
    dpo_button = mo.ui.run_button(label="Train on comparisons (about 30 seconds)")
    dpo_button
    return (dpo_button,)


@app.cell
def _(base, dpo_button, prompts):
    mo.stop(not dpo_button.value, mo.md("*Press the button to train.*"))
    torch.manual_seed(0)
    pair_prompts, better, worse = judged_pairs(base, prompts, 16)
    dpo_model = pretrained()
    train_dpo(dpo_model, base, pair_prompts, better, worse, 300)
    len(pair_prompts), share_right(dpo_model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 1,500 pairs, and a share right of about 0.4, from 0.01. Nobody told the model a single correct answer. It learned from "this one is less wrong", across all 100 prompts.
    It doesn't get further because its pairs came from the pretrained model's answers, and for many prompts the right answer was never among them. A comparison can only teach about answers somebody thought to compare.

    ## Way 3: score it

    The third way needs no examples at all. For some tasks a program can **check** an answer: does the code pass its tests, is the proof valid, does the sum add up?
    Then the model can practise. It answers, the checker gives a **reward**, 1 for right and 0 for wrong, and training makes rewarded answers more likely. A reward that a program can compute is called a **verifiable reward**.
    Learning from rewards is called **reinforcement learning**.

    /// admonition | 🧰 Toolbox: text into numbers
    - A string can be indexed like a list: `"7+2="[0]` is `"7"`, one character of text.
    - `int(text)` turns text made of digits into a whole number: `int("7")` is 7, and `int("012")` is 12.
    ///

    ## Exercise 3: reward

    Write the checker, `reward(prompt, answer)`. `prompt` is a string like `"3+4="` with single digits at positions 0 and 2, and `answer` is a string of two digits like `"07"`.
    It gives back `1.0` if the answer is the sum of the two digits and `0.0` if not.

    `reward("3+4=", "07")` must give `1.0`, and `reward("3+4=", "08")` must give `0.0`.
    """)
    return


@app.function
def reward(prompt, answer):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "reward", reward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three pieces of text have to become numbers before you can add and compare: the two digits of the prompt, and the answer as a whole.""",
        r"""It's 1 to 4 lines: an `if` with two `return`s, or chapter 18's `1.0 if ... else 0.0`. The test is `int(prompt[0]) + int(prompt[2]) == int(answer)`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The slope of a reward

    Here's the difficulty. The checker is a program, not a formula with slopes: there's no going backward through `reward`. And the model's answer is *drawn at random* from its chances, which has no slope either. What can gradient descent hold on to?

    The chances. Suppose that for one prompt the model can give answers $1, 2, 3, \ldots$ with chances $p_1, p_2, p_3, \ldots$, and they'd earn rewards $r_1, r_2, r_3, \ldots$ The reward it gets **on average** is each reward weighted by its chance, called the **expected reward**:

    > $$\text{expected reward} = p_1 \times r_1 + p_2 \times r_2 + p_3 \times r_3 + \ldots$$

    ## By hand: an expected reward

    For `3+4=`, a model answers `07` with a chance of 0.5, `08` with 0.3 and `17` with 0.2. Only `07` earns a reward of 1. What's the expected reward? And what would it be if training moved the chances to 0.8, 0.1 and 0.1?
    """)
    return


@app.cell(hide_code=True)
def _():
    expect_boxes = hand.boxes(['expected reward', 'after training'], step=0.1)
    return (expect_boxes,)


@app.cell(hide_code=True)
def _(expect_boxes):
    hand.marked(expect_boxes, {'expected reward': 0.5, 'after training': 0.8}, {
        'expected reward': r"""$0.5 \times 1 + 0.3 \times 0 + 0.2 \times 0$""",
        'after training': r"""the same sum with the new chances""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The expected reward is what we want to raise, and it *does* have slopes: the rewards are fixed numbers, and the chances come from the model's knobs. Its slope for any knob is

    > $$r_1 \times \text{slope of } p_1 + r_2 \times \text{slope of } p_2 + \ldots$$

    We can't compute that sum directly: it runs over every possible answer, and for a real model that means every possible text. One step of algebra fixes it. Chapter 07 found that the slope of $\log p$ is the slope of $p$ divided by $p$. Turn that around:
    the slope of $p$ is $p$ times the slope of $\log p$. (Quick numerical check: when $p = 0.5$, raising $\log p$ by `0.02` raises $p$ by $0.5 \times 0.02 = 0.01$.) Put it in:

    > $$p_1 \times \big(r_1 \times \text{slope of } \log p_1\big) + p_2 \times \big(r_2 \times \text{slope of } \log p_2\big) + \ldots$$

    Compare this with the expected reward above. It has the same form: probabilities times something. So it's also an average over the model's own probabilities—which we can estimate without visiting every answer. Why? Draw 100 answers from the model's current probabilities and an answer with probability `0.5` naturally shows up about 50 times, so simply averaging over the drawn samples automatically weights each one by $p_i$!

    **Draw a batch of answers from the model, check their rewards, and average over those.** In PyTorch terms: draw answers, compute their rewards, and run backpropagation from:

    > $$\text{loss} = \text{mean of } -r \times \log p(\text{drawn answer})$$

    Here the reward $r$ (or advantage, in a moment) is treated as a plain number with no gradient of its own, so gradients flow *only* through $\log p$. The minus sign is there because gradient descent walks downhill while we want reward to go up. This result is the **policy gradient**. In plain words:
    make the answers you tried more likely, each in proportion to its reward.

    ## Better or worse than what?

    Used as it stands, that loss learns slowly. With rewards of 0 and 1, a wrong answer has $r = 0$ and teaches nothing, and a right answer says "more of this" whether the model gets this prompt right once in 16 tries or 15 times in 16.

    The fix is to judge each answer against the others for the same prompt. Draw a **group** of answers per prompt, and replace each reward by its **advantage**: how much better it did than its group, measured in units of the group's spread.

    > $$\text{advantage} = \frac{r - \text{mean of the group}}{\text{spread of the group} + 0.000001}$$

    The spread is chapter 09's typical size: the square root of the mean of the squared distances from the mean. The tiny number avoids dividing by zero when a whole group earned the same.
    Now a wrong answer has a negative advantage and is pushed *down*, and a rare success among failures gets a large advantage. Using groups this way is the heart of a method called **GRPO**.

    ## By hand: a group of four

    Four answers to one prompt earned the rewards 1, 0, 0 and 1. What's the group's mean? Its spread? And the advantage of a right answer and of a wrong one? (Leave the tiny number out.)
    """)
    return


@app.cell(hide_code=True)
def _():
    group_boxes = hand.boxes(['mean', 'spread', 'advantage of a right answer', 'advantage of a wrong answer'], step=0.1)
    return (group_boxes,)


@app.cell(hide_code=True)
def _(group_boxes):
    hand.marked(group_boxes, {'mean': 0.5, 'spread': 0.5, 'advantage of a right answer': 1, 'advantage of a wrong answer': -1}, {
        'mean': r"""$(1 + 0 + 0 + 1) / 4$""",
        'spread': r"""every distance from the mean is $0.5$ or $-0.5$, so every square is $0.25$: the root of their mean""",
        'advantage of a right answer': r"""$(1 - 0.5) / 0.5$""",
        'advantage of a wrong answer': r"""$(0 - 0.5) / 0.5$""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And if all four had earned 1? Every distance from the mean would be 0, and so would every advantage. A group that's all alike, all right or all wrong, teaches nothing. Keep that in mind for later.

    ## Exercise 4: advantages

    Write `advantages(rewards)`. `rewards` is a tensor with the shape (prompts, group size): each row holds the rewards of one prompt's group. It gives back a tensor of the same shape, with every reward replaced by its advantage within its row.

    You'll need means along a row that keep their column shape, so that they broadcast: `torch.mean(x, dim=1, keepdim=True)` (chapter 17 used `keepdim` this way).

    `[[1, 0, 0, 1]]` must give `[[1, -1, -1, 1]]`, as by hand, to within the tiny number.
    """)
    return


@app.function
def advantages(rewards):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "advantages", advantages, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two steps. First the centred rewards: each row minus its own mean. Then the spread of each row, from the centred rewards: square them, take the mean along the row, take `torch.sqrt`.""",
        r"""It's 3 lines: `centred`, `spread`, and the division, with `1e-6` added to the spread. Both means need `dim=1, keepdim=True`. Without `keepdim`, a table with as many rows as columns would still subtract, but along the wrong direction.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 5: policy_loss

    Write `policy_loss(logprobs, advantage)`. Both are tensors with one entry per drawn answer: its log-probability under the model, and its advantage. It gives back the loss of the section "The slope of a reward", with the advantage in the place of $r$.

    For log-probabilities `[-1, -2]` and advantages `[1, -1]` it must give $-0.5$.
    """)
    return


@app.function
def policy_loss(logprobs, advantage):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("26", "policy_loss", policy_loss, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Multiply the two tensors position by position, take the mean, and turn the sign.""",
        r"""It's one line, starting with `return -torch.mean(`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The training function below puts the pieces together. Each step, `sample_answers` draws a group of 16 answers per prompt from the model as it is now. It gives back the prompts repeated 16 times, and the answers beside them, 1,600 rows each.
    `score_samples` runs **your** `reward` on every row, as text. (An answer that isn't two digits, like `=3`, gets 0 without your checker being asked.) Your `advantages` then compares within each prompt's group, which is why the rewards are reshaped to `(100, 16)` and back—here `-1` in `reshape(-1, 16)` and `.reshape(-1)` tells PyTorch to infer that dimension's length automatically (`1600 / 16 = 100`, then back to `1600`).
    """)
    return


@app.function
def train_rewards(model, prompts, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for step in range(steps):
        many, answers = sample_answers(model, prompts, 16)
        rewards = score_samples(reward, many, answers)
        advantage = advantages(rewards.reshape(-1, 16)).reshape(-1)
        optimizer.zero_grad()
        logprobs = answer_logprob(answer_scores(model, many, answers), answers)
        policy_loss(logprobs, advantage).backward()
        optimizer.step()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: read the loop

    Look at `train_rewards`, called with the 100 prompts and 400 steps. How many answers does the model write in one step? How many times is your `reward` called in the whole run? And how many right answers does the loop get from us?
    """)
    return


@app.cell(hide_code=True)
def _():
    loop_boxes = hand.boxes(['answers in one step', 'calls of reward', 'right answers given'])
    return (loop_boxes,)


@app.cell(hide_code=True)
def _(loop_boxes):
    hand.marked(loop_boxes, {'answers in one step': 1600, 'calls of reward': 640000, 'right answers given': 0}, {
        'answers in one step': r"""16 for each of the 100 prompts""",
        'calls of reward': r"""once per answer, in each of the 400 steps""",
        'right answers given': r"""look at the function's inputs: is `right` among them?""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    More than half a million checked attempts, and not one answer from us. That's the trade of this method: it needs no teacher, and a great deal of practice.
    """)
    return


@app.cell(hide_code=True)
def _():
    reward_button = mo.ui.run_button(label="Train on rewards (about 40 seconds)")
    reward_button
    return (reward_button,)


@app.cell
def _(prompts, reward_button):
    mo.stop(not reward_button.value, mo.md("*Press the button to train.*"))
    reward_model = pretrained()
    torch.manual_seed(0)
    train_rewards(reward_model, prompts, 400)
    to_text(prompts[34:38]), to_text(best_answers(reward_model, prompts[34:38])), share_right(reward_model)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 0.9, from 0.01: the model has taught itself to add, from a checker and its own attempts. (Random draws decide a lot here. With other seeds we measured between 0.8 and 0.9.)

    Why not 1? Remember the group that's all alike. For a few prompts the model settled early on one wrong answer and now gives it 16 times out of 16. Every advantage in that group is 0, so nothing pushes it away, and it will never try the right answer.
    A model can only be rewarded for what it tries. Keeping it trying new things, called **exploration**, is the standing difficulty of learning from rewards.

    Here are the three ways side by side, with our measurements:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    | way | what we must supply | share right |
    |---|---|---:|
    | none (pretrained) | | 0.01 |
    | show: SFT | 30 right answers | 0.31 |
    | compare: DPO | a judge for about 1,500 pairs | 0.40 |
    | score: rewards | a checker, and 640,000 attempts | 0.92 |

    Don't read this as a ranking. In this world a perfect checker exists, so rewards win easily. For "write a kind reply" there's no checker, and comparisons are the best signal there is.
    Real models are aligned with all three in turn: demonstrations first, to get the form right, then comparisons and rewards on top.
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("pretrained", "continues text"), ("shown", "answers"), ("compared", "answers well"), ("rewarded", "solves tasks")], ["SFT", "DPO", "rewards"], "The usual order of the three ways to align a model")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you'd better know

    Each way teaches the model exactly what its signal says, which isn't always what you meant.

    - Demonstrations teach the answers shown, including their mistakes and their style.
    - A judge's preferences are taught faithfully too. If the judge likes long, confident answers, the model becomes long and confident, right or wrong.
    - A checker is obeyed to the letter. If there's a way to earn the reward without doing the task, a model that tries enough things will find it. Your `reward` gives `1.0` for `07`. Would it for `7`, or for `+7`? A model would find out.

    Checking that an aligned model does what was meant, and not only what was scored, is most of the real work.

    ## What's next

    That completes training: a tokenizer, a modern transformer, a proper training run, a budget, clean data, and alignment. Part 4 is about the other half of a model's life, **serving** it.
    **Chapter 27** loads the weights of a real, published model into code you've written.

    ## What you can do now

    - say what alignment is and why a pretrained model needs it
    - compute the log-probability of a whole answer from its tokens' scores
    - fine-tune on demonstrations, and say what that can't reach
    - explain the reference model and the margin, and write the DPO loss
    - write a checker, and explain why a reward has no slope but an expected reward does
    - turn rewards into advantages within a group, and train a model on its own attempts
    - name what each of the three signals asks of you, and how each can mislead

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** DPO is from Rafailov et al., *Direct Preference Optimization* (2023). The policy gradient goes back to Williams's *REINFORCE* (1992). GRPO, *group relative policy optimization*, was introduced by Shao et al. in *DeepSeekMath* (2024);
    the full method also limits how far each step may move the model, which we left out. Training on human comparisons through a separate reward model is known as *RLHF*.
    """)
    return


if __name__ == "__main__":
    app.run()
