# learn-llm — Scope (v0.1 scope approved 2026-10-24)

## One-sentence goal
Take someone who knows almost no math or programming to a working, from-scratch
understanding of modern LLMs — training *and* serving — along one unbroken path
that covers the essential knowledge of CS231n → CS224N → CS336 → tiny-llm.

## Who it's for
- Learners with high-school arithmetic and the ability to install one app. No calculus,
  no linear algebra, no Python assumed.
- A laptop (Linux, macOS or Windows via WSL/Codespaces). No GPU required; a GPU or
  free cloud GPU unlocks optional "scale it up" sections.

## Design principles (from your brief)
1. **Prerequisites just in time.** Each math/Python idea appears in the chapter that needs it,
   in a short "Toolbox" box, then is used immediately. No standalone math unit up front.
2. **Taught once.** Overlapping topics across the four courses (backprop, optimizers, attention,
   transformers, evaluation) get one canonical chapter; later chapters build on it, never re-teach it.
3. **Zero configuration, browser only.** One command (`make start`) sets up the environment and opens
   the course on a localhost web page. Learners read, code and check exercises there; they never need
   the terminal again.
4. **No jumping.** Everything needed to finish lives in the repo. External links are only
   optional "go deeper" footnotes at the end of a chapter.
5. **One growing project.** The code you write in chapter 3 is the code you're still using in
   chapter 28. You end with your own LLM stack.
6. **Original material.** Same knowledge as the source courses, our own words, code and
   exercises. No copied lecture notes, slides or assignment text.

## Chapter map (draft)

| # | Chapter | You build | Prereqs taught here | Source coverage |
|---|---|---|---|---|
| **Part 0 — Start** |
| 00 | Your first program | Run code, write functions, pass on-page checks | Notebook cells; values, variables, functions, lists, loops; the mean | — |
| **Part 1 — How machines learn** (CS231n core) |
| 01 | Fitting a line | Predict a number from data; the loss | zip, charts, error, squaring, mean squared error | 231n linear models |
| 02 | Going downhill | Gradient descent with nudged slopes; exact derivatives | Rates, slope, limits (as closing in), derivative rules | 231n optimization |
| 03 | Blame flows backwards | Backprop by hand and in plain code; exact gradients | Computation graphs, chain rule, blames add, gradient check | 231n backprop |
| 04 | Keeping the books | A small automatic backprop engine (scalar autograd) | Classes, objects, recording a calculation, reverse order | 231n backprop |
| 05 | Many numbers at once | A model with several inputs; its loss and gradient in NumPy | Vectors, dot product, matrices, shapes, `@`, broadcasting, NumPy | 231n numpy |
| 06 | Recognizing digits | Scores for ten digits on a bundled MNIST subset; the guess and its accuracy | Indexing, reshape, matrix @ matrix, argmax, axis | 231n softmax/SVM |
| 07 | A loss for choices | Softmax classifier trained by gradient descent (about 90%) | e^x and log with their slopes, softmax, cross-entropy, transpose | 231n softmax/SVM |
| 08 | Neural networks | One-hidden-layer network with matrix backprop (about 94%) | ReLU, hidden layers, blame through matrices, random start | 231n NN, A1/A2 |
| 09 | Training that works | Mini-batches, He start, momentum, Adam, weight decay | Variance, slicing, moving averages, overfitting | 231n training, A2 |
| 10 | Hello PyTorch | Port your engine, compare to torch | Tensors, modules, GPU concept | 231n PyTorch |
| 11 | Seeing with convolutions | Small CNN and residual blocks on shifted digits | Convolution, pooling, residuals | 231n CNNs (condensed) |
| **Part 2 — Language** (CS224N core) |
| 12 | Words as vectors | word2vec (skip-gram, neg. sampling) | Dot product as similarity, sampling | 224N word vectors |
| 13 | Predicting the next word | n-gram LM, perplexity, sampling | Conditional probability, log-likelihood | 224N LMs |
| 14 | Memory: RNNs | RNN/LSTM language model | Recurrence, vanishing gradients | 224N RNNs |
| 15 | Attention | Seq2seq with attention | Weighted averages, softmax again | 224N attention |
| 16 | Self-attention | A masked, multi-head self-attention layer | Query, key and value; masking; heads | 224N + 336 (taught once) |
| 17 | The Transformer | Tiny GPT trained on TinyStories | Positional encodings, layer norm, the block | 224N + 336 (taught once) |
| 18 | Pretrain, fine-tune, evaluate | Fine-tuning a GPT on a task made from the stories; BERT explained, not trained | Baselines, precision and recall, margin of error, leakage | 224N pretraining |
| **Part 3 — Building a real LLM** (CS336 core) |
| 19 | Tokenization | Byte-level BPE tokenizer | Bytes, Unicode, hashing/merging | 336 A1 |
| 20 | A modern transformer | RMSNorm, RoPE, SwiGLU | Angles, cos and sin, rotations | 336 A1 |
| 21 | Training it properly | AdamW, LR schedule, checkpoints; the leaderboard run | Schedules, saving and loading | 336 A1 |
| 22 | Making it fast | Profiling, FLOPs/memory math, mixed precision, tiled (Flash-style) attention | Big-O, memory hierarchy, floats | 336 A2 |
| 23 | Many devices | Data parallel on real CPU processes, gradient accumulation, sharding, tensor parallel | Processes, all-reduce | 336 A2 |
| 24 | Scaling laws | Fit laws to your own small runs | Log-log plots, power laws | 336 A3 |
| 25 | Data | Filtering, dedup, quality classifiers | Hashing (MinHash), classifiers | 336 A4 |
| 26 | Alignment | SFT, DPO, RL with verifiable rewards (GRPO-style) | Expectations, policy gradients | 336 A5 |
| **Part 4 — Serving** (tiny-llm, ported from MLX to PyTorch/NumPy) |
| 27 | Running a real model | Load Qwen3-0.6B weights into *your* code; GQA; decoding loop | Weight files, dtypes | tiny-llm wk 1 |
| 28 | Fast single request | KV cache, quantized matmul, fused ops | Integer quantization | tiny-llm wk 2 |
| 29 | A mini vLLM | Continuous batching, paged KV cache, chunked prefill | Queues, paging | tiny-llm wk 3 |
| 30 | Capstone (optional) | Speculative decoding, MoE, or a small agent loop | — | tiny-llm wk 3–4 |

Each chapter = one marimo notebook (a plain `.py` file) served on localhost: the lesson text,
editable exercise cells with stubs, and live check cells that grade them on the page. Reference
solutions and the CI test harness live outside the learner's notebook. Real local Python is required
because PyTorch does not run in browser-only Python (Pyodide).

**Agents track (optional, after chapter 19; chapter numbers start at 40, so the main course keeps 24–30)**

| # | Chapter | You build | Prereqs taught here | Source studied |
|---|---|---|---|---|
| 40 | The agent loop | Tools, errors as results, a loop with a turn limit | try/except, raise, f-strings | Pi `agent-loop.ts` |
| 41 | Inside Pi's loop | Two-level loop with steering and follow-ups | Reading TypeScript, `lambda`, `or`, `list.pop` | Pi `agent` package |
| 42 | One API for every model | Stream assembly, two provider converters, tool deltas | JSON, `isinstance` | Pi `ai` package |
| 43 | Tools that survive the real world | Truncation, unique edits, fuzzy match, result ordering | Text methods | Pi `coding-agent/core/tools` |
| 44 | A conversation is a tree | Append-only session, branching, JSONL | `is None`, `reverse` | Pi `session-manager.ts` |
| 45 | Running out of room | Token estimate, cut points, retry and overflow decisions | `math.ceil` | Pi `compaction.ts`, `agent-session.ts` |
| 46 | A harness you can make your own | Prompt sections, skill listing, handler chain | `continue` | Pi extensions, skills, trust |
| 47 | Super-agent: one Router, many Workers | Idempotent intake, direct delivery | | DeepSeek `dsh-superagent-prototype`: docs 000-001, `core/intake.ts` |
| 48 | Who may ask for work | Admission cell, gate, assessment outcome, Eval deadline | | `policy.ts`, `core/evals.ts` |
| 49 | Subscriptions, Tickets and follow-ups | Fan-out and resume rules, Ticket stages, pull-request ends | | `core/intake.ts`, `core/stage.ts`, `core/tickets.ts` |
| 50 | An engine that never loses work | Effect coalescing, capped backoff, worker slots | | `core/engine.ts`, `app/effects.ts`, `executor/dsh.ts` |
| 51 | Prompts, context and the exam | Prompt order, card preview; a graded exam | Regular expressions | `docs/004-prompts.md`, `app/cards.ts` |

## Non-goals
- Not a survey of every model or paper; one path, chosen depth.
- Not a replacement for the source courses' videos; no re-hosting of their materials.
- No mandatory GPU, paid API, or account sign-up to finish Parts 0–2.
- No framework zoo: NumPy → PyTorch only (Triton optional in ch. 20).
- Not production MLOps, a survey of agent frameworks, or RAG app tutorials. The agents track is two projects read in depth, not a framework tour.

## v0.1 acceptance check (machine-runnable)
On a fresh clone, on Ubuntu and macOS CI runners with no GPU:
1. `make start` installs everything and serves chapters 00–03 on localhost in under 10 minutes,
   with no manual steps (CI checks the server answers, then stops it).
2. Headless `make check-solutions` passes chapters 00–03 against the reference solutions.
3. Headless `make check CH=03` **fails** against the untouched exercise stubs (proves the checks check something).
4. A link checker finds no required external link in chapters 00–03 (footnotes allowed).
5. Chapters 00–03 exist in `en` and `zh-CN`; checks pass in both, and no `zh-CN` chapter is stale
   against its `en` source.

## Languages (i18n)
- Locales use BCP 47 tags: `en` (source language) and `zh-CN` first. More are added the same way.
- **Prose is translated per file; code is shared.** Each locale has its own copy of every lesson
  notebook (`lessons/<locale>/NN-slug.py`). Exercise checks, helpers, data and reference solutions
  live once in the shared package and are imported by every locale, so a fix to a check reaches all
  languages at once.
- **Code stays English** (identifiers, function names) so learners can search errors and read later
  chapters' code; comments and on-page explanations are translated.
- **UI strings** (check results, hints, navigation) come from one message catalog per locale
  (`learn_llm/locales/<locale>.json`), looked up by key. Missing keys fall back to `en`.
- **Fallback:** if a chapter has no translation yet, the course shows the `en` version with a notice.
- **Staleness:** each translated notebook records the `en` revision it was translated from; CI flags
  translations whose source has changed since.
- **Glossary:** `i18n/glossary.zh-CN.md` fixes one translation per technical term (梯度 = gradient,
  etc.) and keeps the English term in brackets on first use.
- The start page has a language picker. It defaults to the system language and the choice is
  remembered in `my-work/.settings.json`.

## Decisions
- License: MIT for code, CC-BY-4.0 for lesson text (user, 2026-10-24)
- Name: learn-llm (user, 2026-10-24; GitHub collision check: only small unrelated repos, PyPI name free)
- Languages: en + zh-CN first, design under "Languages" above (user chose languages; design agent-default)
- Format: lessons and exercises in the browser only, served from localhost (user, 2026-10-24)
- Web app: marimo notebooks (user, 2026-10-24)
- v0.1: chapters 00–03 plus the one-command environment (user, 2026-10-24); acceptance check green in CI on Ubuntu, macOS, Windows (PR #1)
- New chapter 04 "Keeping the books" (automatic engine) before "Many numbers at once"; later chapters shift by one (user, 2026-10-25)
- NumPy (`numpy>=2.5`, BSD-3-Clause, no dependencies of its own) is the second runtime dependency, from chapter 05; Python floor raised to 3.12, which NumPy 2.5 needs (user, 2026-10-25)

- Installer: uv (pinned version) behind `start.sh` / `make start`; Windows uses `start.cmd` → `start.ps1` (user, 2026-10-24)

- Route: R3, numbers → digits → text, with one text dataset and one perplexity leaderboard
  from ch 12 to the end (user, 2026-10-24; see docs/design/curriculum.md)
- Gradients/backprop: nudge experiment → rules earned by nudge algebra → chain rule as
  exchange rates → hand-filled tables → autograd code; matrix gradients one entry at a time;
  every gradient auto-checked against nudging; d/dx notation from ch 02 (user, 2026-10-24)
- Text dataset from ch 12: the same English text for every language edition (user, 2026-10-24)
- Every chapter passes the novice-review gate (docs/design/novice-review.md) with zero blockers

- Chapters 10–12 batch: PyTorch is the third runtime dependency, CPU builds on Linux and Windows through uv's CPU index (default builds bring several GB of GPU libraries); ch 11 uses the bundled digits pasted at random spots instead of CIFAR, which has no clear licence and is bigger and slower (it also shows directly why convolutions help); the course's text is a bundled 13,000-story TinyStories subset, CDLA-Sharing-1.0, with its own notice (user, 2026-10-31)

- Chapters 13–15 batch: the leaderboard scores perplexity per word on a 2,002-entry word vocabulary and lives in each learner's `my-work/.leaderboard.json`; chapter 15 teaches attention on reversing text instead of translation, since the course has no bundled translated text and every answer can be checked at a glance (agent-default, 2026-11-01)

- Chapters 16–19 batch: chapter 18's task is naming a blanked-out animal, made from the stories, because the course has no hand-labelled data; it reports that pretraining gave no clear gain on this small model, and that a small BERT-style model scored below the baseline, instead of the textbook result (agent-default, 2026-11-01)

- Agents track, chapters 40–51: added at the user's request (2026-11-01), which amends the earlier non-goal on agents; it studies Pi (40–46) and DeepSeek's Super-agent (47–51) to source level. "Deepseek Super-Agent" was first misread as ByteDance's DeerFlow; on 2026-11-03 the user pointed to the AgentsPulse review of DeepSeek's essay and its prototype `deepseek-ai/dsh-superagent-prototype`, and chapters 47–51 were rewritten on it (user). Numbers first started at 30 so the existing chapters, checks, ledger and navigation worked unchanged; the track moved to 40–51 in the 2026-11-09 renumbering (agent-default). A scripted model keeps every exercise offline and free (agent-default). Excerpts stay short, under their MIT licences, pinned by commit in `learn_llm/data/AGENT-SOURCES.md` and checked by `python -m learn_llm.source_check` (agent-default). The prototype is a research reference, not a product, and differs from DeepSeek's essay; lessons say which source each claim comes from (agent-default)

- Chapters 20–23 batch: token models read the same text as the word models, with rare words as `<unk>`, and are scored per word; spelling rare words out costs about 40% more perplexity, so mixing the two would rig the leaderboard. Merges for the stories are bundled (`learn_llm/data/tinystories-bpe.json`, rebuilt with `python -m learn_llm.tokenizer`). Chapter 23 trains on real CPU processes with PyTorch's `gloo` backend (`learn_llm/parallel.py`); chapters 20–22 use NumPy's `cos`, `sin` and `exp` so the agent track can keep introducing the `math` module (agent-default, 2026-11-09)

- Even depth, chapters 06–23 (user, 2026-11-09; plan in docs/design/depth-plan.md): a review found that chapters 06–20 (old numbers) taught about twice as many ideas per page as 00–05 and printed 78–100% of each exercise's answer in the lesson. Three chapters are split (old 06 → 06, 07; old 15 → 16, 17; old 18 → 20, 21), later chapters are renumbered, and the agents track moves from 30–41 to 40–51 because the main course now reaches 30. Until a split chapter is rewritten, its second number (07, 17, 21) has no lesson. Exercises give a spec and a worked example, not the solution's lines; the main idea of a chapter is done by hand first

## Open decisions
- (none)
