# learn-llm

**English** · [简体中文](README.zh-CN.md)

Go from almost no maths or programming to building your own large language model, then
serving it the way production systems do. Everything happens on one page in your browser.

> **Status: early.** Chapters 00–23 are ready: from your first line of Python to a modern
> transformer you build yourself, trained across several processes. Chapter 24 (Scaling laws)
> is next. Chapters 06–23 are being reworked for a more even pace; number 21 is
> reserved for a chapter being split out of 20.
> See the [full chapter map](SCOPE.md).

**New: an optional agents track.** Chapters 40–51 take you through two real open-source agents,
[Pi](https://github.com/badlogic/pi-mono) and DeepSeek's [Super-agent](https://github.com/deepseek-ai/dsh-superagent-prototype),
down to their source code. You read the real code on the page and rebuild each mechanism yourself.
It needs Python up to chapter 19, and runs offline with no API key.

## Start

You need a computer running macOS, Linux or Windows, and an internet connection the first
time. No GPU, no accounts, no Python install needed.

1. Download this repository ([Code → Download ZIP](../../archive/refs/heads/main.zip)) and unzip it,
   or `git clone` it.
2. Start the course:
   - **Windows:** double-click `start.cmd`.
   - **macOS / Linux:** open a terminal in the folder and run `./start.sh`.
3. Your browser opens the course. Pick your language and begin with chapter 00.

The first start installs [uv](https://docs.astral.sh/uv/), which downloads the right Python
and everything the course needs into this folder (about 1 GB, mostly PyTorch). It takes a few
minutes, once. Later starts take seconds.

Chapters 10 onward use PyTorch, which runs on Linux, Windows and Macs with Apple chips on macOS
14 or later. On an Intel Mac the course still starts, and chapters 00–09 work.

Your work is saved in `my-work/`, your own copy of every lesson. Updating the course never
overwrites it. To restart a chapter from scratch, delete its file in `my-work/` and start
again.

Every chapter ends with links to the previous and next chapters, and a small bar in the
bottom-left corner (`← 04 · All chapters · 06 →`) moves between chapters from anywhere on the
page. The start script adds this navigation to your copies, including ones you already
started, without touching your answers; it keeps the original in `my-work/.navigation-backups/`
first, and never adds it twice. (Editing the source lessons directly with
`marimo edit lessons/...` shows no navigation.) If the editor asks before leaving a chapter,
your latest edit hasn't been saved yet: wait a moment for autosave, or save, then switch.

## How it's different

This course covers the knowledge of Stanford CS231n, CS224N and CS336 and the
[tiny-llm](https://github.com/skyzh/tiny-llm) serving course, rebuilt as one continuous path:

- **Prerequisites when you need them.** No maths unit up front. Each idea is taught on the
  page that uses it.
- **Each topic taught once.** Backprop, attention and transformers each get one chapter,
  and later chapters build on it.
- **Nothing to set up.** One start command; the checks for every exercise run on the page.
- **No jumping around.** Everything you need is in the course. Links are optional extras.
- **One growing project.** The code you write early is still running in the last chapter.

It is original material, not copied lecture notes or assignments, and it is not affiliated
with Stanford or the tiny-llm authors.

## Support

This is a personal, best-effort project with no guaranteed response times. Bug reports and
questions are welcome as GitHub issues.

## License

Code: [MIT](LICENSE). Lesson text: [CC BY 4.0](LICENSE-CONTENT). The bundled MNIST digits
(from chapter 06): [CC BY-SA 3.0](learn_llm/data/MNIST-NOTICE.md), by Yann LeCun and Corinna Cortes.
Source excerpts in chapters 40–51: [MIT, from Pi and the DeepSeek Super-agent prototype](learn_llm/data/AGENT-SOURCES.md).
