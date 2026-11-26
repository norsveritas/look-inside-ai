# Look Inside AI: how does a small AI handle "Kurdish"?

**By Nashat**

This is my first experiment looking **inside** an AI language model. I wanted to find out where it keeps the idea of "Kurdish": the Kurdish people, Kurdistan, cities like Zakho and Duhok, and the Kurdish language (Kurmanji). Then I tested what happens if we turn that idea up or down.

The model is **GPT-2 small**, a small, older AI from 2019 that runs on a normal computer without a graphics card. This field of research is called **interpretability**, which means trying to understand what is happening inside an AI instead of only looking at its answers.

![Steering chart](charts/5_steering_before_after.png)

## What I found (short version)

- **The idea of "Kurdish" can be found inside the model.** It shows up as a clear pattern (a "direction") in every layer of the model, and it is strongest in the early-middle layers. Many of the neurons that react to Kurdish sit in the very first layer and switch on when they see words like *Kurdistan*, *Kurds* or *Peshmerga*.
- **Turning it up changes what the AI writes.** When we pushed the "Kurdish" pattern in, a sentence like *"Today I want to eat…"* turned into *"…some vegetables in the city of Dohuk"*. At the strongest settings the model started inventing fake words like **"Peshistan"**.
- **Turning it down makes the AI "forget" Kurdistan.** *"Erbil is the capital of the…"* became *"…Baltic"*, and *"Masoud Barzani is a famous…"* became *"…French"*. With the pattern removed, the model's continuations of Kurdish prompts mentioned Kurdish things **0 times out of 24**, compared with **13 out of 24** normally.
- **The AI connects "Kurdish" with war news.** Pushing the Kurdish pattern often brought up words like ISIS, militants and fighters. This reflects the news articles the model learned from. It is not a fact about Kurdish people, and finding this kind of bias is one of the reasons this research matters.
- **GPT-2 barely knows the Kurmanji language.** It mostly notices the special letters (ê, î, û, ş), but it does not understand the words.

## Experiment 2: the "war bias"

In experiment 1, war words kept appearing whenever "Kurdish" was pushed into the model. In experiment 2 I measured this properly by comparing Kurdish with 9 other peoples (Palestinian, Chechen, Armenian, Tibetan, Basque, Scottish, Welsh, Japanese, Dutch).

- When GPT-2 completes sentences like *"The Kurdish people are known for their…"*, **80%** of its war-or-culture guesses are war words. That is the **highest of all 10 groups**; Scottish, Welsh, Japanese and Dutch are around 30%.
- Inside the model, even the word "Kurdish" in *"This is a Kurdish song."* leans towards "war". This is strongest in the first layers.
- Holding the model's internal "war direction" at a peaceful level cut war words in texts about Kurdish people from **16 out of 48 to 0 out of 48**, while culture words went up. A random-direction control changed nothing. However, this fix silences war talk for every group, so it is a blunt tool rather than real fairness.

![War bias chart](charts/war_bias/1_next_word_war_share.png)

👉 **Full story: [REPORT_WAR_BIAS.md](REPORT_WAR_BIAS.md).**

👉 **For the full story of experiment 1 in simple English, read [REPORT.md](REPORT.md).** It covers what was done, the charts, the funniest examples, the honest limits and ideas for the next experiment.

## What's in this project

- `REPORT.md`: experiment 1 report, written for non-programmers
- `REPORT_WAR_BIAS.md`: experiment 2 report (the war bias)
- `charts/`: six pictures of the results
- `results/`: all the numbers, plus every text the AI wrote during the tests
- `experiment/`: the Python code (sentences, analysis, steering, removal, charts)

## How to run it yourself

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cd experiment
python step1_find_kurdish.py     # find the Kurdish layers, neurons and direction (~1 minute)
python step2_steering.py         # steering test (~10-20 minutes on a normal computer)
python step2b_removal.py         # removal test (~5 minutes)
python step3_make_charts.py      # make the charts
python war_bias.py               # experiment 2: war bias (~15 minutes)
python war_bias_charts.py        # experiment 2 charts
```

You need an internet connection the first time, so the GPT-2 model can be downloaded from Hugging Face.

## A note on how this was made

The idea, the questions and the direction of this experiment are mine (Nashat). The code, the charts and the first draft of the report were written by **Claude Code** (an AI coding assistant from Anthropic), working under my direction. The Kurmanji example sentences were also written by the AI, so a native speaker may find small mistakes in them.
