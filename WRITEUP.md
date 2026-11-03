# <repo name>

<!-- Write-up format for every experiment. Replace README.md with this when the
run is done. Fill in Question and Predictions BEFORE launching. Keep it to a
page: someone skimming should get the answer from the first two lines, and
someone digging in should be able to reproduce it from the rest. Drop optional
sections that have nothing to say. -->

**Question:** <one sentence, plain words, no jargon>

**Answer (at this scale):** <yes / no / unclear, in 2–3 sentences: the verdict
first, then the main number, then the biggest caveat>

## Setup

- **Model and data:** <size, dataset, tokens or steps, metric>
- **Arms:** <each arm in one line, controls included>
- <seeds, the hardware, wall-clock time and cost; the rule for what counts as a
  real difference (e.g. > 2 seed sd)>

## Results (mean ± sd over N seeds)

| arm | <main metric> | vs baseline | <one or two more columns> |
|---|---|---|---|

<units, and what chance level is>

## Findings

1. **<Bold one-line claim.>** <The evidence in 1–3 sentences, with numbers.>

## Predictions vs outcome   <!-- optional: only if written before the run -->

| outcome | predicted odds | happened? |
|---|---|---|

## Next, if revisited

- <the most informative next run first>

## Prior art   <!-- optional -->

## Reproduce

```bash
<install, data prep, one training run, eval: copy-pasteable>
```

<Where per-run logs, metrics and checkpoints live (e.g. the `results` branch).>
