# GPU experiment repo (from gpu-experiment-template)

- The box must finish and destroy itself without any Claude session. Don't add
  steps that need an agent to watch, SSH in, or approve something.
- Per experiment, edit only `box/experiment.sh` and `experiment.conf`. Changes to
  the other `box/` scripts or `launch.sh` belong in hextobin/gpu-experiment-template.
- `experiment.sh` rules: keep outputs in `$OUT` under 95MB per file; make each
  unit skip if its output exists; call `progress_start` after data prep and
  `progress i n` after each unit; exit non-zero on failure.
- Before launch: write the Question and Predictions (with odds) into the README,
  following `WRITEUP.md`; run a CPU smoke test if possible; commit and push.
- The user runs `./launch.sh` and types tokens at the prompts. Never ask for
  tokens in chat.
- Check progress from the `results` branch (`git fetch origin results`), not by
  watching the box.
- After the run: confirm `doctl compute droplet list` is empty, then rewrite
  README.md in the `WRITEUP.md` format (Question/Answer first, results table,
  findings, predictions vs outcome, reproduce). If work continues, add a short
  handoff section here; if the experiment is finished, delete this file.
- When the user is ready to publish (repo going public):
  1. Add a row, newest first, to the table in hextobin/hextobin `README.md`
     (the profile page): the question, a one-line answer and a repo link,
     matching this repo's README.
  2. Set the About box: description in the form
     `Q: <short question> A: <short answer>. Write-up, code, <what's included>.`
     (under 350 characters) plus 3-5 topics, with
     `gh repo edit hextobin/<repo> --description "..." --add-topic a,b,c`.
     Run it if `gh` is available and logged in; otherwise (e.g. in a cloud
     session) give the user the ready-to-paste command.

## Handoff: build done (2026-11-02), next phase is launch

Design, arms and predictions are in README.md. Don't change the predictions.

- **Code:** `polytope/proj.py` (box, l1, shrink, and `NormProj` = P(RMSNorm(x))
  with per-call stats), `polytope/model.py` (attention-only GPT, `--arm`
  floor|box|l1|shrink|ceiling), `train.py` and `scripts/{sweep,summarize}.py`
  ported from hextobin/polydeq. RMSNorm has no gain, on purpose.
- **Logged per eval** (`proj` in log.jsonl), for `emb` and `L0`..`L5`: fraction
  of tokens changed, fraction of coordinates changed (= clipped for box),
  nonzero fraction (mean and median over tokens), mean tau for l1.
- **Tests:** `python -m pytest -q tests` (33 pass): L1 projection vs brute-force
  face enumeration, feasibility, KKT, identity inside, gradcheck.
- **CPU smoke test:** all 5 arms, 60 steps on synthetic data, plus l1 under
  `--compile`; all fine.
- **Box:** `box/experiment.sh` downloads enwik8, checks the projection on the
  GPU, then runs 15 units (all arms at seed 0, then 1, then 2) to
  `$OUT/runs/<arm>_s<seed>` and writes `$OUT/summary.md`. `DEADLINE_HOURS=3`.
- **Next:** the user runs `./launch.sh`. Then check `git fetch origin results`.
