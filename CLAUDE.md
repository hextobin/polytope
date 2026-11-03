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

## Handoff: design done (2026-11-02), next phase is build

The design, arms and predictions are in README.md. Don't change the predictions.

- **Code base:** reuse hextobin/polydeq (clone it from GitHub). Take its
  byte-level GPT, enwik8 data prep, `train.py`, and the sweep and summarize
  scripts. Drop the MLP, and add an `--arm` with values floor|box|l1|shrink|ceiling.
- **Structure for every arm:** attention reads through a pre-norm, and after
  each add `x ← P(RMSNorm(x + attn(x)))`. Also apply it once to the
  embeddings, then a final norm and the unembedding. `ceiling` adds a GELU MLP
  sublayer, followed by RMSNorm only.
- **P:** floor = identity; box = clamp(−1, 1); l1 = exact projection onto the
  L1 ball of radius d/4 (sort, cumsum, shared threshold τ, soft-threshold;
  identity if already inside); shrink = sign(z)·max(|z| − 0.78, 0).
- **Logging per layer, at each eval:** fraction of tokens the projection
  changes; fraction of nonzero coordinates (l1, shrink) or clipped coordinates (box).
- **Tests:** check the L1 projection against a brute-force reference (the
  output's L1 norm ≤ r, KKT conditions hold, the input is unchanged when it's
  already inside), and run gradcheck.
- **Run plan:** 5 arms × seeds 0,1,2 at 5000 steps, each run one resumable
  unit in `box/experiment.sh`. Set `DEADLINE_HOURS` to about 3.
- **Prior art:** checked on 2026-11-02 and recorded in the README. CRATE and
  attention-only white-box models (fixed-threshold soft-threshold after
  attention) are the nearest work, which is why the `shrink` control exists.
- **Before launch:** run a CPU smoke test of all arms for a few steps, then
  commit and push. The user runs `./launch.sh`.
