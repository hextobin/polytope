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
- Before the user launches, recommend a box and get an OK on the price: run
  `./launch.sh --check` (it lists each size and region in launch order with
  $/hr and the cost at the deadline, and rents nothing). If `doctl` can't reach
  DigitalOcean from your session, ask the user to run it and paste the output.
  Recommend one option with its hourly price, the expected cost (estimated
  runtime x price) and the max cost (deadline x price), and wait for the user
  to confirm. `launch.sh` then asks y/n, with the price, before every droplet
  it creates.
- The user runs `./launch.sh` and types tokens at the prompts. Never ask for
  tokens in chat.
- Check progress from the `results` branch (`git fetch origin results`), not by
  watching the box.
- After launch, offer an on-box session, which the user can check in with from
  the sidebar. It needs no API key and no box script changes: the user saves an
  SSH connection in the Claude app to the IP `launch.sh` printed (`root@<ip>`),
  then start it with `start_session` (`target: "ssh"`, `host` from
  `list_start_targets`, repo at `/root/exp`). It uses the user's plan and ends
  with the box. It's optional: nothing on the box may depend on it.
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

## Handoff: launched (2026-11-04), next phase is analysis

- **Box:** `polytope-run`, `gpu-h100x1-80gb` in ams3 at $4.41/hr, max $13.23.
  Hard deadline 04:55 UTC. This was the second launch; the first was cancelled.
- **On-box session:** `polytope-box`, on the SSH connection `root@174.138.0.12`.
  It ends with the box, and the saved connection can be removed afterwards.
- **First crash:** both boxes died on the first training step because `Python.h`
  was missing for `--compile`. Fixed by hand on this box, then the user restored
  the deadline and restarted `exp-run`; `box/run.sh` now installs `python3-dev`
  here and in the template. On the restart, training past step 1 was not yet
  confirmed when this was written.
- **`results` branch:** the first logs there include the cancelled box's, so
  the early `floor_s0` files and `self_destruct.log` are from that run.
- **Write-up:** the setup line says "one 48GB GPU"; name the H100 instead. Don't
  change the predictions.
- **Next:** check `git fetch origin results`. After the run, the user confirms
  `doctl compute droplet list` is empty (doctl can't run in the sandbox), then
  rewrite README.md per `WRITEUP.md`.
