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
