# gpu-experiment-template

Run a GPU experiment on a DigitalOcean droplet that sets itself up, pushes
results to git, and destroys itself. Nothing has to watch it: no terminal,
no agent session.

## You need

- A DigitalOcean account with an SSH key uploaded. Put its ID in
  `experiment.conf` (`doctl compute ssh-key list` shows it).
- `doctl` (logged in), `jq` and `git` on your machine.
- Two short-lived tokens per run, typed at `launch.sh`'s prompts and never
  saved anywhere else:
  - DigitalOcean: custom scopes **droplet: read + delete**, 1-day expiry.
  - GitHub fine-grained: the experiment repo only, **Contents: read and write**,
    7-day expiry.

## Use it without an agent

1. On GitHub, click **Use this template** to make a new repo, and clone it.
2. Add your code. Edit `box/experiment.sh` (what runs on the box; the only box
   script you change) and `experiment.conf` (sizes, regions, deadline, pip
   packages).
3. Write the Question and Predictions sections of `WRITEUP.md` before you
   know the answer.
4. Commit and push to `main`.
5. Run `./launch.sh` and type the tokens at the prompts.
6. Results show up on the `results` branch every 15 minutes:
   `git fetch origin results && git show origin/results:logs/run.log | tail`
7. Afterwards, check `doctl compute droplet list` is empty, and revoke the tokens.
8. Finish `WRITEUP.md` and make it your `README.md`.

## Use it with Claude Code (or another agent)

The box works the same way; the agent does the writing and checking around it.
`CLAUDE.md` holds the agent's rules. Claude Code reads it automatically.
`AGENTS.md` points to the same file for other agents.

1. Make a repo from the template, clone it, and start a session in that folder.
2. Describe the experiment. The agent writes the code, `box/experiment.sh` and
   `experiment.conf`, smoke-tests them on CPU, and writes the Question and
   Predictions into the README before launch.
3. You make the two tokens and run `./launch.sh` yourself. The agent never asks
   for tokens in chat.
4. The agent checks progress from the `results` branch, not by logging in to
   the box. Nothing depends on the session staying open.
5. After the run, the agent confirms the droplet is gone, writes the results up
   in the `WRITEUP.md` format, and (when you publish) drafts the repo's About
   box.

## What the box does

| When | What |
|---|---|
| Boot | Arms the dead-man timer (hard cap `DEADLINE_HOURS`), starts `exp-run.service` |
| Every 15 min | `sync.sh` pushes `/root/logs` and `out/` to the `results` branch |
| Every 5 min | `deadman.sh`: destroys the box after the deadline or if `control/stop` is on `main`; if the experiment service died without finishing, pulls the deadline in to 1 hour |
| During the run | `progress i n` tightens the deadline to 1.5× the projected finish + 20 min |
| Success | Final push, then self-destruct (or wait `KEEP_AFTER_DONE_MIN`) |
| Failure | Deadline pulled in to 1 hour so you can fix it; then self-destruct |

**Kill it remotely:** push a file named `control/stop` to `main`. It's gone
within 5 minutes. Delete the file before the next launch (`launch.sh` refuses
while it exists).

**Fix a crash by hand** (SSH as root): fix, push, `git -C /root/exp pull`,
`systemctl restart exp-run.service`, then write a later epoch to
`/root/deadline` if you need more time. Finished units are skipped if
`experiment.sh` is resumable.
