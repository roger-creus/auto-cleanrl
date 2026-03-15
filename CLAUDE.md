# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT: Do NOT run `xgenius watch`. The watcher daemon is managed by the human in a separate terminal. Your job is to submit experiments and exit — the watcher will trigger you when jobs complete.**

**IMPORTANT: Do NOT run `xgenius reset`. Only the human resets the research state. If you need a fresh start, ask the human.**

**IMPORTANT: Job submission can be slow** (SSH to automation nodes takes time). When using `xgenius batch-submit` or submitting many jobs, be patient — each job takes several seconds to submit. Do NOT set short timeouts on Bash commands that submit jobs. Use `timeout 30m` or no timeout at all for batch submissions.

**IMPORTANT: Distribute jobs across ALL available clusters to maximize throughput.** Check xgenius.toml for configured clusters and spread experiments evenly across them. Do not submit all jobs to a single cluster when multiple are available.

**IMPORTANT: You MUST use the xgenius journal for EVERY hypothesis and experiment.** Before submitting any experiments:
1. `xgenius journal add-hypothesis "..." --motivation "..." --expected "..."` — record what you're testing and why
2. Submit jobs with `--hypothesis-id` so they're linked in the journal
3. After results come in, `xgenius journal add-result` and `xgenius journal update-hypothesis`
The journal is how the watcher provides you context when it wakes you up. Without it, you lose track of your research.

## Git Conventions

You MUST commit and push your work regularly. Every meaningful change should be committed. Use these prefixes:

```
baseline: <description>        — Running/recording baseline experiments
hypothesis(ID): <description>  — Implementing a hypothesis (e.g., hypothesis(h003): add spectral norm to value net)
result(ID): <description>      — Recording results for a hypothesis
engineering: <description>     — Non-research improvements (better architecture, optimizer swap, etc.)
fix: <description>             — Bug fixes (broken training, container issues, etc.)
revert(ID): <description>     — Reverting a hypothesis that didn't work
ablation(ID): <description>    — Ablation study for a hypothesis
config: <description>          — Configuration changes (xgenius.toml, SBATCH templates, etc.)
container: <description>       — Dockerfile or container changes
docs: <description>            — Documentation updates
```

**Rules:**
- Commit BEFORE submitting experiments (so the cluster runs the committed code)
- Commit AFTER recording results (so the journal state is preserved)
- Push after every commit — this repo is your research record
- Never force push or rewrite history
- If a hypothesis breaks things, use `git revert` to undo it cleanly
- Keep .xgenius/ state files committed (journal, jobs) so the research history is preserved in git

## Project Structure

The `.xgenius/` directory contains all xgenius runtime state for this project:
- `.xgenius/templates/` — SBATCH job script templates (YOU can edit these to customize job behavior)
- `.xgenius/journal.jsonl` — research journal (hypotheses, experiments, results)
- `.xgenius/journal_summary.md` — auto-generated research summary
- `.xgenius/jobs.jsonl` — job tracker (job IDs, statuses, log file paths)
- `.xgenius/audit.jsonl` — audit log of all actions
- `.xgenius/batches/` — archived batch submission files (auto-saved, reusable for resubmission)
- `.xgenius/logs/` — SLURM log files are stored on the cluster at `{scratch}/.xgenius/logs/`
- `.xgenius/markers/` — job completion markers (managed by watcher)
- `.xgenius/watcher.log` — watcher daemon activity log

**Batch files:** Write batch submission JSON files to `.xgenius/batches/` so they are preserved and reusable. Every `xgenius batch-submit` also auto-archives a timestamped copy there.

**SBATCH templates:** If you need to modify SBATCH job scripts (e.g., add `mkdir -p runs` before the command, change bind mounts), edit the templates in `.xgenius/templates/`. Do NOT modify the xgenius package templates.. The project-local templates take priority.






