# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**IMPORTANT: Do NOT run `xgenius watch`. The watcher daemon is managed by the human in a separate terminal. Your job is to submit experiments and exit — the watcher will trigger you when jobs complete.**

**IMPORTANT: Distribute jobs across ALL available clusters to maximize throughput.** Check xgenius.toml for configured clusters and spread experiments evenly across them. Do not submit all jobs to a single cluster when multiple are available.

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

**SBATCH templates:** If you need to modify SBATCH job scripts (e.g., add `mkdir -p runs` before the command, change bind mounts), edit the templates in `.xgenius/templates/`. Do NOT modify the xgenius package templates at `/home/roger/Desktop/xgenius/`. The project-local templates take priority.






## xgenius — Autonomous Research Tools

This project uses xgenius for autonomous research on SLURM clusters.
Configuration is in `xgenius.toml`. Research goal is in `research_goal.md`.
Runtime state is in `.xgenius/` (journal, jobs, audit log).

### Available Commands

**Research Loop:**
- `xgenius journal context` — Full research context: goal, hypotheses, experiments, results, what to try next
- `xgenius journal summary` — Concise progress summary
- `xgenius journal add-hypothesis "text" --motivation "why" --expected "outcome"` — Record a hypothesis
- `xgenius journal add-result --experiment-id ID --metrics '{"key": value}' --analysis "text"` — Record results
- `xgenius journal update-hypothesis --id ID --status confirmed|rejected|partially_confirmed --conclusion "text"`

**Job Management:**
- `xgenius submit --cluster NAME --command "python script.py --args" [--experiment-id ID] [--hypothesis-id ID] [--gpus N] [--cpus N] [--memory "16G"] [--walltime "04:00:00"]` — Submit a job
- `xgenius batch-submit --file experiments.json` — Submit multiple jobs
- `xgenius status [--cluster NAME] [--json]` — Check job statuses
- `xgenius cancel --cluster NAME --job-ids ID1,ID2` — Cancel specific jobs
- `xgenius logs --experiment-id ID --json` — View job stdout (can also use --cluster NAME --job-id ID)
- `xgenius errors --experiment-id ID --json` — View job stderr/crashes/tracebacks (can also use --cluster NAME --job-id ID)
- `xgenius check-completions [--json]` — Check for newly completed jobs
- `xgenius reconcile --json` — Sync local job tracker with actual SLURM state (fixes stale jobs)

**Code & Data:**
- `xgenius sync --cluster NAME` — Rsync project code to cluster
- `xgenius pull --cluster NAME [--job-id ID]` — Pull results from cluster
- `xgenius ls --cluster NAME [--path PATH]` — List files on cluster

**Container (you must handle building intelligently):**
- `xgenius build --json` — Full pipeline: docker build → test → singularity convert. Returns structured step-by-step results.
- `xgenius build --step docker --json` — Docker build only. If it fails, read the error, fix the Dockerfile, and retry.
- `xgenius build --step test --json` — Run tests inside the Docker container. Use `--test-command "..."` for custom commands.
- `xgenius build --step singularity --json` — Convert Docker image to Singularity .sif.
- `xgenius build --skip-tests --json` — Full pipeline but skip tests.
- `xgenius push-image --cluster NAME [--image PATH] --json` — Push .sif to cluster, verify it runs.
- `xgenius verify-image --cluster NAME --json` — Verify container works on cluster.

**CRITICAL — Container design:**
The Docker/Singularity image must contain ONLY dependencies (CUDA, Python, pip packages), NOT the project code.
Code is mounted at runtime via Singularity --bind flags in the SBATCH template. This means:
- The Dockerfile should install all pip/system dependencies but should NOT COPY source code into the image.
- Removing `COPY ./src /src` or similar lines from the Dockerfile is correct — code goes via `xgenius sync`.
- Keep `COPY pyproject.toml` + `pip install` to install dependencies, but remove any final COPY of source code.
- Remove any ENTRYPOINT that expects code to be baked in — the SBATCH template handles command execution.
- After building, verify the image has all deps by running a test import, not by running the full code.

**IMPORTANT for container building:** When `xgenius build` fails, YOU are responsible for diagnosing the error.
Read the Dockerfile, understand the project structure, fix the issue, and re-run `xgenius build`.
You can run individual steps (`--step docker`, `--step test`, `--step singularity`) to iterate on specific failures.
Typical issues: outdated base images, missing dependencies, wrong Python version, broken COPY paths.

**Safety & Budget:**
- `xgenius budget` — Check remaining compute budget
- `xgenius validate --command "python script.py"` — Dry-run safety check
- `xgenius audit [--limit N]` — View audit log
- `xgenius job-history [--limit N] --json` — View past jobs with walltime, resources, log file paths, and status
- `xgenius reset` — Clear all state for a fresh research run

All commands support `--json` for structured output.

### Debugging Failed Jobs
When a job fails:
1. Run `xgenius errors --experiment-id EXPERIMENT_ID --json` to see tracebacks and error messages
2. Run `xgenius logs --experiment-id EXPERIMENT_ID --json` to see full stdout
3. Run `xgenius job-history --json` to see all jobs with their log file paths, statuses, and walltimes
4. Log files are stored at `{scratch}/.xgenius/logs/{experiment_id}_{job_id}.out` on the cluster

### Resource Management
The xgenius.toml [safety] section defines MAXIMUM resource limits. You can request LESS:
- `--gpus 1` instead of the max 4
- `--walltime "02:00:00"` for a quick test instead of the max 24h
- `--memory "16G"` if the job doesn't need much RAM
- `--cpus 4` for a lightweight job

Use `xgenius job-history --json` to see how long past jobs took, then adjust walltime accordingly.
Use `xgenius status --json` to see pending/running jobs with their elapsed time, submit time, and pending reason.

### Research Workflow
1. Run `xgenius journal context` to understand current state
2. Formulate a hypothesis and record it
3. Modify code to test the hypothesis
4. Run `xgenius sync` to push code to cluster
5. Run `xgenius submit` to start experiments
6. Wait for `xgenius watch` daemon to trigger you on completion
7. Analyze results, record findings, iterate

### Container Build Workflow
When you need to build/rebuild the container:
1. Read the Dockerfile and understand what it does
2. Ensure the Dockerfile does NOT bake source code into the image (remove COPY of source dirs, remove ENTRYPOINT)
3. Ensure the Dockerfile installs all system + pip dependencies
4. Run `xgenius build --json` for the full pipeline
5. If docker build fails: read the error, fix the Dockerfile, retry with `xgenius build --step docker --json`
6. Verify deps work: `xgenius build --step test --test-command "python -c 'import torch; print(torch.cuda.is_available())'" --json`
7. If singularity conversion fails: check if apptainer/singularity is installed
8. Push to cluster: `xgenius push-image --cluster NAME --json`
9. You only need to rebuild the container when dependencies change. For code changes, use `xgenius sync`.

### Safety
- All commands are validated against limits in xgenius.toml [safety]
- Commands must start with allowed prefixes (e.g., "python")
- Resource requests are checked against max GPU/CPU/memory/walltime
- All actions are logged to .xgenius/audit.jsonl
