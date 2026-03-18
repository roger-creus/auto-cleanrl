# Debug Log

Errors and issues encountered during autonomous research.

## 2026-03-18 ~07:00 — Phase 2 output-dir /output bug (all h003-h009 jobs)

**Problem:** All 105 Phase 2 pilot jobs (h003-h009) were submitted with `--output-dir /output`, but the Singularity container only bind-mounts the output directory to `/runs` (configured in xgenius.toml as `output_dir_container = "/runs"`). Training completed successfully but CSV saving failed with `OSError: [Errno 30] Read-only file system: '/output'`.

**Root cause:** Session 3 batch submission script hardcoded `--output-dir /output` in the commands, but the scripts' default is `/runs` (correct). The explicit `--output-dir /output` flag overrode the correct default.

**Impact:** All 105 Phase 2 jobs ran to completion but lost results. ~105 GPU-hours wasted.

**Fix:** Cancelled all running Phase 2 jobs, resubmitted 105 jobs WITHOUT `--output-dir` flag (using script default of `/runs`).

**Lesson:** Always verify output paths match container bind mounts. The correct container output path is `/runs`, not `/output`.

## 2026-03-18 ~08:30 — h010 IMPALA CNN channel bug (Conv2d input channels)

**Problem:** All h010 (PPO + IMPALA CNN) jobs crashed with `RuntimeError: Given groups=1, weight of size [16, 84, 3, 3], expected input[128, 4, 84, 84] to have 84 channels, but got 4 channels instead`.

**Root cause:** In `ppo_atari_envpool_impala.py` line 173, observation shape was unpacked as `h, w, c = envs.single_observation_space.shape`. Envpool returns CHW format (4, 84, 84), so this gave `h=4, w=84, c=84` and the first Conv layer was created with 84 input channels instead of 4.

**Fix:** Changed `h, w, c` to `c, h, w` on line 173. The PQN IMPALA script (h011) was already correct (`c, h, w`). Cancelled 1 remaining h010 job, resubmitted all 15 with fix.

**Impact:** ~15 GPU-minutes wasted (jobs crash immediately at first forward pass).

**Lesson:** Always check observation space format — envpool uses CHW (channel-first), not HWC.

## 2026-03-18 ~08:30 — h003-h009 running at 10M not 40M timesteps

**Problem:** All h003-h009 Phase 2 pilot jobs were submitted WITHOUT `--total-timesteps 40000000`, so they default to the script's `total_timesteps: int = 10000000` (10M). Baselines used 40M.

**Root cause:** Session 3 batch submission did not include `--total-timesteps` flag for h003-h009. Session 6 (h010-h015) correctly included it.

**Impact:** h003-h009 results are at 10M steps, not directly comparable to 40M baselines. However, the 10M results still provide strong directional signal. Both h003 and h004 show excellent sample efficiency at 10M.

**Lesson:** Always explicitly set `--total-timesteps 40000000` in submission commands. Do not rely on script defaults.

## 2026-03-18 09:30 — Rorqual container image corruption after rebuild

### What happened
Session 8 rebuilt the container to add `schedulefree` dependency (~09:03-09:11 UTC). The new .sif pushed to rorqual was corrupted — local file is 4,945,575,936 bytes but rorqual had 4,945,334,272 bytes (~242KB short).

### Impact
33 rorqual h003/h004/h005 40M 3-seed eval jobs started at ~09:13 (right after corrupt push) all failed with:
```
FATAL: container creation failed: image driver mount failure: squashfuse_ll exited: Something went wrong trying to read the squashfs image.
```

Also a secondary path quoting issue: `couldn't chdir to '/home/rogercc/'/scratch/rogercc/cleanrl''` (extra quotes in -H flag).

### Resolution
1. Re-pushed the image to rorqual (took ~20 min for 5GB SCP)
2. Cancelled 16 still-running at-risk jobs
3. 14 had already disappeared from SLURM
4. 3 had "completed" with exit code 0 (but actually failed)
5. Resubmitted all 33 jobs distributed across all 4 clusters

### Lesson
Container pushes to rorqual may silently truncate/corrupt the .sif. After push, should verify file size matches local file. The verify-image command only checks existence, not integrity.
