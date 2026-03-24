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

## 2026-03-18 ~15:30 — h027/h030/h031 output-dir /output bug (recurrence)

**Problem:** h027, h030, h031 were submitted with `--output-dir /output` instead of `/runs`. Same bug as the Phase 2 issue (see 07:00 entry). h028, h029, h032, h033 correctly used `/runs`.

**Impact:** New-format CSVs (with q4_return, auc, etc.) lost for h027/h030/h031 completed jobs. Watcher-generated old-format CSVs were used instead. Was able to compute q4 from curve files as a workaround.

**Root cause:** Batch submission JSON files for h027/h030/h031 had hardcoded `--output-dir /output`.

**Workaround:** Computed q4 metrics from `{hypothesis_id}__{experiment_id}__curve.csv` files which contain per-episode learning curves. These are saved alongside the old-format summary CSVs.

**Fix:** Re-synced code to all 4 clusters. Jobs already running will still use old code (Python loads scripts at start time). Only NEW submissions will use the corrected code.

**Additional finding:** Code on clusters was OLD version for ALL h030-h033 (even h032/h033 with correct output-dir). The old code produces old-format summary CSVs from the training script itself. The new-format CSV code was committed locally but not synced before submission.

## 2026-03-19 02:00 — h034 new-code results IDENTICAL to h029

**Issue:** h034 (CVaR+Dueling+DrQ) produces results identical to h029 (CVaR+QR+DrQ) to 13 decimal places on all 3 games tested (Phoenix, Qbert, SpaceInvaders). Same n_episodes, mean_return, q4_return, auc, final_avg20.

**Investigation:**
- Verified code on fir cluster matches local (md5sum identical)
- Local test confirms architectures DO produce different outputs with same seed
- Both scripts use `torch.manual_seed(1)`, CNN backbone is identical, heads differ (h034 adds hidden layers for value/advantage streams)
- The architectures are structurally different: h029 has direct Linear heads, h034 has 2-layer Sequential heads with ReLU
- Yet the training trajectories are perfectly identical across 40M steps

**Root cause:** Unknown. Possible theories:
1. Stale __pycache__/.pyc files on cluster from older version of h034 that was identical to h029
2. Some PyTorch/CUDA determinism quirk that makes different architectures converge to identical behavior
3. Singularity container caching issue

**Impact:** h034 IQM (previously 0.0082 from curve-derived data) was based on the SAME algorithm as h029. h034 closed, 12 running jobs cancelled.

**Action:** h036 (CVaR+Dueling+SEM) confirmed to produce DIFFERENT results (includes SEM which fundamentally changes representation). h035 (CVaR+SEM) also confirmed different. Both pilots still running.

## 2026-03-19 07:45 — h056 PPO Wide results IDENTICAL to PPO baseline (stale code)

**Problem:** h056 (PPO + Wider NatureCNN) first 3 completed results — Amidar (fir) and Phoenix (narval) are bit-for-bit identical to h001 PPO baseline (same n_episodes, mean_return, q4_return, auc, final_avg20 to full floating-point precision). MsPacman (fir) shows slight differences but SPS ~3510 matches standard PPO (wider network should have ~30-40% lower SPS due to 4x parameters).

**Investigation:**
- `ppo_atari_envpool_wide.py` on fir confirmed correct architecture (Conv2d channels 64→128→128, hidden 1024)
- File size (16062 bytes) and timestamp match local version
- SPS across all 3 completed jobs: 3486-3510 (consistent with standard NatureCNN, not wide)
- Commit timestamp (08:30 UTC) vs submission time (08:27 UTC) — jobs submitted 3 min BEFORE commit

**Root cause:** Most likely the sync happened before the architecture changes were fully written to disk. Jobs were submitted and started with the old file content (standard PPO copy). By the time the file was updated, Python had already loaded the old version.

**Impact:** All 15 h056 jobs running stale code. 12 cancelled, 3 completed results discarded (NOT added to experiments.csv).

**Action:** Synced code to all 4 clusters, verified correct content, resubmitted all 15 h056 games with 4h walltime. This is the THIRD occurrence of stale-code issue (h051 CReLU was the second, h034 dueling was the first).

## 2026-03-19 17:20 — h051/h056 stale CSVs persist despite correct cluster code
16 CSVs pulled from fir/rorqual for h051 (CReLU) and h056 (Wide) were bit-for-bit identical to PPO h001 baseline.
Verified: ppo_atari_envpool_crelu.py (17005 bytes) and ppo_atari_envpool_wide.py (16062 bytes) exist on ALL 4 clusters with correct file sizes matching local.
Root cause: OLD jobs from sessions 88-90 (which ran before code was properly synced or with incomplete commands) completed and left output CSVs in /scratch/rogercc/runs/. The watcher pulled these stale CSVs when triggered.
Currently running jobs (submitted with correct commands AND code present on cluster) should produce genuine results.
Deleted 16 stale CSVs. Re-synced all clusters as precaution.
Resubmitted 15 gap-filling jobs across all 4 clusters.

## 2026-03-19 23:30 — h051/h056 stale plague root cause diagnosed

**Problem:** h051 (CReLU) and h056 (Wide) experiments produced stale results matching h001 PPO baseline exactly for 100+ sessions. Only 4/15 genuine results for each after dozens of resubmissions across all clusters.

**Root cause:** Some resubmission batches omitted `--hypothesis-id` and `--total-timesteps` flags. The scripts defaulted to `hypothesis_id="h000"` and `total_timesteps=10000000` (10M). This caused:
1. Results saved to wrong filenames (e.g., `h000__BattleZone-v5_s1.csv` instead of `h051__h051-battlezone-s1.csv`)
2. Training ran for only 10M steps instead of 40M
3. The watcher pulled OLD leftover h001 CSVs from the output directory instead of genuine h051/h056 results

Additionally, many correctly-flagged jobs were stuck running 44+ hours (SLURM elapsed MM:SS misread as HH:MM — actually only 44 minutes). Jobs with stale code on cluster ignored the new flags since the script version didn't have those CLI args.

**Fix:**
- Changed default `total_timesteps` from 10M to 40M in both scripts
- Changed default `hypothesis_id` to "h051" (crelu) and "h056" (wide)
- Synced fresh code to all 4 clusters
- Cancelled 74 wasted/duplicate jobs
- Resubmitted 22 jobs (11 h051 + 11 h056) with correct flags and fresh code

**Lesson:** Always use explicit flags AND update defaults when scripts are used for a specific hypothesis. Never rely on defaults being correct across resubmissions.

## 2026-03-23 ~05:00 — h063-qbert-s2 narval resubmit failed (path quoting)
Job 58120116 on narval completed in 47 seconds with exit_code=0 but produced no CSV. SLURM log shows:
- `slurmstepd: error: couldn't chdir to /home/rogercc/'/scratch/rogercc/cleanrl'` — embedded single quotes around project_path
- `-H` flag in SBATCH template rendered with quotes, causing bind mount failure
- `python: can't open file '/src/cleanrl/dqn_atari_envpool_iqn.py': [Errno 2] No such file or directory`
- This is the 2nd failure for h063-qbert-s2 (1st was CUDA error on fir job 28939139)
- Resubmitted to rorqual as job 8909264
- Other narval jobs from same batch (submitted 22:17 UTC) worked fine — issue may be intermittent or specific to the resubmission path

**Note:** 5 other narval jobs from the original batch are still running normally at ~6.5h elapsed.

## 2026-03-23 ~05:00 — h063-qbert-s2 rorqual resubmit ALSO failed (same path quoting)
Job 8909264 on rorqual had the SAME path quoting failure as narval:
- `slurmstepd: error: couldn't chdir to /home/rogercc/'/scratch/rogercc/cleanrl'`
- `python: can't open file '/src/cleanrl/dqn_atari_envpool_iqn.py': [Errno 2] No such file or directory`
- Exit code 0 but no results produced
- This is the 3rd failure for h063-qbert-s2 (fir CUDA error, narval path quoting, rorqual path quoting)
- Resubmitted to nibi as job 10786553 (attempt #4)
- The path quoting issue appears to be a xgenius bug in individual resubmissions (vs batch submissions which work fine)
- All batch-submitted narval jobs are running normally — only individual `xgenius submit` calls produce the quoting error

## 2026-03-23 17:00 — Phase 4 h001/h002 missing --total-timesteps 40000000

The Phase 4 Atari57 batch (batch_phase4_atari57.json, Session 229) omitted `--total-timesteps 40000000` 
from h001 (PPO) and h002 (PQN) commands. Both scripts default to 10M steps, so all 252 h001/h002 Phase 4 
experiments ran at 10M instead of the required 40M.

h064 (Rainbow-lite) was NOT affected — dqn_atari_envpool_rainbow_lite.py defaults to 40M.

**Symptoms**: Jobs completed in ~35 min (expected ~3h for PPO/PQN at 40M). CSVs showed total_timesteps=10000000.

**Fix**: Created corrected batch (batch_phase4_h001_h002_40M_fix.json) with explicit --total-timesteps 40000000 
and resubmitted all 252 experiments.

**Root cause**: Batch generation script didn't include total_timesteps flag, relying on script defaults which 
differ between PPO/PQN (10M default) and DQN/Rainbow-lite (40M default).

**Prevention**: Always include --total-timesteps explicitly in batch commands regardless of script defaults.

## 2026-03-24 04:00 — 32 h064 experiments repeatedly fail silently (5th resubmission)

**Problem:** The same 32 h064 (Rainbow-lite) experiments have failed 4 times (sessions 276-286). Jobs start, rsync 4.9GB container, Python imports begin (FutureWarning + Gym deprecation visible), then jobs die silently with no training output and no error message. All 4 clusters (rorqual, narval, nibi, fir) show 0 running/0 pending after each batch.

**Symptoms:**
- SLURM .out files show container rsync + Python import warnings, then EOF
- No `global_step=` or `SPS:` output (training never starts)
- No error files available
- Successful h064 jobs (139/171) used identical commands and ran fine

**Investigation:**
- Commands identical to successful experiments
- Script exists on all clusters (verified via `xgenius ls`)
- Code synced to all clusters
- No env-id-specific pattern (standard Atari games that work elsewhere)

**Actions for attempt 5:**
1. Added diagnostics to SBATCH template: hostname, GPU info via nvidia-smi, timestamps, exit code capture
2. Increased memory from 32G to 48G (replay buffer alone needs ~6GB)
3. Resubmitted all 32 to same distribution of clusters

**Next steps if this fails again:**
- Check nvidia-smi output to verify GPU allocation on MIG slices
- Check dmesg/journal for OOM killer events
- Try submitting with larger MIG slices (2g.20gb instead of 1g.10gb)
- Try submitting without MIG (full GPU) on one cluster as test
