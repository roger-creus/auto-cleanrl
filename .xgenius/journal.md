
---
**[2026-03-18 03:22 UTC]**

## Session 1: Fresh Start — Phase 1 Baseline Submission

### What was done:
1. **Code fixes applied to PPO and PQN scripts:**
   - Fixed envpool config: episodic_life=True, reward_clip=True (community standard)
   - Fixed logging: aggregate multiple episodes at same global_step into single log entry (mean)
   - Added --hypothesis-id, --experiment-id, --output-dir CLI args
   - Added CSV output at end of training with robust metrics: mean_return, q4_return, auc, final_avg20

2. **Results bank created:**
   - results/experiments.csv (empty, ready for data)
   - results/hypotheses.csv (h001 PPO, h002 PQN)

3. **Submitted 90 baseline jobs (all 4 clusters: rorqual, narval, nibi, fir):**
   - h001 (PPO): 15 games × 3 seeds = 45 jobs, 40M env steps each
   - h002 (PQN): 15 games × 3 seeds = 45 jobs, 40M env steps each

### Games: Amidar, Alien, Breakout, Enduro, PrivateEye, Solaris, BattleZone, DoubleDunk, NameThisGame, Phoenix, Qbert, SpaceInvaders, MsPacman, Venture, MontezumaRevenge

### What to do next:
1. Wait for baseline jobs to complete
2. Parse CSV results from results/CLUSTER/ directories
3. Compute HNS using compute_hns.py for cross-game comparison
4. Populate results bank with all baseline data
5. Analyze which games each baseline struggles on — these are the targets for Phase 2
6. Begin Phase 2: formulate first hypotheses based on baseline analysis and literature review

### Context from previous research runs (memory):
- Previous runs showed PPO > PQN on HNS (mean 0.034 vs -0.181)
- Best PQN variant was h034 (Entropy-Adaptive + Regenerative) with 8W/2L/2F vs base PQN
- Key techniques to explore: NaP (weight projection), CHAIN (churn reduction), SPR (self-predictive), network sparsity
- PPO + entropy-adaptive was highest priority direction
- Must avoid the word 'eval' in xgenius commands (blocked by safety filter)

---
**[2026-03-18 05:28 UTC]**

## Session 2: Processing Early Baseline Results

### State:
- 23/90 baseline results collected (11 PPO games, 10 PQN games)
- 9 nibi jobs 'disappeared' from SLURM but all had valid CSV results — no resubmission needed
- 1 narval job (Phoenix-v5_s3 h001) disappeared and was resubmitted (job 57935917), but the original also left a valid CSV
- ~67 jobs still running across 4 clusters

### CRITICAL FINDING: Return Metric Issue
The custom RecordEpisodeStatistics wrapper resets episode_returns on `infos['terminated']`.
With `episodic_life=True`, envpool sets terminated=True on EVERY life loss (not just game over).
This means `info['r']` only contains the LAST LIFE's unclipped Atari return, not the full game return.

However:
- The returns ARE true unclipped Atari scores (confirmed from envpool source: info_reward uses unclipped reward)
- All algorithms have the same measurement bias → relative comparisons are valid
- Full-game HNS cannot be computed from these numbers (but we can compute last-life HNS)
- We should NOT change baselines mid-run; fix this in Phase 2 scripts if needed

### Preliminary PPO vs PQN (8 common games, 1 seed each):
PPO wins: MsPacman, NameThisGame, Phoenix (huge: 735 vs 87), Qbert
PQN wins: Amidar (huge: 31 vs 2), Breakout (marginal), MontezumaRevenge (both 0), SpaceInvaders
Score: 4-4 tie. Very different from previous research iteration that found PPO >> PQN.
But single seeds are noisy — need full 3-seed results before conclusions.

### Literature Research Done:
- CHAIN-SP paper: CHAIN + LayerNorm is most effective for PPO plasticity on Atari
- Plasticine benchmark: 13+ plasticity methods; normalization (LayerNorm, NaP) consistently top
- On-policy plasticity study: Soft shrink+perturb + LayerNorm is optimal combination
- CReLU and Plasticity Injection FAILED in on-policy settings

### Phase 2 Priority Hypotheses (to implement when baselines complete):
1. PPO + LayerNorm + CHAIN (CHAIN-SP approach) — plasticity maintenance
2. PQN + NaP (weight projection) — complement existing LayerNorm
3. PPO + Regenerative Regularization (shrink+perturb) — proven in on-policy RL
4. Network Sparsity (random pruning at init) — ICML 2025 oral, very simple
5. SPR auxiliary loss — self-predictive representations

### Next Steps:
1. Wait for remaining ~67 baseline jobs to complete
2. When complete: full analysis with 3-seed stats per game
3. Begin Phase 2 implementation starting with most promising directions
4. Consider fixing the return metric (accumulate across lives) for Phase 2 scripts

---
**[2026-03-18 05:36 UTC]**

## Session 2 Final Summary (closing at 36/90 results)

### Baselines collected: 36/90 (PPO 17/45, PQN 19/45)
- All nibi jobs completed (many 'disappeared' from SLURM but all had valid CSV results)
- Jobs completing at ~2h walltime each, remaining ~54 jobs across rorqual/narval/fir
- 1 resubmitted job (Phoenix-v5_s3 h001 on narval) — original also completed

### Return Metric Issue (IMPORTANT):
RecordEpisodeStatistics resets on infos['terminated'] which fires on every life loss.
So info['r'] gives LAST-LIFE unclipped Atari return, not full-game return.
All algorithms have same bias — relative comparisons are valid. Documented in experiments.csv.

### Preliminary PPO vs PQN (12 games, mostly 1 seed):
Score: PPO 3 / PQN 7 / Tie 2
- PPO dominates: Phoenix (9x), NameThisGame (1.6x), DoubleDunk
- PQN dominates: Amidar (15x), SpaceInvaders (2x), plus several marginal wins
- Both fail: Breakout, MontezumaRevenge, Venture, Enduro (all ~0)

### Phase 2 Scripts Prepared (NOT submitted yet — waiting for baselines):
- h003: PPO + LayerNorm (ppo_atari_envpool_layernorm.py)
- h004: PQN + NaP weight projection (pqn_atari_envpool_nap.py) 
- h005: PPO + LayerNorm + CHAIN-SP (ppo_atari_envpool_chainsp.py)
All scripts synced to all 4 clusters.

### Literature Research:
- CHAIN-SP paper: CHAIN + LayerNorm most effective for PPO plasticity on Atari
- Plasticine benchmark: 13+ methods; normalization + churn reduction = best
- On-policy plasticity: soft shrink+perturb + LayerNorm optimal
- CReLU and Plasticity Injection FAILED in on-policy settings (avoid these)

### Next Session TODO:
1. Process remaining ~54 baseline results
2. Full 3-seed analysis when baselines complete
3. Submit Phase 2 pilots (h003/h004/h005) — 15 games x 1 seed each
4. Consider additional hypotheses: network sparsity, SPR aux loss, RND for hard-exploration games

---
**[2026-03-18 06:02 UTC]**

## Session 3: Phase 2 Broad Exploration Launched

### Results processed:
- 6 new baseline results (MontezumaRevenge s3, BattleZone s3, SpaceInvaders s1/s3, MsPacman s2, Enduro s3)
- Total: 42/90 baseline results in experiments.csv (19 PPO, 23 PQN)

### Updated PPO vs PQN comparison (13 games, partial seeds):
Score: PPO 5 / PQN 4 / Tie 4 on Q4 metric
- PQN dominates: Amidar (15x), BattleZone, SpaceInvaders (2x), PrivateEye
- PPO dominates: Phoenix (9x), NameThisGame (1.6x), Qbert, MsPacman, DoubleDunk
- Both fail: Breakout (~1.5), Enduro (0), MontezumaRevenge (0), Venture (0)
- Missing: Alien, Solaris (still running)

### Phase 2 pilots submitted (105 new jobs total):
h003: PPO + LayerNorm — 15 games × 1 seed (plasticity)
h004: PQN + NaP weight projection — 15 games × 1 seed (plasticity)
h005: PPO + CHAIN-SP — 15 games × 1 seed (plasticity + churn reduction)
h006: PPO + Symlog value transform — 15 games × 1 seed (reward processing, DreamerV3-style)
h007: PPO + Soft Shrink-and-Perturb — 15 games × 1 seed (plasticity/regularization)
h008: PQN + LSTM — 15 games × 1 seed (memory architecture)
h009: PPO + RND exploration — 15 games × 1 seed (exploration/intrinsic motivation)

### Category coverage:
- Cat 5 (Architecture/Plasticity): h003, h004, h005, h007
- Cat 7 (Reward/return processing): h006
- Cat 5 (Memory): h008
- Cat 6 (Exploration): h009

### Scripts created this session:
- cleanrl/ppo_atari_envpool_symlog.py (h006) — symlog/symexp value targets
- cleanrl/ppo_atari_envpool_sp.py (h007) — soft shrink-and-perturb after optimizer steps
- cleanrl/pqn_atari_envpool_lstm_v2.py (h008) — adapted existing PQN LSTM with pipeline changes
- cleanrl/ppo_rnd_envpool_v2.py (h009) — adapted existing PPO-RND with pipeline changes

### Active jobs: ~49 baselines + 105 Phase 2 = ~154 total across 4 clusters
All distributed evenly across nibi, fir, narval, rorqual.

### Still uncovered research directions for future hypotheses:
- Cat 1: Off-policy/hybrid methods
- Cat 3: Distributional RL (C51 exists in cleanrl)
- Cat 8: Modern optimizers (Lion, schedule-free Adam)
- Cat 9: Representation learning (SPR, BYOL, DrQ)
- Cat 10: Ensemble methods

### Next session TODO:
1. Process completed baseline and Phase 2 results
2. Full baseline analysis once h001/h002 complete
3. Analyze Phase 2 pilot results — identify winners
4. Prepare more hypotheses from uncovered categories
5. If any Phase 2 method shows promise: run full 3-seed evaluation

---
**[2026-03-18 06:24 UTC]**

## Session 4: h008 Bug Fix + Baseline Processing (57/90 baselines)

### h008 Bug Fix (CRITICAL)
Found and fixed a Conv2d channel bug in pqn_atari_envpool_lstm_v2.py: the CNN expected 1 input channel but envpool returns 4-frame-stacked observations (shape 128x4x84x84). The h008-amidar-s1 job on rorqual crashed with this error. All 14 remaining h008 jobs were cancelled and 15 resubmitted with the fix.

### Baseline Results Update
- h001 (PPO): 28/45 results, 14/15 games (missing Solaris — all 3 seeds still running)
- h002 (PQN): 29/45 results, 15/15 games
- Total: 57/90 baseline results collected

### PPO vs PQN Head-to-Head (14 common games, matched seeds):
Score: PPO 6 / PQN 4 / Tie 4
- PPO dominates: Phoenix (9x), NameThisGame (1.6x), Alien, DoubleDunk, MsPacman, PrivateEye
- PQN dominates: Amidar (16x), BattleZone (2x), SpaceInvaders (2x), Breakout (marginal)
- Both fail: Enduro (0), MontezumaRevenge (0), Venture (0)
- Close: Qbert (tie)

### IMPORTANT: Old Iteration Data Contamination
The results/CLUSTER/ directories contain CSV files from BOTH the current research iteration AND a previous iteration. The old iteration used experiment IDs like 'h003-game-sN' which happen to match the current iteration's experiment IDs. The parser MUST filter by DB experiment IDs AND only accept results for jobs with completed/disappeared status. For h001/h002, the 'Game-v5_sN' format is unique to the current iteration. For h003-h009, results should only be trusted once DB shows them as completed.

### Current State
- h001/h002: ~48 jobs still running across 4 clusters
- h003-h009: all 15-game pilots running (h008 resubmitted after fix)
- Total active: ~183 jobs across 4 clusters
- No action needed — waiting for completions

### Next Session TODO
1. Process remaining baseline results when h001/h002 complete
2. Process h003-h009 pilot results when they complete
3. Full PPO vs PQN baseline analysis once all 90 baselines are in
4. Identify which Phase 2 methods beat baselines
5. Begin Phase 3 hypotheses for remaining uncovered directions

---
**[2026-03-18 06:54 UTC]**

## Session 5: Phase 2 Output-Dir Bug Fix + Baseline Update (72/90)

### CRITICAL BUG FOUND & FIXED
All 105 Phase 2 pilot jobs (h003-h009) had --output-dir /output in commands, but container only binds /runs. Training completed but CSV saving crashed with 'Read-only file system: /output'. All jobs wasted ~105 GPU-hours.

**Fix:** Cancelled all 105 running Phase 2 jobs across 4 clusters. Resubmitted all 105 without --output-dir flag (scripts default to /runs, which is correct). All resubmissions successful.

### Baseline Results Updated: 72/90 (38 PPO + 34 PQN)
15 new results found and parsed (mostly from fir and narval pulls).
All 15 games now have at least 2 seeds for both PPO and PQN.

### PPO vs PQN Head-to-Head (15 games):
Score: PPO 8 / PQN 4 / Tie 3
Average Q4: PPO=540 PQN=396

PPO dominates: Phoenix (9x), Solaris (4x), NameThisGame (1.6x), MsPacman, DoubleDunk, PrivateEye, Qbert, Alien
PQN dominates: Amidar (16x), BattleZone (1.5x), SpaceInvaders (2x), Breakout (marginal)
Both fail: Enduro (0), MontezumaRevenge (0), Venture (0)

### Still missing baselines (20/90):
h001: 7 more seeds needed (1 each for several games)
h002: 11 more seeds needed

### Current State:
- h001/h002: ~47 baseline jobs still running
- h003-h009: 105 Phase 2 pilots resubmitted (all running)
- Total active: ~152 jobs across 4 clusters

### Next Session TODO:
1. Process remaining baseline results when h001/h002 complete
2. Process Phase 2 pilot results when h003-h009 complete (this time CSVs will save correctly!)
3. Full baseline analysis once all 90 baselines are in
4. Identify Phase 2 winners to scale to full 3-seed evaluation

---
**[2026-03-18 07:35 UTC]**

## Session 6: Baselines COMPLETE + 6 New Hypotheses Submitted

### Baselines Complete: 90/90 (45 PPO + 45 PQN)
Parsed 18 remaining baseline CSVs. All 15 games × 3 seeds for both algorithms.

### FULL BASELINE ANALYSIS (15 games × 3 seeds):
PPO avg Q4=550, PQN avg Q4=391. Score: PPO 7 / PQN 4 / Tie 4.

PPO dominates:
- Phoenix (9.2x: 796 vs 87)
- Solaris (5.7x: 2280 vs 399)
- NameThisGame (1.7x: 2348 vs 1381)
- MsPacman (1.3x: 319 vs 249)
- DoubleDunk (-17.9 vs -24.0)
- Alien (1.1x: 200 vs 178)
- PrivateEye (-135 vs -173)

PQN dominates:
- Amidar (14.5x: 31.7 vs 2.2!!)
- SpaceInvaders (1.9x: 287 vs 147)
- BattleZone (1.5x: 3296 vs 2150)
- Breakout (1.3x: 1.8 vs 1.4)

Both fail (Q4<1): Enduro(0), MontezumaRevenge(0), Venture(0)
Tie: Qbert (158 vs 151)

### Key Insights:
1. PQN's 14.5x advantage on Amidar is remarkable — Q-learning excels at maze navigation
2. PPO's 9.2x advantage on Phoenix — policy gradient better for reactive shooting
3. Both algorithms completely fail on hard exploration (Enduro, MontezumaRevenge, Venture)
4. Low variance across seeds for most games — results are stable
5. Breakout is very low for both (~1.5) — neither algorithm learns to play well

### Phase 2 Pilots Running (h003-h009):
All 105 jobs (7 hypotheses × 15 games × 1 seed) still running. CSVs in results/ are old-iteration contamination (confirmed by different column format). Must wait for current DB jobs to complete.

### NEW Phase 2 Hypotheses Submitted (h010-h015):
90 new jobs (6 hypotheses × 15 games × 1 seed) submitted across 4 clusters:
- h010: PPO + IMPALA CNN (Cat 5: Architecture — stronger encoder)
- h011: PQN + IMPALA CNN (Cat 5: Architecture — stronger encoder)
- h012: PPO + DrQ-style Random Shift Augmentation (Cat 9: Representation)
- h013: PPO + Spectral Normalization (Cat 5: Stability)
- h014: PPO + Entropy Annealing 0.03→0.001 (Cat 4: Policy Gradient)
- h015: PPO + PopArt Adaptive Value Normalization (Cat 7: Value)

### Active Jobs: ~106 running (h003-h009) + 90 submitted (h010-h015) = ~196 total
Distributed across nibi, fir, narval, rorqual.

### Category Coverage So Far:
- Cat 1 (Algorithmic paradigms): NOT YET — consider SAC-discrete, DQN variants
- Cat 2 (Experience management): NOT YET — consider replay buffers
- Cat 3 (Distributional RL): NOT YET — consider C51, Rainbow components
- Cat 4 (Policy gradient): h014 (entropy annealing)
- Cat 5 (Architecture/Plasticity): h003, h004, h005, h007, h008, h010, h011, h013
- Cat 6 (Exploration): h009 (RND)
- Cat 7 (Reward/value processing): h006 (symlog), h015 (PopArt)
- Cat 8 (Optimization): NOT YET — consider schedule-free Adam
- Cat 9 (Representation): h012 (DrQ aug)
- Cat 10 (Ensemble): NOT YET

### Next Session TODO:
1. Check h003-h009 pilot completions — analyze results
2. Check h010-h015 pilot completions — analyze results
3. Prepare hypotheses for remaining uncovered categories (Cat 1, 2, 3, 8, 10)
4. Once pilots complete: identify winners, scale to 3-seed full evaluation
5. Consider novel combinations of winning techniques

---
**[2026-03-18 08:02 UTC]**

## Session 7: Bug Fixes, Early Results Analysis, New Hypotheses (h016-h017)

### Bugs Found & Fixed
1. **h010 IMPALA CNN channel bug**: Conv2d expected 84 input channels due to `h,w,c = obs.shape` instead of `c,h,w`. All h010 rorqual jobs crashed. Fixed and resubmitted 15 jobs.
2. **h003-h009 running at 10M not 40M**: Submission commands omitted `--total-timesteps` flag, defaulting to 10M. h010-h015 were correctly submitted with 40M. The 10M results still provide directional signal.

### Phase 2 Early Results (h003/h004 at 10M steps)
**h003 (PPO + LayerNorm, 14/15 games at 10M):**
- avg Q4=503 at 10M vs PPO baseline avg Q4=550 at 40M
- 8 games already match/exceed 40M baseline with 4x less training!
- Notable: PrivateEye Q4=71 vs baseline -135 (sign reversal). NameThisGame exceeds baseline.
- Missing: SpaceInvaders

**h004 (PQN + NaP, 15/15 games at 10M):**
- avg Q4=380 at 10M vs PQN baseline avg Q4=391 at 40M
- 9 games match/exceed 40M baseline at only 10M!
- Notable: BattleZone +15%, Solaris +26%, Phoenix +7% over 40M baseline
- Only NameThisGame (68%) and PrivateEye (worse) below baseline

**CONCLUSION: Both LayerNorm (h003) and NaP (h004) dramatically improve sample efficiency.**

### h005 partial (PPO + CHAIN-SP): 3 games, looks similar to h003
Alien Q4=220 (vs base 200), MsPacman Q4=294 (vs 319), PrivateEye Q4=71 (vs -135)

### New Hypotheses Implemented and Submitted
- **h016: PPO + Network Sparsity** (Cat 5, ICML 2025 oral) — Random 50% pruning at init, static mask. 15 games × 1 seed @ 40M submitted.
- **h017: PPO + SPO/TV divergence** (Cat 4, ICLR 2025) — Replace PPO clipping with soft TV penalty. 15 games × 1 seed @ 40M submitted.

### Old Iteration Data Contamination Warning
Results directories (results/CLUSTER/) contain CSVs from BOTH old and current iterations. Old CSVs use format: env_id,seed,total_timesteps,mean_return_last_25,... Current CSVs use: env_id,seed,hypothesis_id,experiment_id,algorithm,...,q4_return,auc,final_avg20. The 'algorithm' column distinguishes current from old. h010-h015 CSVs from narval/nibi/fir are ALL from the old iteration.

### Active Jobs Summary
- h003-h009: ~107 running (10M steps, will finish soon)
- h010: 15 resubmitted (40M, fixed IMPALA bug)
- h011-h015: ~75 running (40M, various clusters)
- h016: 15 submitted (40M, network sparsity)
- h017: 15 submitted (40M, SPO/TV divergence)
- Total active: ~227 jobs

### Category Coverage Update
- Cat 1 (Off-policy): NOT YET
- Cat 2 (Replay): NOT YET
- Cat 3 (Distributional): NOT YET
- Cat 4 (Policy gradient): h014 (entropy anneal), h017 (SPO)
- Cat 5 (Architecture/Plasticity): h003, h004, h005, h007, h008, h010, h011, h013, h016
- Cat 6 (Exploration): h009 (RND)
- Cat 7 (Value processing): h006 (symlog), h015 (PopArt)
- Cat 8 (Optimization): NOT YET — schedule-free Adam, Fast TRAC are top priorities
- Cat 9 (Representation): h012 (DrQ aug)
- Cat 10 (Ensemble): NOT YET

### Literature Findings (new this session)
Top uncovered directions from recent literature:
- SPO/TV divergence (ICLR 2025) — implemented as h017
- Network Sparsity (ICML 2025 oral) — implemented as h016
- Fast TRAC optimizer (NeurIPS 2024) — drop-in optimizer wrapper, tested on Atari
- Schedule-Free AdamW (NeurIPS 2024) — untested in RL, high novelty
- PFO (Proximal Feature Optimization, NeurIPS 2024) — fixes PPO representation collapse
- Symmetric PPO (ICML 2025) — robust to non-stationary targets
- ADDQ (Adaptive Double Q, ICML 2025) — few-line change for PQN

### Next Session TODO
1. Process completed h003-h009 pilot results (10M) — full comparison table
2. Process h010-h015 pilot results when they complete (40M)
3. Process h016-h017 pilot results when they complete
4. Implement more hypotheses for uncovered categories (Cat 1, 2, 3, 8, 10)
5. If h003/h004 confirm at 40M: begin combination experiments (LayerNorm + NaP + IMPALA)
6. Implement: Fast TRAC optimizer, Schedule-Free Adam, PFO for PPO

---
**[2026-03-18 09:03 UTC]**

## Session 8: Phase 2 Analysis + New Hypotheses (h018-h020)

### h003/h004/h005 Complete Results at 10M
All three plasticity pilots are complete (15/15 games at 10M steps):

**h003 (PPO + LayerNorm, 10M):** Avg Q4=479 vs PPO(40M)=550
- 8/15 games >= 40M baseline at 1/4 training (4x sample efficiency!)
- PrivateEye Q4=71 vs -135 (sign reversal — huge win)
- Weak: Solaris (0.51x), Phoenix (0.79x)
- Score vs PPO(40M): wins 3, loses 5, ties 7

**h004 (PQN + NaP, 10M):** Avg Q4=380 vs PQN(40M)=391
- BEST sample efficiency: 12/15 games >= 40M baseline at 10M!
- BattleZone +15%, Solaris +26%, Phoenix +7%
- Only NameThisGame below 75%
- Score vs PQN(40M): wins 6, loses 4, ties 5

**h005 (PPO + CHAIN-SP, 10M):** Avg Q4=515 vs PPO(40M)=550
- 7/15 games >= 40M baseline at 10M
- Phoenix 968 vs base 796 (1.22x at 10M!), NameThisGame 2567 vs 2348
- Score vs PPO(40M): wins 6, loses 6, ties 3

### Actions Taken
1. Added h003/h004/h005 10M results (45 rows) to experiments.csv
2. Updated hypotheses.csv with detailed analysis
3. Submitted h003/h004/h005 FULL 3-seed evaluation at 40M (135 jobs across 4 clusters)
4. Implemented 3 new hypotheses:
   - h018: PPO + Schedule-Free AdamW (Cat 8 Optimization) — eliminates LR schedule
   - h019: PPO + Muon Optimizer (Cat 8 Optimization) — Newton-Schulz gradient orthogonalization
   - h020: PPO + Dueling Architecture (Cat 3/5) — separate V+A streams
5. Rebuilt container with schedulefree dependency, pushing to all clusters
6. Submitted h019/h020 pilots (30 jobs)
7. h018 batch ready, will submit after container image push completes

### Current Active Jobs
- h003-h005 (40M 3-seed): 135 new jobs submitted
- h006-h009 (10M pilots): 60 still running (should finish soon)
- h010-h017 (40M pilots): ~120 still running
- h019-h020 (40M pilots): 30 new jobs submitted
- h018: pending container image push

### Category Coverage Update
- Cat 1 (Off-policy): NOT YET
- Cat 2 (Replay): NOT YET
- Cat 3 (Distributional/Value): h006 (symlog), h015 (PopArt), h020 (dueling)
- Cat 4 (Policy gradient): h014 (entropy anneal), h017 (SPO)
- Cat 5 (Architecture/Plasticity): h003, h004, h005, h007, h008, h010, h011, h013, h016
- Cat 6 (Exploration): h009 (RND)
- Cat 7 (Value processing): h006 (symlog), h015 (PopArt)
- Cat 8 (Optimization): h018 (schedule-free), h019 (Muon) — NEW
- Cat 9 (Representation): h012 (DrQ aug)
- Cat 10 (Ensemble): NOT YET

### Key Findings So Far
1. LayerNorm (h003) and NaP (h004) show DRAMATIC sample efficiency gains
2. CHAIN-SP (h005) has highest avg Q4 at 10M (515) — benefits on Phoenix and NameThisGame
3. Both h003 and h005 fix PrivateEye (71 vs -135 for PPO base)
4. Still need 40M results for definitive comparison

### Next Session TODO
1. Check h003-h005 40M results when complete — definitive comparison
2. Process h006-h009 10M pilot results (should be done soon)
3. Process h010-h017 40M pilot results
4. Submit h018 after container push
5. Analyze all pilots and identify top performers for combination experiments
6. Implement remaining categories: Cat 1 (off-policy), Cat 2 (replay), Cat 10 (ensemble)
7. Begin thinking about novel combinations of winning techniques

---
**[2026-03-18 09:11 UTC]**

### Session 8 Addendum: h018 submitted
Container with schedulefree pushed to all 4 clusters. h018 (PPO + Schedule-Free AdamW) pilot submitted: 15 games × 1 seed across rorqual/narval/nibi/fir.

Total active jobs after this session: ~370
- h003-h005 (40M 3-seed eval): 135 jobs
- h006-h009 (10M pilots): ~60 running
- h010-h017 (40M pilots): ~120 running 
- h018 (schedule-free pilot): 15 jobs
- h019 (Muon pilot): 15 jobs
- h020 (dueling pilot): 15 jobs

---
**[2026-03-18 09:54 UTC]**

## Session 9: Container Fix + Partial Results (h006/h007)

### Rorqual Container Corruption — FIXED
Session 8's container rebuild (for schedulefree dep) corrupted the .sif on rorqual (~242KB short). 33 h003/h004/h005 40M 3-seed eval jobs failed with 'squashfuse_ll: Something went wrong trying to read the squashfs image'.
- Re-pushed image (20 min SCP)
- Cancelled/identified all 33 failed jobs
- Resubmitted all 33 distributed across all 4 clusters
- Jobs on other clusters (narval, nibi, fir) unaffected

### New Results Added: 12 rows (h006: 7/15, h007: 5/15 at 40M)

**h006 (PPO + Symlog, 40M pilot, 7/15 games):**
- Score vs PPO: 3W / 1L / 3T
- Wins: Phoenix (+60), PrivateEye (+22), SpaceInvaders (+10)
- Loss: NameThisGame (-300) — significant regression
- 8 games still running/pending

**h007 (PPO + Shrink-and-Perturb, 40M pilot, 5/15 games):**
- Score vs PPO: 1W / 2L / 2T
- Win: PrivateEye Q4=34 vs -135 (sign reversal!)
- Losses: MsPacman (-53), NameThisGame (-86)
- 10 games still running/pending

### Current Job Status (post-reconcile + resubmission)
- Total active: ~250+ jobs across 4 clusters
- h003/h004/h005: 40M 3-seed evals running (33 resubmitted)
- h006/h007: 40M pilots partially done (8+10 games remaining)
- h008-h009: 40M pilots running (on correct image, pre-rebuild)
- h010-h017: 40M pilots running
- h018-h020: pending/early running

### Results Bank: 148 rows (90 baselines + 45 h003-h005@10M + 7 h006 + 5 h007 + 1 h005 duplicate)

### Next Session TODO:
1. Process h003/h004/h005 40M results when complete — definitive comparison
2. Process remaining h006/h007 pilot results
3. Process h008-h009 pilot results (PQN+LSTM, PPO+RND)
4. Process h010-h020 pilot results as they complete
5. Analyze all Phase 2 pilots, identify winners
6. Begin combination experiments with winning techniques
7. Consider more hypotheses for uncovered categories (Cat 1 off-policy, Cat 2 replay, Cat 10 ensemble)

---
**[2026-03-18 10:09 UTC]**

## Session 10: Phase 2 Analysis — 126 New Results

### Results Processed
- 126 new rows added to experiments.csv (total: 273 rows)
- Pulled from all 4 clusters, cross-referenced with DB to filter old-iteration contamination
- Ran reconcile: 296 still active, 234 completed/disappeared, 134 cancelled

### KEY FINDINGS — Phase 2 Pilot Rankings

**TIER 1 — Best performers (recommend full 3-seed eval):**
1. **h013 (Spectral Norm): 4W/0L/5T on 9/15 games** — ZERO LOSSES. Phoenix +19%, Solaris +34%, Amidar +40%. Best stability-enhancing technique. Still 6 games running.
2. **h007 (Shrink-and-Perturb): 4W/2L/8T on 14/15 games** — Solaris massive +78% (4054 vs 2280!), BattleZone +16%. Only loses: MsPacman (-17%), PrivateEye. Near-complete pilot.
3. **h012 (DrQ augmentation): 3W/1L/4T on 8/15 games** — BattleZone +33%, Phoenix +11%. Only loss: MsPacman. 7 games running.

**TIER 2 — Safe/promising:**
4. **h015 (PopArt): 2W/0L/5T on 7/15 games** — No losses, mostly ties. Safe but modest.
5. **h020 (Dueling): 7W/5L/3T on 15/15 games** — COMPLETE. Most raw wins (BattleZone 2.4x!) but also many losses (Phoenix 0.07x, Solaris 0.19x). High variance.

**TIER 3 — Neutral/bad:**
6. **h006 (Symlog): 2W/3L/9T on 14/15 games** — Neutral. No dramatic effects.
7. **h003 (LayerNorm 40M): 5W/7L/3T on 15/15 games** — DISAPPOINTING. Catastrophic at 40M: Phoenix 0.01x, SpaceInvaders 0.03x, Qbert 0x. 10M results were misleading — LayerNorm causes late-training collapse.
8. **h018 (Schedule-Free): 4W/7L/4T on 15/15 games** — BAD. Phoenix 0.02x, Solaris 0x. Closed.

**TIER 4 — Too few results (3/15 games only — all show Amidar/Breakout/Qbert):**
h008 (PQN LSTM), h009 (RND), h010 (IMPALA), h011 (PQN IMPALA), h016 (Sparse), h017 (SPO), h019 (Muon)

### Critical Insight: Phoenix/Solaris/SpaceInvaders as Discriminators
- PPO baseline is STRONG on Phoenix (796), Solaris (2280), SpaceInvaders (147)
- Most modifications HURT these games — they're the hardest to maintain
- h013 (Spectral Norm) is special: it IMPROVES Phoenix/Solaris while maintaining everything else
- h007 (S&P) excels specifically on Solaris (+78%)
- Anything that hurts Phoenix/Solaris by >50% should be considered harmful

### Metric Format Issue (IMPORTANT for next sessions)
- Baselines (h001/h002): use q4_return metric (robust, last 25% of all episodes)
- Many Phase 2 scripts: use mean_return_last_25 (old format, noisy, only last 25 episodes)
- Comparison uses final_avg20 from baselines when comparing with old-format results
- Scripts h006, h012, h013, h015 have proper q4_return format
- Scripts h003(40M), h008-h011, h014, h016-h020 have old format
- Consider fixing CSV output in remaining scripts before combination experiments

### Actions Taken
- Resubmitted h006-alien-s1 (disappeared on fir, no output) → narval job 57940886
- h010 results (3 games) may be from pre-bugfix batch — marked as uncertain
- Updated hypotheses.csv with full analysis notes

### Active Jobs: ~296 across 4 clusters
- h003/h004/h005: 40M 3-seed evals (135 total, many running)
- h006: 1 resubmitted (alien-s1)
- h007: 1 remaining (enduro-s1)
- h008-h017: 40M pilots (6-15 games each remaining)
- h018: COMPLETE
- h019-h020: 40M pilots running

### Next Session TODO
1. Process newly completed results from h006-h017 pilots
2. Full analysis of h013 and h007 when their 15/15 pilots complete
3. Process h004/h005 40M 3-seed results (check if LayerNorm collapse affects NaP/CHAIN-SP)
4. Submit h013+h007 combination experiment if both confirm strong results
5. Fix CSV output format in remaining scripts for future consistency
6. Still need hypotheses for: Cat 1 (off-policy), Cat 2 (replay), Cat 10 (ensemble)

---
**[2026-03-18 10:30 UTC]**

## Session 11: Results Processing + h007 3-Seed Eval Submitted

### Results Processed
- 21 new rows added to experiments.csv (total: 294 rows)
- New results for: h006 (1 game), h009 (4 games), h012 (5 games), h013 (3 games), h014 (5 games), h015 (2 games), h017 (1 game)
- Pulled results from all 4 clusters and ran reconcile

### Updated Pilot Analysis (key hypotheses with 8+ games):

**TIER 1 — Top performers:**
1. **h007 (Shrink-and-Perturb): 4W/0L/10T on 14/15** — BEST W/L ratio. ZERO losses! Solaris +78% (4054 vs 2280), PrivateEye sign reversal (34 vs -135), Breakout 3x. Missing only Enduro (always 0). **3-SEED EVAL SUBMITTED (30 jobs).**
2. **h012 (DrQ augmentation): 3W/1L/9T on 13/15** — PrivateEye Q4=502 (!!) is best ever. BattleZone +33%. Only Breakout loss. Missing: SpaceInvaders, Venture.
3. **h013 (Spectral Norm): 3W/1L/8T on 12/15** — Solaris +34%, Phoenix +19%, Breakout 3.6x. One Qbert loss (125 vs 158). Missing: Alien, MontezumaRevenge, PrivateEye.

**TIER 2 — Decent:**
4. **h014 (Entropy Anneal): 3W/1L/6T on 10/15** — PrivateEye improved, Breakout 3x. MsPacman -24% loss. 5 games still running.
5. **h006 (Symlog): 0W/1L/14T on 15/15 — CLOSED.** Essentially neutral. Only MsPacman loss.

**TIER 3 — Too early:**
h008-h011, h015-h017, h019 all have <10/15 games. Need more data.

### IMPORTANT FINDING: Amidar Seed=1 Artifact
Almost ALL PPO modifications get Amidar=31 with seed=1, while PPO baseline gets 2.0. This is a seed-specific effect — PQN baseline also gets ~32. Techniques that DON'T fix Amidar with seed=1: LayerNorm (h003), CHAIN-SP (h005), symlog (h006), spectral norm (h013), PopArt (h015). The Amidar 'win' from a single seed is not informative — 3-seed eval will clarify.

### Actions Taken
1. Parsed 21 new experiment results into experiments.csv
2. Updated hypotheses.csv with refined analysis (h006 closed, h007/h012/h013/h014 updated)
3. Submitted h007 3-seed evaluation: 30 jobs (15 games × seeds 2,3) across all 4 clusters
4. Resubmitted 2 disappeared h017 jobs (phoenix-s1, privateeye-s1) that had no output
5. Reconciled DB state

### Active Jobs: ~321 across 4 clusters
- h003-h005: 40M 3-seed evals (~93 running/pending)
- h007: 30 NEW 3-seed eval jobs just submitted
- h008-h017, h019: pilots in progress (~168 running)
- h018/h020: complete (closed)

### Next Session TODO
1. Process h007 3-seed results when complete — definitive comparison with PPO baseline
2. Complete h012/h013 pilots (1-3 games remaining each) → submit 3-seed evals
3. Process h004/h005 40M 3-seed results (check for late-training collapse like h003)
4. Process remaining pilots (h008-h011, h014-h017, h019)
5. Plan combination experiments: h007+h012, h007+h013, h007+h012+h013
6. Still need hypotheses for: Cat 1 (off-policy), Cat 2 (replay), Cat 10 (ensemble)
7. Consider novel research directions beyond just combining techniques

---
**[2026-03-18 10:55 UTC]**

## Session 12: Results Processing + 3-Seed Eval Submissions

### Results Processed
- 28 new rows added to experiments.csv (total: 277 rows)
- New results from: h008(2), h009(5), h012(2), h013(3), h014(2), h015(2), h016(7), h017(5)
- Pulled from all 4 clusters

### h003 Full 3-Seed Analysis — CLOSED as CATASTROPHIC
h003 (PPO+LayerNorm) 45/45 at 40M: 3W/7L/5T. CONFIRMED late-training collapse:
- Phoenix: 0.27x (seed 2=0, seed 3=20 vs base 796)
- Qbert: 0.31x (seeds 2,3 collapse to 0)
- Solaris: 0.32x, SpaceInvaders: 0.36x, PrivateEye: 4.77x worse
- Only Alien (1.85x) and BattleZone (1.26x) are genuine wins
- LayerNorm is definitively harmful for long PPO training

### Updated Pilot Rankings (all 15 games where COMPLETE)

**TIER 1 — Best performers:**
1. h013 (Spectral Norm): 5W/1L/9T on 15/15 — EXCELLENT. Solaris +34%, Phoenix +19%, Breakout 3.7x. Only Qbert -21%.
2. h007 (Shrink&Perturb): 5W/1L/8T on 14/15 — Solaris +78%, BattleZone +16%. Only MsPacman -17%.
3. h012 (DrQ Augment): 3W/2L/10T on 15/15 — PrivateEye Q4=502 (BEST!), BattleZone +33%.

**TIER 2 — Promising but incomplete:**
4. h016 (Sparsity): 4W/0L/6T on 10/15 — ZERO LOSSES! Missing 5 key games.
5. h014 (Entropy Anneal): 3W/1L/8T on 12/15 — Breakout 3x. Missing 3 games.

**NOTE: Amidar seed=1 artifact:** Nearly ALL PPO modifications get Amidar~31 with seed=1 (PPO base=2.2). This is a seed-specific effect, NOT a real technique advantage.

### Actions Taken
1. Submitted h012 3-seed eval: 30 jobs (15 games × seeds 2,3) across 4 clusters
2. Submitted h013 3-seed eval: 30 jobs (15 games × seeds 2,3) across 4 clusters
3. Resubmitted 14 disappeared pilot gaps:
   - h008: Solaris (1)
   - h009: Enduro, Solaris (2)
   - h014: Enduro, SpaceInvaders (2)
   - h015: NameThisGame (1)
   - h016: DoubleDunk, NameThisGame, Phoenix (3)
   - h019: BattleZone, MontezumaRevenge, MsPacman, PrivateEye, Venture (5)
4. Updated hypotheses.csv with full analysis
5. h020 closed as HIGH VARIANCE (5W/4L, Phoenix 0.08x, Solaris 0.15x)

### Active Jobs After This Session
- h004/h005: 40M 3-seed evals (RUNNING)
- h007: 3-seed eval (30 jobs RUNNING, submitted session 11)
- h008: 11 running + 1 resubmitted (12 active)
- h009: 2 running + 2 resubmitted (needs Venture from active)
- h010/h011: 15 running each
- h012: 30 NEW 3-seed eval just submitted
- h013: 30 NEW 3-seed eval just submitted + 1 pilot still running
- h014-h016: partial pilots running + resubmits
- h017: 8 running
- h018: CLOSED
- h019: 9 running + 5 resubmitted
- h020: CLOSED
- Total: ~350+ active jobs

### Critical Next Steps
1. When h007/h012/h013 3-seed evals complete: definitive ranking of top techniques
2. When h004/h005 3-seed evals complete: check if CHAIN-SP shares h003's collapse
3. Complete all remaining pilots (h008-h017, h019)
4. Plan combination experiments: h007+h012, h007+h013, h012+h013 (top 3 techniques combined)
5. Still need to explore: Cat 1 (off-policy), Cat 2 (replay), Cat 10 (ensemble)
6. Consider novel research directions beyond combining techniques

---
**[2026-03-18 11:09 UTC]**

## Session 13: Results Processing + Corrected Analysis + Gap Resubmissions

### Results Processed
- Pulled results from all 4 clusters
- Cleaned experiments.csv: removed old-run data (h001B, h002B, h021-h045), deduplicated baselines, fixed q4_return from old-format CSVs
- Final experiments.csv: 396 rows (h001-h020 only)
- Updated 21 entries with proper q4_return from CSV files (were using mean_return_last_25 proxy)

### CRITICAL FINDING: Previous analysis was overoptimistic
Earlier sessions compared single-seed pilots (seed=1) against single-seed baselines. With proper 3-seed baseline means, the picture changes DRAMATICALLY:
- PPO baseline 3-seed means: Alien=198, BattleZone=3700, Solaris=2615, Phoenix=796, NameThisGame=2161
- Amidar seed=1 artifact: nearly ALL PPO modifications get ~31 vs baseline 1.4 — NOT a real win

### FAIR RANKINGS (excluding Amidar artifact, using 3-seed baseline):

**TIER 1 — Zero losses:**
- h017 (SPO): 3W/0L/6T on 10/15 — BEST! Breakout 1.2x, PrivateEye 4.2x, Phoenix 1.1x. 5 games still running.

**TIER 2 — More wins than losses (but with caveats):**
- h005 (CHAIN-SP): 4W/5L/5T — Phoenix/Qbert/NameThisGame wins but Solaris 0.5x collapse
- h004 (NaP): 4W/5L/5T — Breakout/SpaceInvaders wins but Solaris 0.2x, Phoenix 0.1x COLLAPSE
- h013 (Spectral Norm): 3W/4L/7T — Solaris/NameThisGame/Phoenix wins but BattleZone 0.5x

**TIER 3 — Minimal or negative impact:**
- h007 (S&P): 1W/3L/10T — ONLY Solaris win. Disappointing vs earlier analysis.
- h012 (DrQ): 1W/3L/10T — PrivateEye massive -11.3x loss
- h016 (Sparsity): 1W/2L/9T — Safe but minimal
- h014 (Entropy Anneal): 2W/4L/7T — More losses than wins
- h015 (PopArt): 1W/3L/8T — Minimal
- h009 (RND): 1W/5L/7T — BAD

**CLOSED as BAD:** h003 (LayerNorm collapse), h006 (neutral), h018 (Schedule-Free), h020 (high variance)

### Gap Resubmissions (7 jobs)
- h016-solaris-s1 (narval, 57942639) — disappeared pilot gap
- h016-mspacman-s1 (nibi, 10528391) — disappeared pilot gap
- h017-doubledunk-s1 (fir, 28236637) — disappeared pilot gap
- h004-enduro-s2 (narval, 57942640) — disappeared 3-seed eval
- h004-montezumarevenge-s3 (nibi, 10528684) — disappeared 3-seed eval
- h004-namethisgame-s3 (fir, 28236642) — disappeared 3-seed eval
- h004-phoenix-s2 (rorqual, 8527980) — disappeared 3-seed eval

### Active Jobs Summary
- ~318 active jobs across all clusters (242 running + 76 pending)
- 3-seed evals: h004(36), h005(45), h007(31), h012(30), h013(31) — all early stage
- Pilots still completing: h008(6/15), h010(3/15), h011(3/15), h014(14/15), h015(13/15), h016(13/15), h017(10/15), h019(3/15)

### Key Insight
h017 (SPO — soft TV divergence replacing PPO clipping) is the ONLY technique with zero losses across 10+ games. If it maintains this as remaining 5 games complete, it deserves immediate 3-seed evaluation. SPO is from ICLR 2025 and represents a genuine algorithmic innovation.

### Next Session TODO
1. Check h017 pilot completion — if 15/15 with 0 losses, submit 3-seed eval immediately
2. Process 3-seed eval results for h004/h005/h007/h012/h013 as they come in
3. Complete remaining pilots: h008, h010, h011, h015, h016, h019
4. Still need hypotheses for: Cat 1 (off-policy), Cat 2 (replay), Cat 10 (ensemble)
5. Consider novel combinations if h017 confirms strong: h017+h013 (SPO+SpectralNorm), h017+h007 (SPO+S&P)

---
**[2026-03-18 11:20 UTC]**

## Session 14: CRITICAL Analysis Bug Fix + Results Processing

### MAJOR FINDING: Ratio-based analysis had a negative-value bug
Previous sessions used ratio = hypothesis_q4 / baseline_q4 to classify wins/losses. This is WRONG for negative values:
- PrivateEye baseline h001 = -44.5. h012 scores 502 → ratio = -11.3 → wrongly classified as LOSS. Actually a MASSIVE WIN (+1228%)!
- h017 PrivateEye = -188 → ratio = 4.2 → wrongly classified as WIN. Actually a LOSS (-322% worse than baseline).

Fixed to use absolute return difference: (hypothesis - baseline) / max(|baseline|, 10). Higher return = better, always.

### CORRECTED RANKINGS (excl Amidar artifact):
**TIER 1:**
- h013 (Spectral Norm): 4W/3L/7T — Most wins. NameThisGame +18%, Phoenix +19%, PrivateEye (43 vs -44), Solaris +17%. BUT BattleZone -50%.
- h008 (PQN+LSTM): 3W/0L/2T — Zero losses on 5/14 games! Breakout, Phoenix, Solaris wins. VERY promising if pattern holds.

**TIER 2:**
- h004 (PQN+NaP): 3W/2L on 10/14. MsPacman/NameThisGame/Phoenix wins. BattleZone/PrivateEye losses.
- h012 (DrQ): 2W/2L — PrivateEye massive WIN (502 vs -44). BattleZone/MsPacman losses.
- h007 (S&P): 2W/2L — PrivateEye went positive (34 vs -44)! Solaris +55%. BattleZone/MsPacman losses.

**DOWNGRADED:**
- h017 (SPO): 1W/1L on 9/14 — WAS '3W/0L BEST'. Only Phoenix win (+14%). PrivateEye is a LOSS. Still missing 5 games.

**NEWLY CLOSED:** h009 (1W/4L), h014 (1W/5L), h015 (1W/3L)

### Key Insight: BattleZone is the universal loss game
Almost every technique loses on BattleZone. Only h020 (Dueling) and h001 baseline perform well there.

### Results Processed
- 61 new 40M result CSVs ingested (19 updated, 42 added)
- Total experiments.csv: 438 rows
- Updated hypotheses.csv with corrected analysis

### Resubmissions (7 jobs)
- h017-alien-s1 (fir, 28237143) — pilot gap
- h017-battlezone-s1 (narval, 57942776) — pilot gap
- h004-alien-s3 (rorqual, 8528197) — 3-seed gap
- h004-privateeye-s1 (fir, 28237146) — 3-seed gap
- h004-solaris-s3 (narval, 57942778) — 3-seed gap
- h004-privateeye-s3 (rorqual, 8528220) — 3-seed gap
- h004-mspacman-s2 (fir, 28237154) — 3-seed gap

### Active Jobs: ~323 (240 running + 83 pending)
Major running batches: h003(35), h004(29+7), h005(31), h007(22), h010(15), h011(15), h012(16), h013(17), h017(7+2), h018(11), h019(11), h020(10)

### Next Session TODO
1. Process any newly completed results
2. Check h017 pilot completion (now only needs doubledunk, mspacman, solaris + resubmitted alien, battlezone)
3. Check h008 pilot progress — if still 0 losses, consider 3-seed eval
4. Monitor h010/h011 (IMPALA CNN) — early results very promising
5. Once more pilots complete, start thinking about COMBINATION experiments: h013+h012 (SpectralNorm+DrQ), h013+h007 (SpectralNorm+S&P)
6. Consider new hypotheses from unexplored categories: off-policy, ensemble, distributional RL
