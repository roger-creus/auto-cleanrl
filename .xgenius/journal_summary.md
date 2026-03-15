# Research Context

## Research Goal

# Research Goal

## Objective
Develop novel deep reinforcement learning algorithms that achieve state-of-the-art performance on Atari, surpassing PPO and PQN baselines. The research should produce publishable algorithmic innovations — not just hyperparameter tuning. Even though we will use PPO and PQN as baselines to beat only, the cleanrl codebase contains many other implementations of algorithms which you should explore and potentially draw inspiration from when building new contributions.

## Benchmark
**Environments (7-game Atari subset via envpool):**
- Amidar-v5
- Alien-v5
- Breakout-v5
- Enduro-v5
- PrivateEye-v5
- Solaris-v5
- BattleZone-v5
- DoubleDunk-v5
- NameThisGame-v5
- Phoenix-v5
- Qbert-v5
- SpaceInvaders-v5
- MsPacman-v5
- Venture-v5
- MontezumaRevenge-v5

**Training budget:** 40M environment steps per run.

**Evaluation metric:** Mean episode return over the last 25 training episodes. Report per-game scores and the mean across all 7 games.

**Seeds:** Each experiment must run 3 seeds (1, 2, 3) for statistical validity.

ALL JOBS NEED TO USE AT LEAST 1 GPU.

## Baselines
Run these first and cache their results before attempting any improvements:

1. **PPO (envpool):** `cleanrl/ppo_atari_envpool.py` — run on all 7 games × 3 seeds
2. **PQN (envpool):** `cleanrl/pqn_atari_envpool.py` — run on all 7 games × 3 seeds

These baseline results are the bar to beat. Every new algorithm must be compared against them.

END GOAL: State-of-the-art performance competitive with published methods, with a clear and publishable algorithmic narrative. **Do not stop iterating if you have done few changes and already increase the performance of PQN and PPO by a smallish margin. You have to continue to hill climb until you have done a full exploration of your hypothesis space.** For instance if we do one change like replace NatureCNN for ResNet and we get big performance gains, that is not too novel because just running the PQN baseline with a ResNet would give the same performance gains and that wouldn't be a novel research contribution.

## Research Directions to Explore

You MUST explore broadly across these categories. Do NOT get stuck in only on-policy or only model-free methods. The goal is to discover the best possible algorithm by combining insights from across the entire deep RL landscape.

### Research Axes

Each axis below lists existing techniques as **starting points only**. Your job is NOT to pick the best existing technique from each category. Your job is to understand WHY each technique works, identify its limitations, and **invent principled improvements**. The existing techniques are inspiration — the novel contribution must go beyond them.

Each axis includes "Where novelty lives" suggestions, but these are **only examples** — you are strongly encouraged to come up with your own novel research directions that we haven't thought of. The best contributions will likely be ideas that aren't on this list.

### 1. Algorithmic paradigms
*Starting points:* on-policy (PPO, A2C), off-policy (DQN, SAC-discrete), hybrid (ACER, V-trace), N-step returns
*Where novelty lives:* How can on-policy and off-policy be unified in a principled new way? Can you design a new objective that smoothly interpolates between them? Can you derive a new update rule from first principles that has better theoretical properties? What if the degree of off-policy-ness was learned rather than fixed?

### 2. Experience management and replay
*Starting points:* prioritized replay, large/small buffers, replay ratio tuning
*Where novelty lives:* Can you design a new prioritization scheme based on learning progress rather than TD error? What if the replay strategy was conditioned on the agent's current uncertainty? Can you invent a replay mechanism that specifically targets catastrophic forgetting in RL?

### 3. Value function design
*Starting points:* distributional RL (C51, QR-DQN, IQN), expectile regression, dueling architectures, Popart normalization
*Where novelty lives:* Can you design a new value decomposition that captures something distributional methods miss? What about a value function that explicitly models temporal structure? Can you derive a new Bellman operator with better contraction properties? What if the value function architecture was game-adaptive?

### 4. Policy optimization
*Starting points:* PPO clipping, trust regions, entropy regularization, off-policy corrections (V-trace, Retrace)
*Where novelty lives:* Can you design a new surrogate objective with tighter bounds than PPO's clipping? What about an adaptive trust region that expands/contracts based on training progress? Can you invent a policy update that is provably monotone-improving under weaker assumptions?

### 5. Architecture design
*Starting points:* ResNet, attention-augmented CNNs, memory (LSTM, GTrXL), spectral norm, network resets
*Where novelty lives:* Can you design a visual encoder specifically tailored for RL's non-stationary data distribution? What about an architecture that explicitly handles the different timescales in Atari (reaction time vs planning)? Can you invent a new normalization technique that addresses plasticity loss in RL specifically? What if the architecture adapted its capacity during training?

### 6. Exploration
*Starting points:* RND, ICM, count-based, noisy nets, entropy bonuses
*Where novelty lives:* Can you design an exploration bonus that is aware of the agent's current policy quality (explore more when stuck, less when improving)? What about exploration that targets specific types of states (e.g., high-reward-variance states)? Can you invent a principled way to combine multiple exploration signals?

### 7. Reward and return processing
*Starting points:* reward clipping, Popart, symlog, discount schedules
*Where novelty lives:* Can you design a return estimator that is robust to reward scale without losing information (unlike clipping)? What about a learned reward transformation? Can you invent a new way to mix TD and MC returns that adapts to the quality of the value function?

### 8. Optimization
*Starting points:* AdamW, Lion, Sophia, Muon, cosine schedules, gradient clipping
*Where novelty lives:* Can you design an optimizer that handles RL's non-stationary loss landscape better than Adam? What about gradient processing that accounts for the policy-value coupling? Can you invent a learning rate schedule that responds to training diagnostics (loss landscape curvature, gradient noise)?

### 9. Representation learning and auxiliary losses
*Starting points:* BYOL, VICReg, SPR, data augmentation (DrQ), forward/inverse dynamics
*Where novelty lives:* Can you design an auxiliary loss specifically for RL that captures reward-relevant features? What about a representation that disentangles controllable vs uncontrollable aspects of the environment? Can you invent a self-supervised objective that directly improves policy learning rather than just learning good features?

### 10. Ensemble and composition methods
*Starting points:* ensembles, mixture of experts, population-based training
*Where novelty lives:* Can you design a principled way to combine diverse agent behaviors into something better than any individual? What about a method that distills insights from failed experiments into a better initialization?

### 11. Hyperparameter optimization (LAST PHASE ONLY)
- Once algorithmic innovations are **FULLY established**, tune: learning rate, number of environments, GAE lambda, number of minibatches, update epochs, clipping epsilon, replay buffer size, target network update frequency, and any other relevant hyperparameter
- Sweep one hyperparameter at a time, not grid search (too expensive)

## Philosophy

**CRITICAL — What constitutes a novel contribution:**
Simply combining existing techniques (e.g., "PPO + ResNet + RND + cosine LR") is NOT novel research. That's engineering. Engineering is the FOUNDATION — build the best possible base system by combining winning techniques. But then the REAL work begins: inventing something new on top of that foundation.

A novel contribution means:
- A new objective function derived from a principled insight
- A new architectural component designed for a specific RL challenge
- A new training procedure motivated by analysis of failure modes
- A new theoretical connection that leads to a practical improvement
- Taking an existing idea and improving it in a principled way (not just tuning it)

**Your workflow should be:**
1. Build the best engineering baseline (combine best existing techniques)
2. Analyze WHERE and WHY the baseline fails on specific games
3. Formulate a hypothesis about the root cause
4. Design a novel solution motivated by that analysis
5. Test it — if it works, understand WHY and whether the principle generalizes
6. Iterate

- **Breadth first, depth second**: Test many different ideas quickly (2-3 games, 1 seed) before committing to full evaluations. Do not over-invest in the first thing that works.
- **Search the literature**: Use web search to look up recent papers, state-of-the-art methods, and implementation tricks. Understand what exists before trying to go beyond it.
- **Learn from failures**: A rejected hypothesis is still valuable. Record WHY it failed — this informs future directions and often points toward the real solution.

## Constraints
- **Single-file implementation** following CleanRL design philosophy (readable, self-contained)
- **Must use envpool** for Atari environment vectorization
- **40M env steps** per training run (no more, for fair comparison)
- **3 seeds** per experiment for statistical validity
- **External dependencies are allowed** — install optimizers, augmentation libraries, or anything else needed via pip in the Singularity container. If new deps are needed, rebuild the container and push it!

## Compute Budget
**You have access to a LOT of compute. Use it aggressively.** You have multiple clusters with H100 GPUs and jobs get scheduled quickly. Do not be conservative with experiments — submit large batches in parallel across all clusters. You can easily run 100+ jobs concurrently. The faster you run experiments, the faster you iterate and the more hypothesis space you cover. When in doubt, submit more experiments rather than fewer. Time spent waiting is time wasted.

## Experiment Protocol

### Phase 1: Baselines (do this first!)
1. Run PPO envpool on all 7 games × 3 seeds
2. Run PQN envpool on all 7 games × 3 seeds
3. Record all baseline results in the journal
4. Analyze which games each baseline struggles on — these are the targets

### Phase 2: Broad exploration
1. Start with the most promising directions based on baseline analysis
2. Implement a single algorithmic change at a time
3. Test on 2-3 games first (the ones baselines struggle on) × 1 seed
4. If promising: run full 7 games × 3 seeds
5. If not promising: record findings, move to next idea
6. **Explore ACROSS categories** — don't just iterate on architecture, also try replay buffers, exploration, reward processing, etc.
7. Combine successful innovations incrementally

### Phase 3: 7-game evaluation and paper-readiness (only after ALL hypothesis space has been thoroughly explored)
1. Run best algorithm on all 7 games × 3 seeds
2. Ablation study: remove each innovation one at a time to measure contribution
3. Generate learning curves and comparison tables
4. Write up the algorithmic narrative

### Final Phase
1. Run PPO and PQN envpool on full Atari57 suite on 40M env steps
2. Run your fully discovered best-performing algorithm on full Atari57 suite on 40M env steps
3. Flag any failure mode/inconsistency and continue to iterate if results don't translate to the full benchmark
4. Write-up full paper with all narrative, ablations, benchmark experiments and methodology

## Output Format
When recording results, always include:
- Per-game mean episode return (last 25 episodes) ± std across seeds
- Mean normalized score across all 7 games
- Training wall-clock time
- Any notable observations (training instability, specific games where it excels/fails)

# SOURCES

CleanRL docs: https://docs.cleanrl.dev/
Envpool atari docs: https://envpool.readthedocs.io/en/latest/env/atari.html
Fir cluster: https://docs.alliancecan.ca/wiki/Fir
Rorqual cluster: https://docs.alliancecan.ca/wiki/Rorqual
Nibi cluster: https://docs.alliancecan.ca/wiki/Nibi
MIG (multi-instance GPUS) docs: https://docs.alliancecan.ca/wiki/Multi-Instance_GPU


## Hypotheses

| ID | Hypothesis | Status | Conclusion |
|---|---|---|---|
| h001 | PPO baseline on 15-game Atari subset (40M steps, envpool, minimal action space) | proposed | - |
| h002 | PQN baseline on 15-game Atari subset (40M steps, envpool, minimal action space) | proposed | - |

## Experiments & Results

### h001: PPO baseline on 15-game Atari subset (40M steps, envpool, minimal action space)

- **e001** (rorqual, job 8404296): `python cleanrl/ppo_atari_envpool.py --env-id Amidar-v5 --see` [PENDING]
- **e002** (rorqual, job 8404297): `python cleanrl/ppo_atari_envpool.py --env-id Amidar-v5 --see` [PENDING]
- **e003** (rorqual, job 8404321): `python cleanrl/ppo_atari_envpool.py --env-id Amidar-v5 --see` [PENDING]
- **e004** (rorqual, job 8404326): `python cleanrl/ppo_atari_envpool.py --env-id Alien-v5 --seed` [PENDING]
- **e005** (rorqual, job 8404327): `python cleanrl/ppo_atari_envpool.py --env-id Alien-v5 --seed` [PENDING]
- **e006** (rorqual, job 8404330): `python cleanrl/ppo_atari_envpool.py --env-id Alien-v5 --seed` [PENDING]
- **e007** (rorqual, job 8404332): `python cleanrl/ppo_atari_envpool.py --env-id Breakout-v5 --s` [PENDING]
- **e008** (rorqual, job 8404333): `python cleanrl/ppo_atari_envpool.py --env-id Breakout-v5 --s` [PENDING]
- **e009** (rorqual, job 8404334): `python cleanrl/ppo_atari_envpool.py --env-id Breakout-v5 --s` [PENDING]
- **e010** (rorqual, job 8404335): `python cleanrl/ppo_atari_envpool.py --env-id Enduro-v5 --see` [PENDING]
- **e011** (rorqual, job 8404336): `python cleanrl/ppo_atari_envpool.py --env-id Enduro-v5 --see` [PENDING]
- **e012** (rorqual, job 8404337): `python cleanrl/ppo_atari_envpool.py --env-id Enduro-v5 --see` [PENDING]
- **e013** (rorqual, job 8404338): `python cleanrl/ppo_atari_envpool.py --env-id PrivateEye-v5 -` [PENDING]
- **e014** (rorqual, job 8404339): `python cleanrl/ppo_atari_envpool.py --env-id PrivateEye-v5 -` [PENDING]
- **e015** (rorqual, job 8404340): `python cleanrl/ppo_atari_envpool.py --env-id PrivateEye-v5 -` [PENDING]
- **e016** (nibi, job 10412748): `python cleanrl/ppo_atari_envpool.py --env-id Solaris-v5 --se` [PENDING]
- **e017** (nibi, job 10412749): `python cleanrl/ppo_atari_envpool.py --env-id Solaris-v5 --se` [PENDING]
- **e018** (nibi, job 10412750): `python cleanrl/ppo_atari_envpool.py --env-id Solaris-v5 --se` [PENDING]
- **e019** (nibi, job 10412751): `python cleanrl/ppo_atari_envpool.py --env-id BattleZone-v5 -` [PENDING]
- **e020** (nibi, job 10412752): `python cleanrl/ppo_atari_envpool.py --env-id BattleZone-v5 -` [PENDING]
- **e021** (nibi, job 10412753): `python cleanrl/ppo_atari_envpool.py --env-id BattleZone-v5 -` [PENDING]
- **e022** (nibi, job 10412754): `python cleanrl/ppo_atari_envpool.py --env-id DoubleDunk-v5 -` [PENDING]
- **e023** (nibi, job 10412755): `python cleanrl/ppo_atari_envpool.py --env-id DoubleDunk-v5 -` [PENDING]
- **e024** (nibi, job 10412756): `python cleanrl/ppo_atari_envpool.py --env-id DoubleDunk-v5 -` [PENDING]
- **e025** (nibi, job 10412757): `python cleanrl/ppo_atari_envpool.py --env-id NameThisGame-v5` [PENDING]
- **e027** (nibi, job 10412759): `python cleanrl/ppo_atari_envpool.py --env-id NameThisGame-v5` [PENDING]
- **e029** (nibi, job 10412760): `python cleanrl/ppo_atari_envpool.py --env-id NameThisGame-v5` [PENDING]
- **e031** (nibi, job 10412761): `python cleanrl/ppo_atari_envpool.py --env-id Phoenix-v5 --se` [PENDING]
- **e033** (nibi, job 10412763): `python cleanrl/ppo_atari_envpool.py --env-id Phoenix-v5 --se` [PENDING]
- **e035** (nibi, job 10412765): `python cleanrl/ppo_atari_envpool.py --env-id Phoenix-v5 --se` [PENDING]
- **e038** (fir, job 27850822): `python cleanrl/ppo_atari_envpool.py --env-id Qbert-v5 --seed` [PENDING]
- **e042** (fir, job 27850829): `python cleanrl/ppo_atari_envpool.py --env-id Qbert-v5 --seed` [PENDING]
- **e045** (fir, job 27850831): `python cleanrl/ppo_atari_envpool.py --env-id Qbert-v5 --seed` [PENDING]
- **e048** (fir, job 27850832): `python cleanrl/ppo_atari_envpool.py --env-id SpaceInvaders-v` [PENDING]
- **e052** (fir, job 27850843): `python cleanrl/ppo_atari_envpool.py --env-id SpaceInvaders-v` [PENDING]
- **e055** (fir, job 27850845): `python cleanrl/ppo_atari_envpool.py --env-id SpaceInvaders-v` [PENDING]
- **e058** (fir, job 27850849): `python cleanrl/ppo_atari_envpool.py --env-id MsPacman-v5 --s` [PENDING]
- **e061** (fir, job 27850855): `python cleanrl/ppo_atari_envpool.py --env-id MsPacman-v5 --s` [PENDING]
- **e064** (fir, job 27850876): `python cleanrl/ppo_atari_envpool.py --env-id MsPacman-v5 --s` [PENDING]
- **e068** (fir, job 27850878): `python cleanrl/ppo_atari_envpool.py --env-id Venture-v5 --se` [PENDING]
- **e071** (fir, job 27850903): `python cleanrl/ppo_atari_envpool.py --env-id Venture-v5 --se` [PENDING]
- **e073** (fir, job 27850927): `python cleanrl/ppo_atari_envpool.py --env-id Venture-v5 --se` [PENDING]
- **e075** (fir, job 27850929): `python cleanrl/ppo_atari_envpool.py --env-id MontezumaReveng` [PENDING]
- **e077** (fir, job 27850935): `python cleanrl/ppo_atari_envpool.py --env-id MontezumaReveng` [PENDING]
- **e079** (fir, job 27850943): `python cleanrl/ppo_atari_envpool.py --env-id MontezumaReveng` [PENDING]

### h002: PQN baseline on 15-game Atari subset (40M steps, envpool, minimal action space)

- **e026** (rorqual, job 8404341): `python cleanrl/pqn_atari_envpool.py --env-id Amidar-v5 --see` [PENDING]
- **e028** (rorqual, job 8404342): `python cleanrl/pqn_atari_envpool.py --env-id Amidar-v5 --see` [PENDING]
- **e030** (rorqual, job 8404343): `python cleanrl/pqn_atari_envpool.py --env-id Amidar-v5 --see` [PENDING]
- **e032** (rorqual, job 8404344): `python cleanrl/pqn_atari_envpool.py --env-id Alien-v5 --seed` [PENDING]
- **e034** (rorqual, job 8404345): `python cleanrl/pqn_atari_envpool.py --env-id Alien-v5 --seed` [PENDING]
- **e036** (rorqual, job 8404346): `python cleanrl/pqn_atari_envpool.py --env-id Alien-v5 --seed` [PENDING]
- **e037** (rorqual, job 8404348): `python cleanrl/pqn_atari_envpool.py --env-id Breakout-v5 --s` [PENDING]
- **e039** (rorqual, job 8404349): `python cleanrl/pqn_atari_envpool.py --env-id Breakout-v5 --s` [PENDING]
- **e040** (rorqual, job 8404351): `python cleanrl/pqn_atari_envpool.py --env-id Breakout-v5 --s` [PENDING]
- **e041** (rorqual, job 8404352): `python cleanrl/pqn_atari_envpool.py --env-id Enduro-v5 --see` [PENDING]
- **e043** (rorqual, job 8404353): `python cleanrl/pqn_atari_envpool.py --env-id Enduro-v5 --see` [PENDING]
- **e044** (rorqual, job 8404354): `python cleanrl/pqn_atari_envpool.py --env-id Enduro-v5 --see` [PENDING]
- **e046** (rorqual, job 8404355): `python cleanrl/pqn_atari_envpool.py --env-id PrivateEye-v5 -` [PENDING]
- **e047** (rorqual, job 8404358): `python cleanrl/pqn_atari_envpool.py --env-id PrivateEye-v5 -` [PENDING]
- **e049** (rorqual, job 8404359): `python cleanrl/pqn_atari_envpool.py --env-id PrivateEye-v5 -` [PENDING]
- **e050** (nibi, job 10412789): `python cleanrl/pqn_atari_envpool.py --env-id Solaris-v5 --se` [PENDING]
- **e051** (nibi, job 10412791): `python cleanrl/pqn_atari_envpool.py --env-id Solaris-v5 --se` [PENDING]
- **e053** (nibi, job 10412792): `python cleanrl/pqn_atari_envpool.py --env-id Solaris-v5 --se` [PENDING]
- **e054** (nibi, job 10412794): `python cleanrl/pqn_atari_envpool.py --env-id BattleZone-v5 -` [PENDING]
- **e056** (nibi, job 10412795): `python cleanrl/pqn_atari_envpool.py --env-id BattleZone-v5 -` [PENDING]
- **e057** (nibi, job 10412796): `python cleanrl/pqn_atari_envpool.py --env-id BattleZone-v5 -` [PENDING]
- **e059** (nibi, job 10412804): `python cleanrl/pqn_atari_envpool.py --env-id DoubleDunk-v5 -` [PENDING]
- **e060** (nibi, job 10412806): `python cleanrl/pqn_atari_envpool.py --env-id DoubleDunk-v5 -` [PENDING]
- **e062** (nibi, job 10412807): `python cleanrl/pqn_atari_envpool.py --env-id DoubleDunk-v5 -` [PENDING]
- **e063** (nibi, job 10412808): `python cleanrl/pqn_atari_envpool.py --env-id NameThisGame-v5` [PENDING]
- **e065** (nibi, job 10412809): `python cleanrl/pqn_atari_envpool.py --env-id NameThisGame-v5` [PENDING]
- **e066** (nibi, job 10412811): `python cleanrl/pqn_atari_envpool.py --env-id NameThisGame-v5` [PENDING]
- **e067** (nibi, job 10412813): `python cleanrl/pqn_atari_envpool.py --env-id Phoenix-v5 --se` [PENDING]
- **e069** (nibi, job 10412815): `python cleanrl/pqn_atari_envpool.py --env-id Phoenix-v5 --se` [PENDING]
- **e070** (nibi, job 10412817): `python cleanrl/pqn_atari_envpool.py --env-id Phoenix-v5 --se` [PENDING]
- **e072** (fir, job 27850926): `python cleanrl/pqn_atari_envpool.py --env-id Qbert-v5 --seed` [PENDING]
- **e074** (fir, job 27850928): `python cleanrl/pqn_atari_envpool.py --env-id Qbert-v5 --seed` [PENDING]
- **e076** (fir, job 27850934): `python cleanrl/pqn_atari_envpool.py --env-id Qbert-v5 --seed` [PENDING]
- **e078** (fir, job 27850936): `python cleanrl/pqn_atari_envpool.py --env-id SpaceInvaders-v` [PENDING]
- **e080** (fir, job 27850947): `python cleanrl/pqn_atari_envpool.py --env-id SpaceInvaders-v` [PENDING]
- **e081** (fir, job 27850981): `python cleanrl/pqn_atari_envpool.py --env-id SpaceInvaders-v` [PENDING]
- **e082** (fir, job 27850983): `python cleanrl/pqn_atari_envpool.py --env-id MsPacman-v5 --s` [PENDING]
- **e083** (fir, job 27850988): `python cleanrl/pqn_atari_envpool.py --env-id MsPacman-v5 --s` [PENDING]
- **e084** (fir, job 27850989): `python cleanrl/pqn_atari_envpool.py --env-id MsPacman-v5 --s` [PENDING]
- **e085** (fir, job 27850990): `python cleanrl/pqn_atari_envpool.py --env-id Venture-v5 --se` [PENDING]
- **e086** (fir, job 27850994): `python cleanrl/pqn_atari_envpool.py --env-id Venture-v5 --se` [PENDING]
- **e087** (fir, job 27850996): `python cleanrl/pqn_atari_envpool.py --env-id Venture-v5 --se` [PENDING]
- **e088** (fir, job 27850997): `python cleanrl/pqn_atari_envpool.py --env-id MontezumaReveng` [PENDING]
- **e089** (fir, job 27851005): `python cleanrl/pqn_atari_envpool.py --env-id MontezumaReveng` [PENDING]
- **e090** (fir, job 27851007): `python cleanrl/pqn_atari_envpool.py --env-id MontezumaReveng` [PENDING]

## Currently Active

- e001 on rorqual (job 8404296): submitted
- e002 on rorqual (job 8404297): submitted
- e003 on rorqual (job 8404321): submitted
- e004 on rorqual (job 8404326): submitted
- e005 on rorqual (job 8404327): submitted
- e006 on rorqual (job 8404330): submitted
- e007 on rorqual (job 8404332): submitted
- e008 on rorqual (job 8404333): submitted
- e009 on rorqual (job 8404334): submitted
- e010 on rorqual (job 8404335): submitted
- e011 on rorqual (job 8404336): submitted
- e012 on rorqual (job 8404337): submitted
- e013 on rorqual (job 8404338): submitted
- e014 on rorqual (job 8404339): submitted
- e015 on rorqual (job 8404340): submitted
- e016 on nibi (job 10412748): submitted
- e017 on nibi (job 10412749): submitted
- e018 on nibi (job 10412750): submitted
- e019 on nibi (job 10412751): submitted
- e020 on nibi (job 10412752): submitted
- e021 on nibi (job 10412753): submitted
- e022 on nibi (job 10412754): submitted
- e023 on nibi (job 10412755): submitted
- e024 on nibi (job 10412756): submitted
- e025 on nibi (job 10412757): submitted
- e026 on rorqual (job 8404341): submitted
- e027 on nibi (job 10412759): submitted
- e028 on rorqual (job 8404342): submitted
- e029 on nibi (job 10412760): submitted
- e030 on rorqual (job 8404343): submitted
- e031 on nibi (job 10412761): submitted
- e032 on rorqual (job 8404344): submitted
- e033 on nibi (job 10412763): submitted
- e034 on rorqual (job 8404345): submitted
- e035 on nibi (job 10412765): submitted
- e036 on rorqual (job 8404346): submitted
- e037 on rorqual (job 8404348): submitted
- e038 on fir (job 27850822): submitted
- e039 on rorqual (job 8404349): submitted
- e040 on rorqual (job 8404351): submitted
- e041 on rorqual (job 8404352): submitted
- e042 on fir (job 27850829): submitted
- e043 on rorqual (job 8404353): submitted
- e044 on rorqual (job 8404354): submitted
- e045 on fir (job 27850831): submitted
- e046 on rorqual (job 8404355): submitted
- e047 on rorqual (job 8404358): submitted
- e048 on fir (job 27850832): submitted
- e049 on rorqual (job 8404359): submitted
- e050 on nibi (job 10412789): submitted
- e051 on nibi (job 10412791): submitted
- e052 on fir (job 27850843): submitted
- e053 on nibi (job 10412792): submitted
- e054 on nibi (job 10412794): submitted
- e055 on fir (job 27850845): submitted
- e056 on nibi (job 10412795): submitted
- e057 on nibi (job 10412796): submitted
- e058 on fir (job 27850849): submitted
- e059 on nibi (job 10412804): submitted
- e060 on nibi (job 10412806): submitted
- e061 on fir (job 27850855): submitted
- e062 on nibi (job 10412807): submitted
- e063 on nibi (job 10412808): submitted
- e064 on fir (job 27850876): submitted
- e065 on nibi (job 10412809): submitted
- e066 on nibi (job 10412811): submitted
- e067 on nibi (job 10412813): submitted
- e068 on fir (job 27850878): submitted
- e069 on nibi (job 10412815): submitted
- e070 on nibi (job 10412817): submitted
- e071 on fir (job 27850903): submitted
- e072 on fir (job 27850926): submitted
- e073 on fir (job 27850927): submitted
- e074 on fir (job 27850928): submitted
- e075 on fir (job 27850929): submitted
- e076 on fir (job 27850934): submitted
- e077 on fir (job 27850935): submitted
- e078 on fir (job 27850936): submitted
- e079 on fir (job 27850943): submitted
- e080 on fir (job 27850947): submitted
- e081 on fir (job 27850981): submitted
- e082 on fir (job 27850983): submitted
- e083 on fir (job 27850988): submitted
- e084 on fir (job 27850989): submitted
- e085 on fir (job 27850990): submitted
- e086 on fir (job 27850994): submitted
- e087 on fir (job 27850996): submitted
- e088 on fir (job 27850997): submitted
- e089 on fir (job 27851005): submitted
- e090 on fir (job 27851007): submitted
