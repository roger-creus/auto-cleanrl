# Research Goal

## Objective
Develop novel deep reinforcement learning algorithms that achieve state-of-the-art performance on Atari, surpassing PPO and PQN baselines. The research should produce publishable algorithmic innovations — not just hyperparameter tuning. Even though we will use PPO and PQN as baselines to beat only, the cleanrl codebase contains many other implementations of algorithms which you should explore and potentially draw inspiration from when building new contributions.

## Benchmark
**Environments (15-game Atari subset via envpool):**
- Amidar-v5, Alien-v5, Breakout-v5, Enduro-v5, PrivateEye-v5
- Solaris-v5, BattleZone-v5, DoubleDunk-v5, NameThisGame-v5, Phoenix-v5
- Qbert-v5, SpaceInvaders-v5, MsPacman-v5, Venture-v5, MontezumaRevenge-v5

**Training budget:** 40M environment steps per run.

**Seeds:** 3 seeds (1, 2, 3) for statistical validity on full evaluations.

ALL JOBS NEED TO USE AT LEAST 1 GPU.

## Environment Configuration (CRITICAL)

The default envpool configuration for ALL training and evaluation MUST use:

```python
envs = envpool.make(
    args.env_id,
    env_type="gym",
    num_envs=args.num_envs,
    episodic_life=True,
    reward_clip=True,
    seed=args.seed,
)
```

This is the community standard Atari benchmark setup with:
- `episodic_life=True` — episode ends on life loss (standard evaluation)
- `reward_clip=True` — rewards clipped to [-1, 1] (standard for Atari)

ALL algorithms (baselines and new) must use this configuration. Do NOT change these settings.

## Metrics (CRITICAL — read carefully)

### Known logging issue
CleanRL's envpool scripts log episode returns from ALL parallel environments at the same `global_step`. With 128 parallel envs, many episodes end simultaneously, producing many log entries with identical `global_step` values. This MUST be fixed before any meaningful analysis:

**FIX REQUIRED:** Modify the logging to ensure exactly ONE metric entry per unique `global_step`. Options:
- Average all episode returns that complete at the same step
- Only log the first episode return per step
- Use a running average that updates per step

**Do NOT draw conclusions from metrics computed on unfixed logging.** Fix this first.

### Performance metric
The "mean return over last 25 episodes" is **unreliable** because:
- If the agent was unlucky in the last 25 episodes, the score is unrepresentative
- Even though the rewards are clipped in the env, info["r"][idx] returns the corect atari score so that's good.

**Use robust metrics instead:**
- **Q4 metric:** Mean return over the last 25% of ALL episodes during training (not just the last 25)
- **AUC:** Area under the episode return curve (captures full training performance)
- **Human Normalized Score (HNS):** If `compute_hns.py` is available, use it for cross-game comparisons
- Report multiple metrics — do not rely on a single number to judge hypothesis validity

### Judging hypothesis validity
Be VERY careful when deciding if a hypothesis is good or bad:
- Never judge from a single game or single seed
- A pilot must cover ALL 15 games (see experiment protocol below)
- Consider per-game breakdown — an algorithm might excel on 10 games but fail on 5
- Statistical noise is real — a 5% difference on 1 seed means nothing

## Baselines
Run these first and cache their results before attempting any improvements:

1. **PPO (envpool):** `cleanrl/ppo_atari_envpool.py` — run on all 15 games × 3 seeds
2. **PQN (envpool):** `cleanrl/pqn_atari_envpool.py` — run on all 15 games × 3 seeds

These baseline results are the bar to beat. Every new algorithm must be compared against them.

END GOAL: State-of-the-art performance competitive with published methods, with a clear and publishable algorithmic narrative. **Do not stop iterating if you have done few changes and already increase the performance of PQN and PPO by a smallish margin. You have to continue to hill climb until you have done a full exploration of your hypothesis space.** For instance if we do one change like replace NatureCNN for ResNet and we get big performance gains, that is not too novel because just running the PQN baseline with a ResNet would give the same performance gains and that wouldn't be a novel research contribution.

## Experiment Protocol (CRITICAL)

### Pilot definition
**Pilot = ALL 15 games × 1 seed.** This is the MINIMUM for evaluating any hypothesis.
- **NEVER use fewer games** (e.g., 3 games) — results from a subset are misleading and unreliable
- A pilot with 3 games is WORTHLESS for judging algorithm quality
- We have plenty of compute — running 15 games × 1 seed is cheap

### Full evaluation
**Full evaluation = ALL 15 games × 3 seeds.** For statistical robustness and final conclusions.

### Phase 1: Baselines (do this first!)
1. Fix the logging issue (one entry per global_step)
2. Run PPO envpool on all 15 games × 3 seeds
3. Run PQN envpool on all 15 games × 3 seeds
4. Record all baseline results in the journal and results bank
5. Analyze which games each baseline struggles on — these are the targets

### Phase 2: Broad exploration (**this is the longest phase. Must not end untill all the hypothesis space is fully explored**)
1. Start with the most promising directions based on baseline analysis
2. Implement a single algorithmic change at a time
3. **Pilot: run ALL 15 games × 1 seed** (this is cheap — do it for every hypothesis)
4. If promising: run full 15 games × 3 seeds
5. If not promising: record detailed findings with comments, move to next idea
6. **Explore ACROSS categories** — don't just iterate on one algorithm family
7. Combine successful innovations incrementally

### Phase 3: Full evaluation and paper-readiness (only after ALL hypothesis space has been thoroughly explored)
1. Run best algorithm on all 15 games × 3 seeds
2. Ablation study: remove each innovation one at a time to measure contribution
3. Generate learning curves and comparison tables
4. Write up the algorithmic narrative

### Final Phase
1. Run PPO and PQN envpool on full Atari57 suite on 40M env steps
2. Run your fully discovered best-performing algorithm on full Atari57 suite on 40M env steps
3. Flag any failure mode/inconsistency and continue to iterate if results don't translate to the full benchmark
4. Write-up full paper with all narrative, ablations, benchmark experiments and methodology

## Research Directions to Explore

You MUST explore broadly across these categories. Do NOT get stuck in only one algorithm family. Read the existing cleanrl implementations for inspiration.

Each axis lists existing techniques as **starting points only**. Your job is to understand WHY each technique works, identify its limitations, and **invent principled improvements**. Come up with your own novel research directions.

### 1. Algorithmic paradigms (explore beyond on-policy!)
- On-policy: PPO variants, A2C
- Off-policy with replay buffers: DQN variants (Rainbow components), SAC-discrete
- Hybrid on/off-policy: ACER, V-trace, Retrace
- Q-learning + policy gradient fusion
- N-step returns, varying bootstrapping depths

### 2. Experience management and replay
- Prioritized experience replay, large/small buffers
- Replay ratio tuning, sequence replay for recurrent architectures

### 3. Value function innovations
- Distributional RL (C51, QR-DQN, IQN), expectile regression
- Dueling architectures, multi-head value functions, Popart normalization

### 4. Policy gradient innovations
- Clipping alternatives, trust region methods
- Entropy scheduling, off-policy corrections (V-trace, Retrace)

### 5. Architecture innovations
- Enhanced visual encoders (ResNet, attention-augmented CNNs)
- Memory (LSTM, GTrXL), spectral norm, network resets / plasticity

### 6. Exploration
- Intrinsic motivation (RND, ICM), count-based, noisy nets
- Adaptive entropy, bootstrapped heads

### 7. Reward and return processing
- Reward normalization, Popart, symlog, discount schedules

### 8. Optimization
- Modern optimizers (Lion, Muon, schedule-free), LR schedules, gradient manipulation

### 9. Representation learning and auxiliary losses
- BYOL, VICReg, SPR, data augmentation (DrQ)

### 10. Ensemble and composition methods
- Ensembles, mixture of experts, population-based training

### 11. Hyperparameter optimization (LAST PHASE ONLY)
- Once algorithmic innovations are FULLY established, tune hyperparameters
- Sweep one at a time, not grid search

## Philosophy

**What constitutes a novel contribution:**
Simply combining existing techniques is NOT novel research. Engineering the best combination is the FOUNDATION. Then invent something new on top.

**Your workflow:**
1. Build the best engineering baseline (combine best existing techniques)
2. Analyze WHERE and WHY it fails on specific games
3. Design a novel solution motivated by that analysis
4. Test it with proper pilots (ALL 15 games)
5. Iterate

- **Breadth first, depth second**: Test many ideas quickly before committing
- **Search the literature**: Use web search constantly for papers and implementation tricks
- **Learn from failures**: Record WHY things failed — this informs future directions

## Compute Budget
You have access to a LOT of compute across 4 clusters with H100 and A100 GPUs. Jobs schedule fast. Submit large batches in parallel across all clusters. Don't be conservative.

# SOURCES

CleanRL docs: https://docs.cleanrl.dev/
Envpool atari docs: https://envpool.readthedocs.io/en/latest/env/atari.html
