#!/usr/bin/env python3
"""Compute IQM of Human Normalized Scores (HNS) for all hypotheses.

Uses q4_return (last 25% of episodes) as the performance metric.
IQM = interquartile mean = trim bottom 25% and top 25% of per-game HNS, mean the middle 50%.

Usage:
    python analyze_iqm_hns.py [--hypothesis H1 H2 ...] [--verbose]
"""

import argparse
import csv
import sys
from collections import defaultdict

# Human and random scores from compute_hns.py — full Atari57
HUMAN_RANDOM = {
    "Alien-v5": (227.8, 7127.7),
    "Amidar-v5": (5.8, 1719.5),
    "Assault-v5": (222.4, 742.0),
    "Asterix-v5": (210.0, 8503.3),
    "Asteroids-v5": (719.1, 47388.7),
    "Atlantis-v5": (12850.0, 29028.1),
    "BankHeist-v5": (14.2, 753.1),
    "BattleZone-v5": (2360.0, 37187.5),
    "BeamRider-v5": (363.9, 16926.5),
    "Berzerk-v5": (123.7, 2630.4),
    "Bowling-v5": (23.1, 160.7),
    "Boxing-v5": (0.1, 12.1),
    "Breakout-v5": (1.7, 30.5),
    "Centipede-v5": (2090.9, 12017.0),
    "ChopperCommand-v5": (811.0, 7387.8),
    "CrazyClimber-v5": (10780.5, 35829.4),
    "Defender-v5": (2874.5, 18688.9),
    "DemonAttack-v5": (152.1, 1971.0),
    "DoubleDunk-v5": (-18.6, -16.4),
    "Enduro-v5": (0.0, 860.5),
    "FishingDerby-v5": (-91.7, -38.7),
    "Freeway-v5": (0.0, 29.6),
    "Frostbite-v5": (65.2, 4334.7),
    "Gopher-v5": (257.6, 2412.5),
    "Gravitar-v5": (173.0, 3351.4),
    "Hero-v5": (1027.0, 30826.4),
    "IceHockey-v5": (-11.2, 0.9),
    "Jamesbond-v5": (29.0, 302.8),
    "Kangaroo-v5": (52.0, 3035.0),
    "Krull-v5": (1598.0, 2665.5),
    "KungFuMaster-v5": (258.5, 22736.3),
    "MontezumaRevenge-v5": (0.0, 4753.3),
    "MsPacman-v5": (307.3, 6951.6),
    "NameThisGame-v5": (2292.3, 8049.0),
    "Phoenix-v5": (761.4, 7242.6),
    "Pitfall-v5": (-229.4, 6463.7),
    "Pong-v5": (-20.7, 14.6),
    "PrivateEye-v5": (24.9, 69571.3),
    "Qbert-v5": (163.9, 13455.0),
    "Riverraid-v5": (1338.5, 17118.0),
    "RoadRunner-v5": (11.5, 7845.0),
    "Robotank-v5": (2.2, 11.9),
    "Seaquest-v5": (68.4, 42054.7),
    "Skiing-v5": (-17098.1, -4336.9),
    "Solaris-v5": (1236.3, 12326.7),
    "SpaceInvaders-v5": (148.0, 1668.7),
    "StarGunner-v5": (664.0, 10250.0),
    "Surround-v5": (-10.0, 6.5),
    "Tennis-v5": (-23.8, -8.3),
    "TimePilot-v5": (3568.0, 5229.2),
    "Tutankham-v5": (11.4, 167.6),
    "UpNDown-v5": (533.4, 11693.2),
    "Venture-v5": (0.0, 1187.5),
    "VideoPinball-v5": (16256.9, 17667.9),
    "WizardOfWor-v5": (563.5, 4756.5),
    "YarsRevenge-v5": (3092.9, 54576.9),
    "Zaxxon-v5": (32.5, 9173.3),
}

# Original 15-game subset for Phase 1-3 comparisons
ORIG_15_GAMES = sorted([
    "Alien-v5", "Amidar-v5", "BattleZone-v5", "Breakout-v5", "DoubleDunk-v5",
    "Enduro-v5", "MontezumaRevenge-v5", "MsPacman-v5", "NameThisGame-v5",
    "Phoenix-v5", "PrivateEye-v5", "Qbert-v5", "Solaris-v5", "SpaceInvaders-v5",
    "Venture-v5",
])

ALL_GAMES = sorted(HUMAN_RANDOM.keys())


def compute_hns(env_id, score):
    """Compute human normalized score: (score - random) / (human - random)."""
    random_score, human_score = HUMAN_RANDOM[env_id]
    if human_score == random_score:
        return 0.0
    return (score - random_score) / (human_score - random_score)


def compute_iqm(values):
    """Compute interquartile mean: mean of middle 50% after sorting."""
    if len(values) < 4:
        # With fewer than 4 values, IQM is just the mean (can't trim meaningfully)
        return sum(values) / len(values) if values else float('nan')
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    # Trim bottom 25% and top 25%
    lower = int(n * 0.25)
    upper = n - lower
    trimmed = sorted_vals[lower:upper]
    return sum(trimmed) / len(trimmed) if trimmed else float('nan')


def load_experiments(csv_path):
    """Load experiments.csv, return list of dicts."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_best_q4_per_game(experiments, hypothesis_id):
    """For a hypothesis, get the best (most recent / highest priority) q4 per game.

    When multiple seeds exist, average them. When multiple entries exist for same
    game+seed, take the last one (most recent).
    """
    # Group by (env_id, seed)
    by_game_seed = defaultdict(list)
    for exp in experiments:
        if exp['hypothesis_id'] != hypothesis_id:
            continue
        env_id = exp.get('env_id', '')
        if env_id not in HUMAN_RANDOM:
            continue
        q4 = exp.get('q4_return', '')
        if q4 == '' or q4 is None:
            continue
        try:
            q4_val = float(q4)
        except (ValueError, TypeError):
            continue
        seed = exp.get('seed', '1')
        by_game_seed[(env_id, seed)].append(q4_val)

    # Take last entry per (game, seed), then average across seeds per game
    by_game = defaultdict(list)
    for (env_id, seed), vals in by_game_seed.items():
        by_game[env_id].append(vals[-1])  # last entry = most recent

    result = {}
    for env_id, seed_vals in by_game.items():
        result[env_id] = sum(seed_vals) / len(seed_vals)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypothesis', nargs='*', help='Specific hypotheses to analyze')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--csv', default='results/experiments.csv')
    parser.add_argument('--min-games', type=int, default=5, help='Min games for IQM (default: 5)')
    parser.add_argument('--game-set', choices=['15', '57', 'all'], default='all',
                        help='Game set: 15 (original), 57 (Atari57), all (whatever is available)')
    args = parser.parse_args()

    experiments = load_experiments(args.csv)

    # Filter game set
    global ALL_GAMES
    if args.game_set == '15':
        ALL_GAMES = ORIG_15_GAMES
    elif args.game_set == '57':
        ALL_GAMES = sorted(HUMAN_RANDOM.keys())
    # else 'all': use whatever games appear in data (but still within HUMAN_RANDOM)

    # Find all hypotheses
    all_hyps = sorted(set(exp['hypothesis_id'] for exp in experiments))
    if args.hypothesis:
        all_hyps = [h for h in all_hyps if h in args.hypothesis]

    # Compute per-hypothesis IQM HNS
    results = []
    for hyp in all_hyps:
        q4_by_game = get_best_q4_per_game(experiments, hyp)
        if len(q4_by_game) < args.min_games:
            continue

        hns_values = []
        game_details = {}
        for env_id in ALL_GAMES:
            if env_id in q4_by_game:
                hns = compute_hns(env_id, q4_by_game[env_id])
                hns_values.append(hns)
                game_details[env_id] = (q4_by_game[env_id], hns)

        iqm = compute_iqm(hns_values)
        mean_hns = sum(hns_values) / len(hns_values) if hns_values else float('nan')
        median_hns = sorted(hns_values)[len(hns_values)//2] if hns_values else float('nan')
        n_games = len(hns_values)
        n_seeds_info = ""

        results.append({
            'hypothesis_id': hyp,
            'iqm_hns': iqm,
            'mean_hns': mean_hns,
            'median_hns': median_hns,
            'n_games': n_games,
            'game_details': game_details,
            'hns_values': hns_values,
        })

    # Sort by IQM HNS descending
    results.sort(key=lambda x: x['iqm_hns'], reverse=True)

    # Print results
    print(f"\n{'='*80}")
    print(f"IQM HNS RANKINGS (q4 metric, {len(results)} hypotheses with >= {args.min_games} games)")
    print(f"{'='*80}")
    print(f"{'Rank':>4}  {'Hypothesis':>12}  {'IQM HNS':>9}  {'Mean HNS':>9}  {'Med HNS':>9}  {'Games':>5}")
    print(f"{'-'*4}  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*5}")
    for i, r in enumerate(results):
        marker = " ***" if r['hypothesis_id'] in ('h001', 'h002') else ""
        print(f"{i+1:>4}  {r['hypothesis_id']:>12}  {r['iqm_hns']:>9.4f}  {r['mean_hns']:>9.4f}  {r['median_hns']:>9.4f}  {r['n_games']:>5}{marker}")

    # Baseline comparison
    h001 = next((r for r in results if r['hypothesis_id'] == 'h001'), None)
    h002 = next((r for r in results if r['hypothesis_id'] == 'h002'), None)

    if h001:
        print(f"\n{'='*80}")
        print(f"RELATIVE TO PPO BASELINE (h001 IQM HNS = {h001['iqm_hns']:.4f})")
        print(f"{'='*80}")
        print(f"{'Rank':>4}  {'Hypothesis':>12}  {'IQM HNS':>9}  {'Delta':>9}  {'% Change':>9}  {'Games':>5}")
        print(f"{'-'*4}  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*5}")
        for i, r in enumerate(results):
            delta = r['iqm_hns'] - h001['iqm_hns']
            pct = (delta / abs(h001['iqm_hns'])) * 100 if h001['iqm_hns'] != 0 else float('inf')
            marker = " <-- BASELINE" if r['hypothesis_id'] == 'h001' else ""
            marker = " <-- PQN BASE" if r['hypothesis_id'] == 'h002' else marker
            print(f"{i+1:>4}  {r['hypothesis_id']:>12}  {r['iqm_hns']:>9.4f}  {delta:>+9.4f}  {pct:>+8.1f}%  {r['n_games']:>5}{marker}")

    if args.verbose:
        print(f"\n{'='*80}")
        print("PER-GAME HNS BREAKDOWN")
        print(f"{'='*80}")
        for r in results:
            print(f"\n--- {r['hypothesis_id']} (IQM HNS = {r['iqm_hns']:.4f}, {r['n_games']} games) ---")
            for env_id in ALL_GAMES:
                if env_id in r['game_details']:
                    q4, hns = r['game_details'][env_id]
                    game_short = env_id.replace('-v5', '')
                    print(f"  {game_short:>20s}: q4={q4:>10.1f}  HNS={hns:>+8.4f}")

    # CSV output for easy comparison
    print(f"\n{'='*80}")
    print("CSV FORMAT (for pasting)")
    print(f"{'='*80}")
    print("hypothesis_id,iqm_hns,mean_hns,median_hns,n_games")
    for r in results:
        print(f"{r['hypothesis_id']},{r['iqm_hns']:.4f},{r['mean_hns']:.4f},{r['median_hns']:.4f},{r['n_games']}")


if __name__ == '__main__':
    main()
