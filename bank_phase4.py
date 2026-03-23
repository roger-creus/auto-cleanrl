#!/usr/bin/env python3
"""Bank Phase 4 Atari57 results from pulled CSVs into results/experiments.csv.

Reads CSVs from results/{cluster}/{hyp}__{hyp}-{game}-s{seed}.csv,
verifies total_timesteps=40000000, and appends to experiments.csv.

Usage:
    python bank_phase4.py [--dry-run] [--hypothesis h001 h002 h064]
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

CLUSTERS = ['narval', 'fir', 'nibi', 'rorqual']
RESULTS_CSV = 'results/experiments.csv'


def load_banked():
    """Load already-banked experiment IDs."""
    banked = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                banked.add((row['hypothesis_id'], row['experiment_id']))
    return banked


def find_phase4_csvs(hypotheses):
    """Find all Phase 4 CSVs with valid 40M results."""
    results = {}  # (hyp, exp_id) -> row dict
    skipped_10m = 0
    errors = 0

    for hyp in hypotheses:
        for cluster in CLUSTERS:
            pattern = f'results/{cluster}/{hyp}__{hyp}-*.csv'
            for fpath in glob.glob(pattern):
                if '__curve' in fpath:
                    continue
                try:
                    with open(fpath) as f:
                        reader = csv.DictReader(f)
                        row = next(reader)

                    ts = int(row.get('total_timesteps', 0))
                    if ts != 40000000:
                        skipped_10m += 1
                        continue

                    env_id = row['env_id']
                    seed = row['seed']
                    exp_id = row.get('experiment_id', '')
                    if not exp_id:
                        exp_id = f'{hyp}-{env_id.replace("-v5","").lower()}-s{seed}'

                    key = (hyp, exp_id)
                    # Keep the first valid result found (don't overwrite)
                    if key not in results:
                        results[key] = {
                            'experiment_id': exp_id,
                            'hypothesis_id': hyp,
                            'env_id': env_id,
                            'seed': seed,
                            'algorithm': row.get('algorithm', ''),
                            'total_timesteps': ts,
                            'q4_return': row.get('q4_return', ''),
                            'auc': row.get('auc', ''),
                            'mean_return': row.get('mean_return', ''),
                            'command': '',
                            'comment': f'Phase 4 Atari57 ({cluster})',
                        }
                except Exception as e:
                    errors += 1

    return results, skipped_10m, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Show what would be banked without writing')
    parser.add_argument('--hypothesis', nargs='*', default=['h001', 'h002', 'h064'])
    args = parser.parse_args()

    banked = load_banked()
    results, skipped_10m, errors = find_phase4_csvs(args.hypothesis)

    # Filter out already-banked
    to_bank = {k: v for k, v in results.items() if k not in banked}

    print(f"Found {len(results)} valid 40M results, {skipped_10m} skipped (10M), {errors} errors")
    print(f"Already banked: {len(results) - len(to_bank)}")
    print(f"New to bank: {len(to_bank)}")

    if not to_bank:
        print("Nothing new to bank.")
        return

    # Group by hypothesis for summary
    by_hyp = defaultdict(list)
    for (hyp, exp_id), row in sorted(to_bank.items()):
        by_hyp[hyp].append(row)

    for hyp in sorted(by_hyp.keys()):
        rows = by_hyp[hyp]
        games = set(r['env_id'] for r in rows)
        print(f"\n{hyp}: {len(rows)} experiments, {len(games)} games")
        for r in rows[:5]:
            print(f"  {r['experiment_id']}: {r['env_id']} q4={r['q4_return']}")
        if len(rows) > 5:
            print(f"  ... and {len(rows)-5} more")

    if args.dry_run:
        print("\n[DRY RUN] Would bank the above. Run without --dry-run to commit.")
        return

    # Read existing CSV to get fieldnames
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    # Append new rows
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        for (hyp, exp_id), row in sorted(to_bank.items()):
            writer.writerow(row)

    print(f"\nBanked {len(to_bank)} new experiments to {RESULTS_CSV}")


if __name__ == '__main__':
    main()
