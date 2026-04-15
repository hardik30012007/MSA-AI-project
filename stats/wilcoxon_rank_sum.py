from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np
from scipy.stats import ranksums

def num_key(label: str) -> int:
    m = re.search(r'(\d+)', str(label))
    return int(m.group(1)) if m else 10**9

def load_raw_csv(path: Path):
    df = pd.read_csv(path)
    label_col = df.columns[0]
    df = df.copy()
    df[label_col] = df[label_col].astype(str)
    df['__order'] = df[label_col].map(num_key)
    df = df.sort_values('__order').drop(columns='__order').reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--reference', default='IMSA')
    ap.add_argument('--algorithms', default='IMSA,MSA,PSO,GWO,WOA,SSA,DE,GA,HHO')
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_order = [a.strip() for a in args.algorithms.split(',') if a.strip()]

    # Load all CSVs
    algo_frames = {}
    for algo in algo_order:
        fp = raw_dir / f'{algo}.csv'
        if not fp.exists():
            raise FileNotFoundError(f'Missing file: {fp}')
        algo_frames[algo] = load_raw_csv(fp)

    label_col = algo_frames[algo_order[0]].columns[0]

    # 🔥 IMPORTANT: use UNION of all labels (handles missing F12 etc.)
    all_labels = set()
    for df in algo_frames.values():
        all_labels.update(df[label_col].tolist())

    labels = sorted(all_labels, key=num_key)

    rows = []

    for label in labels:
        row = {label_col: label}

        # Get reference row
        ref_df = algo_frames[args.reference]
        ref_row = ref_df[ref_df[label_col] == label]

        if ref_row.empty:
            continue  # skip if reference doesn't have it

        ref_vals = ref_row.iloc[0, 1:].astype(float).to_numpy()

        wins = losses = ties = 0

        for algo in algo_order:
            if algo == args.reference:
                continue

            algo_df = algo_frames[algo]
            algo_row = algo_df[algo_df[label_col] == label]

            if algo_row.empty:
                continue  # skip missing functions safely

            vals = algo_row.iloc[0, 1:].astype(float).to_numpy()

            stat, p = ranksums(ref_vals, vals)

            ref_mean = float(np.mean(ref_vals))
            other_mean = float(np.mean(vals))

            if p < 0.05:
                if ref_mean < other_mean:
                    result = 'Win'
                    wins += 1
                else:
                    result = 'Loss'
                    losses += 1
            else:
                result = 'Tie'
                ties += 1

            row[f'{args.reference}_vs_{algo}_p'] = p
            row[f'{args.reference}_vs_{algo}_result'] = result

        row[f'{args.reference}_wins'] = wins
        row[f'{args.reference}_losses'] = losses
        row[f'{args.reference}_ties'] = ties

        rows.append(row)

    out_df = pd.DataFrame(rows)

    out_file = output_dir / f'wilcoxon_{args.reference}_vs_all.csv'
    out_df.to_csv(out_file, index=False)

    # Summary
    summary = []
    for algo in algo_order:
        if algo == args.reference:
            continue

        col = f'{args.reference}_vs_{algo}_result'
        if col not in out_df.columns:
            continue

        summary.append({
            'Algorithm': algo,
            'Win': int((out_df[col] == 'Win').sum()),
            'Loss': int((out_df[col] == 'Loss').sum()),
            'Tie': int((out_df[col] == 'Tie').sum())
        })

    pd.DataFrame(summary).to_csv(
        output_dir / f'wilcoxon_summary_{args.reference}.csv',
        index=False
    )

    print(f'Saved {out_file}')

if __name__ == '__main__':
    main()