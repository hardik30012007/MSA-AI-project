from pathlib import Path
import argparse
import re
import pandas as pd

def num_key(label: str) -> int:
    m = re.search(r'(\d+)', str(label))
    return int(m.group(1)) if m else 10**9

def load_processed_csv(path: Path, row_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if row_label not in df.columns:
        row_label = df.columns[0]
    df = df.copy()
    df[row_label] = df[row_label].astype(str)
    df['__order'] = df[row_label].map(num_key)
    df = df.sort_values('__order').drop(columns='__order').reset_index(drop=True)
    return df.set_index(row_label)[['Mean', 'Std']]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-file', required=True)
    ap.add_argument('--algorithms', default='IMSA,MSA,PSO,GWO,WOA,SSA,DE,GA,HHO')
    ap.add_argument('--row-label', default='Function')
    args = ap.parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    algo_order = [a.strip() for a in args.algorithms.split(',') if a.strip()]
    algo_data = {}
    for algo in algo_order:
        fp = input_dir / f'{algo}.csv'
        if not fp.exists():
            raise FileNotFoundError(f'Missing file: {fp}')
        algo_data[algo] = load_processed_csv(fp, args.row_label)
    common_labels = set.intersection(*(set(df.index) for df in algo_data.values()))
    labels = sorted(common_labels, key=num_key)
    rows = []
    for label in labels:
        means = {algo: float(algo_data[algo].loc[label, 'Mean']) for algo in algo_order}
        stds = {algo: float(algo_data[algo].loc[label, 'Std']) for algo in algo_order}
        ranked_algos = sorted(algo_order, key=lambda a: (means[a], algo_order.index(a)))
        ranks = {algo: i + 1 for i, algo in enumerate(ranked_algos)}
        row = {'algo': label}
        for algo in algo_order:
            row[f'{algo}.MEAN'] = means[algo]
            row[f'{algo}.STD'] = stds[algo]
            row[f'{algo}.RANK'] = ranks[algo]
        rows.append(row)
    cols = ['algo'] + [f'{a}.{s}' for a in algo_order for s in ['MEAN', 'STD', 'RANK']]
    final_df = pd.DataFrame(rows, columns=cols)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_file, index=False)
    print(f'Saved {output_file}')

if __name__ == '__main__':
    main()
