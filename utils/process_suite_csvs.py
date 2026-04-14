from pathlib import Path
import argparse
import re
import pandas as pd

def func_key(label: str) -> int:
    m = re.search(r'(\d+)', str(label))
    return int(m.group(1)) if m else 10**9

def process_one_csv(input_path: Path, output_dir: Path, row_label: str):
    df = pd.read_csv(input_path)
    first_col = df.columns[0]
    run_cols = [c for c in df.columns if str(c).lower().startswith('run')]
    if not run_cols:
        raise ValueError(f'No Run columns found in {input_path.name}')
    run_data = df[run_cols].astype(float)
    out = pd.DataFrame({row_label: df[first_col].astype(str), 'Mean': run_data.mean(axis=1), 'Std': run_data.std(axis=1, ddof=1)})
    out['__order'] = out[row_label].map(func_key)
    out = out.sort_values('__order').drop(columns='__order').reset_index(drop=True)
    output_path = output_dir / input_path.name
    out.to_csv(output_path, index=False)
    print(f'Saved {output_path}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--row-label', default='Function')
    args = ap.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_file in sorted(input_dir.glob('*.csv')):
        process_one_csv(csv_file, output_dir, args.row_label)

if __name__ == '__main__':
    main()
