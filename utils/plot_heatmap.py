from pathlib import Path
import argparse
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def num_key(label: str) -> int:
    m = re.search(r'(\d+)', str(label))
    return int(m.group(1)) if m else 10**9

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', required=True)
    ap.add_argument('--output-file', required=True)
    ap.add_argument('--title', default='Rank Heatmap')
    args = ap.parse_args()
    df = pd.read_csv(args.input_file)
    rank_cols = [c for c in df.columns if c.endswith('.RANK')]
    if not rank_cols:
        raise SystemExit('No .RANK columns found')
    heat = df[rank_cols].copy()
    heat.columns = [c.split('.')[0] for c in rank_cols]
    heat.index = df['algo']
    heat['__order'] = heat.index.map(num_key)
    heat = heat.sort_values('__order').drop(columns='__order')
    plt.figure(figsize=(12, 8))
    sns.heatmap(heat, annot=True, fmt='.0f', cmap='RdYlGn_r', linewidths=0.5, cbar_kws={'label': 'Rank'})
    plt.title(args.title)
    plt.xlabel('Algorithms')
    plt.ylabel('Functions')
    plt.tight_layout()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved {output_path}')

if __name__ == '__main__':
    main()
