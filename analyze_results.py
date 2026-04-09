import argparse, glob, os, csv
import numpy as np
import pandas as pd
from scipy.stats import ranksums
from bench_utils import summarize_runs


def load_raw_csv(path):
    df=pd.read_csv(path)
    name=os.path.basename(path).split('_')[-2] if 'engineering' in os.path.basename(path) else os.path.basename(path).split('_')[-2]
    # fallback parse
    parts=os.path.basename(path).replace('.csv','').split('_')
    algo=parts[-2] if parts[-1]=='raw' else parts[-1]
    keycol=df.columns[0]
    return algo, keycol, df


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--indir', required=True)
    ap.add_argument('--prefix', required=True, help='e.g. cec2017 or engineering')
    ap.add_argument('--imsa_name', default='IMSA')
    args=ap.parse_args()

    files=sorted(glob.glob(os.path.join(args.indir, f'{args.prefix}_*_raw.csv')))
    if not files:
        raise SystemExit('No raw csv files found')

    alg_dfs={}
    item_names=None
    for path in files:
        algo, keycol, df = load_raw_csv(path)
        # robust algo parse
        algo = os.path.basename(path).replace(f'{args.prefix}_','').replace('_raw.csv','')
        alg_dfs[algo]=df
        if item_names is None:
            item_names=df[keycol].tolist(); first_col=keycol

    # mean/std/rank
    summary_rows=[]
    rank_rows=[]
    for idx, item in enumerate(item_names):
        means={}; stds={}
        for algo, df in alg_dfs.items():
            vals=df.iloc[idx,1:].astype(float).values
            _, m, s, _ = summarize_runs(vals)
            means[algo]=m; stds[algo]=s
        ranks={a:r+1 for r,(a,_) in enumerate(sorted(means.items(), key=lambda kv: kv[1]))}
        rank_row={first_col:item}
        summary_row={first_col:item}
        for algo in alg_dfs:
            summary_row[algo]=f"{means[algo]:.6g} ± {stds[algo]:.6g}"
            rank_row[algo]=ranks[algo]
        summary_rows.append(summary_row); rank_rows.append(rank_row)

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.indir, f'{args.prefix}_table_mean_std.csv'), index=False)
    pd.DataFrame(rank_rows).to_csv(os.path.join(args.indir, f'{args.prefix}_table_rank.csv'), index=False)

    # average rank
    rank_df=pd.DataFrame(rank_rows)
    avg = {algo: rank_df[algo].mean() for algo in alg_dfs}
    pd.DataFrame([avg]).to_csv(os.path.join(args.indir, f'{args.prefix}_average_rank.csv'), index=False)

    # Wilcoxon IMSA vs all
    rows=[]
    if args.imsa_name in alg_dfs:
        ref=alg_dfs[args.imsa_name]
        for idx, item in enumerate(item_names):
            refvals=ref.iloc[idx,1:].astype(float).values
            row={first_col:item}
            for algo, df in alg_dfs.items():
                if algo==args.imsa_name:
                    continue
                vals=df.iloc[idx,1:].astype(float).values
                stat,p = ranksums(refvals, vals)
                mref=np.mean(refvals); malt=np.mean(vals)
                sign='=' if p>=0.05 else ('+' if mref<malt else '-')
                row[algo]=f'{sign} (p={p:.3g})'
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(args.indir, f'{args.prefix}_wilcoxon_{args.imsa_name}_vs_all.csv'), index=False)

    print('Saved analysis files in', args.indir)

if __name__=='__main__':
    main()
