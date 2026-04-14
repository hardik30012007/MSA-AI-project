from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np
from scipy.stats import ranksums


def num_key(label: str) -> int:
    m = re.search(r"(\d+)", str(label))
    return int(m.group(1)) if m else 10**9


def load_raw_csv(path: Path):
    df = pd.read_csv(path)
    label_col = df.columns[0]
    df = df.copy()
    df[label_col] = df[label_col].astype(str)
    df["__order"] = df[label_col].map(num_key)
    df = df.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="Folder with raw CSVs")
    ap.add_argument("--reference", default="IMSA", help="Reference algorithm")
    ap.add_argument(
        "--algorithms",
        default="IMSA,MSA,PSO,GWO,WOA,SSA,DE,GA,HHO",
        help="Comma-separated algorithm order",
    )
    ap.add_argument("--output-dir", required=True, help="Folder for Wilcoxon outputs")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    algo_order = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    algo_frames = {}

    for algo in algo_order:
        path = raw_dir / f"{algo}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        algo_frames[algo] = load_raw_csv(path)

    label_col = algo_frames[algo_order[0]].columns[0]
    labels = algo_frames[algo_order[0]][label_col].tolist()

    # function-wise pairwise test table
    rows = []
    for i, label in enumerate(labels):
        row = {label_col: label}
        ref_vals = algo_frames[args.reference].iloc[i, 1:].astype(float).to_numpy()

        wins = losses = ties = 0
        for algo in algo_order:
            if algo == args.reference:
                continue
            vals = algo_frames[algo].iloc[i, 1:].astype(float).to_numpy()
            stat, p = ranksums(ref_vals, vals)

            ref_mean = float(np.mean(ref_vals))
            other_mean = float(np.mean(vals))

            if p < 0.05:
                if ref_mean < other_mean:
                    result = "Win"
                    wins += 1
                else:
                    result = "Loss"
                    losses += 1
            else:
                result = "Tie"
                ties += 1

            row[f"{args.reference}_vs_{algo}_p"] = p
            row[f"{args.reference}_vs_{algo}_result"] = result

        row[f"{args.reference}_wins"] = wins
        row[f"{args.reference}_losses"] = losses
        row[f"{args.reference}_ties"] = ties
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_file = output_dir / f"wilcoxon_{args.reference}_vs_all.csv"
    out_df.to_csv(out_file, index=False)

    # simple summary table
    summary = []
    for algo in algo_order:
        if algo == args.reference:
            continue
        wins = (out_df[f"{args.reference}_vs_{algo}_result"] == "Win").sum()
        losses = (out_df[f"{args.reference}_vs_{algo}_result"] == "Loss").sum()
        ties = (out_df[f"{args.reference}_vs_{algo}_result"] == "Tie").sum()
        summary.append(
            {
                "Algorithm": algo,
                "Win": int(wins),
                "Loss": int(losses),
                "Tie": int(ties),
            }
        )

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_dir / f"wilcoxon_summary_{args.reference}.csv", index=False)

    print(f"Saved {out_file}")
    print(f"Saved {output_dir / f'wilcoxon_summary_{args.reference}.csv'}")


if __name__ == "__main__":
    main()