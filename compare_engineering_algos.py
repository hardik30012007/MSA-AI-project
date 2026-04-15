import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from optimizers import ALGORITHMS
from engineering_problems import PROBLEMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", required=True, help="Comma-separated, e.g. MSA,IMSA,PSO,GWO,DE,WOA,GA,SSA,HHO")
    parser.add_argument("--problems", required=True, help="Comma-separated problem names from PROBLEMS")
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--max_fes", type=int, default=60000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    algos = [a.strip().upper() for a in args.algorithms.split(",") if a.strip()]
    problems = [p.strip() for p in args.problems.split(",") if p.strip()]

    for problem_name in problems:
        if problem_name not in PROBLEMS:
            raise ValueError(f"Unknown problem: {problem_name}. Available: {list(PROBLEMS.keys())}")

        problem = PROBLEMS[problem_name]
        func = problem["func"]
        bounds = problem["bounds"]

        # ---------- Convergence Plot ----------
        plt.figure(figsize=(10, 6))

        raw_boxplot_rows = []

        for algo_name in algos:
            if algo_name not in ALGORITHMS:
                print(f"Skipping unknown algorithm: {algo_name}")
                continue

            runner = ALGORITHMS[algo_name]

            # one representative run for convergence
            _, _, history = runner(
                func=func,
                bounds=bounds,
                pop_size=args.pop_size,
                max_fes=args.max_fes,
                seed=args.seed
            )

            plt.plot(history, label=algo_name, linewidth=2)

            # multiple runs for boxplot/raw csv
            run_vals = []
            for r in range(args.runs):
                _, best_val, _ = runner(
                    func=func,
                    bounds=bounds,
                    pop_size=args.pop_size,
                    max_fes=args.max_fes,
                    seed=args.seed + r
                )
                run_vals.append(best_val)

            row = {"Problem": problem_name, "Algorithm": algo_name}
            for i, v in enumerate(run_vals, start=1):
                row[f"Run{i}"] = v
            raw_boxplot_rows.append(row)

        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (log scale)")
        plt.title(f"Convergence Comparison - {problem_name}")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()

        conv_path = os.path.join(args.outdir, f"{problem_name}_convergence.png")
        plt.savefig(conv_path, dpi=200)
        plt.close()

        # ---------- Raw CSV for this problem ----------
        raw_df = pd.DataFrame(raw_boxplot_rows)
        raw_csv_path = os.path.join(args.outdir, f"{problem_name}_raw.csv")
        raw_df.to_csv(raw_csv_path, index=False)

        # ---------- Boxplot ----------
        plt.figure(figsize=(10, 6))
        data = []
        labels = []

        for _, row in raw_df.iterrows():
            vals = [row[f"Run{i}"] for i in range(1, args.runs + 1)]
            data.append(vals)
            labels.append(row["Algorithm"])

        plt.boxplot(data, labels=labels)
        plt.yscale("log")
        plt.xlabel("Algorithm")
        plt.ylabel("Best Fitness (log scale)")
        plt.title(f"Boxplot Comparison - {problem_name}")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        box_path = os.path.join(args.outdir, f"{problem_name}_boxplot.png")
        plt.savefig(box_path, dpi=200)
        plt.close()

        print(f"Saved: {conv_path}")
        print(f"Saved: {raw_csv_path}")
        print(f"Saved: {box_path}")


if __name__ == "__main__":
    main()