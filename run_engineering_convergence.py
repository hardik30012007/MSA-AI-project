import os
import argparse
import matplotlib.pyplot as plt

from engineering_problems import PROBLEMS
from optimizers import run_msa, run_imsa


def get_algo(name):
    name = name.strip().upper()
    if name == "MSA":
        return run_msa
    elif name == "IMSA":
        return run_imsa
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", required=True, help="Comma-separated, e.g. IMSA or MSA,IMSA")
    parser.add_argument("--problems", required=True, help="Comma-separated, e.g. pressure_vessel,spring")
    parser.add_argument("--max_fes", type=int, default=60000)
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    algorithms = [x.strip() for x in args.algorithms.split(",") if x.strip()]
    problems = [x.strip() for x in args.problems.split(",") if x.strip()]

    for problem_name in problems:
        if problem_name not in PROBLEMS:
            raise ValueError(f"Unknown problem: {problem_name}. Available: {list(PROBLEMS.keys())}")

        problem = PROBLEMS[problem_name]
        func = problem["func"]
        bounds = problem["bounds"]

        plt.figure(figsize=(8, 5))

        for algo_name in algorithms:
            algo = get_algo(algo_name)

            best_sol, best_val, history = algo(
                func=func,
                bounds=bounds,
                pop_size=args.pop_size,
                max_fes=args.max_fes,
                seed=42
            )

            

        plt.plot(history, label=algo_name)

        plt.yscale("log")   # ⭐ YE LINE ADD KARNA HAI

        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness (log scale)")
        plt.title(f"Convergence - {problem_name}")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        out_path = os.path.join(args.outdir, f"{problem_name}_convergence.png")
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()