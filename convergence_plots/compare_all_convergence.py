import argparse
import matplotlib.pyplot as plt
from optimizers import ALGORITHMS
from opfunu.cec_based import cec2017, cec2014


def get_problem(suite, function_id, dim):
    suite = suite.lower()

    if suite == "cec2017":
        cls = getattr(cec2017, f"F{function_id}2017")
    elif suite == "cec2014":
        cls = getattr(cec2014, f"F{function_id}2014")
    else:
        raise ValueError("suite must be cec2014 or cec2017")

    problem = cls(ndim=dim)
    func = problem.evaluate
    bounds = list(zip(problem.lb, problem.ub))
    return func, bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, help="cec2014 or cec2017")
    parser.add_argument("--function", type=int, required=True, help="Function number")
    parser.add_argument(
        "--algorithms",
        default="MSA,IMSA,PSO,GWO,DE,WOA,GA,SSA,HHO",
        help="Comma-separated list of algorithms"
    )
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--max_fes", type=int, default=60000)
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output png path")
    args = parser.parse_args()

    algo_list = [a.strip().upper() for a in args.algorithms.split(",") if a.strip()]

    func, bounds = get_problem(args.suite, args.function, args.dim)

    plt.figure(figsize=(10, 6))

    for algo_name in algo_list:
        if algo_name not in ALGORITHMS:
            print(f"Skipping unknown algorithm: {algo_name}")
            continue

        print(f"Running {algo_name}...")
        runner = ALGORITHMS[algo_name]

        best_sol, best_val, history = runner(
            func=func,
            bounds=bounds,
            pop_size=args.pop_size,
            max_fes=args.max_fes,
            seed=args.seed
        )

        plt.plot(history, label=algo_name, linewidth=2)

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (log scale)")
    plt.title(f"Convergence Comparison - {args.suite.upper()} F{args.function}")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.show()

    print(f"\nSaved plot to: {args.out}")


if __name__ == "__main__":
    main()