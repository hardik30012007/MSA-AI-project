import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support, get_context

import numpy as np

# Keep BLAS / OpenMP from spawning extra threads inside every worker.
# This matters a lot when we parallelize across processes.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

from optimizers import ALGORITHMS

# pkg_resources shim support for opfunu on py3.13
import opfunu.utils.operator as op


def different_powers_safe(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    ndim = len(x)
    if ndim <= 1:
        return float(np.sum(x ** 2))
    idx = np.arange(ndim)
    return float(np.sum(10 ** (6.0 * idx / (ndim - 1)) * x ** 2))


if hasattr(op, "different_powers"):
    op.different_powers = different_powers_safe
if hasattr(op, "different_powers__"):
    op.different_powers__ = different_powers_safe

from opfunu.cec_based import cec2014, cec2017

SUITES = {
    "cec2014": cec2014,
    "cec2017": cec2017,
}


def get_problem(module, suite, fid, dim):
    cname = f"F{fid}{suite[-4:]}"
    return getattr(module, cname)(ndim=dim)


def _parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def _run_function_task(task):
    """
    Run all repeated runs for a single benchmark function.

    This keeps the expensive benchmark object creation inside the worker
    and avoids shipping large objects between processes.
    """
    suite, algo, fid, dim, runs, pop_size, max_fes = task
    module = SUITES[suite]
    prob = get_problem(module, suite, fid, dim)
    bounds = list(zip(prob.lb, prob.ub))
    runner = ALGORITHMS[algo]

    results = []
    for run_idx in range(runs):
        seed = 1000 * fid + run_idx
        _, best, _ = runner(
            prob.evaluate,
            bounds,
            pop_size=pop_size,
            max_fes=max_fes,
            seed=seed,
        )
        results.append(float(best))

    return fid, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["cec2014", "cec2017"], required=True)
    ap.add_argument(
        "--algorithms",
        default="MSA,IMSA,PSO,GWO,DE,WOA,GA,SSA,HHO",
    )
    ap.add_argument(
        "--functions",
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30",
        help="Comma-separated function ids.",
    )
    ap.add_argument("--dim", type=int, default=30)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--max_fes", type=int, default=60000)
    ap.add_argument("--pop_size", type=int, default=50)
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of parallel worker processes. Default uses roughly the physical core count.",
    )
    ap.add_argument("--outdir", default="results_cec")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    module = SUITES[args.suite]
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    functions = _parse_int_list(args.functions)

    # Use one process pool per algorithm so each algorithm's output stays clean
    # and we can write results to CSV in function order.
    ctx = get_context("spawn")
    max_workers = max(1, min(args.workers, len(functions)))

    for algo in algos:
        raw_path = os.path.join(args.outdir, f"{args.suite}_{algo}_raw.csv")
        print(f"\n=== Running {args.suite} with {algo} on {len(functions)} functions ===")
        print(f"Workers: {max_workers} | dim={args.dim} | runs={args.runs} | pop={args.pop_size} | max_fes={args.max_fes}")

        tasks = [
            (args.suite, algo, fid, args.dim, args.runs, args.pop_size, args.max_fes)
            for fid in functions
        ]

        ordered = {}
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            for fid, values in ex.map(_run_function_task, tasks, chunksize=1):
                ordered[fid] = values
                print(f"Done {args.suite} F{fid} {algo}")

        with open(raw_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Function"] + [f"Run{i}" for i in range(1, args.runs + 1)])
            for fid in functions:
                w.writerow([f"F{fid}"] + ordered[fid])

        print("Saved", raw_path)


if __name__ == "__main__":
    freeze_support()
    main()
