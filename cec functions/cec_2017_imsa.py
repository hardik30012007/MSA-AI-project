import argparse, csv, os
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algos.imsa import run_imsa

# 🔥 Prevent thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# opfunu fix
import opfunu.utils.operator as op

def different_powers_safe(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    ndim = len(x)
    if ndim <= 1:
        return float(np.sum(x ** 2))
    idx = np.arange(ndim)
    return float(np.sum(10 ** (6.0 * idx / (ndim - 1)) * x ** 2))

if hasattr(op, 'different_powers'): op.different_powers = different_powers_safe
if hasattr(op, 'different_powers__'): op.different_powers__ = different_powers_safe

from opfunu.cec_based import cec2017

def get_problem(fid, dim):
    cname = f'F{fid}2017'
    return getattr(cec2017, cname)(ndim=dim)


# 🔥 ONE TASK = ONE RUN (MAX PARALLELISM)
def run_single(args):
    fid, r = args

    dim = 30
    max_fes = 60000
    pop_size = 50

    prob = get_problem(fid, dim)
    bounds = list(zip(prob.lb, prob.ub))

    _, best, _ = run_imsa(
        prob.evaluate,
        bounds,
        pop_size=pop_size,
        max_fes=max_fes,
        seed=1000*fid + r
    )

    return fid, r, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results_cec2017_imsa')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    functions = list(range(1, 31))
    runs = 50

    raw_path = os.path.join(args.outdir, 'cec2017_IMSA_raw.csv')

    # Prepare all tasks
    tasks = [(fid, r) for fid in functions for r in range(runs)]

    # Storage
    results = {fid: [None]*runs for fid in functions}

    print(f"Total tasks: {len(tasks)} (30 functions × 50 runs)")
    print(f"Using {args.workers} workers")

    # 🔥 FULL PARALLEL EXECUTION
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_single, t) for t in tasks]

        for future in as_completed(futures):
            fid, r, best = future.result()
            results[fid][r] = best
            print(f"Done F{fid} Run{r}")

    # Save CSV
    with open(raw_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Function'] + [f'Run{i}' for i in range(1, runs+1)])

        for fid in functions:
            w.writerow([f'F{fid}'] + results[fid])

    print(f"Saved {raw_path}")


if __name__ == '__main__':
    freeze_support()
    main()