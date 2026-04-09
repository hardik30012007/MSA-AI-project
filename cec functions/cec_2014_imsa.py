import argparse, csv, os
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algos.imsa import run_imsa

# 🔥 Limit internal threading (VERY IMPORTANT for performance)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# pkg_resources shim support for opfunu on py3.13
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

# CEC 2014 import
from opfunu.cec_based import cec2014

def get_problem(fid, dim):
    cname = f'F{fid}2014'
    return getattr(cec2014, cname)(ndim=dim)


# 🔥 Parallel worker: runs all 50 runs for one function
def run_function(fid):
    dim = 30
    runs = 50
    max_fes = 60000
    pop_size = 50

    prob = get_problem(fid, dim)
    bounds = list(zip(prob.lb, prob.ub))

    vals = []
    print(f'Running CEC2014 F{fid} IMSA...')

    for r in range(runs):
        _, best, _ = run_imsa(
            prob.evaluate,
            bounds,
            pop_size=pop_size,
            max_fes=max_fes,
            seed=1000*fid + r
        )
        vals.append(best)

    return fid, vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results_cec2014_imsa')
    ap.add_argument('--workers', type=int, default=12)  # 🔥 Ultra 9 → use more cores
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    functions = list(range(1, 31))  # F1–F30

    raw_path = os.path.join(args.outdir, 'cec2014_IMSA_raw.csv')

    results = {}

    # 🔥 Parallel execution
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for fid, vals in executor.map(run_function, functions):
            results[fid] = vals
            print(f'Finished F{fid}')

    # 🔥 Save results
    with open(raw_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Function'] + [f'Run{i}' for i in range(1, 51)])

        for fid in functions:
            w.writerow([f'F{fid}'] + results[fid])

    print(f'Saved {raw_path}')


if __name__ == '__main__':
    freeze_support()
    main()