import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support, get_context

import numpy as np
from optimizers import ALGORITHMS

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import opfunu.utils.operator as op

def different_powers_safe(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    ndim = len(x)
    if ndim <= 1:
        return float(np.sum(x ** 2))
    idx = np.arange(ndim)
    return float(np.sum(10 ** (6.0 * idx / (ndim - 1)) * x ** 2))

if hasattr(op, 'different_powers'):
    op.different_powers = different_powers_safe
if hasattr(op, 'different_powers__'):
    op.different_powers__ = different_powers_safe

from opfunu.cec_based import cec2014, cec2017
SUITES = {'cec2014': cec2014, 'cec2017': cec2017}

def get_problem(module, suite, fid, dim):
    cname = f'F{fid}{suite[-4:]}'
    return getattr(module, cname)(ndim=dim)

def task_runner(task):
    suite, algo, fid, dim, runs, pop_size, max_fes = task
    module = SUITES[suite]
    prob = get_problem(module, suite, fid, dim)
    bounds = list(zip(prob.lb, prob.ub))
    runner = ALGORITHMS[algo]
    vals = []
    for r in range(runs):
        seed = 1000 * fid + r
        _, best, _ = runner(prob.evaluate, bounds, pop_size=pop_size, max_fes=max_fes, seed=seed)
        vals.append(float(best))
    return fid, vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', choices=['cec2014', 'cec2017'], required=True)
    ap.add_argument('--algorithms', default='IMSA,MSA,PSO,GWO,WOA,SSA,DE,GA,HHO')
    ap.add_argument('--functions', default='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30')
    ap.add_argument('--dim', type=int, default=30)
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--max-fes', type=int, default=60000)
    ap.add_argument('--pop-size', type=int, default=50)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 1) // 2))
    ap.add_argument('--outdir', default='benchmarks')
    args = ap.parse_args()

    functions = [int(x) for x in args.functions.split(',') if x.strip()]
    algos = [a.strip() for a in args.algorithms.split(',') if a.strip()]

    base = os.path.join(args.outdir, args.suite)
    raw_dir = os.path.join(base, 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    ctx = get_context('spawn')
    for algo in algos:
        raw_path = os.path.join(raw_dir, f'{algo}.csv')
        tasks = [(args.suite, algo, fid, args.dim, args.runs, args.pop_size, args.max_fes) for fid in functions]
        results = {}
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)), mp_context=ctx) as ex:
            for fid, vals in ex.map(task_runner, tasks, chunksize=1):
                results[fid] = vals
                print(f'Done {args.suite} F{fid} {algo}')
        with open(raw_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Function'] + [f'Run{i}' for i in range(1, args.runs + 1)])
            for fid in functions:
                w.writerow([f'F{fid}'] + results[fid])
        print('Saved', raw_path)

if __name__ == '__main__':
    freeze_support()
    main()
