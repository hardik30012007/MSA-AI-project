import argparse, csv, os
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algos.msa import run_msa

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

from opfunu.cec_based import cec2017

def get_problem(fid, dim):
    cname = f'F{fid}2017'
    return getattr(cec2017, cname)(ndim=dim)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results_cec2017_msa')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    dim = 30
    runs = 50
    max_fes = 60000
    pop_size = 50
    functions = list(range(1, 30))  # CEC2017 has 29 functions (F1-F29)

    raw_path = os.path.join(args.outdir, 'cec2017_MSA_raw.csv')
    with open(raw_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Function'] + [f'Run{i}' for i in range(1, runs+1)])
        for fid in functions:
            prob = get_problem(fid, dim)
            bounds = list(zip(prob.lb, prob.ub))
            vals = []
            print(f'Running CEC2017 F{fid} MSA (Dim={dim}) ...')
            for r in range(runs):
                _, best, _ = run_msa(prob.evaluate, bounds, pop_size=pop_size, max_fes=max_fes, seed=1000*fid + r)
                vals.append(best)
            w.writerow([f'F{fid}'] + vals)
    
    print(f'Saved {raw_path}')

if __name__ == '__main__':
    main()