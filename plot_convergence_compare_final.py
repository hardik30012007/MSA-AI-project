import argparse, os
import matplotlib.pyplot as plt
from optimizers import ALGORITHMS

# opfunu patch
import numpy as np
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
from opfunu.cec_based import cec2014, cec2017
SUITES={'cec2014': cec2014, 'cec2017': cec2017}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite', choices=['cec2014','cec2017'], required=True)
    ap.add_argument('--function', type=int, required=True)
    ap.add_argument('--algorithms', default='MSA,IMSA,PSO,GWO')
    ap.add_argument('--dim', type=int, default=30)
    ap.add_argument('--max_fes', type=int, default=60000)
    ap.add_argument('--pop_size', type=int, default=50)
    ap.add_argument('--out', default='convergence_compare.png')
    args=ap.parse_args()
    module=SUITES[args.suite]
    cname=f'F{args.function}{args.suite[-4:]}'
    prob=getattr(module, cname)(ndim=args.dim)
    bounds=list(zip(prob.lb, prob.ub))

    plt.figure(figsize=(8,5))
    for algo in [a.strip() for a in args.algorithms.split(',') if a.strip()]:
        _, _, hist = ALGORITHMS[algo](prob.evaluate, bounds, pop_size=args.pop_size, max_fes=args.max_fes, seed=123)
        plt.plot(hist, label=algo)
    plt.yscale('log')
    plt.xlabel('Iteration-like update steps')
    plt.ylabel('Best fitness so far')
    plt.title(f'{args.suite.upper()} F{args.function} convergence')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(args.out, dpi=180)
    print('Saved', args.out)

if __name__=='__main__':
    main()
