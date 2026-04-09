import argparse, csv, os
import numpy as np
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
if hasattr(op, 'different_powers'): op.different_powers = different_powers_safe
if hasattr(op, 'different_powers__'): op.different_powers__ = different_powers_safe

from opfunu.cec_based import cec2014, cec2017

SUITES={'cec2014': cec2014, 'cec2017': cec2017}


def get_problem(module, suite, fid, dim):
    cname = f'F{fid}{suite[-4:]}'
    return getattr(module, cname)(ndim=dim)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--suite', choices=['cec2014','cec2017'], required=True)
    ap.add_argument('--algorithms', default='MSA,IMSA,PSO,GWO,DE,WOA,GA,SSA,HHO')
    ap.add_argument('--functions', default='1,2,3,4,5,6,7,8,9,10')
    ap.add_argument('--dim', type=int, default=30)
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--max_fes', type=int, default=60000)
    ap.add_argument('--pop_size', type=int, default=50)
    ap.add_argument('--outdir', default='results_cec')
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    module=SUITES[args.suite]
    algos=[a.strip() for a in args.algorithms.split(',') if a.strip()]
    functions=[int(x) for x in args.functions.split(',') if x.strip()]

    for algo in algos:
        raw_path=os.path.join(args.outdir, f'{args.suite}_{algo}_raw.csv')
        with open(raw_path,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['Function']+[f'Run{i}' for i in range(1,args.runs+1)])
            for fid in functions:
                prob=get_problem(module, args.suite, fid, args.dim)
                bounds=list(zip(prob.lb, prob.ub))
                vals=[]
                print(f'Running {args.suite} F{fid} {algo} ...')
                for r in range(args.runs):
                    _, best, _ = ALGORITHMS[algo](prob.evaluate, bounds, pop_size=args.pop_size, max_fes=args.max_fes, seed=1000*fid+r)
                    vals.append(best)
                w.writerow([f'F{fid}']+vals)
        print('Saved', raw_path)

if __name__=='__main__':
    main()
