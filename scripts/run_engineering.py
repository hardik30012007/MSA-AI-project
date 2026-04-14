import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

from optimizers import ALGORITHMS
from engineering import PROBLEMS

def run_problem(task):
    algo, pname, runs, pop_size, max_fes = task
    pdata = PROBLEMS[pname]
    vals = []
    for r in range(runs):
        _, best, _ = ALGORITHMS[algo](pdata['func'], pdata['bounds'], pop_size=pop_size, max_fes=max_fes, seed=5000 + r)
        vals.append(float(best))
    return pname, vals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algorithms', default='MSA,IMSA,PSO,GWO,DE,WOA,GA,SSA,HHO')
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--max-fes', type=int, default=60000)
    ap.add_argument('--pop-size', type=int, default=50)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 1) // 2))
    ap.add_argument('--outdir', default='engineering/raw')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    algos = [a.strip() for a in args.algorithms.split(',') if a.strip()]
    problems = list(PROBLEMS.keys())

    for algo in algos:
        out_path = os.path.join(args.outdir, f'{algo}.csv')
        tasks = [(algo, pname, args.runs, args.pop_size, args.max_fes) for pname in problems]
        results = {}
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            for pname, vals in ex.map(run_problem, tasks, chunksize=1):
                results[pname] = vals
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Problem'] + [f'Run{i}' for i in range(1, args.runs + 1)])
            for pname in problems:
                w.writerow([pname] + results[pname])
        print('Saved', out_path)

if __name__ == '__main__':
    freeze_support()
    main()
