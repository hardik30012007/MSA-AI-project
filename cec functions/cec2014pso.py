import argparse, csv, os
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

# Path setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algos.other_algos import run_pso

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

from opfunu.cec_based import cec2014


# ---------- SAFE PROBLEM FETCH ----------
def get_problem(fid, dim):
    cname = f'F{fid}2014'
    if not hasattr(cec2014, cname):
        return None
    return getattr(cec2014, cname)(ndim=dim)


# ---------- LOAD CHECKPOINT ----------
def load_existing(path, functions, runs):
    results = {fid: [None]*runs for fid in functions}

    if not os.path.exists(path):
        return results

    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            fid = int(row[0][1:])
            vals = row[1:]
            for i, v in enumerate(vals):
                if v != '':
                    results[fid][i] = float(v)

    return results


# ---------- SAVE CHECKPOINT ----------
def save_progress(path, results, functions, runs):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Function'] + [f'Run{i}' for i in range(1, runs+1)])

        for fid in functions:
            row = [f'F{fid}']
            for v in results[fid]:
                row.append(v if v is not None else '')
            w.writerow(row)


# ---------- WORKER ----------
def run_single(args):
    fid, r = args

    dim = 30
    max_fes = 60000
    pop_size = 50

    prob = get_problem(fid, dim)
    if prob is None:
        return fid, r, None

    bounds = list(zip(prob.lb, prob.ub))

    _, best, _ = run_pso(
        prob.evaluate,
        bounds,
        pop_size=pop_size,
        max_fes=max_fes,
        seed=1000*fid + r
    )

    return fid, r, best


# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results_cec2014_pso')
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    functions = [fid for fid in range(1, 31) if hasattr(cec2014, f'F{fid}2014')]
    runs = 50

    raw_path = os.path.join(args.outdir, 'cec2014_PSO_raw.csv')

    # 🔥 LOAD PREVIOUS PROGRESS
    results = load_existing(raw_path, functions, runs)

    # 🔥 SKIP COMPLETED RUNS
    tasks = [
        (fid, r)
        for fid in functions
        for r in range(runs)
        if results[fid][r] is None
    ]

    print(f"Remaining tasks: {len(tasks)}")
    print(f"Using {args.workers} workers")

    # ---------- PARALLEL ----------
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_single, t) for t in tasks]

        for future in as_completed(futures):
            try:
                fid, r, best = future.result()

                if best is None:
                    print(f"[SKIP] F{fid}")
                    continue

                results[fid][r] = best

                # 🔥 SAVE AFTER EVERY RUN
                save_progress(raw_path, results, functions, runs)

                print(f"[SAVED] F{fid} Run{r}")

            except Exception as e:
                print(f"[ERROR] {e}")

    print(f"\nFINAL RESULTS SAVED: {raw_path}")


if __name__ == '__main__':
    freeze_support()
    main()