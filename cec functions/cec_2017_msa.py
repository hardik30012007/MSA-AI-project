import argparse, csv, os
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algos.msa import run_msa

# Limit internal threading (VERY IMPORTANT)
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


# ---------- SAFE PROBLEM ----------
def get_problem(fid, dim):
    cname = f'F{fid}2017'
    if not hasattr(cec2017, cname):
        return None
    return getattr(cec2017, cname)(ndim=dim)


# ---------- LOAD CHECKPOINT ----------
def load_existing(path, functions):
    results = {}

    if not os.path.exists(path):
        return results

    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            fid = int(row[0][1:])
            vals = list(map(float, row[1:]))
            results[fid] = vals

    return results


# ---------- SAVE CHECKPOINT ----------
def save_progress(path, results):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Function'] + [f'Run{i}' for i in range(1, 51)])

        for fid in sorted(results.keys()):
            w.writerow([f'F{fid}'] + results[fid])


# ---------- WORKER ----------
def run_function(fid):
    dim = 30
    runs = 50
    max_fes = 60000
    pop_size = 50

    prob = get_problem(fid, dim)
    if prob is None:
        return fid, None

    bounds = list(zip(prob.lb, prob.ub))

    vals = []
    print(f'[START] F{fid}')

    for r in range(runs):
        _, best, _ = run_msa(
            prob.evaluate,
            bounds,
            pop_size=pop_size,
            max_fes=max_fes,
            seed=1000 * fid + r
        )
        vals.append(best)

    print(f'[DONE] F{fid}')
    return fid, vals


# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='results_cec2017_msa')
    ap.add_argument('--workers', type=int, default=12)  # ✅ as requested
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ✅ ONLY F1–F29
    functions = list(range(1, 30))

    raw_path = os.path.join(args.outdir, 'cec2017_MSA_raw.csv')

    # LOAD EXISTING
    results = load_existing(raw_path, functions)

    remaining = [fid for fid in functions if fid not in results]

    print("Already done:", sorted(results.keys()))
    print("Remaining:", remaining)

    # PARALLEL EXECUTION
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_function, fid): fid for fid in remaining}

        for future in as_completed(futures):
            fid = futures[future]

            try:
                f_id, vals = future.result()

                if vals is None:
                    print(f'[SKIP] F{fid}')
                    continue

                results[f_id] = vals

                # 🔥 SAVE AFTER EACH FUNCTION
                save_progress(raw_path, results)

                print(f'[SAVED] F{fid}')

            except Exception as e:
                print(f'[ERROR] F{fid}: {e}')

    print(f'\nFINAL SAVED: {raw_path}')


if __name__ == '__main__':
    freeze_support()
    main()