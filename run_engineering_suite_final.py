import argparse, csv, os
from optimizers import ALGORITHMS
from engineering_problems import PROBLEMS


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--algorithms', default='MSA,IMSA,PSO,GWO,DE,WOA,GA,SSA,HHO')
    ap.add_argument('--runs', type=int, default=50)
    ap.add_argument('--max_fes', type=int, default=60000)
    ap.add_argument('--pop_size', type=int, default=50)
    ap.add_argument('--outdir', default='results_engineering')
    args=ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    algos=[a.strip() for a in args.algorithms.split(',') if a.strip()]

    for algo in algos:
        raw_path=os.path.join(args.outdir, f'engineering_{algo}_raw.csv')
        with open(raw_path,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(['Problem']+[f'Run{i}' for i in range(1,args.runs+1)])
            for pname, pdata in PROBLEMS.items():
                vals=[]
                print(f'Running {pname} {algo} ...')
                for r in range(args.runs):
                    _, best, _ = ALGORITHMS[algo](pdata['func'], pdata['bounds'], pop_size=args.pop_size, max_fes=args.max_fes, seed=5000+r)
                    vals.append(best)
                w.writerow([pname]+vals)
        print('Saved', raw_path)

if __name__=='__main__':
    main()
