# Final project folder: what to run and where

This folder keeps your original files and adds a final-evaluation pipeline around them.

## What is kept from your original zip
- `msa.py`
- `improved_msa.py`
- your earlier CSVs / plots / pressure vessel script

## New files added
- `optimizers.py` -> standardized MSA/IMSA + 7 baselines (PSO, GWO, DE, WOA, GA, SSA, HHO)
- `engineering_problems.py` -> 4 engineering problems
- `run_cec_suite_final.py` -> raw 50-run benchmark CSV generator for CEC2014/2017
- `run_engineering_suite_final.py` -> raw 50-run engineering CSV generator
- `analyze_results.py` -> mean/std, rank, average rank, Wilcoxon tables
- `plot_convergence_compare_final.py`
- `plot_boxplots_final.py`

## Required install
```bash
pip install -r requirements.txt
```
If Python 3.13 gives `pkg_resources` issue, keep your local `pkg_resources.py` file in this folder.

## IMPORTANT settings for teacher requirements
- 60000 evaluations per function per run
- 50 runs per function
- use `--max_fes 60000 --runs 50`

## Fastest completion strategy
### Run locally in VS Code
- `MSA`
- `IMSA`

### Run in Google Colab (same settings)
- `PSO,GWO,DE,WOA,GA,SSA,HHO`

Keep same:
- dimension
- function list
- pop_size
- max_fes
- runs

## Recommended subset to finish quickly
Teacher mentioned CEC-2014, 2017 and 4-5 engineering problems. Fastest realistic subset:
- CEC2014 functions: `1,2,3,4,5,6,7,8,9,10`
- CEC2017 functions: `1,2,3,4,5,6,7,8,9,10`
- all 4 engineering problems in `engineering_problems.py`

If you have time, extend the function list.

## Step-by-step commands
### 1) Generate raw CSVs for CEC2017 (example with MSA + IMSA only)
```bash
python run_cec_suite_final.py --suite cec2017 --algorithms MSA,IMSA --functions 1,2,3,4,5,6,7,8,9,10 --dim 30 --runs 50 --max_fes 60000 --pop_size 50 --outdir results_cec2017
```

### 2) Generate raw CSVs for CEC2014
```bash
python run_cec_suite_final.py --suite cec2014 --algorithms MSA,IMSA --functions 1,2,3,4,5,6,7,8,9,10 --dim 30 --runs 50 --max_fes 60000 --pop_size 50 --outdir results_cec2014
```

### 3) Generate engineering raw CSVs
```bash
python run_engineering_suite_final.py --algorithms MSA,IMSA --runs 50 --max_fes 60000 --pop_size 50 --outdir results_engineering
```

### 4) Analyze tables for any folder
```bash
python analyze_results.py --indir results_cec2017 --prefix cec2017 --imsa_name IMSA
python analyze_results.py --indir results_cec2014 --prefix cec2014 --imsa_name IMSA
python analyze_results.py --indir results_engineering --prefix engineering --imsa_name IMSA
```
This creates:
- `*_table_mean_std.csv`
- `*_table_rank.csv`
- `*_average_rank.csv`
- `*_wilcoxon_IMSA_vs_all.csv`

### 5) Convergence plot
```bash
python plot_convergence_compare_final.py --suite cec2017 --function 10 --algorithms MSA,IMSA,PSO,GWO --dim 30 --max_fes 60000 --pop_size 50 --out cec2017_f10_compare.png
```

### 6) Boxplot from raw CSVs
```bash
python plot_boxplots_final.py --csvs results_cec2017/cec2017_MSA_raw.csv,results_cec2017/cec2017_IMSA_raw.csv --item F10 --out boxplot_f10.png
```

## What to submit / show
- mean/std tables
- rank tables
- Wilcoxon table
- convergence curve (2–3 functions)
- boxplot comparison (1–2 hard functions)
- engineering summary tables

## IMPORTANT honesty note
The baseline algorithms added here are compact educational implementations so that you can complete the assignment under equal settings. They are not official author codes for each algorithm.
