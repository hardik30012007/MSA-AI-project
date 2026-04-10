import pandas as pd
import numpy as np
from scipy.stats import ranksums

# ===== INPUT FILES =====
# put all your CSVs here
FILES = {
    "MSA": "msa.csv",
    "IMSA": "imsa.csv",
    "PSO": "PSO.csv",
    "GWO": "GWO.csv",
    "DE": "DE.csv",
    "WOA": "WOA.csv",
    "GA": "GA.csv",
    "SSA": "SSA.csv",
    "HHO": "HHO.csv",
}

# ===== LOAD DATA =====
data = {}
for name, file in FILES.items():
    df = pd.read_csv(file)
    data[name] = df.set_index("Problem")

# ===== GET PROBLEMS =====
problems = data[list(data.keys())[0]].index

results = []

for problem in problems:
    row = {"Problem": problem}
    
    values = {}

    # mean/std
    for algo in data:
        runs = data[algo].loc[problem].values.astype(float)
        values[algo] = runs
        row[f"{algo}_mean"] = np.mean(runs)
        row[f"{algo}_std"] = np.std(runs)

    # ranking (lower is better)
    means = {algo: row[f"{algo}_mean"] for algo in data}
    sorted_algos = sorted(means, key=means.get)
    
    for i, algo in enumerate(sorted_algos):
        row[f"{algo}_rank"] = i + 1

    # Wilcoxon IMSA vs others
    imsa_vals = values["IMSA"]
    for algo in data:
        if algo == "IMSA":
            continue
        stat, p = ranksums(imsa_vals, values[algo])
        row[f"IMSA_vs_{algo}"] = "Win" if p < 0.05 and np.mean(imsa_vals) < np.mean(values[algo]) else \
                                 "Lose" if p < 0.05 else "Tie"

    results.append(row)

# ===== SAVE =====
final_df = pd.DataFrame(results)
final_df.to_csv("rank wise result cec 2017.csv", index=False)

print("DONE: rank wise result cec 2017.csv generated")