import pandas as pd
import os

# -------- Paths --------
input_folder = "cec functions/cec2014_processed_csvs"
output_file = "cec functions/cec2014_final_result.csv"

# -------- Algorithm order (IMPORTANT for format) --------
algo_order = ["IMSA", "MSA", "PSO", "GWO", "WOA", "SSA", "DE", "GA", "HHO"]

# -------- Load all CSVs --------
algo_data = {}

for algo in algo_order:
    path = os.path.join(input_folder, f"{algo}.csv")
    algo_data[algo] = pd.read_csv(path)

# -------- Functions --------
functions = algo_data[algo_order[0]]["Function"]

# -------- Build header --------
header = ["Function"]
for algo in algo_order:
    header.extend([f"{algo}", f"{algo}", f"{algo}"])  # Mean Std Rank

# -------- Build rows --------
rows = []

for i, func in enumerate(functions):
    row = [func]
    
    means = {}
    
    # collect means/std
    for algo in algo_order:
        mean = algo_data[algo].loc[i, "Mean"]
        std = algo_data[algo].loc[i, "Std"]
        means[algo] = mean
    
    # ranking (lower = better)
    sorted_algos = sorted(means, key=means.get)
    ranks = {algo: rank+1 for rank, algo in enumerate(sorted_algos)}
    
    # fill row in same order
    for algo in algo_order:
        mean = algo_data[algo].loc[i, "Mean"]
        std = algo_data[algo].loc[i, "Std"]
        rank = ranks[algo]
        
        row.extend([mean, std, rank])
    
    rows.append(row)

# -------- Create DataFrame --------
final_df = pd.DataFrame(rows, columns=header)

# -------- Add second header row (like your sample) --------
second_header = ["algo"]
for _ in algo_order:
    second_header.extend(["Mean", "Std", "Rank"])

final_df.loc[-1] = second_header
final_df.index = final_df.index + 1
final_df = final_df.sort_index()

# -------- Save --------
final_df.to_csv(output_file, index=False)

print("🔥 Final formatted CSV saved at:", output_file)