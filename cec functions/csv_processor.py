import pandas as pd
import os

# -------- File paths --------
input_file = "results_cec2017_imsa/cec2017_IMSA_raw.csv"   # your raw CSV file
output_file = "results_cec2017_imsa/processed_cec2017_imsa.csv"  # output file

# -------- Load CSV --------
df = pd.read_csv(input_file)

# -------- Extract run columns --------
# Select all columns except 'Function'
run_columns = [col for col in df.columns if col.startswith("Run")]

run_data = df[run_columns].astype(float)

# -------- Compute statistics --------
df["Mean"] = run_data.mean(axis=1)
df["Std"] = run_data.std(axis=1)

# -------- Create final dataframe --------
processed_df = df[["Function", "Mean", "Std"]]

# -------- Save to project folder --------
# (saves in same directory as script)
processed_df.to_csv(output_file, index=False)

print("Processed CSV saved at:", os.path.abspath(output_file))
print(processed_df.head())