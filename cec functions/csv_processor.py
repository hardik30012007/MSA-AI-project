import pandas as pd
import os

# -------- Input & Output folders --------
input_folder = "cec functions/cec2014_split_csvs"
output_folder = "cec functions/cec2014_processed_csvs"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# -------- Process each CSV --------
for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        input_path = os.path.join(input_folder, file)
        
        # Output file (same name)
        output_path = os.path.join(output_folder, file)

        # -------- Load CSV --------
        df = pd.read_csv(input_path)

        # -------- Extract run columns --------
        run_columns = [col for col in df.columns if col.startswith("Run")]
        run_data = df[run_columns].astype(float)

        # -------- Compute statistics --------
        df["Mean"] = run_data.mean(axis=1)
        df["Std"] = run_data.std(axis=1)

        # -------- Final dataframe --------
        processed_df = df[["Function", "Mean", "Std"]]

        # -------- Save --------
        processed_df.to_csv(output_path, index=False)

        print(f"Processed: {file} → {output_path}")
print("All files processed.")