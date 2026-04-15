import os
import pandas as pd

INPUT_DIR = "engineering_plots"

def main():
    all_rows = []

    for file in os.listdir(INPUT_DIR):
        if file.endswith("_raw.csv"):
            path = os.path.join(INPUT_DIR, file)
            df = pd.read_csv(path)

            # Example expected format:
            # Problem,Algorithm,Run1,Run2,...Run10

            if "Algorithm" not in df.columns:
                print(f"Skipping {file} because 'Algorithm' column not found.")
                continue

            if "Problem" in df.columns:
                problem_name = str(df["Problem"].iloc[0])
            else:
                problem_name = file.replace("_raw.csv", "")

            means = {}

            for _, row in df.iterrows():
                algo = row["Algorithm"]
                run_values = pd.to_numeric(row.iloc[2:], errors="coerce")  # skip Problem, Algorithm
                means[algo] = run_values.mean()

            # lower mean = better rank
            sorted_algos = sorted(means.items(), key=lambda x: x[1])

            rank_row = {"Problem": problem_name}
            for rank, (algo, _) in enumerate(sorted_algos, start=1):
                rank_row[algo] = rank

            all_rows.append(rank_row)

    if not all_rows:
        print("No valid *_raw.csv files found in engineering_plots")
        return

    rank_df = pd.DataFrame(all_rows)
    rank_df = rank_df.set_index("Problem")

    # fill missing algos if any
    rank_df = rank_df.sort_index(axis=1)

    # average rank
    rank_df.loc["Average"] = rank_df.mean(numeric_only=True)

    print("\n==== RANK TABLE ====\n")
    print(rank_df)

    rank_df.to_csv("rank_table.csv")
    print("\nSaved: rank_table.csv")

if __name__ == "__main__":
    main()