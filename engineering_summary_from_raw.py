import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    results = []

    for _, row in df.iterrows():
        problem = row["Problem"]

        # get all run values
        runs = row[1:].values.astype(float)

        best_value = runs.min()
        best_run = runs.argmin() + 1  # Run index (1-based)
        mean = runs.mean()
        std = runs.std(ddof=1)

        results.append({
            "Problem": problem,
            "Best Value": best_value,
            "Best Run": best_run,
            "Mean": mean,
            "Std": std,
            "Worst": runs.max()
        })

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(args.output_csv, index=False)

    print("Summary saved at:", args.output_csv)
    print(summary_df.head())


if __name__ == "__main__":
    main()