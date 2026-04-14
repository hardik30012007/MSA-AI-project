import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    results = []
    for _, row in df.iterrows():
        problem = row.iloc[0]
        runs = row.iloc[1:].astype(float).to_numpy()
        results.append({'Problem': problem, 'Best Value': runs.min(), 'Best Run': int(runs.argmin()) + 1, 'Mean': runs.mean(), 'Std': runs.std(ddof=1), 'Worst': runs.max()})
    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print('Saved:', args.output_csv)

if __name__ == '__main__':
    main()
