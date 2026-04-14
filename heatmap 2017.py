import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------- CHANGE FILE PATH HERE --------
file_path = "cec2017_rankwise.csv"
title = "CEC 2017 Rank Heatmap"

# file_path = "cec functions/cec2017_rankwise_result.csv"
# title = "CEC 2017 Rank Heatmap"

# -------- Load CSV --------
df = pd.read_csv(file_path)

# -------- Extract rank columns --------
rank_cols = [col for col in df.columns if col.endswith(".RANK")]

# -------- Create heatmap data --------
heatmap_data = df[rank_cols].copy()

# Rename columns → cleaner names
heatmap_data.columns = [col.split(".")[0] for col in rank_cols]

# Set function names as index
heatmap_data.index = df["algo"]

# -------- Plot --------
plt.figure(figsize=(12, 8))

sns.heatmap(
    heatmap_data,
    annot=True,        # show numbers
    cmap="coolwarm",   # color scheme
    linewidths=0.5,
    cbar_kws={"label": "Rank"}
)

plt.title(title, fontsize=14)
plt.xlabel("Algorithms")
plt.ylabel("Functions")

plt.tight_layout()
plt.show()