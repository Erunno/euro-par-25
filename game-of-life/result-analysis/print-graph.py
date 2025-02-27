import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("report.csv", delimiter=";")

# Compute mean execution time per cell
df["time_per_cell"] = df["time"] / df["iterations"] / (df["dim"] ** 2)

# Aggregate by algorithm and dimension
df_grouped = df.groupby(["id", "dim"]).agg({"time_per_cell": "mean"}).reset_index()

# Define a color palette
colors = ["b", "g", "r", "c", "m", "y", "k"]
markers = ["o", "s", "D", "^", "v", "<", ">"]

# Convert dimension values to strings for equal spacing
df_grouped["dim_str"] = df_grouped["dim"].astype(str)

# Create the plot
plt.figure(figsize=(8, 6))

for i, (algo, data) in enumerate(df_grouped.groupby("id")):
    plt.plot(data["dim_str"], data["time_per_cell"], label=algo, marker=markers[i % len(markers)], linestyle="--", color=colors[i % len(colors)])

# Labels and title
plt.xlabel("Grid Size")
plt.ylabel("Time per one cell (ps)")
plt.xticks(df_grouped["dim_str"].unique(), labels=df_grouped["dim_str"].unique())  # Ensure equal spacing
plt.legend(title="Algorithm", loc="best")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save plot as PNG
plt.savefig("time_analysis.png", dpi=300)
plt.show()
