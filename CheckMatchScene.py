import pandas as pd
import numpy as np
import os

# === Base paths ===
base_dir_1 = "C://VLNLP//generated_scenes_with_a_star"
base_dir_2 = "C://VLNLP//Test"

file1 = os.path.join(base_dir_1, "scene_objects1.csv")
file2 = os.path.join(base_dir_2, "scene_objects.csv")

# === Load data ===
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# === Merge on scene_id and object_id ===
merged = pd.merge(
    df1, df2,
    on=["scene_id", "label"],
    suffixes=("_base", "_test")
)
print(df1[:10])
# === Focus on size_z column ===
merged["squared_error"] = (merged["size_z_base"] - merged["size_z_test"]) ** 2
merged["abs_error"] = abs(merged["size_z_base"] - merged["size_z_test"])

# === First 228 rows ===
first_228 = merged.iloc[:228]
mse_228 = first_228["squared_error"].mean()
max_err_228 = first_228["abs_error"].max()
 

# === Full dataset ===
mse_full = merged["squared_error"].mean()
max_err_full = merged["abs_error"].max()

# === Results ===
print("📊 Comparison Results for `size_z`:")
print(f"▶️ First 228 objects:")
print(f"   Mean Squared Error: {mse_228:.6f}")
print(f"   Max Absolute Error: {max_err_228:.6f}")

print(f"\n▶️ Full dataset:")
print(f"   Mean Squared Error: {mse_full:.15f}")
print(f"   Max Absolute Error: {max_err_full:.6f}")
