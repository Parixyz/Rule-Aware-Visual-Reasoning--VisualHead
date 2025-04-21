import pandas as pd
import os

# === Config ===
base_dir = "C://VLNLP//Test"
target_total = 8000
label_count = 8
ideal_max_per_class = target_total // label_count

# === Load violation labels and compute rule_code on the fly
violation_file = os.path.join(base_dir, "violation_labels_updated.csv")
df_viol = pd.read_csv(violation_file)
df_viol['_rule1'] = df_viol['violations'].str.contains('rule1', na=False).astype(int)
df_viol['_rule2'] = df_viol['violations'].str.contains('rule2', na=False).astype(int)
df_viol['_rule3'] = df_viol['violations'].str.contains('rule3', na=False).astype(int)
df_viol['rule_code'] = df_viol['_rule1'] * 1 + df_viol['_rule2'] * 2 + df_viol['_rule3'] * 4

# === Load full object and solution logs
df_objs = pd.read_csv(os.path.join(base_dir, "scene_objects.csv"))
df_sols = pd.read_csv(os.path.join(base_dir, "solution_log_updated.csv"))

# === Step 1: stratified balanced sampling
rule_counts = df_viol['rule_code'].value_counts()
samples = []
remaining_quota = target_total
overflow_codes = []

for code, count in rule_counts.items():
    group = df_viol[df_viol['rule_code'] == code]
    if count >= ideal_max_per_class:
        samples.append(group.sample(n=ideal_max_per_class, random_state=42))
        remaining_quota -= ideal_max_per_class
    else:
        samples.append(group)
        remaining_quota -= len(group)
        overflow_codes.append(code)

# === Step 2: fill from heavy classes
if remaining_quota > 0:
    heavy_codes = [code for code in rule_counts.index if rule_counts[code] > ideal_max_per_class and code not in overflow_codes]
    additional_per_code = remaining_quota // len(heavy_codes)
    for code in heavy_codes:
        extra = df_viol[df_viol['rule_code'] == code].sample(n=additional_per_code, random_state=99)
        samples.append(extra)

# === Final balanced dataset
df_balanced_viol = pd.concat(samples)
if len(df_balanced_viol) > target_total:
    df_balanced_viol = df_balanced_viol.sample(n=target_total, random_state=7)

scene_ids_balanced = df_balanced_viol['scene_id'].unique()
df_balanced_objs = df_objs[df_objs['scene_id'].isin(scene_ids_balanced)].copy()
df_balanced_sols = df_sols[df_sols['scene_id'].isin(scene_ids_balanced)].copy()

# === Save
df_balanced_viol.to_csv(os.path.join(base_dir, "violation_labels_balanced_7k.csv"), index=False)
df_balanced_objs.to_csv(os.path.join(base_dir, "scene_objects_balanced_7k.csv"), index=False)
df_balanced_sols.to_csv(os.path.join(base_dir, "solution_log_balanced_7k.csv"), index=False)

# === Print summary
print("✅ Balanced subset created and saved:")
print(f"  - violation_labels_balanced.csv ({len(df_balanced_viol)} rows)")
print(f"  - scene_objects_balanced.csv     ({len(df_balanced_objs)} rows)")
print(f"  - solution_log_balanced.csv      ({len(df_balanced_sols)} rows)\n")
print("📊 Final Distribution:")
print(df_balanced_viol['rule_code'].value_counts().sort_index())
