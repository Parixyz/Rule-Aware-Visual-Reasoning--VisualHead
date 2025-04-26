import pandas as pd
import os


base_dir = "C://VLNLP//Test"
violation_file = os.path.join(base_dir, "violation_labels_updated.csv")


df = pd.read_csv(violation_file)


df['_rule1'] = df['violations'].str.contains('rule1', na=False).astype(int)
df['_rule2'] = df['violations'].str.contains('rule2', na=False).astype(int)
df['_rule3'] = df['violations'].str.contains('rule3', na=False).astype(int)

#  Binary encoding: 2^x to cover all varitions
df['rule_code'] = df['_rule1'] * 1 + df['_rule2'] * 2 + df['_rule3'] * 4


print(" Rule Code Distribution:")
print(df['rule_code'].value_counts().sort_index())
print(sum(df['rule_code'].value_counts()))
