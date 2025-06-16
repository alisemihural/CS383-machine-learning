# author: Ali Ural
# date: 04-07-2025
# description: PART II: INSURANCE PLOTS

import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("insurance.csv")

# Convert continuous features to binary
for col in ['age', 'bmi', 'children', 'charges']:
    mean = df[col].mean()
    df[col] = (df[col] >= mean).astype(int)

# One-hot Encode Categorical Features
categorical_cols = ['sex', 'smoker', 'region']
df_encoded = pd.get_dummies(df, columns=categorical_cols)
X = df_encoded.values 

# Plot Creation
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(X[:, 0], X[:, 1], alpha=0.7, c="orange")

ax.set_title("Binary Feature 1 vs Feature 2")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.grid(True)

plt.tight_layout()
plt.show()