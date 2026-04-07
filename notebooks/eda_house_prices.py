# ============================================================
# House Prices EDA - MLOps Project (Week 1)
# ============================================================
# Setup: pip install pandas numpy matplotlib seaborn scikit-learn
# Place this file in: notebooks/
# Place train.csv in:  data/
# Run: python eda_house_prices.py  (from inside notebooks/ folder)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ──────────────────────────────────────────────────
TRAIN_PATH = "../data/train.csv"   # path from notebooks/ to data/
TARGET     = "SalePrice"
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("muted")

# ============================================================
# 1. Load Data
# ============================================================
df = pd.read_csv(TRAIN_PATH)
print(f"Shape: {df.shape}")
print(df.head())

# ============================================================
# 2. Basic Info
# ============================================================
print("\n── dtypes ──")
print(df.dtypes.value_counts())

print("\n── Descriptive stats (numeric) ──")
print(df.describe().T)

# ============================================================
# 3. Target Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df[TARGET], bins=50, edgecolor="white")
axes[0].set_title("SalePrice distribution")
axes[0].set_xlabel("SalePrice")

axes[1].hist(np.log1p(df[TARGET]), bins=50, edgecolor="white", color="steelblue")
axes[1].set_title("log1p(SalePrice) distribution")
axes[1].set_xlabel("log1p(SalePrice)")

plt.tight_layout()
plt.savefig("target_distribution.png", dpi=120)
plt.show()
print("Tip: log1p(SalePrice) is more normal — use it as your training target.")

# ============================================================
# 4. Missing Values
# ============================================================
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
print(f"\n── Columns with missing values ({len(missing_df)}) ──")
print(missing_df.head(20))

plt.figure(figsize=(10, 5))
missing_pct.head(20).plot(kind="bar")
plt.title("Top 20 columns with missing values (%)")
plt.ylabel("Missing %")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("missing_values.png", dpi=120)
plt.show()

# ============================================================
# 5. Correlation with Target
# ============================================================
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()[TARGET].drop(TARGET).sort_values(ascending=False)

print("\n── Top 10 positive correlations with SalePrice ──")
print(corr.head(10))
print("\n── Top 10 negative correlations with SalePrice ──")
print(corr.tail(10))

plt.figure(figsize=(8, 6))
corr.head(15).plot(kind="barh")
plt.title("Top 15 features correlated with SalePrice")
plt.xlabel("Pearson correlation")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("top_correlations.png", dpi=120)
plt.show()

# ============================================================
# 6. Key Feature Scatter Plots
# ============================================================
top_features = corr.head(5).index.tolist()
fig, axes = plt.subplots(1, len(top_features), figsize=(18, 4))

for ax, feat in zip(axes, top_features):
    ax.scatter(df[feat], df[TARGET], alpha=0.4, s=15)
    ax.set_xlabel(feat)
    ax.set_ylabel(TARGET)
    ax.set_title(f"{feat} vs SalePrice")

plt.tight_layout()
plt.savefig("scatter_top_features.png", dpi=120)
plt.show()

# ============================================================
# 7. Categorical Feature Analysis
# ============================================================
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"\nCategorical columns: {len(cat_cols)}")

plt.figure(figsize=(14, 5))
order = df.groupby("Neighborhood")[TARGET].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="Neighborhood", y=TARGET, order=order)
plt.xticks(rotation=45, ha="right")
plt.title("SalePrice by Neighborhood")
plt.tight_layout()
plt.savefig("neighborhood_prices.png", dpi=120)
plt.show()

# ============================================================
# 8. Outlier Detection
# ============================================================
plt.figure(figsize=(7, 5))
plt.scatter(df["GrLivArea"], df[TARGET], alpha=0.5, s=15)
plt.xlabel("GrLivArea (sq ft)")
plt.ylabel("SalePrice")
plt.title("GrLivArea vs SalePrice — check top-right outliers")
plt.tight_layout()
plt.savefig("outlier_check.png", dpi=120)
plt.show()

outliers = df[(df["GrLivArea"] > 4000) & (df[TARGET] < 300000)]
print(f"\nSuspect outliers (large area, low price): {len(outliers)} rows")
print(outliers[["GrLivArea", TARGET, "Neighborhood"]])

# ============================================================
# 9. Summary + Next Steps
# ============================================================
print("""
════════════════════════════════════════════
EDA SUMMARY
════════════════════════════════════════════
Total rows      : {rows}
Total features  : {cols}
Missing columns : {miss}
Top 3 correlated: {top3}

NEXT STEPS (Day 2):
1. Drop or impute missing values
   - High missing (>80%): drop column
   - Numeric: fill with median
   - Categorical: fill with None or mode

2. Remove 2 outliers: GrLivArea > 4000 and SalePrice < 300k

3. Use log1p(SalePrice) as your target

4. Encode categoricals with pd.get_dummies()

5. Train a baseline Ridge model with sklearn

6. Log experiment with MLflow
════════════════════════════════════════════
""".format(
    rows=df.shape[0],
    cols=df.shape[1],
    miss=len(missing_df),
    top3=", ".join(corr.head(3).index.tolist())
))
