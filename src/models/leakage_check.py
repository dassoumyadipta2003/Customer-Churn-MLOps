# ============================================================
# LEAKAGE INVESTIGATION
# ============================================================

import pandas as pd

DATA_PATH = "data/processed/gold_customer_churn.csv"

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)

# ------------------------------------------------------------
# 1. Numerical correlation
# ------------------------------------------------------------

numeric_features = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

numeric_features.remove("churn")
numeric_features.remove("customerid")

if "ingestion_timestamp" in numeric_features:
    numeric_features.remove("ingestion_timestamp")

correlations = (
    df[numeric_features + ["churn"]]
    .corr()["churn"]
    .drop("churn")
    .sort_values(ascending=False)
)

print("\n===== NUMERICAL CORRELATION WITH CHURN =====")
print(correlations)

# ------------------------------------------------------------
# 2. Categorical features
# ------------------------------------------------------------

categorical_features = [
    "gender",
    "subscription_type",
    "contract_length"
]

for column in categorical_features:
    print(f"\n===== {column.upper()} VS CHURN =====")

    print(
        pd.crosstab(
            df[column],
            df["churn"],
            normalize="index"
        ).round(4)
    )

# ------------------------------------------------------------
# 3. Engineered features
# ------------------------------------------------------------

engineered_features = [
    "usage_per_tenure",
    "support_call_rate",
    "payment_delay_flag",
    "high_support_flag"
]

print("\n===== ENGINEERED FEATURES BY CHURN =====")

for column in engineered_features:
    print(f"\n{column}")

    print(
        df.groupby("churn")[column]
        .mean()
    )

# ------------------------------------------------------------
# 4. Duplicate rows
# ------------------------------------------------------------

print("\n===== DUPLICATES =====")

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

# ============================================================
# 5. CHECK CHURN RATE FOR IMPORTANT FEATURES
# ============================================================

print("\n===== CONTRACT LENGTH CHURN RATE =====")

print(
    df.groupby("contract_length")["churn"]
    .agg(["count", "mean"])
    .sort_values("mean", ascending=False)
)


print("\n===== HIGH SUPPORT FLAG CHURN RATE =====")

print(
    df.groupby("high_support_flag")["churn"]
    .agg(["count", "mean"])
)


print("\n===== SUPPORT CALLS CHURN RATE =====")

print(
    df.groupby("support_calls")["churn"]
    .agg(["count", "mean"])
    .sort_index()
)