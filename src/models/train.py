# ============================================================
# 1. LOAD GOLD DATASET
# ============================================================

import pandas as pd

DATA_PATH = "data/processed/gold_customer_churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

TARGET = "churn"

X = df.drop(
    columns=[
        "churn",
        "customerid",
        "ingestion_timestamp"
    ]
)

y = df[TARGET]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 3. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 5. CREATE PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("Preprocessor created successfully.")


# ============================================================
# 6. LOGISTIC REGRESSION — BASELINE MODEL
# ============================================================

from sklearn.linear_model import LogisticRegression

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)

print("\nLogistic Regression trained successfully.")


# ============================================================
# 7. EVALUATE LOGISTIC REGRESSION
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

print("\n===== LOGISTIC REGRESSION RESULTS =====")

print("Accuracy :", accuracy_score(y_test, logistic_pred))
print("Precision:", precision_score(y_test, logistic_pred))
print("Recall   :", recall_score(y_test, logistic_pred))
print("F1 Score :", f1_score(y_test, logistic_pred))
print("ROC-AUC  :", roc_auc_score(y_test, logistic_prob))

print("\nClassification Report:")
print(classification_report(y_test, logistic_pred))


# ============================================================
# 8. RANDOM FOREST — NONLINEAR MODEL
# ============================================================

from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest trained successfully.")


# ============================================================
# 9. EVALUATE RANDOM FOREST
# ============================================================

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n===== RANDOM FOREST RESULTS =====")

print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1 Score :", f1_score(y_test, rf_pred))
print("ROC-AUC  :", roc_auc_score(y_test, rf_prob))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))


# ============================================================
# 10. SAVE LOGISTIC REGRESSION MODEL
# ============================================================

import joblib
from pathlib import Path

MODEL_PATH = Path("models/churn_logistic_regression.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(logistic_model, MODEL_PATH)

print(f"\nLogistic Regression saved to: {MODEL_PATH}")


# ============================================================
# 11. SAVE RANDOM FOREST MODEL
# ============================================================

RF_MODEL_PATH = Path("models/churn_random_forest.joblib")

joblib.dump(rf_model, RF_MODEL_PATH)

print(f"Random Forest saved to: {RF_MODEL_PATH}")


# ============================================================
# 12. XGBOOST — ADVANCED MODEL
# ============================================================

from xgboost import XGBClassifier

xgb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss"
            )
        )
    ]
)

xgb_model.fit(X_train, y_train)

print("\nXGBoost trained successfully.")


# ============================================================
# 13. EVALUATE XGBOOST
# ============================================================

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\n===== XGBOOST RESULTS =====")

print("Accuracy :", accuracy_score(y_test, xgb_pred))
print("Precision:", precision_score(y_test, xgb_pred))
print("Recall   :", recall_score(y_test, xgb_pred))
print("F1 Score :", f1_score(y_test, xgb_pred))
print("ROC-AUC  :", roc_auc_score(y_test, xgb_prob))

print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))


# ============================================================
# 14. SAVE XGBOOST MODEL
# ============================================================

XGB_MODEL_PATH = Path("models/churn_xgboost.joblib")

joblib.dump(xgb_model, XGB_MODEL_PATH)

print(f"\nXGBoost saved to: {XGB_MODEL_PATH}")


# ============================================================
# 15. CATBOOST — CATEGORICAL BOOSTING MODEL
# ============================================================

from catboost import CatBoostClassifier

cat_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False,
    thread_count=-1,
    allow_writing_files=False
      )
        )
    ]
)

cat_model.fit(X_train, y_train)

print("\nCatBoost trained successfully.")


# ============================================================
# 16. EVALUATE CATBOOST
# ============================================================

cat_pred = cat_model.predict(X_test)
cat_prob = cat_model.predict_proba(X_test)[:, 1]

print("\n===== CATBOOST RESULTS =====")

print("Accuracy :", accuracy_score(y_test, cat_pred))
print("Precision:", precision_score(y_test, cat_pred))
print("Recall   :", recall_score(y_test, cat_pred))
print("F1 Score :", f1_score(y_test, cat_pred))
print("ROC-AUC  :", roc_auc_score(y_test, cat_prob))

print("\nClassification Report:")
print(classification_report(y_test, cat_pred))


# ============================================================
# 17. SAVE CATBOOST MODEL
# ============================================================

CAT_MODEL_PATH = Path("models/churn_catboost.joblib")

joblib.dump(cat_model, CAT_MODEL_PATH)

print(f"\nCatBoost saved to: {CAT_MODEL_PATH}")