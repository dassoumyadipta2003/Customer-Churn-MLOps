# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ============================================================
# 2. LOAD DATA
# ============================================================

DATA_PATH = "data/processed/gold_customer_churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
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


# ============================================================
# 4. IDENTIFY FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. PREPROCESSING PIPELINE
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


# ============================================================
# 7. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline(
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
    ),

    "Random Forest": Pipeline(
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
    ),

    "XGBoost": Pipeline(
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
    ),

    "CatBoost": Pipeline(
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
}


# ============================================================
# 8. CREATE MLFLOW EXPERIMENT
# ============================================================

# Linux-compatible paths for Airflow Docker container
MLFLOW_DB = "sqlite:////opt/airflow/mlflow/mlflow.db"
MLFLOW_ARTIFACTS = "/opt/airflow/mlflow/mlruns"

mlflow.set_tracking_uri(MLFLOW_DB)

EXPERIMENT_NAME = "Customer_Churn_Model_Comparison"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    mlflow.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=f"file://{MLFLOW_ARTIFACTS}"
    )

mlflow.set_experiment(EXPERIMENT_NAME)

print("\nMLflow experiment created.")


# ============================================================
# 9. TRAIN + TRACK EACH MODEL
# ============================================================

for model_name, model in models.items():

    print(f"\n{'=' * 60}")
    print(f"Training: {model_name}")
    print(f"{'=' * 60}")

    with mlflow.start_run(run_name=model_name):

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        # ----------------------------------------------------
        # PRINT METRICS
        # ----------------------------------------------------

        print(f"Accuracy : {accuracy:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall   : {recall:.6f}")
        print(f"F1 Score : {f1:.6f}")
        print(f"ROC-AUC  : {roc_auc:.6f}")

        # ----------------------------------------------------
        # LOG PARAMETERS
        # ----------------------------------------------------

        mlflow.log_param(
            "model_name",
            model_name
        )

        # ----------------------------------------------------
        # LOG METRICS
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        # ----------------------------------------------------
        # LOG MODEL
        # ----------------------------------------------------

        if model_name in [
            "Logistic Regression",
            "Random Forest"
        ]:

            mlflow.sklearn.log_model(
                model,
                name="model"
            )

        elif model_name == "XGBoost":

            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier"
                ]
            )

        elif model_name == "CatBoost":

            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "catboost.core.CatBoostClassifier"
                ]
            )

        print(
            f"MLflow logged: {model_name}"
        )


# ============================================================
# 10. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("MLflow experiment completed successfully.")
print("=" * 60)