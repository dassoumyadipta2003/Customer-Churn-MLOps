# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import mlflow
from mlflow import MlflowClient


# ============================================================
# 2. MLFLOW CONFIGURATION
# ============================================================

MLFLOW_DB = "sqlite:////opt/airflow/mlflow/mlflow.db"

mlflow.set_tracking_uri(MLFLOW_DB)

EXPERIMENT_NAME = "Customer_Churn_Model_Comparison"

mlflow.set_experiment(EXPERIMENT_NAME)

experiment = mlflow.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:
    raise RuntimeError(
        "MLflow experiment not found."
    )

print("Experiment ID:", experiment.experiment_id)


# ============================================================
# 3. FIND BEST RUN
# ============================================================

client = MlflowClient()

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"],
    max_results=1
)

if not runs:
    raise RuntimeError(
        "No MLflow runs found."
    )

best_run = runs[0]

best_run_id = best_run.info.run_id

best_model_name = best_run.data.params.get(
    "model_name"
)

best_f1 = best_run.data.metrics.get(
    "f1_score"
)

best_accuracy = best_run.data.metrics.get(
    "accuracy"
)

best_roc_auc = best_run.data.metrics.get(
    "roc_auc"
)

print("\n===== BEST MODEL =====")
print("Model   :", best_model_name)
print("F1      :", best_f1)
print("Accuracy:", best_accuracy)
print("ROC-AUC :", best_roc_auc)
print("Run ID  :", best_run_id)


# ============================================================
# 4. REGISTER BEST MODEL
# ============================================================

MODEL_NAME = "Customer_Churn_Model"

model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME
)

print("\n===== MODEL REGISTERED =====")
print("Model name:", MODEL_NAME)
print("Version   :", registered_model.version)


# ============================================================
# 5. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("MLflow Model Registry completed successfully.")
print("=" * 60)