from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="customer_churn_pipeline",
    start_date=datetime(2026, 9, 2),
    schedule=None,
    catchup=False,
    tags=["customer-churn", "mlops"],
) as dag:

    load_data = BashOperator(
        task_id="load_gold_data",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/data/s3_loader.py"
        ),
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/models/train.py"
        ),
    )

    mlflow_tracking = BashOperator(
        task_id="mlflow_tracking",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/models/train_mlflow.py"
        ),
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/models/register_model.py"
        ),
    )

    load_data >> train_models >> mlflow_tracking >> register_model