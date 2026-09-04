# Customer Churn MLOps

End-to-end Machine Learning Operations pipeline for customer churn prediction — from cloud data engineering and model training to experiment tracking, model registry, containerization, and production API deployment.

## Project Overview

This project implements a complete Customer Churn Prediction MLOps workflow designed to automate and operationalize the machine learning lifecycle.

The system combines:

- **AWS S3** — cloud data storage
- **Databricks + PySpark** — data engineering and feature engineering
- **Scikit-learn, XGBoost & CatBoost** — machine learning
- **MLflow** — experiment tracking and model registry
- **Apache Airflow** — ML pipeline orchestration
- **FastAPI** — prediction API
- **Docker** — containerization
- **AWS ECS + ECR** — production deployment

The final system allows a user to send customer information to a REST API and receive a churn prediction.

## Architecture

```
                         ┌──────────────────┐
                         │    Raw Data      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     AWS S3       │
                         │   Raw Dataset    │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │       Databricks        │
                    │                         │
                    │  Bronze → Silver → Gold │
                    │                         │
                    │  Cleaning               │
                    │  Transformation         │
                    │  Feature Engineering    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                         ┌──────────────────┐
                         │    S3 Gold       │
                         │  ML Dataset      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Airflow      │
                         │   Orchestration  │
                         └────────┬─────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │    Model Training    │
                       │                      │
                       │ Logistic Regression  │
                       │ Random Forest        │
                       │ XGBoost              │
                       │ CatBoost             │
                       └──────────┬───────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     MLflow       │
                         │ Experiment       │
                         │ Tracking         │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Champion Model   │
                         │   Selection      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ MLflow Registry  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │      Docker      │
                         │   FastAPI App    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     AWS ECS      │
                         │    Production    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Churn Prediction│
                         │      API         │
                         └──────────────────┘
```

## End-to-End Workflow

### 1. Data Engineering

Customer data is stored in AWS S3.

Databricks processes the data using a multi-layer architecture:

```
Raw
 ↓
Bronze
 ↓
Silver
 ↓
Gold
```

**Bronze**

Raw customer data is ingested into the Bronze layer.

**Silver**

The Silver layer performs data cleaning and transformation.

**Gold**

The Gold layer contains the machine-learning-ready dataset.

Additional engineered features include:

- `usage_per_tenure`
- `support_call_rate`
- `payment_delay_flag`
- `high_support_flag`

The final ML dataset is exported back to S3 as CSV.

## Machine Learning

Four classification models were evaluated:

| Model | F1 Score | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.923471 | 0.966388 |
| Random Forest | 0.994692 | 0.999968 |
| XGBoost | 0.999900 | 0.999999 |
| CatBoost | 0.999930 | 0.9999996 |

### Champion Model

**CatBoost** was selected as the champion model based on the highest F1 score.

- **Champion:** CatBoost
- **F1 Score:** 0.999930
- **ROC-AUC:** 0.9999996

> **Important:** The dataset is synthetic and contains very strong deterministic relationships between certain customer attributes and churn. The exceptionally high model scores should therefore not be interpreted as expected real-world performance.

## Data Leakage Analysis

A dedicated leakage/data-quality analysis was performed before finalizing the models.

Important observations included strong churn patterns such as:

- Monthly-contract customers showing extremely high churn
- High support-call counts strongly associated with churn
- Engineered `high_support_flag` having a strong relationship with the target
- No duplicate records detected

These patterns are documented rather than hidden because they are characteristics of the synthetic dataset.

## MLflow

MLflow is used for:

- Experiment tracking
- Parameter logging
- Metric logging
- Model artifact logging
- Champion model selection
- Model Registry

The training pipeline evaluates all four models and logs:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The best-performing model is then registered as:

```
Customer_Churn_Model
```

## Apache Airflow

Airflow orchestrates the machine-learning workflow.

The current DAG:

```
load_gold_data
       ↓
train_models
       ↓
mlflow_tracking
       ↓
register_model
```

The complete DAG has been successfully executed end-to-end.

### Airflow responsibilities

Instead of manually running individual scripts, the pipeline can be triggered from Airflow and execute the ML workflow in the correct order.

This makes the training process:

- Reproducible
- Ordered
- Easier to rerun
- Easier to schedule
- Easier to monitor

**Current architecture note:** Databricks ETL and Airflow ML orchestration are separate pipelines. The current Airflow DAG starts from the already-generated Gold dataset rather than triggering the Databricks ETL job itself.

## FastAPI Prediction API

The trained champion model is served through a FastAPI REST API.

### Example request

`POST /predict`

Example payload:

```json
{
  "tenure": 12,
  "monthly_charges": 75.5,
  "total_spend": 906.0,
  "support_calls": 2,
  "usage_frequency": 25,
  "payment_delay": 0
}
```

Example response:

```json
{
  "churn_prediction": 0
}
```

Where:

- `0` → Customer predicted not to churn
- `1` → Customer predicted to churn

## Docker

The FastAPI application is containerized using Docker.

The Docker image contains:

- Python runtime
- ML dependencies
- FastAPI application
- Model artifact
- API dependencies

This provides a reproducible environment for deployment.

## AWS Deployment

The API is deployed using:

```
Docker
  ↓
Amazon ECR
  ↓
Amazon ECS
  ↓
FastAPI
```

### AWS Components

| Service | Purpose |
|---|---|
| Amazon S3 | Data storage |
| Amazon ECR | Docker image registry |
| Amazon ECS | Container deployment |
| Databricks | Data engineering |
| MLflow | ML lifecycle management |

### Live API

The current ECS deployment exposes the FastAPI Swagger interface at the public ECS endpoint.

Current demo URL:

```
http://13.127.101.218:8000/docs
```

Note: This is a current ECS task public IP and should be considered a temporary/demo URL rather than a permanent production address. A future improvement is to place the service behind a stable domain/load balancer.

## Project Structure

```
Customer_Churn_MLOps/
│
├── airflow/
│   ├── dags/
│   │   └── customer_churn_pipeline.py
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── src/
│   ├── api/
│   │   └── predict.py
│   │
│   ├── data/
│   │   └── s3_loader.py
│   │
│   └── models/
│       ├── leakage_check.py
│       ├── train.py
│       ├── train_mlflow.py
│       └── register_model.py
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── ecs-task-definition.json
├── README.md
└── requirements.txt
```

## Technologies

### Data Engineering
- Python
- PySpark
- Databricks
- AWS S3

### Machine Learning
- Scikit-learn
- XGBoost
- CatBoost
- Pandas
- NumPy

### MLOps
- MLflow
- Apache Airflow
- Docker

### Deployment
- FastAPI
- Amazon ECR
- Amazon ECS

## Running the Project

### Clone

```bash
git clone <your-github-repository-url>
cd Customer_Churn_MLOps
```

### Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API locally

```bash
uvicorn src.api.predict:app --reload
```

Swagger UI:

```
http://localhost:8000/docs
```

## Run the ML Pipeline

The ML workflow can be executed through Airflow.

Start Airflow:

```bash
cd airflow
docker compose up -d
```

Open:

```
http://localhost:8080
```

Then trigger:

```
customer_churn_pipeline
```

The DAG executes:

```
Load Gold Data
      ↓
Train Models
      ↓
MLflow Tracking
      ↓
Register Champion
```

## MLOps Lifecycle

The project demonstrates the following lifecycle:

```
Data
 ↓
Data Engineering
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Model Evaluation
 ↓
Experiment Tracking
 ↓
Champion Selection
 ↓
Model Registry
 ↓
Containerization
 ↓
Cloud Deployment
 ↓
Prediction API
```

## Key MLOps Concepts Demonstrated

- Cloud-based data storage
- Medallion-style data architecture
- Feature engineering
- Model comparison
- Reproducible model training
- Data leakage analysis
- Experiment tracking
- Model registry
- Pipeline orchestration
- Containerization
- REST API serving
- AWS cloud deployment
- Production-oriented ML workflow

## Limitations & Future Improvements

### Current limitations
- Dataset is synthetic.
- Extremely high model scores are influenced by deterministic patterns in the dataset.
- Databricks ETL and Airflow orchestration are currently separate.
- The current ECS public IP is not a permanent API address.
- Production monitoring and automated model drift detection are not yet implemented.

### Future improvements
- Connect Airflow directly to Databricks Jobs.
- Add automated data-quality validation.
- Add model performance gates before registration.
- Add model promotion from staging → production.
- Add automated ECS model deployment.
- Add model/data drift monitoring.
- Add centralized MLflow server.
- Put the API behind an AWS Load Balancer and custom domain.
- Add CI/CD using GitHub Actions.

## Author

**Soumyadipta Das**

MSc Statistics & Data Science — LMU Munich

[LinkedIn](https://www.linkedin.com/in/soumyadiptadas3)

Interested in:

- Data Science
- Machine Learning
- MLOps
- Data Engineering
- Cloud Deployment