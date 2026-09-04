FROM apache/airflow:2.10.5

USER airflow

RUN pip install --no-cache-dir \
    apache-airflow-providers-databricks \
    pandas \
    numpy \
    scikit-learn \
    mlflow \
    boto3 \
    joblib \
    xgboost==3.4.1 \
    catboost==1.2.10 \
    pyOpenSSL==24.3.0