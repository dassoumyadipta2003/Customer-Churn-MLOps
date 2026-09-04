from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc


app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML API for predicting customer churn",
    version="1.0.0"
)


# Load the exact model version packaged into the Docker image
MODEL_PATH = "models/churn_model"

model = mlflow.pyfunc.load_model(MODEL_PATH)

print("Churn model loaded successfully.")


class CustomerData(BaseModel):
    age: int
    gender: str
    tenure: int
    usage_frequency: int
    support_calls: int
    payment_delay: int
    subscription_type: str
    contract_length: str
    total_spend: float
    last_interaction: int


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    data = pd.DataFrame([customer.model_dump()])

    # Recreate Gold-layer engineered features
    data["usage_per_tenure"] = (
        data["usage_frequency"] /
        data["tenure"].replace(0, 1)
    )

    data["support_call_rate"] = (
        data["support_calls"] /
        data["tenure"].replace(0, 1)
    )

    data["payment_delay_flag"] = (
        data["payment_delay"] > 0
    ).astype(int)

    data["high_support_flag"] = (
        data["support_calls"] >= 5
    ).astype(int)

    prediction = model.predict(data)

    return {
        "churn_prediction": int(prediction[0])
    }