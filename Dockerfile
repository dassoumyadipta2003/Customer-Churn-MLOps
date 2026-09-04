FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

COPY models/churn_model ./models/churn_model

EXPOSE 8000

CMD ["uvicorn", "src.api.predict:app", "--host", "0.0.0.0", "--port", "8000"]