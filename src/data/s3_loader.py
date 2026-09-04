import boto3
from pathlib import Path


BUCKET = "customer-churn-mlops-soumya-458781646240-ap-south-1-an"
S3_PREFIX = "gold/customer_churn_ml_csv/"
LOCAL_PATH = Path("data/processed/gold_customer_churn.csv")


def download_gold_dataset():
    s3 = boto3.client("s3")

    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=S3_PREFIX
    )

    csv_files = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError("No Gold CSV found in S3.")

    # Our current Spark export has one CSV part file.
    s3_key = csv_files[0]

    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    s3.download_file(
        BUCKET,
        s3_key,
        str(LOCAL_PATH)
    )

    print(f"Downloaded: {s3_key}")
    print(f"Saved to: {LOCAL_PATH}")


if __name__ == "__main__":
    download_gold_dataset()