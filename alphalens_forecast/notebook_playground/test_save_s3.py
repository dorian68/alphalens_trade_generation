# %%
import boto3
from pathlib import Path

BUCKET_NAME = "alphalens-forecast-model-store"
S3_PREFIX = "production"

def upload_py_to_s3(local_file: Path):
    if not local_file.exists():
        raise FileNotFoundError(f"{local_file} does not exist")

    s3 = boto3.client("s3")

    s3_key = f"{S3_PREFIX}/{local_file.name}"

    s3.upload_file(
        Filename=str(local_file),
        Bucket=BUCKET_NAME,
        Key=s3_key
    )

    print(f"✅ Uploaded {local_file} → s3://{BUCKET_NAME}/{s3_key}")


if __name__ == "__main__":
    upload_py_to_s3(Path(r"C:\Users\Labry\Documents\ALPHALENS_PRJOECT_FORECAST\alphalens_trade_generation\alphalens_forecast\notebook_playground\azerty.py"))
# %%
