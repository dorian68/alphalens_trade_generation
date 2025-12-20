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
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

BUCKET_NAME = "alphalens-forecast-model-store"
PREFIX = ""  # ex: "production/" ou "" pour tout lister


def list_s3_files(bucket: str, prefix: str = "") -> list[str]:
    s3 = boto3.client("s3")

    files = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                files.append(obj["Key"])

        return files

    except NoCredentialsError:
        print("❌ AWS credentials not found (check aws configure or IAM role).")
        return []

    except ClientError as e:
        print(f"❌ AWS error: {e}")
        return []


if __name__ == "__main__":
    print(f"📦 Listing S3 files in s3://{BUCKET_NAME}/{PREFIX}\n")

    files = list_s3_files(BUCKET_NAME, PREFIX)

    if not files:
        print("⚠️ No files found or access denied.")
    else:
        for f in files:
            print(" -", f)

        print(f"\n✅ {len(files)} file(s) found.")


# %%
