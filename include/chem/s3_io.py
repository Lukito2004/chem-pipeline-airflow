import io
import pandas as pd
import os
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

BUCKET = os.environ.get("CHEM_S3_BUCKET", "chem-pipeline-datasets")
AWS_CONN_ID = os.environ.get("CHEM_AWS_CONN_ID", "aws_default")

def _hook() -> S3Hook:
    return S3Hook(aws_conn_id=AWS_CONN_ID)

def read_csv(key: str) -> pd.DataFrame:
    body = _hook().read_key(key, bucket_name=BUCKET)
    return pd.read_csv(io.StringIO(body))

def write_csv(df: pd.DataFrame, key: str) -> None:
    _hook().load_string(df.to_csv(index=False), key=key,
                        bucket_name=BUCKET, replace=True)

def list_keys(prefix: str = "") -> list[str]:
    return _hook().list_keys(bucket_name=BUCKET, prefix=prefix) or []

def exists(key: str) -> bool:
    return _hook().check_for_key(key, bucket_name=BUCKET)