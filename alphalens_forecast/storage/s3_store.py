"""S3 storage utilities for AlphaLens artifacts."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional


class S3UnavailableError(RuntimeError):
    """Raised when S3 credentials are unavailable."""


class S3Store:
    """Minimal S3 wrapper for model artifact IO."""

    def __init__(self, bucket: str, prefix: Optional[str], logger: logging.Logger) -> None:
        self._bucket = bucket
        self._prefix = (prefix or "").strip().strip("/")
        self._logger = logger
        self._client = None

    @classmethod
    def from_env(cls, logger: logging.Logger) -> Optional["S3Store"]:
        bucket = (os.getenv("ALPHALENS_MODEL_BUCKET") or "").strip()
        if not bucket:
            return None
        prefix = (os.getenv("ALPHALENS_MODEL_PREFIX") or "").strip()
        return cls(bucket=bucket, prefix=prefix or None, logger=logger)

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError("boto3 is required for S3 model storage.") from exc
            self._client = boto3.client("s3")
        return self._client

    def _join_prefix(self, key: str) -> str:
        cleaned = key.lstrip("/")
        if not self._prefix:
            return cleaned
        if not cleaned:
            return self._prefix
        return f"{self._prefix}/{cleaned}"

    def _strip_prefix(self, key: str, prefix: str) -> str:
        if not prefix:
            return key
        if key.startswith(prefix):
            return key[len(prefix) :].lstrip("/")
        return key

    def _is_not_found(self, exc: Exception) -> bool:
        try:
            from botocore.exceptions import ClientError
        except ImportError:  # pragma: no cover - environment-specific
            return False
        if not isinstance(exc, ClientError):
            return False
        code = exc.response.get("Error", {}).get("Code", "")
        return code in {"404", "NoSuchKey", "NotFound"}

    def _is_credentials_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        if "unable to locate credentials" in message:
            return True
        try:
            from botocore.exceptions import NoCredentialsError, PartialCredentialsError
        except ImportError:  # pragma: no cover - environment-specific
            return False
        return isinstance(exc, (NoCredentialsError, PartialCredentialsError))

    def exists(self, path: str) -> bool:
        key = self._join_prefix(path)
        client = self._get_client()
        try:
            client.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception as exc:  # noqa: BLE001
            if self._is_credentials_error(exc):
                self._logger.warning("S3 credentials unavailable; skipping S3 checks.")
                raise S3UnavailableError("S3 credentials unavailable.") from exc
            if self._is_not_found(exc):
                return False
            self._logger.warning("S3 exists() failed for %s: %s", key, exc)
            raise

    def upload_file(self, local_path: Path, remote_path: str) -> None:
        key = self._join_prefix(remote_path)
        client = self._get_client()
        try:
            client.upload_file(str(local_path), self._bucket, key)
        except Exception as exc:  # noqa: BLE001
            if self._is_credentials_error(exc):
                self._logger.warning("S3 credentials unavailable; skipping upload.")
                raise S3UnavailableError("S3 credentials unavailable.") from exc
            raise

    def download_file(self, remote_path: str, local_path: Path) -> None:
        key = self._join_prefix(remote_path)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client = self._get_client()
        try:
            client.download_file(self._bucket, key, str(local_path))
        except Exception as exc:  # noqa: BLE001
            if self._is_credentials_error(exc):
                self._logger.warning("S3 credentials unavailable; skipping download.")
                raise S3UnavailableError("S3 credentials unavailable.") from exc
            raise

    def list(self, prefix: str) -> List[str]:
        client = self._get_client()
        joined_prefix = self._join_prefix(prefix).rstrip("/")
        paginator = client.get_paginator("list_objects_v2")
        keys: List[str] = []
        try:
            for page in paginator.paginate(Bucket=self._bucket, Prefix=joined_prefix):
                for entry in page.get("Contents", []):
                    key = entry.get("Key", "")
                    if not key:
                        continue
                    relative = self._strip_prefix(key, joined_prefix)
                    if relative:
                        keys.append(relative)
        except Exception as exc:  # noqa: BLE001
            if self._is_credentials_error(exc):
                self._logger.warning("S3 credentials unavailable; skipping list.")
                raise S3UnavailableError("S3 credentials unavailable.") from exc
            raise
        return keys


__all__ = ["S3Store", "S3UnavailableError"]
