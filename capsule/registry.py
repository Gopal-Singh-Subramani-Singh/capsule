from __future__ import annotations
import hashlib
import os
from pathlib import Path
from typing import List, Optional
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import structlog

logger = structlog.get_logger(__name__)

# Maximum model file size accepted for upload (2 GB)
_MAX_MODEL_SIZE_BYTES = 2 * 1024 ** 3


class ModelRegistry:
    """
    MinIO-backed model registry. S3-compatible API.

    Production hardening:
    - Credentials never logged
    - Short connect timeout + single retry so CLI startup is fast
    - Content-addressable digests stored alongside image tags
    - All write operations are no-ops (with a warning) when offline
    - Paginated list_versions — handles buckets with many objects
    - upload validates file size before starting the transfer
    """

    def __init__(
        self,
        endpoint: str = "http://localhost:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        bucket: str = "capsule-models",
        secure: bool = False,
    ):
        self.bucket = bucket
        self._available = False
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="us-east-1",
            config=Config(
                connect_timeout=3,
                retries={"max_attempts": 1, "mode": "standard"},
            ),
        )
        self._try_connect()

    # ── connectivity ──────────────────────────────────────────────────────

    def _try_connect(self):
        try:
            self._ensure_bucket()
            self._available = True
            logger.info("registry.connected", bucket=self.bucket)
        except Exception as exc:
            # Do NOT log exc directly — it may contain credential hints
            logger.warning(
                "registry.unavailable",
                reason="Could not reach registry endpoint",
                detail=type(exc).__name__,
            )

    def _ensure_bucket(self):
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchBucket"):
                self._client.create_bucket(Bucket=self.bucket)
                logger.info("registry.bucket_created", bucket=self.bucket)
            else:
                raise

    def _require_available(self, operation: str) -> bool:
        if not self._available:
            logger.warning("registry.offline", operation=operation)
        return self._available

    # ── model artifacts ───────────────────────────────────────────────────

    def push_model(self, name: str, version: str, local_path: str, framework: str) -> str:
        """Upload a model artifact; returns the S3 key."""
        if not self._require_available("push_model"):
            return f"models/{name}/{version}/model"

        size = os.path.getsize(local_path)
        if size > _MAX_MODEL_SIZE_BYTES:
            raise ValueError(
                f"Model file {local_path} is {size / 1024**2:.0f} MB, "
                f"exceeds limit of {_MAX_MODEL_SIZE_BYTES // 1024**2} MB"
            )

        ext = Path(local_path).suffix
        key = f"models/{name}/{version}/model{ext}"
        self._client.upload_file(local_path, self.bucket, key)
        logger.info(
            "registry.model_pushed",
            name=name, version=version, key=key,
            size_mb=round(size / 1024 ** 2, 2),
        )
        return key

    def pull_model(self, name: str, version: str, local_dir: str) -> str:
        """Download a model artifact; returns local path."""
        if not self._available:
            raise ConnectionError("Registry unavailable — cannot pull model")

        prefix = f"models/{name}/{version}/"
        response = self._client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        objects = [
            o for o in response.get("Contents", [])
            if not o["Key"].endswith(("/image.tag", "/Dockerfile", "/digest"))
        ]
        if not objects:
            raise FileNotFoundError(f"No model artifact found for {name}:{version}")

        key = objects[0]["Key"]
        local_path = os.path.join(local_dir, os.path.basename(key))
        os.makedirs(local_dir, exist_ok=True)
        self._client.download_file(self.bucket, key, local_path)
        logger.info("registry.model_pulled", name=name, version=version)
        return local_path

    # ── image metadata ────────────────────────────────────────────────────

    def push_image_tag(self, name: str, version: str, image_tag: str):
        """Store the mutable Docker image tag string."""
        if not self._require_available("push_image_tag"):
            return
        key = f"models/{name}/{version}/image.tag"
        self._client.put_object(Bucket=self.bucket, Key=key, Body=image_tag.encode())

    def push_image_digest(self, name: str, version: str, digest: str):
        """
        Store the content-addressable image digest (sha256:...).
        Rollback uses this instead of the mutable tag so it always
        pulls the exact binary that was verified at package time.
        """
        if not self._require_available("push_image_digest"):
            return
        key = f"models/{name}/{version}/image.digest"
        self._client.put_object(Bucket=self.bucket, Key=key, Body=digest.encode())

    def get_image_tag(self, name: str, version: str) -> Optional[str]:
        if not self._available:
            return None
        try:
            resp = self._client.get_object(
                Bucket=self.bucket, Key=f"models/{name}/{version}/image.tag"
            )
            return resp["Body"].read().decode().strip()
        except ClientError:
            return None

    def get_image_digest(self, name: str, version: str) -> Optional[str]:
        """Return the pinned sha256 digest, or None if not stored."""
        if not self._available:
            return None
        try:
            resp = self._client.get_object(
                Bucket=self.bucket, Key=f"models/{name}/{version}/image.digest"
            )
            return resp["Body"].read().decode().strip()
        except ClientError:
            return None

    def get_deploy_ref(self, name: str, version: str) -> Optional[str]:
        """
        Return the best available deploy reference:
        - sha256 digest if available (immutable)
        - mutable tag as fallback
        """
        digest = self.get_image_digest(name, version)
        if digest:
            tag = self.get_image_tag(name, version)
            if tag:
                # repo@sha256:... is the fully pinned form
                repo = tag.rsplit(":", 1)[0]
                return f"{repo}@{digest}"
        return self.get_image_tag(name, version)

    # ── dockerfile storage ────────────────────────────────────────────────

    def push_dockerfile(self, name: str, version: str, dockerfile_content: str):
        if not self._require_available("push_dockerfile"):
            return
        key = f"models/{name}/{version}/Dockerfile"
        self._client.put_object(
            Bucket=self.bucket, Key=key, Body=dockerfile_content.encode()
        )

    # ── version listing ───────────────────────────────────────────────────

    def list_versions(self, name: str) -> List[str]:
        if not self._available:
            return []
        versions = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=f"models/{name}/",
            Delimiter="/",
        ):
            for cp in page.get("CommonPrefixes", []):
                ver = cp["Prefix"].rstrip("/").split("/")[-1]
                versions.append(ver)
        return sorted(versions, reverse=True)

    def model_exists(self, name: str, version: str) -> bool:
        if not self._available:
            return False
        response = self._client.list_objects_v2(
            Bucket=self.bucket, Prefix=f"models/{name}/{version}/", MaxKeys=1
        )
        return bool(response.get("Contents"))
