"""MinIO client factory backed by the `minio` service in docker-compose.yml."""

from functools import lru_cache

from minio import Minio

from pipeline.config import get_settings


@lru_cache
def get_minio_client() -> Minio:
    settings = get_settings()
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_root_user,
        secret_key=settings.minio_root_password,
        secure=settings.minio_secure,
    )
