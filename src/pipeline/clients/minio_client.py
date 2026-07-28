"""MinIO client factory backed by the `minio` service in docker-compose.yml. Generated media
(character images, video clips, voice audio) is uploaded here; `object_uri` fields on the
per-shot generation records store `minio://<bucket>/<key>` locations, resolved to a temporary
signed URL via `presigned_url` whenever a human needs to actually view/listen to the asset
(shot review)."""

from datetime import timedelta
from functools import lru_cache
from io import BytesIO

from minio import Minio

from pipeline.config import get_settings

_URI_PREFIX = "minio://"


@lru_cache
def get_minio_client() -> Minio:
    settings = get_settings()
    return Minio(
        settings.minio_endpoint,
        access_key=settings.minio_root_user,
        secret_key=settings.minio_root_password,
        secure=settings.minio_secure,
    )


def _ensure_bucket(bucket: str) -> None:
    client = get_minio_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def _parse_uri(object_uri: str, default_bucket: str) -> tuple[str, str]:
    if object_uri.startswith(_URI_PREFIX):
        bucket, _, key = object_uri[len(_URI_PREFIX) :].partition("/")
        return bucket, key
    return default_bucket, object_uri


def upload_bytes(object_name: str, data: bytes, content_type: str) -> str:
    """Uploads to the configured bucket and returns the `minio://` uri to store on the run."""
    settings = get_settings()
    _ensure_bucket(settings.minio_bucket)
    get_minio_client().put_object(
        settings.minio_bucket,
        object_name,
        BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return f"{_URI_PREFIX}{settings.minio_bucket}/{object_name}"


def presigned_url(object_uri: str, expires_minutes: int = 60) -> str:
    """Resolves a `minio://` uri to a temporary signed URL a human reviewer can open directly."""
    settings = get_settings()
    bucket, key = _parse_uri(object_uri, settings.minio_bucket)
    return get_minio_client().presigned_get_object(bucket, key, expires=timedelta(minutes=expires_minutes))
