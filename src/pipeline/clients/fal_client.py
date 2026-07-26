"""fal.ai client factory (character/video/voice/lip-sync generation)."""

from functools import lru_cache

import fal_client

from pipeline.config import get_settings


@lru_cache
def get_fal_client() -> fal_client.SyncClient:
    return fal_client.SyncClient(key=get_settings().fal_key)
