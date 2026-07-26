"""Redis client factory backed by the `redis` service in docker-compose.yml."""

from functools import lru_cache

import redis

from pipeline.config import get_settings


@lru_cache
def get_redis_client() -> redis.Redis:
    return redis.from_url(get_settings().redis_url, decode_responses=True)
