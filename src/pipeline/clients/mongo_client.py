"""MongoDB client factory backed by the `mongodb` service in docker-compose.yml."""

from functools import lru_cache

from pymongo import MongoClient

from pipeline.config import get_settings


@lru_cache
def get_mongo_client() -> MongoClient:
    return MongoClient(get_settings().mongo_url)
