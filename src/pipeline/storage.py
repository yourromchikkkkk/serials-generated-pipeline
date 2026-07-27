"""Persistence for pipeline run/result records (structured artifacts) in MongoDB — generated
media itself lives in MinIO (see clients/minio_client.py); Mongo holds the Run documents that
reference it.
"""

from pymongo.collection import Collection

from pipeline.clients.mongo_client import get_mongo_client
from pipeline.config import get_settings
from pipeline.graph.state import Run


def get_runs_collection() -> Collection:
    settings = get_settings()
    return get_mongo_client()[settings.mongo_database]["runs"]


def save_run(run: Run) -> None:
    """Upsert the run's current state as a single document, keyed by run_id."""
    get_runs_collection().replace_one(
        {"run_id": run.run_id},
        run.model_dump(),
        upsert=True,
    )


def load_run(run_id: str) -> Run | None:
    document = get_runs_collection().find_one({"run_id": run_id})
    if document is None:
        return None
    document.pop("_id", None)
    return Run.model_validate(document)
