import logging
from io import BytesIO
from pathlib import PurePosixPath

from infra.queue.celery_app import celery_app
from infra.settings.settings import settings
from infra.storage.minio import get_minio_client, download_object_bytes
from domains.exams.preprocess_service import preprocess_retina_image

logger = logging.getLogger(__name__)

def _extract_filename_from_key(object_key: str) -> str:
    return object_key.split("/")[-1]

def _build_processed_key(object_key: str) -> str:
    path = PurePosixPath(object_key)
    processed_dir = path.parent / "processed"
    new_name = f"{path.stem}-processed.png"
    return str(processed_dir / new_name)

def _upload_png_bytes(client, object_name: str, data: bytes) -> None:
    client.put_object(
        bucket_name=settings.MINIO_BUCKET_EXAMS,
        object_name=object_name,
        data=BytesIO(data),
        length=len(data),
        content_type="image/png",
    )

@celery_app.task(
    bind=True,
    name="domains.exams.steps.fetch_and_preprocess_images",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 4},
)
def fetch_and_preprocess_images(self, payload: dict) -> dict:
    exam_id = payload["exam_id"]
    left_key = payload["left_image_key"]
    right_key = payload["right_image_key"]

    logger.info("Iniciando fetch_and_preprocess_images | exam_id=%s", exam_id)

    minio_client = get_minio_client()

    left_original_bytes = download_object_bytes(minio_client, left_key)
    right_original_bytes = download_object_bytes(minio_client, right_key)

    left_processed_png = preprocess_retina_image(
        image_bytes=left_original_bytes,
        filename=_extract_filename_from_key(left_key),
    )
    right_processed_png = preprocess_retina_image(
        image_bytes=right_original_bytes,
        filename=_extract_filename_from_key(right_key),
    )

    left_processed_key = _build_processed_key(left_key)
    right_processed_key = _build_processed_key(right_key)

    _upload_png_bytes(minio_client, left_processed_key, left_processed_png)
    _upload_png_bytes(minio_client, right_processed_key, right_processed_png)

    payload["artifacts"]["left_processed_key"] = left_processed_key
    payload["artifacts"]["right_processed_key"] = right_processed_key
    payload["meta"]["preprocessing_done"] = True

    logger.info(
        "Pré-processamento concluído | exam_id=%s | left=%s | right=%s",
        exam_id,
        left_processed_key,
        right_processed_key,
    )

    return payload