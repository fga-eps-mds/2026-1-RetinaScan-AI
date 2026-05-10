# domains/exams/tasks/steps/fetch_and_preprocess_images.py
import logging
from io import BytesIO

from infra.queue.celery_app import celery_app
from infra.settings.settings import settings
from infra.storage.minio import get_minio_client, download_object_bytes
from domains.exams.image_loader import load_and_clean_image
from domains.exams.preprocess_service import preprocess_retina_image

logger = logging.getLogger(__name__)


def _extract_filename_from_key(object_key: str) -> str:
    return object_key.split("/")[-1]


def _upload_png_bytes(client, object_name: str, data: bytes) -> None:
    client.put_object(
        bucket_name=settings.MINIO_BUCKET_EXAMS,
        object_name=object_name,
        data=BytesIO(data),
        length=len(data),
        content_type="image/png",
    )


@celery_app.task(name="domains.exams.steps.fetch_and_preprocess_images")
def fetch_and_preprocess_images(payload: dict) -> dict:
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

    left_processed_key = f"exams/{exam_id}/processed/OE-clean.png"
    right_processed_key = f"exams/{exam_id}/processed/OD-clean.png"

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