import logging
from io import BytesIO

from domains.exams.preprocess_service import preprocess_retina_image
from infra.queue.celery_app import celery_app
from infra.settings.settings import settings
from infra.storage.minio import download_object_bytes, get_minio_client

from .on_task_failure import ErrorWebhookTask

logger = logging.getLogger(__name__)


def _extract_filename_from_key(object_key: str) -> str:
    return object_key.split("/")[-1]


# def _build_processed_key(object_key: str) -> str:
#     path = PurePosixPath(object_key)
#     processed_dir = path.parent / "processed"
#     new_name = f"{path.stem}-processed.png"
#     return str(processed_dir / new_name)


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
    base=ErrorWebhookTask,
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

    new_left_key = (
        left_key.rsplit(".", 1)[0] + ".png" if "." in left_key else left_key + ".png"
    )
    new_right_key = (
        right_key.rsplit(".", 1)[0] + ".png" if "." in right_key else right_key + ".png"
    )

    _upload_png_bytes(minio_client, new_left_key, left_processed_png)
    _upload_png_bytes(minio_client, new_right_key, right_processed_png)

    if new_left_key != left_key:
        minio_client.remove_object(settings.MINIO_BUCKET_EXAMS, left_key)
    if new_right_key != right_key:
        minio_client.remove_object(settings.MINIO_BUCKET_EXAMS, right_key)

    payload["left_image_key"] = new_left_key
    payload["right_image_key"] = new_right_key

    payload["meta"]["preprocessing_done"] = True

    logger.info(
        "Pré-processamento concluído | exam_id=%s | left=%s | right=%s",
        exam_id,
        new_left_key,
        new_right_key,
    )

    return payload
