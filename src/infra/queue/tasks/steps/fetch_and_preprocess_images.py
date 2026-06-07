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


def _process_side(minio_client, key: str | None) -> str | None:
    """Função que processa um lado do exame, salva o png e deleta o jpg."""
    if not key:
        return None

    original_bytes = download_object_bytes(minio_client, key)

    processed_png = preprocess_retina_image(
        image_bytes=original_bytes,
        filename=_extract_filename_from_key(key),
    )

    new_key = key.rsplit(".", 1)[0] + ".png" if "." in key else key + ".png"

    _upload_png_bytes(minio_client, new_key, processed_png)

    if new_key != key:
        minio_client.remove_object(settings.MINIO_BUCKET_EXAMS, key)

    return new_key


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

    logger.info("Iniciando fetch_and_preprocess_images | exam_id=%s", exam_id)

    minio_client = get_minio_client()

    payload["left_image_key"] = _process_side(
        minio_client, payload.get("left_image_key")
    )
    payload["right_image_key"] = _process_side(
        minio_client, payload.get("right_image_key")
    )

    payload["meta"]["preprocessing_done"] = True

    logger.info("Pré-processamento concluído | exam_id=%s", exam_id)

    return payload
