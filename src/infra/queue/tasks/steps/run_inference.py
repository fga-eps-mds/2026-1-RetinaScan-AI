# infra/queue/tasks/steps/run_inference.py
import httpx
from domains.exams.models.retina_scan_model import get_retina_scan_model
from infra.logger.logger import logger
from infra.queue.celery_app import celery_app
from infra.storage.minio import download_object_bytes, get_minio_client

from .on_task_failure import ErrorWebhookTask


@celery_app.task(
    bind=True,
    base=ErrorWebhookTask,
    name="domains.exams.steps.run_inference",
    autoretry_for=(httpx.RequestError, TimeoutError, ConnectionError),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
)
def run_inference(self, payload: dict) -> dict:
    exam_id = payload["exam_id"]

    # Usamos .get() para evitar erro caso a chave não exista (ex: exame de apenas um olho)
    left_key = payload.get("left_image_key")
    right_key = payload.get("right_image_key")

    logger.info(
        "Iniciando inferência | exam_id=%s | task_id=%s | left=%s | right=%s",
        exam_id,
        self.request.id,
        left_key,
        right_key,
    )

    minio_client = get_minio_client()
    predictor = get_retina_scan_model()
    results = {}

    if left_key:
        left_bytes = download_object_bytes(minio_client, left_key)
        results["left_eye"] = predictor.predict_bytes(left_bytes)

    if right_key:
        right_bytes = download_object_bytes(minio_client, right_key)
        results["right_eye"] = predictor.predict_bytes(right_bytes)

    payload["result"] = results
    payload["meta"]["inference_done"] = True

    logger.info(
        "Inferência concluída | exam_id=%s | task_id=%s", exam_id, self.request.id
    )
    return payload
