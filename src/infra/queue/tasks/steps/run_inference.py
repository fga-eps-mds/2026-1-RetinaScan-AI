# infra/queue/tasks/steps/run_inference.py
import httpx

from infra.queue.celery_app import celery_app
from infra.storage.minio import get_minio_client, download_object_bytes, upload_object_bytes
from domains.exams.models.retina_scan_model import get_retina_scan_model
from infra.logger.logger import logger
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
    left_key = payload["artifacts"]["left_processed_key"]
    right_key = payload["artifacts"]["right_processed_key"]

    minio_client = get_minio_client()
    left_bytes = download_object_bytes(minio_client, left_key)
    right_bytes = download_object_bytes(minio_client, right_key)

    predictor = get_retina_scan_model()

    left_pred = predictor.predict_bytes(left_bytes)
    right_pred = predictor.predict_bytes(right_bytes)

    left_gradcam_bytes = left_pred.pop("gradcam_png")
    right_gradcam_bytes = right_pred.pop("gradcam_png")

    left_gradcam_key = f"exams/{exam_id}/OE-{exam_id}-gradcam.png"
    right_gradcam_key = f"exams/{exam_id}/OD-{exam_id}-gradcam.png"

    upload_object_bytes(minio_client, left_gradcam_key, left_gradcam_bytes, content_type="image/png")
    upload_object_bytes(minio_client, right_gradcam_key, right_gradcam_bytes, content_type="image/png")

    payload["artifacts"]["left_gradcam_key"] = left_gradcam_key
    payload["artifacts"]["right_gradcam_key"] = right_gradcam_key

    payload["result"] = {
        "left_eye": left_pred,
        "right_eye": right_pred,
    }
    payload["meta"]["inference_done"] = True

    return payload