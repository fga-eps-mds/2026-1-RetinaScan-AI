# infra/queue/tasks/steps/run_inference.py
import logging

from infra.queue.celery_app import celery_app
from infra.storage.minio import get_minio_client, download_object_bytes
from domains.exams.models.retina_scan_model import get_retina_scan_model
from infra.logger.logger import logger


@celery_app.task(name="domains.exams.steps.run_inference")
def run_inference(payload: dict) -> dict:
    exam_id = payload["exam_id"]
    left_key = payload["artifacts"]["left_processed_key"]
    right_key = payload["artifacts"]["right_processed_key"]

    logger.info("Iniciando inferência | exam_id=%s | left=%s | right=%s", 
                exam_id, left_key, right_key)

    minio_client = get_minio_client()
    left_bytes = download_object_bytes(minio_client, left_key)
    right_bytes = download_object_bytes(minio_client, right_key)

    predictor = get_retina_scan_model()

    left_pred = predictor.predict_bytes(left_bytes)
    right_pred = predictor.predict_bytes(right_bytes)

    payload["result"] = {
        "left_eye": left_pred,
        "right_eye": right_pred,
    }
    payload["meta"]["inference_done"] = True

    logger.info("Inferência concluída | exam_id=%s", exam_id)
    return payload