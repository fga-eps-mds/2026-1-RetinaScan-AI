# infra/queue/tasks/steps/finalize_exam.py
import logging
import httpx
from typing import Dict, Any, List

from infra.queue.celery_app import celery_app
from infra.settings.settings import settings
from infra.logger.logger import logger


@celery_app.task(name="domains.exams.steps.finalize_exam")
def finalize_exam(payload: dict) -> dict:
    exam_id = payload["exam_id"]
    
    logger.info("Finalizando exame | exam_id=%s | enviando webhook", exam_id)

    # extrai resultados para formato do webhook
    left_result = payload["result"]["left_eye"]
    right_result = payload["result"]["right_eye"]

    # nomes originais dos arquivos
    left_filename = payload["left_image_key"].split("/")[-1]
    right_filename = payload["right_image_key"].split("/")[-1]

    webhook_payload = {
        "total_images": 2,
        "exam_id": exam_id,
        "results": [
            {
                "filename": left_filename,
                "content_type": "image/png",
                **left_result,
            },
            {
                "filename": right_filename,
                "content_type": "image/png",
                **right_result,
            },
        ],
    }

    try:
        response = httpx.post(
            settings.WEBHOOK_URL,
            json=webhook_payload,
            timeout=30.0,
            headers={
                "Content-Type": "application/json",
            },
        )

        logger.info(
            "Webhook enviado | exam_id=%s | status=%s | payload=%s",
            exam_id,
            response.status_code,
            webhook_payload,
        )

        if response.status_code not in (200, 201, 202):
            logger.warning(
                "Webhook retornou status não 2xx | exam_id=%s | status=%s",
                exam_id,
                response.status_code,
            )

    except httpx.TimeoutException:
        logger.error("Webhook timeout | exam_id=%s", exam_id)
    except httpx.RequestError as e:
        logger.error("Webhook erro de rede | exam_id=%s | erro=%s", exam_id, str(e))
    except Exception as e:
        logger.error("Webhook erro inesperado | exam_id=%s | erro=%s", exam_id, str(e))

    payload["meta"]["pipeline_completed"] = True
    payload["meta"]["webhook_sent"] = True

    return payload