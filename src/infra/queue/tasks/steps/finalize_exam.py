import mimetypes
import httpx

from infra.queue.celery_app import celery_app
from infra.settings.settings import settings
from infra.logger.logger import logger


def _content_type(path: str) -> str:
    content_type, _ = mimetypes.guess_type(path)
    return content_type or "application/octet-stream"


@celery_app.task(
    bind=True,
    name="domains.exams.steps.finalize_exam",
    autoretry_for=(httpx.TimeoutException, httpx.RequestError),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 5},
)
def finalize_exam(self, payload: dict) -> dict:
    exam_id = payload["exam_id"]

    logger.info(
        "Finalizando exame | exam_id=%s | task_id=%s | enviando webhook",
        exam_id,
        self.request.id,
    )

    left_result = payload["result"]["left_eye"]
    right_result = payload["result"]["right_eye"]

    left_filename = payload["left_image_key"]
    right_filename = payload["right_image_key"]

    webhook_payload = {
        "total_images": 2,
        "exam_id": exam_id,
        "results": [
            {
                "filename": left_filename,
                "content_type": _content_type(left_filename),
                **left_result,
            },
            {
                "filename": right_filename,
                "content_type": _content_type(right_filename),
                **right_result,
            },
        ],
    }

    try:
        response = httpx.post(
            f"{settings.WEBHOOK_URL}/api/exams/{exam_id}/webhook",
            json=webhook_payload,
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )

        response.raise_for_status()

        logger.info(
            "Webhook enviado com sucesso | exam_id=%s | status=%s",
            exam_id,
            response.status_code,
        )

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code

        logger.error(
            "Webhook retornou erro HTTP | exam_id=%s | status=%s | body=%s | payload=%s",
            exam_id,
            status_code,
            exc.response.text,
            webhook_payload,
        )

        if status_code >= 500:
            raise self.retry(exc=exc)

        raise

    payload["meta"]["pipeline_completed"] = True
    payload["meta"]["webhook_sent"] = True

    return payload