import httpx
from celery import Task

from infra.settings.settings import settings
from infra.logger.logger import logger


class ErrorWebhookTask(Task):
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        payload = None

        if kwargs and isinstance(kwargs, dict):
            payload = kwargs.get("payload")

        if payload is None and args:
            payload = args[0] if isinstance(args[0], dict) else None

        exam_id = payload.get("exam_id") if isinstance(payload, dict) else None

        logger.error(
            "Task falhou | task=%s | exam_id=%s | task_id=%s | retries=%s | erro=%s",
            self.name,
            exam_id,
            task_id,
            self.request.retries,
            exc,
            exc_info=True,
        )

        if not exam_id:
            logger.error("Webhook de erro não enviado: exam_id ausente")
            return

        try:
            response = httpx.post(
                f"{settings.WEBHOOK_URL}/api/exams/{exam_id}/webhook/error",
                json={
                    "exam_id": exam_id,
                    "task_id": task_id,
                    "task_name": self.name,
                    "error": str(exc),
                    "traceback": str(einfo) if einfo else None,
                    "args": payload,
                },
                timeout=10.0,
            )
            response.raise_for_status()
        except Exception as notify_err:
            logger.error(
                "Falha ao enviar webhook de erro | exam_id=%s | erro=%s",
                exam_id,
                notify_err,
                exc_info=True,
            )