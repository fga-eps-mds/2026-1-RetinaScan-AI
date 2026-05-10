import httpx
from celery import Task
from infra.settings.settings import settings
from infra.logger.logger import logger

def on_task_failure(self: Task, exc: Exception, task_id: str, args: dict, kwargs: dict, einfo: str) -> None:    
    exam_id = args.get("exam_id") if args else "unknown"
    
    logger.error(
        "Task falhou após retries | exam_id=%s | task_id=%s | erro=%s",
        exam_id,
        task_id,
        exc,
        exc_info=True,
    )
    
    try:
        httpx.post(
            settings.ERROR_WEBHOOK_URL,
            json={
                "exam_id": exam_id,
                "task_id": task_id,
                "task_name": self.name,
                "error": str(exc),
                "traceback": einfo,
                "args": args,
            },
            timeout=10.0,
        )
    except Exception as notify_err:
        logger.error("Falha ao notificar erro | %s", notify_err)