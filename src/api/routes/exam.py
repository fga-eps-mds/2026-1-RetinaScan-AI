from fastapi import APIRouter, status

from infra.logger.logger import logger
from api.schemas.exams import EnqueueExamRequest, EnqueueExamResponse
from domains.exams.utils import normalize_object_key
from infra.queue.tasks.pipeline import build_exam_pipeline

router = APIRouter()


@router.post(
    "/analyze",
    response_model=EnqueueExamResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enfileirar exame para processamento",
    description="Enfileira um exame para processamento assíncrono.",
)
async def enqueue_exam(payload: EnqueueExamRequest):
    try:
        exam_payload = {
            "exam_id": payload.exam_id,
            "left_image_key": normalize_object_key(payload.left_image_key),
            "right_image_key": normalize_object_key(payload.right_image_key),
            "artifacts": {},
            "result": {},
            "meta": {},
        }

        workflow = build_exam_pipeline(exam_payload)
        task_result = workflow.apply_async()

        logger.info(
            "Pipeline enfileirada | exam_id=%s | task_id=%s",
            payload.exam_id,
            task_result.id,
        )

        return {
            "message": "Pipeline enfileirada com sucesso",
            "exam_id": payload.exam_id,
            "task_id": task_result.id,
            "status": "queued",
        }

    except Exception as e:
        logger.exception("Erro ao enfileirar exame: %s", str(e))
        raise