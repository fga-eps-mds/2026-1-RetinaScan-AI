from typing import Optional

from pydantic import BaseModel


class EnqueueExamRequest(BaseModel):
    exam_id: str
    left_image_key: Optional[str] = None
    right_image_key: Optional[str] = None


class EnqueueExamResponse(BaseModel):
    message: str
    exam_id: str
    task_id: str
    status: str
