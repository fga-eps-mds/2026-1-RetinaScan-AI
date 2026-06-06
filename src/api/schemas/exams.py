from pydantic import BaseModel

class EnqueueExamRequest(BaseModel):
    exam_id: str
    left_image_key: str
    right_image_key: str

class EnqueueExamResponse(BaseModel):
    message: str
    exam_id: str
    task_id: str
    status: str