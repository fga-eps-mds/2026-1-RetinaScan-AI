from .pipeline_config import EXAM_PIPELINE_STEPS
from .registry import STEP_REGISTRY

def build_exam_pipeline(payload: dict):
    if not EXAM_PIPELINE_STEPS:
        raise ValueError("A pipeline de exame está vazia")

    first_step_name = EXAM_PIPELINE_STEPS[0]
    first_task = STEP_REGISTRY.get(first_step_name)

    if not first_task:
        raise ValueError(f"Step inválido na pipeline: {first_step_name}")

    workflow = first_task.s(payload)

    for step_name in EXAM_PIPELINE_STEPS[1:]:
        task = STEP_REGISTRY.get(step_name)

        if not task:
            raise ValueError(f"Step inválido na pipeline: {step_name}")

        workflow |= task.s()

    return workflow