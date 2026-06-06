from .steps.fetch_and_preprocess_images import fetch_and_preprocess_images
from .steps.run_inference import run_inference
from .steps.finalize_exam import finalize_exam

STEP_REGISTRY = {
    "fetch_and_preprocess_images": fetch_and_preprocess_images,
    "run_inference": run_inference,
    "finalize_exam": finalize_exam,
}