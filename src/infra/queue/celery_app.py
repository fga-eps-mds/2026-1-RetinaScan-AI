from celery import Celery
from infra.settings.settings import settings
from kombu import Queue

celery_app = Celery(
    "retinascan",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,

    include=[
        "infra.queue.tasks.steps.fetch_and_preprocess_images",
        "infra.queue.tasks.steps.run_inference",
        "infra.queue.tasks.steps.finalize_exam",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="America/Sao_Paulo",
    enable_utc=True,
    task_track_started=True,

    task_default_queue="retinal_scan_queue",
    task_default_routing_key="retinal_scan_queue",
    task_queues=(
        Queue("retinal_scan_queue", routing_key="retinal_scan_queue"),
    ),
)