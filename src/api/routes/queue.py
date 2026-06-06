# src/api/routes/queue.py
import json
from celery.result import AsyncResult
from typing import Any

from fastapi import APIRouter

from infra.queue.celery_app import celery_app
from infra.queue.redis_client import get_redis_client

router = APIRouter()

QUEUE_NAME = "retinal_scan_queue"
RESULT_KEY_PREFIX = "celery-task-meta-"


def _flatten_tasks(tasks_by_worker: dict | None, state: str) -> list[dict]:
    if not tasks_by_worker:
        return []

    flattened: list[dict] = []

    for worker_name, tasks in tasks_by_worker.items():
        for task in tasks:
            request = task.get("request", {})

            flattened.append(
                {
                    "worker": worker_name,
                    "task_id": task.get("id") or request.get("id"),
                    "name": task.get("name") or request.get("task"),
                    "args": task.get("args") or request.get("args", []),
                    "kwargs": task.get("kwargs") or request.get("kwargs", {}),
                    "delivery_info": task.get("delivery_info", {}),
                    "hostname": task.get("hostname"),
                    "time_start": task.get("time_start"),
                    "acknowledged": task.get("acknowledged"),
                    "state": state,
                }
            )

    return flattened


def _decode_redis_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _safe_json_loads(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except Exception:
        return None


def _list_broker_preview(redis_client, queue_name: str, limit: int = 10) -> list[dict]:
    raw_items = redis_client.lrange(queue_name, 0, max(limit - 1, 0))
    preview: list[dict] = []

    for idx, item in enumerate(raw_items):
        decoded = _decode_redis_value(item)
        parsed = _safe_json_loads(decoded)

        preview.append(
            {
                "index": idx,
                "raw": decoded,
                "parsed": parsed,
            }
        )

    return preview


def _list_finished_tasks(redis_client, limit: int = 50) -> list[dict]:
    finished: list[dict] = []

    for key in redis_client.scan_iter(f"{RESULT_KEY_PREFIX}*"):
        key_str = _decode_redis_value(key)
        raw = redis_client.get(key)
        if raw is None:
            continue

        decoded = _decode_redis_value(raw)
        parsed = _safe_json_loads(decoded)

        task_id = key_str.replace(RESULT_KEY_PREFIX, "", 1)

        item = {
            "task_id": task_id,
            "redis_key": key_str,
            "raw": decoded,
            "status": None,
            "result": None,
            "traceback": None,
            "date_done": None,
            "children": None,
        }

        if isinstance(parsed, dict):
            item["status"] = parsed.get("status")
            item["result"] = parsed.get("result")
            item["traceback"] = parsed.get("traceback")
            item["date_done"] = parsed.get("date_done")
            item["children"] = parsed.get("children")

        finished.append(item)

    finished.sort(key=lambda x: x.get("date_done") or "", reverse=True)
    return finished[:limit]


@router.get("/tasks")
async def list_queue_tasks():
    inspector = celery_app.control.inspect(timeout=1.0)

    active = inspector.active() or {}
    reserved = inspector.reserved() or {}
    scheduled = inspector.scheduled() or {}
    registered = inspector.registered() or {}
    active_queues = inspector.active_queues() or {}

    active_tasks = _flatten_tasks(active, "active")
    reserved_tasks = _flatten_tasks(reserved, "reserved")
    scheduled_tasks = _flatten_tasks(scheduled, "scheduled")

    redis_client = get_redis_client()

    broker_queue_size = redis_client.llen(QUEUE_NAME)
    broker_preview = _list_broker_preview(redis_client, QUEUE_NAME, limit=10)
    finished_tasks = _list_finished_tasks(redis_client, limit=50)

    worker_names = sorted(
        set(
            list(active.keys())
            + list(reserved.keys())
            + list(scheduled.keys())
            + list(registered.keys())
            + list(active_queues.keys())
        )
    )

    return {
        "workers": worker_names,
        "summary": {
            "queue_name": QUEUE_NAME,
            "active": len(active_tasks),
            "reserved": len(reserved_tasks),
            "scheduled": len(scheduled_tasks),
            "broker_queue_size": broker_queue_size,
            "finished": len(finished_tasks),
            "total_visible_tasks": len(active_tasks)
            + len(reserved_tasks)
            + len(scheduled_tasks),
        },
        "workers_details": {
            "registered": registered,
            "active_queues": active_queues,
        },
        "tasks": {
            "active": active_tasks,
            "reserved": reserved_tasks,
            "scheduled": scheduled_tasks,
            "finished": finished_tasks,
        },
        "broker_preview": broker_preview,
    }

@router.get("/status/{task_id}")
async def get_exam_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else False,
        "failed": result.failed(),
    }

    if result.status == "FAILURE":
        response["error"] = str(result.result)
        response["traceback"] = result.traceback

    elif result.status == "SUCCESS":
        response["result"] = result.result

    return response