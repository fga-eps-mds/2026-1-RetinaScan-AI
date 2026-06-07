from infra.settings.settings import settings


def normalize_object_key(key: str) -> str:
    if key is None:
        return None

    prefix = f"{settings.MINIO_BUCKET_EXAMS}/"
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key
