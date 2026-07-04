from minio import Minio

from infra.settings.settings import settings
from infra.logger.logger import logger
from io import BytesIO

def get_minio_client() -> Minio:
    return Minio(
        endpoint=settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )

def check_minio_bucket(client: Minio) -> None:
    exists = client.bucket_exists(settings.MINIO_BUCKET_EXAMS)
    if not exists:
        raise RuntimeError(
            f"Bucket '{settings.MINIO_BUCKET_EXAMS}' não encontrado no MinIO."
        )
    
from minio import Minio

from infra.settings.settings import settings
from infra.logger.logger import logger


def download_object_bytes(
    client: Minio,
    object_name: str,
) -> bytes:
    response = None
    try:
        logger.info(
            "Baixando objeto do MinIO | bucket=%s | object=%s",
            settings.MINIO_BUCKET_EXAMS,
            object_name,
        )

        response = client.get_object(
            bucket_name=settings.MINIO_BUCKET_EXAMS,
            object_name=object_name,
        )
        data = response.read()

        logger.info(
            "Download concluído | object=%s | bytes=%s",
            object_name,
            len(data),
        )

        return data
    finally:
        if response is not None:
            response.close()
            response.release_conn()

def upload_object_bytes(
    client: Minio,
    object_name: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> None:
    try:
        logger.info(
            "Enviando objeto para o MinIO | bucket=%s | object=%s | bytes=%s",
            settings.MINIO_BUCKET_EXAMS,
            object_name,
            len(data),
        )

        client.put_object(
            bucket_name=settings.MINIO_BUCKET_EXAMS,
            object_name=object_name,
            data=BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

        logger.info(
            "Upload concluído | object=%s",
            object_name,
        )
    except Exception:
        logger.exception(
            "Falha ao enviar objeto para o MinIO | object=%s",
            object_name,
        )
        raise