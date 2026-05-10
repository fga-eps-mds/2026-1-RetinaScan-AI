from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from contextlib import asynccontextmanager

from infra.settings.settings import settings
from api.routes.index import api_router
from infra.storage.minio import get_minio_client, check_minio_bucket
from infra.logger.logger import logger
from infra.queue.redis_client import get_redis_client, check_redis_connection

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        logger.info("Conectando ao MinIO em %s", settings.MINIO_ENDPOINT)
        minio_client = get_minio_client()
        check_minio_bucket(minio_client)

        app.state.minio = minio_client

    except Exception as e:
        logger.exception("Erro ao conectar ao MinIO: %s", str(e))
        raise RuntimeError("Falha na inicialização do MinIO") from e
    
    try:
        logger.info("Conectando ao Redis em %s", settings.REDIS_URL)

        redis_client = get_redis_client()
        check_redis_connection(redis_client)

        app.state.redis = redis_client
        logger.info("Redis conectado com sucesso.")

    except Exception as e:
        logger.exception("Erro ao conectar ao Redis: %s", str(e))
        raise RuntimeError("Falha na inicialização do Redis") from e
    
    logger.info("Startup finalizado com sucesso.")
    yield
    logger.info("Encerrando API...")

def create_app() -> FastAPI:
    app = FastAPI(
        title="RetinaScan AI",
        description="API para análise de imagens de retina usando IA",
        version="1.0.0",
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    logger.info("CORS liberado para: %s", settings.cors_origins)
    logger.info("Rotas registradas com prefixo /api")

    app.include_router(api_router, prefix="/api")

    for route in app.routes:
        logger.info("ROTA -> %s %s", getattr(route, "methods", []), route.path)


    return app

app = create_app()
    