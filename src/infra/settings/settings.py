from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_SECURE: bool = False
    MINIO_BUCKET_EXAMS: str

    REDIS_URL: str

    WEBHOOK_URL: str = "https://webhook.site/206a0b59-a9c1-4fef-8608-d93e2869dda2"
    WEBHOOK_TIMEOUT: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )    

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

settings = Settings()