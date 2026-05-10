import redis

from infra.settings.settings import settings

def get_redis_client() -> redis.Redis:
    return redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
    )


def check_redis_connection(client: redis.Redis) -> None:
    pong = client.ping()
    if pong is not True:
        raise RuntimeError("Redis respondeu de forma inesperada ao ping.")