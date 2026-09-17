from app.config import Settings
from app.services.cache_service import CacheService, make_cache_key


def test_cache_service_defaults_to_memory_backend_without_redis_url() -> None:
    settings = Settings(REDIS_URL="")
    cache = CacheService(settings)

    assert cache.backend == "memory"


def test_cache_service_set_and_get_json_roundtrip() -> None:
    settings = Settings(REDIS_URL="")
    cache = CacheService(settings)

    key = make_cache_key("test", "value")
    cache.set_json(key, {"hello": "world"}, ttl_seconds=60)

    result = cache.get_json(key)
    assert result == {"hello": "world"}


def test_cache_service_missing_key_returns_none() -> None:
    settings = Settings(REDIS_URL="")
    cache = CacheService(settings)

    assert cache.get_json("does-not-exist-key") is None


def test_make_cache_key_is_deterministic() -> None:
    key_a = make_cache_key("part1", "part2")
    key_b = make_cache_key("part1", "part2")
    key_c = make_cache_key("part1", "different")

    assert key_a == key_b
    assert key_a != key_c
