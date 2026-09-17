from __future__ import annotations

"""
Cache service (v3.0): caches repeated chatbot/RAG queries so the same
FAQ-style question asked by many farmers doesn't re-run retrieval (and, if
configured, an LLM generation call) every single time.

Backed by Redis when REDIS_URL is configured and the `redis` package is
installed; otherwise falls back to a simple process-local dict with the same
TTL semantics, so the app behaves identically (just without cross-process
sharing) on a machine with no Redis server - consistent with this project's
existing "optional infra, always runnable" philosophy.
"""

import hashlib
import json
import time
from typing import Any, Optional

from app.config import Settings


def make_cache_key(*parts: str) -> str:
    joined = "||".join(parts)
    return "prsv:" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


class _InMemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._store[key] = (time.time() + ttl_seconds, value)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


class CacheService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._redis_client = None
        self._backend = "memory"
        self._memory_cache = _InMemoryCache()

        if settings.redis_url:
            try:
                import redis

                client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                client.ping()
                self._redis_client = client
                self._backend = "redis"
            except Exception:  # noqa: BLE001 - cache is an optimization, never fatal
                self._redis_client = None
                self._backend = "memory"

    @property
    def backend(self) -> str:
        return self._backend

    def get_json(self, key: str) -> Optional[Any]:
        raw = self._get_raw(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds if ttl_seconds is not None else self.settings.cache_ttl_seconds
        self._set_raw(key, json.dumps(value), ttl)

    def _get_raw(self, key: str) -> Optional[str]:
        if self._backend == "redis" and self._redis_client is not None:
            try:
                return self._redis_client.get(key)
            except Exception:  # noqa: BLE001
                return None
        return self._memory_cache.get(key)

    def _set_raw(self, key: str, value: str, ttl_seconds: int) -> None:
        if self._backend == "redis" and self._redis_client is not None:
            try:
                self._redis_client.setex(key, ttl_seconds, value)
                return
            except Exception:  # noqa: BLE001
                pass
        self._memory_cache.set(key, value, ttl_seconds)

    def stats(self) -> dict:
        return {
            "backend": self._backend,
            "memory_entries": self._memory_cache.size if self._backend == "memory" else None,
        }
