from __future__ import annotations

# HasSessionCache
import logging
from abc import abstractmethod
from collections import namedtuple
from types import SimpleNamespace
from typing import Callable, Hashable, List, Optional, Tuple, Type, TypeVar

from sqlalchemy import Column
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from . import get_one

T = TypeVar("T", bound="HasSessionCache")


class HasSessionCache:
    CacheTuple = namedtuple("CacheTuple", ["id", "key", "stats"])
    CACHE_ATTRIBUTE = "_palace_cache"

    """
    A mixin class for ORM classes that maintain an in-memory cache of
    items previously requested from the database table for performance reasons.

    Items in this cache are always maintained in the same database session.
    """

    @property
    @abstractmethod
    def id(self) -> Column[int]:
        ...

    @abstractmethod
    def cache_key(self) -> Hashable:
        ...

    @classmethod
    def log(cls):
        return logging.getLogger(cls.__class__.__name__)

    @classmethod
    def cache_warm(
        cls: Type[T],
        db: Session,
        get_objects: Optional[Callable[[], List[T]]] = None,
    ):
        """
        Populate the cache with the contents of `get_objects`. Useful to populate
        the cache in advance with items we know we will use.
        """
        cache = cls._cache_from_session(db)
        if get_objects is None:
            # Populate the cache with the whole table
            get_objects = db.query(cls).all
        objects = get_objects()
        for obj in objects:
            cls._cache_insert(obj, cache)

    @classmethod
    def _cache_insert(cls: Type[T], obj: T, cache: CacheTuple):
        """Cache an object for later retrieval."""
        key = obj.cache_key()
        id = obj.id
        cache.id[id] = obj
        cache.key[key] = obj

    @classmethod
    def _cache_remove(cls: Type[T], obj: T, cache: CacheTuple):
        """Remove an object from the cache"""
        try:
            key = obj.cache_key()
            id = obj.id
            cache.id.pop(id)
            cache.key.pop(key)
        except (KeyError, SQLAlchemyError):
            # We couldn't find the item we want to remove, or there was an exception
            # getting its cache key or ID, perhaps because its id or key values are
            # no longer valid. Reset the cache to be safe.
            cls.log().warning("Unable to remove object from cache. Resetting cache.")
            cache.id.clear()
            cache.key.clear()

    @classmethod
    def _cache_lookup(
        cls: Type[T],
        db: Session,
        cache: CacheTuple,
        cache_name: str,
        cache_key: Hashable,
        cache_miss_hook: Callable,
    ) -> Tuple[Optional[T], bool]:
        """Helper method used by both by_id and by_cache_key.

        Looks up `cache_key` in the `cache_name` property of `cache`, returning
        the item if its found, or calling `cache_miss_hook` and adding the item to
        the cache if its not. This method also updates our cache statistics to
        keep track of cache hits and misses.
        """
        lookup_cache = getattr(cache, cache_name)
        if cache_key in lookup_cache:
            obj = lookup_cache[cache_key]
            if obj not in db or obj in db.deleted:
                # This object has been deleted since it was cached. Remove it from
                # cache and do another lookup.
                cls._cache_remove(obj, cache)
                return cls._cache_lookup(
                    db, cache, cache_name, cache_key, cache_miss_hook
                )

            else:
                # Object is good, return it from cache
                cache.stats.hits += 1
                return obj, False

        else:
            cache.stats.misses += 1
            obj, new = cache_miss_hook()
            if obj is not None:
                cls._cache_insert(obj, cache)
            return obj, new

    @classmethod
    def _cache_from_session(cls, _db: Session):
        """Get cache from database session."""

        # https://docs.sqlalchemy.org/en/14/orm/session_api.html#sqlalchemy.orm.Session.info
        if cls.CACHE_ATTRIBUTE not in _db.info:
            _db.info[cls.CACHE_ATTRIBUTE] = {}
        cache = _db.info[cls.CACHE_ATTRIBUTE]
        if cls.__name__ not in cache:
            cache[cls.__name__] = cls.CacheTuple(
                {}, {}, SimpleNamespace(hits=0, misses=0)
            )
        return cache[cls.__name__]

    @classmethod
    def by_id(cls: Type[T], db: Session, id: int) -> Optional[T]:
        """Look up an item by its unique database ID."""
        cache = cls._cache_from_session(db)

        def lookup_hook():
            return get_one(db, cls, id=id), False

        obj, _ = cls._cache_lookup(db, cache, "id", id, lookup_hook)
        return obj

    @classmethod
    def by_cache_key(
        cls: Type[T], db: Session, cache_key: Hashable, cache_miss_hook: Callable
    ) -> Tuple[Optional[T], bool]:
        """Look up an item by its cache key."""
        cache = cls._cache_from_session(db)
        return cls._cache_lookup(db, cache, "key", cache_key, cache_miss_hook)
