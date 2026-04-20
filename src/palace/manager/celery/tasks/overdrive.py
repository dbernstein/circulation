import datetime
from typing import Any, Literal, TypedDict, TypeGuard
from uuid import uuid4

from celery import chain, chord, group, shared_task
from celery.exceptions import Ignore

from palace.util.datetime_helpers import utc_now

from palace.manager.celery.importer import (
    import_all as create_import_tasks,
    import_key,
    import_workflow_lock,
)
from palace.manager.celery.task import Task
from palace.manager.celery.tasks import apply
from palace.manager.celery.utils import load_from_id
from palace.manager.integration.license.overdrive.api import (
    BookInfoEndpoint,
    OverdriveAPI,
)
from palace.manager.integration.license.overdrive.importer import OverdriveImporter
from palace.manager.service.celery.celery import QueueNames
from palace.manager.service.redis.models.set import IdentifierSet
from palace.manager.sqlalchemy.model.collection import Collection
from palace.manager.util.http.exception import (
    BadResponseException,
    RemoteIntegrationException,
    RequestTimedOut,
)

IMPORT_SKIPPED: str = "import_skipped"


class ImportSkippedPayload(TypedDict):
    """Payload returned when import is skipped (workflow lock already held)."""

    import_skipped: Literal[True]


class ImportRouterResult(TypedDict, total=False):
    """Result of import_result_router: chord_id when chord runs, import_skipped when skipped."""

    import_skipped: Literal[True]
    chord_id: str | None


@shared_task(
    queue=QueueNames.default,
    bind=True,
    max_retries=4,
    autoretry_for=(BadResponseException, RequestTimedOut),
    throws=(RemoteIntegrationException,),
    retry_backoff=60,
)
def import_title_updates(
    task: Task,
    collection_id: int,
    *,
    import_all: bool = False,
    page: str | None = None,
    total_items: int | None = None,
    modified_since: datetime.datetime | None = None,
    start_time: datetime.datetime | None = None,
    lock_value: str | None = None,
) -> IdentifierSet | ImportSkippedPayload | None:
    """Phase 1 import: process title-metadata changes in reverse chronological order.

    Iterates the product list sorted by ``lastTitleUpdateTime`` starting from the last
    page (most recently changed) and works backwards.  Stops when a book whose
    bibliographic hash is unchanged is encountered (unless ``import_all=True``).

    Uses ``task.replace()`` to chain itself for subsequent pages while holding the
    workflow lock.  Returns an :class:`~palace.manager.service.redis.models.set.IdentifierSet`
    of all identifiers whose metadata was queued for update so that phase 2 can skip
    redundant metadata fetches.

    :param collection_id: The ID of the collection to import.
    :param import_all: When ``True`` every record is processed regardless of whether
        it has changed.
    :param page: URL of the page to process.  ``None`` on the initial call.
    :param total_items: Total items in the result set, forwarded across pages.
    :param modified_since: Lower bound for the ``lastTitleUpdateTime`` filter.
    :param start_time: When this import run began; set on the first page.
    :param lock_value: UUID identifying this import workflow across page boundaries.
    :return: :class:`IdentifierSet` of updated identifiers, or an
        :class:`ImportSkippedPayload` when another import is in progress.
    """
    redis = task.services.redis().client()
    registry = task.services.integration_registry().license_providers()

    if start_time is None:
        start_time = utc_now()

    is_first_page = page is None and lock_value is None
    if lock_value is None:
        lock_value = str(uuid4())

    workflow_lock = import_workflow_lock(redis, collection_id, lock_value)

    with workflow_lock.lock(
        raise_when_not_acquired=False,
        ignored_exceptions=(Ignore, BadResponseException, RequestTimedOut),
    ) as workflow_lock_acquired:
        if not workflow_lock_acquired and is_first_page:
            task.log.warning(
                f"OverDrive title-update import skipped for collection {collection_id}: "
                "another import is already in progress."
            )
            return _import_skipped_payload()
        if not workflow_lock_acquired and not is_first_page:
            task.log.warning(
                f"OverDrive title-update import for collection {collection_id}: workflow lock expired "
                "between pages; continuing (another import may be running)."
            )

        with task.transaction() as session:
            collection = load_from_id(session, Collection, collection_id)
            collection_name = collection.name

            identifier_set = IdentifierSet(
                redis, import_key(collection.id, task.request.id)
            )

            if collection.marked_for_deletion:
                task.log.warning(
                    f"This collection is marked for deletion. "
                    f"Skipping title-update import of '{collection_name}'."
                )
                return identifier_set

            importer = OverdriveImporter(
                db=session,
                collection=collection,
                registry=registry,
                identifier_set=identifier_set,
            )

            task.log.info(
                f"OverDrive title-update import started: '{collection_name}' "
                f"modified_since={modified_since}, page={page}"
            )

            endpoint = None if not page else BookInfoEndpoint(page)

            result = importer.import_title_updates(
                apply_bibliographic=apply.bibliographic_apply.delay,
                import_all=import_all,
                endpoint=endpoint,
                modified_since=modified_since,
                total_items=total_items,
            )

            task.log.info(
                f"OverDrive title-update import page complete: '{collection_name}' "
                f"Page: {result.current_page}. Processed: {result.processed_count}."
            )

        if result.next_page is not None:
            task.log.info(
                f"OverDrive title-update import re-queueing: '{collection_name}' "
                f"Next page: {result.next_page}."
            )
            raise task.replace(
                task.s(
                    collection_id=collection_id,
                    import_all=import_all,
                    page=result.next_page.url,
                    total_items=result.total_items,
                    modified_since=modified_since,
                    start_time=start_time,
                    lock_value=lock_value,
                )
            )
        else:
            return identifier_set


@shared_task(queue=QueueNames.default, bind=True)
def import_availability_bridge(
    task: Task,
    title_update_result: IdentifierSet | dict[str, Any] | ImportSkippedPayload | None,
    collection_id: int,
    *,
    import_all: bool = False,
    modified_since: datetime.datetime | None = None,
    start_time: datetime.datetime | None = None,
) -> ImportSkippedPayload | None:
    """Bridge between the title-update phase and the availability phase.

    Receives the phase-1 :class:`IdentifierSet` (or skip payload) from the Celery
    chain and immediately replaces itself with :func:`import_collection`, preserving
    the chain's callback to :func:`import_result_router`.

    :param title_update_result: Result from :func:`import_title_updates`.
    :param collection_id: The collection to import.
    :param import_all: Forwarded to :func:`import_collection`.
    :param modified_since: Forwarded to :func:`import_collection`.
    :param start_time: Forwarded to :func:`import_collection`.
    """
    if _is_import_skipped(title_update_result):
        task.log.info(
            f"OverDrive availability import skipped for collection {collection_id}: "
            "title-update phase was skipped."
        )
        return _import_skipped_payload()

    title_update_identifiers: dict[str, Any] | None = None
    if title_update_result is not None:
        title_update_identifiers = (
            title_update_result.__json__()
            if isinstance(title_update_result, IdentifierSet)
            else title_update_result  # already a serialised dict
        )

    raise task.replace(
        import_collection.s(
            collection_id=collection_id,
            import_all=import_all,
            modified_since=modified_since,
            start_time=start_time,
            title_update_identifiers=title_update_identifiers,
        )
    )


@shared_task(
    queue=QueueNames.default,
    bind=True,
    max_retries=4,
    autoretry_for=(BadResponseException, RequestTimedOut),
    throws=(RemoteIntegrationException,),
    retry_backoff=60,
)
def import_collection(
    task: Task,
    collection_id: int,
    *,
    import_all: bool = False,
    page: str | None = None,
    total_items: int | None = None,
    modified_since: datetime.datetime | None = None,
    start_time: datetime.datetime | None = None,
    return_identifiers: bool = True,
    parent_identifiers: dict[str, Any] | None = None,
    title_update_identifiers: dict[str, Any] | None = None,
    lock_value: str | None = None,
) -> IdentifierSet | ImportSkippedPayload | None:
    """Phase 2 import: process availability (circulation) changes in reverse chronological order.

    Iterates the product list sorted by ``lastUpdateTime`` starting from the last page
    and works backwards.  Stops when a book whose circulation hash is unchanged is
    encountered (unless ``import_all=True``).

    :param collection_id: The ID of the collection to import.
    :param import_all: When ``True`` every record is processed regardless of whether
        it has changed.
    :param page: URL of the page to process.  ``None`` on the initial call.
    :param total_items: Total items in the result set, forwarded across pages.
    :param modified_since: Only process titles modified after this datetime.
    :param start_time: When this import run began.
    :param return_identifiers: When ``True`` build and return an
        :class:`IdentifierSet` of all processed identifiers.
    :param parent_identifiers: Serialised :class:`IdentifierSet` from the parent
        collection (Advantage collections only).
    :param title_update_identifiers: Serialised :class:`IdentifierSet` from phase 1.
        Identifiers in this set skip the metadata fetch in phase 2.
    :param lock_value: UUID identifying this import workflow across page boundaries.
    :return: :class:`IdentifierSet` when ``return_identifiers`` is ``True``;
        ``None`` otherwise; :class:`ImportSkippedPayload` when skipped.
    """
    redis = task.services.redis().client()
    registry = task.services.integration_registry().license_providers()

    if start_time is None:
        start_time = utc_now()

    # Both page and lock_value are None only on the first page of a fresh import.
    # They are always set together when task.replace() chains to the next page.
    is_first_page = page is None and lock_value is None
    if lock_value is None:
        lock_value = str(uuid4())

    workflow_lock = import_workflow_lock(redis, collection_id, lock_value)

    # Ignore is raised by task.replace() and Retry is raised by autoretry_for exceptions.
    # Neither should release the workflow lock: replace() hands it to the next page task,
    # and retries should continue holding the lock across the backoff window.
    with workflow_lock.lock(
        raise_when_not_acquired=False,
        ignored_exceptions=(Ignore, BadResponseException, RequestTimedOut),
    ) as workflow_lock_acquired:
        if not workflow_lock_acquired and is_first_page:
            task.log.warning(
                f"OverDrive import skipped for collection {collection_id}: "
                "another import is already in progress."
            )
            return _import_skipped_payload()
        if not workflow_lock_acquired and not is_first_page:
            task.log.warning(
                f"OverDrive import for collection {collection_id}: workflow lock expired "
                "between pages; continuing (another import may be running)."
            )

        with task.transaction() as session:
            collection = load_from_id(session, Collection, collection_id)
            collection_name = collection.name

            identifier_set = (
                IdentifierSet(redis, import_key(collection.id, task.request.id))
                if return_identifiers
                else None
            )

            if collection.marked_for_deletion:
                task.log.warning(
                    f"This collection is marked for deletion. "
                    f"Skipping import of '{collection_name}'."
                )
                return identifier_set

            parent_identifier_set = (
                rehydrate_identifier_set(task, parent_identifiers)
                if parent_identifiers
                else None
            )

            title_update_identifier_set = (
                rehydrate_identifier_set(task, title_update_identifiers)
                if title_update_identifiers
                else None
            )

            importer = OverdriveImporter(
                db=session,
                collection=collection,
                registry=registry,
                identifier_set=identifier_set,
                parent_identifier_set=parent_identifier_set,
                title_update_identifier_set=title_update_identifier_set,
            )

            if modified_since is None:
                if import_all:
                    modified_since = None
                else:
                    timestamp = importer.get_timestamp()
                    modified_since = timestamp.start

            task.log.info(
                f"OverDrive import started: '{collection_name}' Modified since: {modified_since}, "
                f"page: {None if not page else page}"
            )

            endpoint = None if not page else BookInfoEndpoint(page)

            result = importer.import_collection(
                apply_bibliographic=apply.bibliographic_apply.delay,
                apply_circulation=apply.circulation_apply.delay,
                import_all=import_all,
                endpoint=endpoint,
                modified_since=modified_since,
                total_items=total_items,
            )

            task.log.info(
                f"OverDrive import page complete: '{collection_name}' Page: {result.current_page}. "
                f"Processed: {result.processed_count}. "
            )

            if identifier_set:
                task.log.info(
                    f"OverDrive collection import '{collection_name}': Total processed in run so far: {identifier_set.len()}"
                )

            if result.next_page is None:
                # We are done. We only update the timestamp once we have processed all pages.
                # To make sure that if we fail or are interrupted, we re-process any
                # titles we may have missed.
                timestamp = importer.get_timestamp()
                timestamp.start = start_time
                timestamp.finish = utc_now()
                task.log.info(
                    f"OverDrive import complete: '{collection_name}' Total time: {timestamp.elapsed}."
                )

        if result.next_page is not None:
            task.log.info(
                f"OverDrive import re-queueing: '{collection_name}' Next page: {result.next_page}."
            )
            # Serialize identifier sets for passing to next task
            serialized_parent_identifiers = (
                parent_identifier_set.__json__() if parent_identifier_set else None
            )
            serialized_title_update_identifiers = (
                title_update_identifier_set.__json__()
                if title_update_identifier_set
                else title_update_identifiers  # already a dict or None
            )
            raise task.replace(
                task.s(
                    collection_id=collection_id,
                    import_all=import_all,
                    parent_identifiers=serialized_parent_identifiers,
                    title_update_identifiers=serialized_title_update_identifiers,
                    return_identifiers=return_identifiers,
                    page=result.next_page.url,
                    total_items=result.total_items,
                    modified_since=modified_since,
                    start_time=start_time,
                    lock_value=lock_value,
                )
            )
        else:
            return identifier_set


@shared_task(
    queue=QueueNames.default,
    bind=True,
    max_retries=4,
    autoretry_for=(BadResponseException, RequestTimedOut),
    throws=(RemoteIntegrationException,),
    retry_backoff=60,
)
def import_collection_group(
    task: Task,
    collection_id: int,
    *,
    import_all: bool = False,
    modified_since: datetime.datetime | None = None,
    start_time: datetime.datetime | None = None,
) -> dict[str, Any] | ImportSkippedPayload:
    """Import an Overdrive collection and all its child (Advantage) collections.

    Orchestrates the two-phase import for the parent collection:

    1. **Phase 1** (:func:`import_title_updates`): iterate the product list sorted by
       ``lastTitleUpdateTime`` in reverse order, updating metadata for changed titles.
    2. **Phase 2** (:func:`import_collection` via :func:`import_availability_bridge`):
       iterate by ``lastUpdateTime`` in reverse order, updating circulation data and
       metadata for any titles not covered in phase 1.
    3. Child (Advantage) collections are imported in parallel after phase 2 completes,
       with the parent's identifier set passed to optimise their metadata fetches.

    :param collection_id: The ID of the parent collection to import.
    :param import_all: When ``True`` import all titles regardless of change status.
    :param modified_since: Lower bound for the time filters in both phases.
    :param start_time: When this import began.
    :return: ``{"chain_id": "..."}`` or an :class:`ImportSkippedPayload`.
    """
    redis = task.services.redis().client()
    # Defense-in-depth: skip chain creation if a workflow is already running.
    if import_workflow_lock(redis, collection_id, str(uuid4())).locked():
        task.log.info(
            f"OverDrive import skipped for collection {collection_id}: "
            "another import is already in progress (skipping at group level)."
        )
        return _import_skipped_payload()

    result = chain(
        import_title_updates.s(
            collection_id=collection_id,
            import_all=import_all,
            modified_since=modified_since,
            start_time=start_time,
        ),
        import_availability_bridge.s(
            collection_id=collection_id,
            import_all=import_all,
            modified_since=modified_since,
            start_time=start_time,
        ),
        import_result_router.s(
            collection_id=collection_id,
            import_all=import_all,
            modified_since=modified_since,
        ),
    )()
    return {"chain_id": result.id}


def _is_import_skipped(
    result: IdentifierSet | dict[str, Any] | None,
) -> TypeGuard[dict[str, Any]]:
    """Type guard: True when result is the skip payload."""
    return isinstance(result, dict) and result.get(IMPORT_SKIPPED) is True


def _import_skipped_payload() -> ImportSkippedPayload:
    """Build the skip payload for return values."""
    return {"import_skipped": True}


@shared_task(queue=QueueNames.default, bind=True)
def import_result_router(
    task: Task,
    import_result: IdentifierSet | dict[str, Any] | None,
    collection_id: int,
    import_all: bool,
    modified_since: datetime.datetime | None,
) -> ImportRouterResult:
    """Route import result to child imports or short-circuit when skipped.

    This task receives the result of import_collection and either invokes the
    child-import chord (when the import ran) or returns early (when the import
    was skipped due to another import already in progress).

    :param import_result: Result from import_collection. Either an IdentifierSet
        (or its serialized form when passed through Celery), ImportSkippedPayload
        when skipped, or None when the import returned no identifier set.
    :param collection_id: The parent collection ID.
    :param import_all: Whether to import all titles in children.
    :param modified_since: Only import titles modified after this datetime.
    :return: {"chord_id": "..."} when chord is invoked, {IMPORT_SKIPPED: True}
        when skipped, or {"chord_id": None} when import_result is None.
    """
    if _is_import_skipped(import_result):
        task.log.info(
            f"OverDrive import skipped for collection {collection_id}: "
            "skipping child imports (another import already in progress)."
        )
        skip_result: ImportRouterResult = {"import_skipped": True}
        return skip_result

    if import_result is None:
        task.log.warning(
            f"OverDrive import for collection {collection_id}: no identifier set "
            "returned; skipping child imports."
        )
        return {"chord_id": None}

    identifier_set_info = (
        import_result.__json__()
        if isinstance(import_result, IdentifierSet)
        else import_result
    )
    async_res = import_children_and_cleanup_chord.apply_async(
        args=[identifier_set_info, collection_id, import_all, modified_since],
    )
    return {"chord_id": async_res.id}


def rehydrate_identifier_set(
    task: Task, identifier_set_info: dict[str, Any]
) -> IdentifierSet:
    """Reconstruct an IdentifierSet from its serialized representation.

    This helper function takes a dictionary containing identifier set metadata
    (specifically the Redis key) and recreates the IdentifierSet object that
    can be used to access the data in Redis.

    :param task: The Celery task instance (provides access to Redis client)
    :param identifier_set_info: Dictionary containing the identifier set's key
                                Format: {"key": ["key", "parts"]}
    :return: Reconstructed IdentifierSet connected to Redis
    """
    return IdentifierSet(task.services.redis().client(), identifier_set_info["key"])


@shared_task(
    queue=QueueNames.default,
    bind=True,
    max_retries=4,
    autoretry_for=(BadResponseException, RequestTimedOut),
    throws=(RemoteIntegrationException,),
    retry_backoff=60,
)
def import_children_and_cleanup_chord(
    task: Task,
    identifier_set_info: dict[str, Any],
    collection_id: int,
    import_all: bool,
    modified_since: datetime.datetime,
) -> dict[str, Any]:
    """Import child (Advantage) collections and clean up the parent identifier set.

    This task is called as the callback/link after a parent collection import completes.
    It receives the parent collection's identifier set and uses a Celery chord to:

    1. Import all child Overdrive Advantage collections in parallel, passing the
       parent's identifier set to optimize metadata fetching (children skip books
       already imported by the parent)
    2. After all child imports complete, remove the shared identifier set from Redis

    The chord pattern ensures the cleanup (step 2) only runs after all child imports
    have finished, preventing premature deletion of the shared identifier set.

    :param identifier_set_info: Serialized parent identifier set info from the parent import.
                                Format: {"key": ["redis", "key", "parts"]}
    :param collection_id: The ID of the parent collection whose children to import
    :param import_all: If True, import all titles in children regardless of change status.
                      If False, only import changed titles.
    :param modified_since: Only process titles modified after this datetime in child collections
    :return: Dictionary containing the chord ID for tracking: {"chord_id": "..."}

    .. note::
       If the parent collection has no children, the chord will still be created
       but with an empty group, and cleanup will proceed normally.
    """
    with task.session() as session:
        collection = load_from_id(session, Collection, collection_id)
        identifier_set = rehydrate_identifier_set(task, identifier_set_info)
        header = group(
            [
                import_collection.si(
                    collection_id=c.id,
                    page=None,
                    import_all=import_all,
                    modified_since=modified_since,
                    parent_identifiers=identifier_set,
                )
                for c in collection.children
            ]
        )
        async_res = chord(
            header=header,
            body=remove_identifier_set.si(identifier_set_info=identifier_set_info),
        ).apply_async()
        return {"chord_id": async_res.id}


@shared_task(
    queue=QueueNames.default,
    bind=True,
    max_retries=4,
    autoretry_for=(BadResponseException, RequestTimedOut),
    throws=(RemoteIntegrationException,),
    retry_backoff=60,
)
def remove_identifier_set(task: Task, identifier_set_info: dict[str, Any]) -> None:
    """Clean up a temporary identifier set from Redis after import completes.

    This task is used as the callback body of the chord in import_children_and_cleanup_chord.
    It deletes the temporary Redis set used to share identifiers between
    parent and child collection imports.  If the set doesn't exist, the operation will
    still succeed.


    :param identifier_set_info: Serialized identifier set info.
                                Format: {"key": ["redis", "key", "parts"]}
    """
    identifier_set = rehydrate_identifier_set(task, identifier_set_info)
    if not identifier_set.exists():
        task.log.warning(
            f"Identifier set (key={identifier_set._key}) does not exist in Redis. Skipping cleanup."
        )
    else:
        identifier_set.delete()


@shared_task(queue=QueueNames.default, bind=True)
def import_all_collections(task: Task, *, import_all: bool = False) -> None:
    """
    A shared task that loops through all OverDrive parent collections and kick off an
    import task for each.
    """
    with task.session() as session:
        registry = task.services.integration_registry().license_providers()
        collection_query = Collection.select_by_protocol(
            OverdriveAPI, registry=registry
        ).where(Collection.parent == None)
        create_import_tasks(
            session.scalars(collection_query).all(),
            import_collection_group.s(
                import_all=import_all,
            ),
            task.log,
        )
