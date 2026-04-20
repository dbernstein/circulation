import asyncio
import datetime
from collections.abc import Set
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from palace.util.exceptions import PalaceValueError
from palace.util.log import LoggerMixin

from palace.manager.celery.tasks.apply import (
    ApplyBibliographicCallable,
    ApplyCirculationCallable,
)
from palace.manager.data_layer.identifier import IdentifierData
from palace.manager.data_layer.policy.replacement import ReplacementPolicy
from palace.manager.integration.license.overdrive.api import (
    BookInfoEndpoint,
    OverdriveAPI,
)
from palace.manager.integration.license.overdrive.model import Availability
from palace.manager.integration.license.overdrive.representation import (
    OverdriveRepresentationExtractor,
)
from palace.manager.service.integration_registry.license_providers import (
    LicenseProvidersRegistry,
)
from palace.manager.service.redis.models.set import IdentifierSet
from palace.manager.sqlalchemy.model.collection import Collection
from palace.manager.sqlalchemy.model.coverage import Timestamp
from palace.manager.sqlalchemy.model.identifier import Identifier
from palace.manager.sqlalchemy.util import get_one_or_create


@dataclass(frozen=True)
class FeedImportResult:
    current_page: BookInfoEndpoint
    next_page: BookInfoEndpoint | None = None
    processed_count: int = 0
    # Propagated across task.replace() calls for reverse-pagination state.
    total_items: int | None = None


class OverdriveImporter(LoggerMixin):
    DEFAULT_PAGE_SIZE = 100

    def __init__(
        self,
        db: Session,
        collection: Collection,
        registry: LicenseProvidersRegistry,
        identifier_set: IdentifierSet | None = None,
        parent_identifier_set: IdentifierSet | None = None,
        title_update_identifier_set: IdentifierSet | None = None,
        api: OverdriveAPI | None = None,
    ) -> None:
        """Constructor for the OverdriveImporter class.

        :param db: The database session.
        :param collection: The collection to import.
        :param registry: The license providers registry.
        :param identifier_set: The identifier set to use for the import.
        :param parent_identifier_set: The parent identifier set to use for the import.
        :param title_update_identifier_set: Identifiers whose metadata was updated in
            phase 1 (``import_title_updates``). Phase 2 skips metadata fetches for
            these identifiers to avoid redundant work.
        :param api: The OverdriveAPI instance to use for the import.
        """
        self._db = db
        self._collection = collection
        self._identifier_set = identifier_set

        self._parent_identifiers: Set[IdentifierData] | None = None
        if parent_identifier_set is not None:
            # create an in-memory set from the redis set  to optimize existence checks for individual identifiers.
            # I don't believe we need to worry about memory here: few redis identifier sets will likely exceed 200K
            # items which should be easily manageable given an identifier is 36 characters (36*200K = 7.2 MB). Most OD
            # collections are much  smaller in the 20-70K range.

            self._parent_identifiers = parent_identifier_set.get()

        self._title_update_identifiers: Set[IdentifierData] | None = None
        if title_update_identifier_set is not None:
            self._title_update_identifiers = title_update_identifier_set.get()

        if not registry.equivalent(collection.protocol, OverdriveAPI):
            raise PalaceValueError(
                f"Collection {collection.name} [id={collection.id} protocol={collection.protocol}] "
                f"is not an OverDrive collection."
            )

        self._api = (
            OverdriveAPI(_db=self._db, collection=self._collection)
            if api is None
            else api
        )

        self._extractor = OverdriveRepresentationExtractor(self._api)

    def get_timestamp(self) -> Timestamp:
        timestamp, _ = get_one_or_create(
            self._db,
            Timestamp,
            service="OverDrive Import",
            service_type=Timestamp.TASK_TYPE,
            collection=self._collection,
        )
        return timestamp

    def _default_replacement_policy(self) -> ReplacementPolicy:
        return ReplacementPolicy(
            identifiers=False,
            subjects=True,
            contributions=True,
            formats=True,
            links=True,
        )

    def _process_book_metadata(
        self,
        book: dict[str, Any],
        policy: ReplacementPolicy,
        apply_bibliographic: ApplyBibliographicCallable,
    ) -> tuple[Identifier, bool]:
        """Process metadata for a single book.

        :param book: Book data dictionary; must contain a ``metadata`` key.
        :param policy: Replacement policy for bibliographic updates.
        :param apply_bibliographic: Callback to apply bibliographic updates.
        :return: ``(identifier, changed)`` where *changed* is ``True`` when the
            bibliographic data differed from what is already stored.
        """
        book = book.copy()

        identifier, _ = Identifier.for_foreign_id(
            self._db,
            foreign_id=book.get("id"),
            foreign_identifier_type=Identifier.OVERDRIVE_ID,
        )
        assert identifier

        if not book.get("metadata"):
            return identifier, False

        bibliographic = self._extractor.book_info_to_bibliographic(book)
        assert bibliographic

        if bibliographic.needs_apply(self._db):
            apply_bibliographic(
                bibliographic,
                collection_id=self._collection.id,
                replace=policy,
            )
            return identifier, True

        return identifier, False

    def _process_book(
        self,
        book: dict[str, Any],
        fetch_metadata: bool,
        policy: ReplacementPolicy,
        apply_bibliographic: ApplyBibliographicCallable,
        apply_circulation: ApplyCirculationCallable,
    ) -> tuple[Identifier, bool]:
        """Process a single book and return (identifier, changed).

        :param book: Book data dictionary from OverDrive API
        :param fetch_metadata: Whether metadata was already fetched
        :param policy: Replacement policy for bibliographic updates
        :param apply_bibliographic: Callback to apply bibliographic updates
        :param apply_circulation: Callback to apply circulation updates
        :return: Tuple of (identifier, changed) where changed is True if any data changed
        """

        # we may need to manipulate values in the book dictionary.  Therefore we make a copy for local changes
        # to avoid unnecessary side effects.
        book = book.copy()

        identifier, _ = Identifier.for_foreign_id(
            self._db,
            foreign_id=book.get("id"),
            foreign_identifier_type=Identifier.OVERDRIVE_ID,
        )

        # the identifier should never be null, because by default autocreate = True in for_foreign_id().
        # however mypy complains throughout without changing type hints or adding an asssertion.
        # An assertion is least verbose solution.
        assert identifier

        changed: bool = False

        identifier_data = IdentifierData.from_identifier(identifier)

        # Skip metadata if it was already processed in the title-update phase.
        already_in_title_update = (
            self._title_update_identifiers is not None
            and identifier_data in self._title_update_identifiers
        )

        # We only need to look up metadata if we didn't already fetch it and it was not in the parent identifier
        # set.  Why? Because the existence of the parent identifier set implies that the parent collection
        # has already been imported which would have included all the metadata.
        if (
            not fetch_metadata
            and not already_in_title_update
            and (
                not self._parent_identifiers
                or identifier_data not in self._parent_identifiers
            )
        ):
            book["metadata"] = self._api.metadata_lookup(identifier=identifier)

        # we need to check that there is metadata because it is possible that we attempted to fetch it, but we
        # didn't get anything back from overdrive (ie from the book list fetch above) or we did not attempt to
        # fetch it because it was already processed by the parent collection.
        if book.get("metadata") and not already_in_title_update:
            bibliographic = self._extractor.book_info_to_bibliographic(book)
            # The bibliographic should never be null here because there is a non-null entry for metadata in the
            # book dictionary.  Mypy complains without an assertion or type hints.
            assert bibliographic

            if bibliographic.needs_apply(self._db):
                changed = True
                apply_bibliographic(
                    bibliographic,
                    collection_id=self._collection.id,
                    replace=policy,
                )

        # availability needs to be checked/updated in all but a few instances so it is
        # probably not worth the compute time to save ourselves a handful of unnecessary updates.
        availability_data = book.get("availabilityV2", None)
        if not availability_data:
            # This is a rare and probably transient case where the availabilityV2
            # was not retrieved due to a 404 from OD.
            self.log.warning(
                f"No availabilityV2 found for book {identifier}. book={book}.  This state can "
                f"arise when the OD returns a 404 for the availability url."
            )
        else:
            availability = Availability.model_validate(availability_data)
            circulation = self._extractor.book_info_to_circulation(availability)
            # The circulation should never be null here because there is a non-null entry for availabilityV2 in the
            # book dictionary.  Mypy complains without an assertion or type hints.
            assert circulation

            if circulation.needs_apply(self._db, self._collection):
                changed = True
                apply_circulation(circulation, collection_id=self._collection.id)

        return identifier, changed

    def import_title_updates(
        self,
        *,
        apply_bibliographic: ApplyBibliographicCallable,
        import_all: bool = False,
        modified_since: datetime.datetime | None = None,
        endpoint: BookInfoEndpoint | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        total_items: int | None = None,
    ) -> FeedImportResult:
        """Phase 1 import: process title-metadata changes in reverse chronological order.

        Iterates the product list sorted by ``lastTitleUpdateTime`` starting from the
        **last** page (most recently changed) and works backwards.  For each book the
        bibliographic hash is checked; when an unchanged book is encountered (and
        ``import_all`` is ``False``) iteration stops immediately, since all remaining
        books are older and also unchanged.

        On the very first call (``endpoint`` and ``total_items`` both ``None``) this
        method fetches the first page solely to determine ``totalItems`` and then
        signals the caller to jump to the last page by returning a ``FeedImportResult``
        whose ``next_page`` points to that last page and ``total_items`` carries the
        count.  On subsequent calls it processes the given page in reverse order.

        :param apply_bibliographic: Callback to queue bibliographic apply tasks.
        :param import_all: When ``True`` the hash-based early exit is disabled and every
            record is processed regardless of whether it has changed.
        :param modified_since: Lower bound for the ``lastTitleUpdateTime`` filter.
        :param endpoint: The page to process. ``None`` on the initial call.
        :param page_size: Items per page (capped by the API limit).
        :param total_items: Total items in the result set; supplied on all calls after
            the initial one so the caller can re-use it without an extra API round-trip.
        :return: :class:`FeedImportResult` with ``next_page`` pointing to the next page
            to process (the *previous* page in chronological order), or ``None`` when
            processing is complete.
        """
        policy = self._default_replacement_policy()

        self.log.info(
            f"Starting title-update import for collection {self._collection.name} "
            f"(id={self._collection.id}), modified_since={modified_since}."
        )

        # --- Initial call: discover total_items and jump to the last page ---
        if endpoint is None and total_items is None:
            base_endpoint = self._api.book_info_initial_endpoint(
                start=modified_since,
                page_size=page_size,
                sort_by="lastTitleUpdateTime",
            )
            book_data, _, discovered_total = asyncio.run(
                self._api.fetch_book_info_list(
                    base_endpoint,
                    fetch_metadata=True,
                    fetch_availability=False,
                )
            )
            if discovered_total <= page_size:
                # Single page: process it directly in reverse.
                return self._process_title_updates_page(
                    books=list(reversed(book_data)),
                    current_endpoint=base_endpoint,
                    page_size=page_size,
                    total_items=discovered_total,
                    import_all=import_all,
                    policy=policy,
                    apply_bibliographic=apply_bibliographic,
                )
            last_offset = ((discovered_total - 1) // page_size) * page_size
            last_page_endpoint = self._api.book_info_endpoint_at_offset(
                base_endpoint, last_offset
            )
            self.log.info(
                f"Title-update import: {discovered_total} total items, "
                f"jumping to last page offset={last_offset}."
            )
            return FeedImportResult(
                current_page=base_endpoint,
                next_page=last_page_endpoint,
                processed_count=0,
                total_items=discovered_total,
            )

        # --- Subsequent pages ---
        assert endpoint is not None
        book_data, _, _ = asyncio.run(
            self._api.fetch_book_info_list(
                endpoint,
                fetch_metadata=True,
                fetch_availability=False,
            )
        )
        return self._process_title_updates_page(
            books=list(reversed(book_data)),
            current_endpoint=endpoint,
            page_size=page_size,
            total_items=total_items,
            import_all=import_all,
            policy=policy,
            apply_bibliographic=apply_bibliographic,
        )

    def _process_title_updates_page(
        self,
        *,
        books: list[dict[str, Any]],
        current_endpoint: BookInfoEndpoint,
        page_size: int,
        total_items: int | None,
        import_all: bool,
        policy: ReplacementPolicy,
        apply_bibliographic: ApplyBibliographicCallable,
    ) -> FeedImportResult:
        """Process a single page of books for the title-update phase.

        Books must already be supplied in the desired processing order (reverse
        chronological, i.e. most recently changed first).

        Returns a :class:`FeedImportResult` whose ``next_page`` is the previous page
        endpoint, or ``None`` if processing should stop (early exit or first page
        reached).
        """
        identifiers: list[Identifier] = []

        for book in books:
            identifier, changed = self._process_book_metadata(
                book, policy, apply_bibliographic
            )
            identifiers.append(identifier)

            if not changed and not import_all:
                self.log.info(
                    f"Title-update import: book {identifier} metadata unchanged — stopping."
                )
                if self._identifier_set is not None:
                    self._identifier_set.add(*identifiers)
                return FeedImportResult(
                    current_page=current_endpoint,
                    next_page=None,
                    processed_count=len(identifiers),
                    total_items=total_items,
                )

        if self._identifier_set is not None:
            self._identifier_set.add(*identifiers)

        prev_endpoint = self._api.book_info_prev_page_endpoint(
            current_endpoint, page_size
        )
        self.log.info(
            f"Title-update import: processed {len(identifiers)} books on page "
            f"{current_endpoint.url}. Next page: {prev_endpoint}."
        )
        return FeedImportResult(
            current_page=current_endpoint,
            next_page=prev_endpoint,
            processed_count=len(identifiers),
            total_items=total_items,
        )

    def import_collection(
        self,
        *,
        apply_bibliographic: ApplyBibliographicCallable,
        apply_circulation: ApplyCirculationCallable,
        import_all: bool = False,
        modified_since: datetime.datetime | None = None,
        endpoint: BookInfoEndpoint | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        total_items: int | None = None,
    ) -> FeedImportResult:
        """Phase 2 import: process availability (circulation) changes in reverse chronological order.

        Iterates the product list sorted by ``lastUpdateTime`` starting from the **last**
        page and works backwards.  For each book:

        * Availability (circulation) data is always fetched and the hash is compared.
        * Metadata is fetched only when the identifier was **not** already updated in
          phase 1 (i.e. not in ``self._title_update_identifiers``) and not already
          present via the parent collection.
        * When an unchanged circulation record is encountered (and ``import_all`` is
          ``False``) iteration stops immediately.

        The initial-call / subsequent-page semantics are identical to
        :meth:`import_title_updates`.

        :param apply_bibliographic: Callback to queue bibliographic apply tasks.
        :param apply_circulation: Callback to queue circulation apply tasks.
        :param import_all: Disable hash-based early exit when ``True``.
        :param modified_since: Lower bound for the ``lastUpdateTime`` filter.
        :param endpoint: The page to process. ``None`` on the initial call.
        :param page_size: Items per page (capped by the API limit).
        :param total_items: Total items in the result set.
        :return: :class:`FeedImportResult` describing the page processed and the next
            page to visit (or ``None`` when done).
        """
        policy = self._default_replacement_policy()

        self.log.info(
            f"Starting availability import for collection {self._collection.name} "
            f"(id={self._collection.id}), modified_since={modified_since}."
        )

        # Fetch metadata upfront only for main (non-advantage) collections when
        # we don't already have it from the title-update phase.
        fetch_metadata = (
            self._parent_identifiers is None and self._title_update_identifiers is None
        )

        # --- Initial call: discover total_items and jump to the last page ---
        if endpoint is None and total_items is None:
            base_endpoint = self._api.book_info_initial_endpoint(
                start=modified_since,
                page_size=page_size,
                sort_by="lastUpdateTime",
            )
            book_data, _, discovered_total = asyncio.run(
                self._api.fetch_book_info_list(
                    base_endpoint,
                    fetch_metadata=fetch_metadata,
                    fetch_availability=True,
                )
            )
            if discovered_total <= page_size:
                return self._process_availability_page(
                    books=list(reversed(book_data)),
                    fetch_metadata=fetch_metadata,
                    current_endpoint=base_endpoint,
                    page_size=page_size,
                    total_items=discovered_total,
                    import_all=import_all,
                    policy=policy,
                    apply_bibliographic=apply_bibliographic,
                    apply_circulation=apply_circulation,
                )
            last_offset = ((discovered_total - 1) // page_size) * page_size
            last_page_endpoint = self._api.book_info_endpoint_at_offset(
                base_endpoint, last_offset
            )
            self.log.info(
                f"Availability import: {discovered_total} total items, "
                f"jumping to last page offset={last_offset}."
            )
            return FeedImportResult(
                current_page=base_endpoint,
                next_page=last_page_endpoint,
                processed_count=0,
                total_items=discovered_total,
            )

        # --- Subsequent pages ---
        assert endpoint is not None
        book_data, _, _ = asyncio.run(
            self._api.fetch_book_info_list(
                endpoint,
                fetch_metadata=fetch_metadata,
                fetch_availability=True,
            )
        )
        return self._process_availability_page(
            books=list(reversed(book_data)),
            fetch_metadata=fetch_metadata,
            current_endpoint=endpoint,
            page_size=page_size,
            total_items=total_items,
            import_all=import_all,
            policy=policy,
            apply_bibliographic=apply_bibliographic,
            apply_circulation=apply_circulation,
        )

    def _process_availability_page(
        self,
        *,
        books: list[dict[str, Any]],
        fetch_metadata: bool,
        current_endpoint: BookInfoEndpoint,
        page_size: int,
        total_items: int | None,
        import_all: bool,
        policy: ReplacementPolicy,
        apply_bibliographic: ApplyBibliographicCallable,
        apply_circulation: ApplyCirculationCallable,
    ) -> FeedImportResult:
        """Process a single page of books for the availability phase.

        Books must already be in reverse chronological order (most recently
        changed first).  Stops early when the first book with an unchanged
        circulation hash is encountered (unless ``import_all`` is ``True``).
        """
        identifiers: list[Identifier] = []
        timestamp = self.get_timestamp()

        for book in books:
            identifier, changed = self._process_book(
                book, fetch_metadata, policy, apply_bibliographic, apply_circulation
            )
            identifiers.append(identifier)

            if not changed and not import_all:
                self.log.info(
                    f"Availability import: book {identifier} circulation unchanged — stopping."
                )
                if self._identifier_set is not None:
                    self._identifier_set.add(*identifiers)
                return FeedImportResult(
                    current_page=current_endpoint,
                    next_page=None,
                    processed_count=len(identifiers),
                    total_items=total_items,
                )

        achievements = [f"Total items queued for import: {len(identifiers)}."]
        if (elapsed_time := timestamp.elapsed_seconds) is not None:
            achievements.append(f"Elapsed time: {elapsed_time:.2f} seconds.")

        if self._identifier_set is not None:
            self._identifier_set.add(*identifiers)

        timestamp.achievements = "\n".join(achievements)

        prev_endpoint = self._api.book_info_prev_page_endpoint(
            current_endpoint, page_size
        )
        self.log.info(
            f"Availability import: processed {len(identifiers)} books on page "
            f"{current_endpoint.url}. Next page: {prev_endpoint}."
        )
        return FeedImportResult(
            current_page=current_endpoint,
            next_page=prev_endpoint,
            processed_count=len(identifiers),
            total_items=total_items,
        )
