"""Tests for the OverdriveImporter class."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock

import pytest

from palace.util.datetime_helpers import datetime_utc
from palace.util.exceptions import PalaceValueError

from palace.manager.data_layer.bibliographic import BibliographicData
from palace.manager.data_layer.circulation import CirculationData
from palace.manager.data_layer.identifier import IdentifierData
from palace.manager.data_layer.policy.replacement import ReplacementPolicy
from palace.manager.integration.license.overdrive.api import (
    BookInfoEndpoint,
    OverdriveAPI,
)
from palace.manager.integration.license.overdrive.importer import (
    FeedImportResult,
    OverdriveImporter,
)
from palace.manager.integration.license.overdrive.representation import (
    OverdriveRepresentationExtractor,
)
from palace.manager.service.redis.models.set import IdentifierSet
from palace.manager.sqlalchemy.model.coverage import Timestamp
from palace.manager.sqlalchemy.model.identifier import Identifier
from tests.fixtures.database import DatabaseTransactionFixture
from tests.fixtures.files import OverdriveFilesFixture
from tests.fixtures.overdrive import OverdriveAPIFixture
from tests.fixtures.redis import RedisFixture
from tests.fixtures.services import ServicesFixture


class TestOverdriveImporter:
    """Tests for the OverdriveImporter class."""

    def test_init_success(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Test successful initialization of OverdriveImporter."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()

        importer = OverdriveImporter(
            db=db.session, collection=collection, registry=registry
        )

        assert importer._db == db.session
        assert importer._collection == collection
        assert importer._identifier_set is None
        assert importer._parent_identifiers is None
        assert isinstance(importer._api, OverdriveAPI)
        assert isinstance(importer._extractor, OverdriveRepresentationExtractor)

    def test_init_with_api_provided(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Test initialization when API instance is provided."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        mock_api = Mock(spec=OverdriveAPI)

        importer = OverdriveImporter(
            db=db.session, collection=collection, registry=registry, api=mock_api
        )

        assert importer._api == mock_api

    def test_init_with_identifier_set(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Test initialization with identifier_set provided."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        mock_identifier_set = Mock(spec=IdentifierSet)

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            identifier_set=mock_identifier_set,
        )

        assert importer._identifier_set == mock_identifier_set

    def test_init_with_parent_identifier_set(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Test initialization with parent_identifier_set."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()

        id1 = IdentifierData(type=Identifier.OVERDRIVE_ID, identifier="id1")
        id2 = IdentifierData(type=Identifier.OVERDRIVE_ID, identifier="id2")
        mock_parent_set = Mock(spec=IdentifierSet)
        mock_parent_set.get.return_value = {id1, id2}

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            parent_identifier_set=mock_parent_set,
        )

        assert importer._parent_identifiers == {id1, id2}

    def test_init_with_title_update_identifier_set(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Test initialization with title_update_identifier_set."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()

        id1 = IdentifierData(type=Identifier.OVERDRIVE_ID, identifier="title-id-1")
        mock_title_set = Mock(spec=IdentifierSet)
        mock_title_set.get.return_value = {id1}

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            title_update_identifier_set=mock_title_set,
        )

        assert importer._title_update_identifiers == {id1}

    def test_init_invalid_collection_protocol(
        self,
        db: DatabaseTransactionFixture,
        services_fixture: ServicesFixture,
    ):
        """Test that initialization fails with invalid collection protocol."""
        collection = db.collection(protocol="Not Overdrive")
        registry = services_fixture.services.integration_registry.license_providers()

        with pytest.raises(PalaceValueError) as exc:
            OverdriveImporter(db=db.session, collection=collection, registry=registry)

        assert "is not an OverDrive collection" in str(exc.value)

    def test_get_timestamp(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Test get_timestamp creates or retrieves a timestamp."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()

        importer = OverdriveImporter(
            db=db.session, collection=collection, registry=registry
        )

        timestamp1 = importer.get_timestamp()
        assert isinstance(timestamp1, Timestamp)
        assert timestamp1.service == "OverDrive Import"
        assert timestamp1.service_type == Timestamp.TASK_TYPE
        assert timestamp1.collection == collection

        timestamp2 = importer.get_timestamp()
        assert timestamp1.id == timestamp2.id

    def test_process_book_skips_metadata_update_if_identifier_in_parent_identifiers(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        overdrive_files_fixture: OverdriveFilesFixture,
        services_fixture: ServicesFixture,
        redis_fixture: RedisFixture,
    ):
        """Ensure _process_book skips metadata update if identifier is in the parent identifier set."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        book_list_data = json.loads(
            overdrive_files_fixture.sample_data("overdrive_book_list.json")
        )
        sample_book = book_list_data["products"][0]
        parent_identifier = IdentifierData(
            type=Identifier.OVERDRIVE_ID, identifier=sample_book["id"]
        )
        redis_client = redis_fixture.client
        parent_identifier_set = IdentifierSet(
            redis_client=redis_client, key="test_parent_set_key"
        )
        parent_identifier_set.add(parent_identifier)

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=overdrive_api_fixture.api,
            parent_identifier_set=parent_identifier_set,
        )

        metadata_lookup_mock = Mock(
            side_effect=AssertionError("Metadata lookup should not be called")
        )
        importer._api.metadata_lookup = metadata_lookup_mock

        book = sample_book.copy()
        book.pop("metadata", None)
        book["availabilityV2"] = json.loads(
            overdrive_files_fixture.sample_data(
                "overdrive_availability_information.json"
            )
        )

        apply_bibliographic = Mock()
        apply_circulation = Mock()

        identifier, changed = importer._process_book(
            book=book,
            fetch_metadata=False,
            policy=ReplacementPolicy(),
            apply_bibliographic=apply_bibliographic,
            apply_circulation=apply_circulation,
        )

        assert isinstance(identifier, Identifier)
        assert isinstance(changed, bool)
        metadata_lookup_mock.assert_not_called()
        apply_bibliographic.assert_not_called()
        assert apply_circulation.called

    def test_process_book_skips_metadata_if_identifier_in_title_update_identifiers(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        overdrive_files_fixture: OverdriveFilesFixture,
        services_fixture: ServicesFixture,
        redis_fixture: RedisFixture,
    ):
        """_process_book skips bibliographic apply if identifier is in the title-update set."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        book_list_data = json.loads(
            overdrive_files_fixture.sample_data("overdrive_book_list.json")
        )
        sample_book = book_list_data["products"][0]

        title_update_id = IdentifierData(
            type=Identifier.OVERDRIVE_ID, identifier=sample_book["id"]
        )
        title_update_set = IdentifierSet(redis_fixture.client, "test_title_update_key")
        title_update_set.add(title_update_id)

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=overdrive_api_fixture.api,
            title_update_identifier_set=title_update_set,
        )

        book = sample_book.copy()
        book["metadata"] = json.loads(
            overdrive_files_fixture.sample_data("overdrive_metadata.json")
        )
        book["availabilityV2"] = json.loads(
            overdrive_files_fixture.sample_data(
                "overdrive_availability_information.json"
            )
        )

        apply_bibliographic = Mock()
        apply_circulation = Mock()

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = True
        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )

        mock_circulation = Mock(spec=CirculationData)
        mock_circulation.needs_apply.return_value = True
        importer._extractor.book_info_to_circulation = Mock(
            return_value=mock_circulation
        )

        identifier, changed = importer._process_book(
            book=book,
            fetch_metadata=True,
            policy=ReplacementPolicy(),
            apply_bibliographic=apply_bibliographic,
            apply_circulation=apply_circulation,
        )

        # Bibliographic apply skipped because identifier is in title_update_identifiers
        apply_bibliographic.assert_not_called()
        # Circulation apply still called
        apply_circulation.assert_called_once()
        assert changed is True

    def test_import_collection_basic(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        overdrive_files_fixture: OverdriveFilesFixture,
        services_fixture: ServicesFixture,
    ):
        """Test import_collection with basic book data (single page)."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        api = overdrive_api_fixture.api

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=api,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()
        modified_since = datetime_utc(2023, 1, 1)

        book_list_data = json.loads(
            overdrive_files_fixture.sample_data("overdrive_book_list.json")
        )
        mock_book_data = [book_list_data["products"][0]]
        mock_book_data[0]["metadata"] = json.loads(
            overdrive_files_fixture.sample_data("overdrive_metadata.json")
        )
        mock_book_data[0]["availabilityV2"] = json.loads(
            overdrive_files_fixture.sample_data(
                "overdrive_availability_information.json"
            )
        )

        # total_items=1 → single page, processes directly
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 1))

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = True
        mock_circulation = Mock(spec=CirculationData)
        mock_circulation.needs_apply.return_value = True

        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )
        importer._extractor.book_info_to_circulation = Mock(
            return_value=mock_circulation
        )

        result = importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
            modified_since=modified_since,
        )

        assert isinstance(result, FeedImportResult)
        assert result.processed_count == 1
        # Single page at offset=0: no previous page
        assert result.next_page is None

        assert mock_apply_bib.call_count == 1
        assert mock_apply_circ.call_count == 1

        identifier, _ = Identifier.for_foreign_id(
            db.session,
            foreign_id="overdrive-id-1",
            foreign_identifier_type=Identifier.OVERDRIVE_ID,
        )
        assert identifier is not None

    def test_import_collection_with_endpoint_provided(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Test import_collection when a specific endpoint (subsequent page) is provided."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        api = overdrive_api_fixture.api

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=api,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()
        custom_endpoint = BookInfoEndpoint(url="http://custom.endpoint")

        api.fetch_book_info_list = AsyncMock(return_value=([], None, 0))

        result = importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
            endpoint=custom_endpoint,
        )

        assert result.current_page == custom_endpoint
        assert result.processed_count == 0

    def test_import_collection_with_identifier_set(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Test import_collection adds identifiers to identifier_set."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        mock_identifier_set = Mock(spec=IdentifierSet)
        api = overdrive_api_fixture.api
        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            identifier_set=mock_identifier_set,
            api=api,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()
        modified_since = datetime_utc(2023, 1, 1)

        mock_book_data = [
            {
                "id": "overdrive-id-1",
                "metadata": {"title": "Test Book"},
                "availabilityV2": {"copiesOwned": 1},
            }
        ]
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 1))

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = True
        mock_circulation = Mock(spec=CirculationData)
        mock_circulation.needs_apply.return_value = True

        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )
        importer._extractor.book_info_to_circulation = Mock(
            return_value=mock_circulation
        )

        importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
            modified_since=modified_since,
        )

        mock_identifier_set.add.assert_called_once()

    def test_import_collection_skips_unchanged_metadata(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Test import_collection skips bibliographic apply when metadata hash is unchanged."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        api = overdrive_api_fixture.api
        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=api,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()
        modified_since = datetime_utc(2023, 1, 1)

        mock_book_data = [
            {
                "id": "overdrive-id-1",
                "metadata": {"title": "Unchanged Book"},
                "availabilityV2": {"copiesOwned": 1},
            }
        ]
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 1))

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = False
        mock_circulation = Mock(spec=CirculationData)
        mock_circulation.needs_apply.return_value = True

        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )
        importer._extractor.book_info_to_circulation = Mock(
            return_value=mock_circulation
        )

        importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
            modified_since=modified_since,
        )

        assert mock_apply_bib.call_count == 0
        assert mock_apply_circ.call_count == 1

    def test_import_collection_stops_on_unchanged_circulation(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Hash-based early exit: stops on the first book whose circulation is unchanged."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        api = overdrive_api_fixture.api

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=api,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()

        mock_book_data = [
            {
                "id": "overdrive-id-1",
                "metadata": {"title": "Book 1"},
                "availabilityV2": {"copiesOwned": 1},
            },
            {
                "id": "overdrive-id-2",
                "metadata": {"title": "Book 2"},
                "availabilityV2": {"copiesOwned": 1},
            },
            {
                "id": "overdrive-id-3",
                "metadata": {"title": "Book 3"},
                "availabilityV2": {"copiesOwned": 1},
            },
        ]
        # single page
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 1))

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = False
        mock_circulation = Mock(spec=CirculationData)
        # Circulation unchanged → triggers early exit on the first processed book
        mock_circulation.needs_apply.return_value = False

        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )
        importer._extractor.book_info_to_circulation = Mock(
            return_value=mock_circulation
        )

        result = importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
            modified_since=datetime_utc(2023, 1, 1),
        )

        # Stops after the first (reversed) book
        assert result.processed_count == 1
        assert result.next_page is None

    def test_import_collection_import_all_disables_early_exit(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """When import_all=True, the hash-based early exit is disabled and all books are processed."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        api = overdrive_api_fixture.api

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=api,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()

        mock_book_data = [
            {
                "id": f"overdrive-id-{i}",
                "metadata": {"title": f"Book {i}"},
                "availabilityV2": {"copiesOwned": 1},
            }
            for i in range(3)
        ]
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 1))

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = False
        mock_circulation = Mock(spec=CirculationData)
        mock_circulation.needs_apply.return_value = False  # All unchanged

        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )
        importer._extractor.book_info_to_circulation = Mock(
            return_value=mock_circulation
        )

        result = importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
            import_all=True,  # disables early exit
        )

        # All 3 books processed even though circulation is unchanged
        assert result.processed_count == 3

    def test_import_collection_reverse_pagination_initial_call_jumps_to_last_page(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """On the initial call with multiple pages, import_collection jumps to the last page."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        api = overdrive_api_fixture.api

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=api,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()

        mock_book_data = [
            {"id": f"overdrive-id-{i}", "availabilityV2": {"copiesOwned": 1}}
            for i in range(5)
        ]
        page_size = 5
        # total_items=10 with page_size=5 → last page at offset=5
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 10))

        result = importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
            endpoint=None,
            page_size=page_size,
        )

        # Initial call returns 0 books processed and a next_page pointing to the last page
        assert result.processed_count == 0
        assert result.next_page is not None
        assert "offset=5" in result.next_page.url
        assert result.total_items == 10

    def test_import_collection_reverse_pagination_prev_page(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """On a subsequent page with all books changed, returns the previous page endpoint."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        api = overdrive_api_fixture.api

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=api,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()

        mock_book_data = [
            {
                "id": f"overdrive-id-{i}",
                "metadata": {"title": f"Book {i}"},
                "availabilityV2": {"copiesOwned": 1},
            }
            for i in range(3)
        ]
        # A subsequent page at offset=300
        page_size = 100
        endpoint = BookInfoEndpoint(
            url="http://api.overdrive.com/products?lastUpdateTime=2020-01-01T00%3A00%3A00Z&limit=100&offset=300"
        )
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 500))

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = True
        mock_circulation = Mock(spec=CirculationData)
        mock_circulation.needs_apply.return_value = True

        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )
        importer._extractor.book_info_to_circulation = Mock(
            return_value=mock_circulation
        )

        result = importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
            endpoint=endpoint,
            total_items=500,
        )

        # All books changed → continues to previous page
        assert result.processed_count == 3
        assert result.next_page is not None
        assert "offset=200" in result.next_page.url

    def test_import_collection_handles_missing_metadata(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Test import_collection handles books with missing metadata."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        api = overdrive_api_fixture.api

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=api,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()
        modified_since = datetime_utc(2023, 1, 1)

        mock_book_data = [
            {
                "id": "overdrive-id-1",
                "metadata": None,
                "availabilityV2": {"copiesOwned": 1},
            }
        ]

        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 1))

        mock_circulation = Mock(spec=CirculationData)
        mock_circulation.needs_apply.return_value = True

        importer._extractor.book_info_to_circulation = Mock(
            return_value=mock_circulation
        )

        result = importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
            modified_since=modified_since,
        )

        assert mock_apply_bib.call_count == 0
        assert mock_apply_circ.call_count == 1
        assert result.processed_count == 1

    def test_import_collection_with_parent_identifiers_fetches_metadata_upfront(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Without parent/title-update identifiers, metadata is fetched upfront for the whole page."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        api = overdrive_api_fixture.api

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=api,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()
        modified_since = datetime_utc(2023, 1, 1)

        mock_book_data = [
            {
                "id": "overdrive-id-1",
                "metadata": {"title": "Test Book"},
                "availabilityV2": {"copiesOwned": 1},
            }
        ]

        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 1))
        api.metadata_lookup = Mock(return_value={"title": "New Book"})

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = True
        mock_circulation = Mock(spec=CirculationData)
        mock_circulation.needs_apply.return_value = True

        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )
        importer._extractor.book_info_to_circulation = Mock(
            return_value=mock_circulation
        )

        result = importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
            modified_since=modified_since,
        )

        api.fetch_book_info_list.assert_called_once()
        call_kwargs = api.fetch_book_info_list.call_args.kwargs
        assert call_kwargs["fetch_metadata"] is True
        assert call_kwargs["fetch_availability"] is True

        api.metadata_lookup.assert_not_called()
        assert result.processed_count == 1

    def test_import_collection_with_parent_identifiers_skips_metadata_for_known_identifiers(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """With a parent identifier set, metadata is fetched lazily, skipping books already in parent."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        api = overdrive_api_fixture.api

        mock_parent_id1 = IdentifierData(
            type=Identifier.OVERDRIVE_ID, identifier="overdrive-id-1"
        )
        mock_parent_id2 = IdentifierData(
            type=Identifier.OVERDRIVE_ID, identifier="overdrive-id-2"
        )

        mock_parent_set = Mock(spec=IdentifierSet)
        mock_parent_set.get.return_value = {mock_parent_id1, mock_parent_id2}

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=api,
            parent_identifier_set=mock_parent_set,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()
        modified_since = datetime_utc(2023, 1, 1)

        mock_book_data = [
            {
                "id": "overdrive-id-1",
                "metadata": None,
                "availabilityV2": {"copiesOwned": 1},
            },
            {
                "id": "overdrive-id-3",
                "metadata": None,
                "availabilityV2": {"copiesOwned": 1},
            },
        ]

        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 1))
        api.metadata_lookup = Mock(return_value={"title": "New Book"})

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = True
        mock_circulation = Mock(spec=CirculationData)
        mock_circulation.needs_apply.return_value = True

        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )
        importer._extractor.book_info_to_circulation = Mock(
            return_value=mock_circulation
        )

        result = importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
            modified_since=modified_since,
        )

        api.fetch_book_info_list.assert_called_once()
        call_kwargs = api.fetch_book_info_list.call_args.kwargs
        assert call_kwargs["fetch_metadata"] is False
        assert call_kwargs["fetch_availability"] is True

        assert api.metadata_lookup.call_count == 1

        assert mock_apply_bib.call_count == 1
        assert mock_apply_circ.call_count == 2
        assert result.processed_count == 2

    def test_import_collection_skips_metadata_for_title_update_identifiers(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
        redis_fixture: RedisFixture,
    ):
        """In phase 2, metadata apply is skipped for identifiers already updated in phase 1."""
        collection = overdrive_api_fixture.collection
        registry = services_fixture.services.integration_registry.license_providers()
        api = overdrive_api_fixture.api

        title_update_id = IdentifierData(
            type=Identifier.OVERDRIVE_ID, identifier="overdrive-id-1"
        )
        title_update_set = IdentifierSet(redis_fixture.client, "phase1_test_key")
        title_update_set.add(title_update_id)

        importer = OverdriveImporter(
            db=db.session,
            collection=collection,
            registry=registry,
            api=api,
            title_update_identifier_set=title_update_set,
        )

        mock_apply_bib = Mock()
        mock_apply_circ = Mock()

        mock_book_data = [
            # This book IS in title_update_identifiers → bibliographic apply skipped
            {
                "id": "overdrive-id-1",
                "metadata": {"title": "Book 1"},
                "availabilityV2": {"copiesOwned": 1},
            },
            # This book is NOT in title_update_identifiers → bibliographic apply runs
            {
                "id": "overdrive-id-2",
                "metadata": {"title": "Book 2"},
                "availabilityV2": {"copiesOwned": 1},
            },
        ]
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 1))

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = True
        mock_circulation = Mock(spec=CirculationData)
        mock_circulation.needs_apply.return_value = True

        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )
        importer._extractor.book_info_to_circulation = Mock(
            return_value=mock_circulation
        )

        result = importer.import_collection(
            apply_bibliographic=mock_apply_bib,
            apply_circulation=mock_apply_circ,
        )

        # apply_bib called only for book 2 (not in title_update_identifiers)
        assert mock_apply_bib.call_count == 1
        # Circulation applied for both books
        assert mock_apply_circ.call_count == 2
        assert result.processed_count == 2


class TestImportTitleUpdates:
    """Tests for OverdriveImporter.import_title_updates()."""

    def _make_importer(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
        identifier_set: IdentifierSet | None = None,
    ) -> OverdriveImporter:
        return OverdriveImporter(
            db=db.session,
            collection=overdrive_api_fixture.collection,
            registry=services_fixture.services.integration_registry.license_providers(),
            api=overdrive_api_fixture.api,
            identifier_set=identifier_set,
        )

    def test_import_title_updates_single_page(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Single-page title import: all books processed and bibliographic tasks queued."""
        api = overdrive_api_fixture.api
        importer = self._make_importer(db, overdrive_api_fixture, services_fixture)
        mock_apply_bib = Mock()

        mock_book_data = [
            {"id": "book-1", "metadata": {"title": "Book 1"}},
            {"id": "book-2", "metadata": {"title": "Book 2"}},
        ]
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 2))

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = True
        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )

        result = importer.import_title_updates(
            apply_bibliographic=mock_apply_bib,
            modified_since=datetime_utc(2023, 1, 1),
        )

        assert result.processed_count == 2
        assert result.next_page is None
        assert mock_apply_bib.call_count == 2

    def test_import_title_updates_fetches_metadata_only(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """import_title_updates fetches metadata but NOT availability."""
        api = overdrive_api_fixture.api
        importer = self._make_importer(db, overdrive_api_fixture, services_fixture)

        api.fetch_book_info_list = AsyncMock(return_value=([], None, 0))

        importer.import_title_updates(apply_bibliographic=Mock())

        api.fetch_book_info_list.assert_called_once()
        kwargs = api.fetch_book_info_list.call_args.kwargs
        assert kwargs["fetch_metadata"] is False  # initial probe: no metadata
        assert kwargs["fetch_availability"] is False

    def test_import_title_updates_subsequent_page_fetches_metadata(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Subsequent pages (not the initial probe) fetch metadata=True."""
        api = overdrive_api_fixture.api
        importer = self._make_importer(db, overdrive_api_fixture, services_fixture)

        endpoint = BookInfoEndpoint(
            "http://example.com?lastTitleUpdateTime=2023-01-01T00%3A00%3A00Z&limit=100&offset=100"
        )
        api.fetch_book_info_list = AsyncMock(return_value=([], None, 200))

        importer.import_title_updates(
            apply_bibliographic=Mock(),
            endpoint=endpoint,
            total_items=200,
        )

        kwargs = api.fetch_book_info_list.call_args.kwargs
        assert kwargs["fetch_metadata"] is True
        assert kwargs["fetch_availability"] is False

    def test_import_title_updates_stops_on_unchanged_metadata(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Stops on the first book whose metadata hash is unchanged (early exit)."""
        api = overdrive_api_fixture.api
        importer = self._make_importer(db, overdrive_api_fixture, services_fixture)
        mock_apply_bib = Mock()

        mock_book_data = [
            {"id": "book-1", "metadata": {"title": "Book 1"}},
            {"id": "book-2", "metadata": {"title": "Book 2"}},
            {"id": "book-3", "metadata": {"title": "Book 3"}},
        ]
        # single page; books will be processed in reverse: book-3, book-2, book-1
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 3))

        mock_bibliographic = Mock(spec=BibliographicData)
        # First call (book-3, most recently updated) unchanged → exit immediately
        mock_bibliographic.needs_apply.return_value = False
        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )

        endpoint = BookInfoEndpoint(
            "http://example.com?lastTitleUpdateTime=2020-01-01T00%3A00%3A00Z&limit=100&offset=0"
        )
        result = importer.import_title_updates(
            apply_bibliographic=mock_apply_bib,
            endpoint=endpoint,
            total_items=3,
        )

        assert result.processed_count == 1
        assert result.next_page is None
        mock_apply_bib.assert_not_called()

    def test_import_title_updates_import_all_disables_early_exit(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """When import_all=True, all books are processed even if metadata is unchanged."""
        api = overdrive_api_fixture.api
        importer = self._make_importer(db, overdrive_api_fixture, services_fixture)

        mock_book_data = [
            {"id": f"book-{i}", "metadata": {"title": f"Book {i}"}} for i in range(4)
        ]
        endpoint = BookInfoEndpoint(
            "http://example.com?lastTitleUpdateTime=2020-01-01T00%3A00%3A00Z&limit=100&offset=0"
        )
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 4))

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = False  # All unchanged
        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )

        result = importer.import_title_updates(
            apply_bibliographic=Mock(),
            import_all=True,
            endpoint=endpoint,
            total_items=4,
        )

        assert result.processed_count == 4

    def test_import_title_updates_adds_changed_identifiers_to_set(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
        redis_fixture: RedisFixture,
    ):
        """Changed identifiers are added to the identifier_set for use in phase 2."""
        api = overdrive_api_fixture.api
        identifier_set = IdentifierSet(redis_fixture.client, "test_title_key")
        importer = self._make_importer(
            db, overdrive_api_fixture, services_fixture, identifier_set=identifier_set
        )

        mock_book_data = [
            {"id": "changed-book-1", "metadata": {"title": "Book 1"}},
            {"id": "changed-book-2", "metadata": {"title": "Book 2"}},
        ]
        endpoint = BookInfoEndpoint(
            "http://example.com?lastTitleUpdateTime=2020-01-01T00%3A00%3A00Z&limit=100&offset=0"
        )
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, 2))

        mock_bibliographic = Mock(spec=BibliographicData)
        mock_bibliographic.needs_apply.return_value = True
        importer._extractor.book_info_to_bibliographic = Mock(
            return_value=mock_bibliographic
        )

        importer.import_title_updates(
            apply_bibliographic=Mock(),
            endpoint=endpoint,
            total_items=2,
        )

        ids_in_set = identifier_set.get()
        assert len(ids_in_set) == 2

    def test_import_title_updates_initial_call_jumps_to_last_page(
        self,
        db: DatabaseTransactionFixture,
        overdrive_api_fixture: OverdriveAPIFixture,
        services_fixture: ServicesFixture,
    ):
        """Initial call with multiple pages: returns a FeedImportResult pointing to the last page."""
        api = overdrive_api_fixture.api
        importer = self._make_importer(db, overdrive_api_fixture, services_fixture)

        page_size = 5
        total = 20
        mock_book_data = [{"id": f"book-{i}"} for i in range(page_size)]
        api.fetch_book_info_list = AsyncMock(return_value=(mock_book_data, None, total))

        result = importer.import_title_updates(
            apply_bibliographic=Mock(),
            page_size=page_size,
        )

        assert result.processed_count == 0
        assert result.next_page is not None
        # last page at offset = ((20-1)//5)*5 = 15
        assert "offset=15" in result.next_page.url
        assert result.total_items == total


class TestFeedImportResult:
    """Tests for the FeedImportResult dataclass."""

    def test_feed_import_result_basic(self):
        """Test FeedImportResult creation with basic data."""
        current_page = BookInfoEndpoint(url="http://current.page")
        result = FeedImportResult(current_page=current_page)

        assert result.current_page == current_page
        assert result.next_page is None
        assert result.processed_count == 0
        assert result.total_items is None

    def test_feed_import_result_with_all_fields(self):
        """Test FeedImportResult with all fields populated."""
        current_page = BookInfoEndpoint(url="http://current.page")
        next_page = BookInfoEndpoint(url="http://next.page")

        result = FeedImportResult(
            current_page=current_page,
            next_page=next_page,
            processed_count=42,
            total_items=500,
        )

        assert result.current_page == current_page
        assert result.next_page == next_page
        assert result.processed_count == 42
        assert result.total_items == 500

    def test_feed_import_result_frozen(self):
        """Test that FeedImportResult is immutable (frozen)."""
        current_page = BookInfoEndpoint(url="http://current.page")
        result = FeedImportResult(current_page=current_page)

        with pytest.raises(AttributeError):
            result.processed_count = 100  # type: ignore[misc]
