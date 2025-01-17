import datetime
import json

from pyld import jsonld

from api.annotations import AnnotationParser, AnnotationWriter
from api.problem_details import *
from core.model import Annotation, create
from core.testing import DatabaseTest
from core.util.datetime_helpers import utc_now

from .test_controller import ControllerTest


class AnnotationTest(DatabaseTest):
    def _patron(self):
        """Create a test patron who has opted in to annotation sync."""
        patron = super(AnnotationTest, self)._patron()
        patron.synchronize_annotations = True
        return patron


class TestAnnotationWriter(AnnotationTest, ControllerTest):
    def test_annotations_for(self):
        patron = self._patron()

        # The patron doesn't have any annotations yet.
        assert [] == AnnotationWriter.annotations_for(patron)

        identifier = self._identifier()
        annotation, ignore = create(
            self._db,
            Annotation,
            patron=patron,
            identifier=identifier,
            motivation=Annotation.IDLING,
        )

        # The patron has one annotation.
        assert [annotation] == AnnotationWriter.annotations_for(patron)
        assert [annotation] == AnnotationWriter.annotations_for(patron, identifier)

        identifier2 = self._identifier()
        annotation2, ignore = create(
            self._db,
            Annotation,
            patron=patron,
            identifier=identifier2,
            motivation=Annotation.IDLING,
        )

        # The patron has two annotations for different identifiers.
        assert set([annotation, annotation2]) == set(
            AnnotationWriter.annotations_for(patron)
        )
        assert [annotation] == AnnotationWriter.annotations_for(patron, identifier)
        assert [annotation2] == AnnotationWriter.annotations_for(patron, identifier2)

    def test_annotation_container_for(self):
        patron = self._patron()

        with self.app.test_request_context("/"):
            container, timestamp = AnnotationWriter.annotation_container_for(patron)

            assert set(
                [AnnotationWriter.JSONLD_CONTEXT, AnnotationWriter.LDP_CONTEXT]
            ) == set(container["@context"])
            assert "annotations" in container["id"]
            assert set(["BasicContainer", "AnnotationCollection"]) == set(
                container["type"]
            )
            assert 0 == container["total"]

            first_page = container["first"]
            assert "AnnotationPage" == first_page["type"]

            # The page doesn't have a context, since it's in the container.
            assert None == first_page.get("@context")

            # The patron doesn't have any annotations yet.
            assert 0 == container["total"]

            # There's no timestamp since the container is empty.
            assert None == timestamp

            # Now, add an annotation.
            identifier = self._identifier()
            annotation, ignore = create(
                self._db,
                Annotation,
                patron=patron,
                identifier=identifier,
                motivation=Annotation.IDLING,
            )
            annotation.timestamp = utc_now()

            container, timestamp = AnnotationWriter.annotation_container_for(patron)

            # The context, type, and id stay the same.
            assert set(
                [AnnotationWriter.JSONLD_CONTEXT, AnnotationWriter.LDP_CONTEXT]
            ) == set(container["@context"])
            assert "annotations" in container["id"]
            assert identifier.identifier not in container["id"]
            assert set(["BasicContainer", "AnnotationCollection"]) == set(
                container["type"]
            )

            # But now there is one item.
            assert 1 == container["total"]

            first_page = container["first"]

            assert 1 == len(first_page["items"])

            # The item doesn't have a context, since it's in the container.
            first_item = first_page["items"][0]
            assert None == first_item.get("@context")

            # The timestamp is the annotation's timestamp.
            assert annotation.timestamp == timestamp

            # If the annotation is deleted, the container will be empty again.
            annotation.active = False

            container, timestamp = AnnotationWriter.annotation_container_for(patron)
            assert 0 == container["total"]
            assert None == timestamp

    def test_annotation_container_for_with_identifier(self):
        patron = self._patron()
        identifier = self._identifier()

        with self.app.test_request_context("/"):
            container, timestamp = AnnotationWriter.annotation_container_for(
                patron, identifier
            )

            assert set(
                [AnnotationWriter.JSONLD_CONTEXT, AnnotationWriter.LDP_CONTEXT]
            ) == set(container["@context"])
            assert "annotations" in container["id"]
            assert identifier.identifier in container["id"]
            assert set(["BasicContainer", "AnnotationCollection"]) == set(
                container["type"]
            )
            assert 0 == container["total"]

            first_page = container["first"]
            assert "AnnotationPage" == first_page["type"]

            # The page doesn't have a context, since it's in the container.
            assert None == first_page.get("@context")

            # The patron doesn't have any annotations yet.
            assert 0 == container["total"]

            # There's no timestamp since the container is empty.
            assert None == timestamp

            # Now, add an annotation for this identifier, and one for a different identifier.
            annotation, ignore = create(
                self._db,
                Annotation,
                patron=patron,
                identifier=identifier,
                motivation=Annotation.IDLING,
            )
            annotation.timestamp = utc_now()

            other_annotation, ignore = create(
                self._db,
                Annotation,
                patron=patron,
                identifier=self._identifier(),
                motivation=Annotation.IDLING,
            )

            container, timestamp = AnnotationWriter.annotation_container_for(
                patron, identifier
            )

            # The context, type, and id stay the same.
            assert set(
                [AnnotationWriter.JSONLD_CONTEXT, AnnotationWriter.LDP_CONTEXT]
            ) == set(container["@context"])
            assert "annotations" in container["id"]
            assert identifier.identifier in container["id"]
            assert set(["BasicContainer", "AnnotationCollection"]) == set(
                container["type"]
            )

            # But now there is one item.
            assert 1 == container["total"]

            first_page = container["first"]

            assert 1 == len(first_page["items"])

            # The item doesn't have a context, since it's in the container.
            first_item = first_page["items"][0]
            assert None == first_item.get("@context")

            # The timestamp is the annotation's timestamp.
            assert annotation.timestamp == timestamp

            # If the annotation is deleted, the container will be empty again.
            annotation.active = False

            container, timestamp = AnnotationWriter.annotation_container_for(
                patron, identifier
            )
            assert 0 == container["total"]
            assert None == timestamp

    def test_annotation_page_for(self):
        patron = self._patron()

        with self.app.test_request_context("/"):
            page = AnnotationWriter.annotation_page_for(patron)

            # The patron doesn't have any annotations, so the page is empty.
            assert AnnotationWriter.JSONLD_CONTEXT == page["@context"]
            assert "annotations" in page["id"]
            assert "AnnotationPage" == page["type"]
            assert 0 == len(page["items"])

            # If we add an annotation, the page will have an item.
            identifier = self._identifier()
            annotation, ignore = create(
                self._db,
                Annotation,
                patron=patron,
                identifier=identifier,
                motivation=Annotation.IDLING,
            )

            page = AnnotationWriter.annotation_page_for(patron)

            assert 1 == len(page["items"])

            # But if the annotation is deleted, the page will be empty again.
            annotation.active = False

            page = AnnotationWriter.annotation_page_for(patron)

            assert 0 == len(page["items"])

    def test_annotation_page_for_with_identifier(self):
        patron = self._patron()
        identifier = self._identifier()

        with self.app.test_request_context("/"):
            page = AnnotationWriter.annotation_page_for(patron, identifier)

            # The patron doesn't have any annotations, so the page is empty.
            assert AnnotationWriter.JSONLD_CONTEXT == page["@context"]
            assert "annotations" in page["id"]
            assert identifier.identifier in page["id"]
            assert "AnnotationPage" == page["type"]
            assert 0 == len(page["items"])

            # If we add an annotation, the page will have an item.
            annotation, ignore = create(
                self._db,
                Annotation,
                patron=patron,
                identifier=identifier,
                motivation=Annotation.IDLING,
            )

            page = AnnotationWriter.annotation_page_for(patron, identifier)
            assert 1 == len(page["items"])

            # If a different identifier has an annotation, the page will still have one item.
            other_annotation, ignore = create(
                self._db,
                Annotation,
                patron=patron,
                identifier=self._identifier(),
                motivation=Annotation.IDLING,
            )

            page = AnnotationWriter.annotation_page_for(patron, identifier)
            assert 1 == len(page["items"])

            # But if the annotation is deleted, the page will be empty again.
            annotation.active = False

            page = AnnotationWriter.annotation_page_for(patron, identifier)
            assert 0 == len(page["items"])

    def test_detail_target(self):
        patron = self._patron()
        identifier = self._identifier()
        target = {
            "http://www.w3.org/ns/oa#hasSource": {"@id": identifier.urn},
            "http://www.w3.org/ns/oa#hasSelector": {
                "@type": "http://www.w3.org/ns/oa#FragmentSelector",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#value": "epubcfi(/6/4[chap01ref]!/4[body01]/10[para05]/3:10)",
            },
        }

        annotation, ignore = create(
            self._db,
            Annotation,
            patron=patron,
            identifier=identifier,
            motivation=Annotation.IDLING,
            target=json.dumps(target),
        )

        with self.app.test_request_context("/"):
            detail = AnnotationWriter.detail(annotation)

            assert "annotations/%i" % annotation.id in detail["id"]
            assert "Annotation" == detail["type"]
            assert Annotation.IDLING == detail["motivation"]
            compacted_target = {
                "source": identifier.urn,
                "selector": {
                    "type": "FragmentSelector",
                    "value": "epubcfi(/6/4[chap01ref]!/4[body01]/10[para05]/3:10)",
                },
            }
            assert compacted_target == detail["target"]

    def test_detail_body(self):
        patron = self._patron()
        identifier = self._identifier()
        body = {
            "@type": "http://www.w3.org/ns/oa#TextualBody",
            "http://www.w3.org/ns/oa#bodyValue": "A good description of the topic that bears further investigation",
            "http://www.w3.org/ns/oa#hasPurpose": {
                "@id": "http://www.w3.org/ns/oa#describing"
            },
        }

        annotation, ignore = create(
            self._db,
            Annotation,
            patron=patron,
            identifier=identifier,
            motivation=Annotation.IDLING,
            content=json.dumps(body),
        )

        with self.app.test_request_context("/"):
            detail = AnnotationWriter.detail(annotation)

            assert "annotations/%i" % annotation.id in detail["id"]
            assert "Annotation" == detail["type"]
            assert Annotation.IDLING == detail["motivation"]
            compacted_body = {
                "type": "TextualBody",
                "bodyValue": "A good description of the topic that bears further investigation",
                "purpose": "describing",
            }
            assert compacted_body == detail["body"]


class TestAnnotationParser(AnnotationTest):
    def setup_method(self):
        super(TestAnnotationParser, self).setup_method()
        self.pool = self._licensepool(None)
        self.identifier = self.pool.identifier
        self.patron = self._patron()

    def _sample_jsonld(self, motivation=Annotation.IDLING):
        data = dict()
        data["@context"] = [
            AnnotationWriter.JSONLD_CONTEXT,
            {"ls": Annotation.LS_NAMESPACE},
        ]
        data["type"] = "Annotation"
        motivation = motivation.replace(Annotation.LS_NAMESPACE, "ls:")
        motivation = motivation.replace(Annotation.OA_NAMESPACE, "oa:")
        data["motivation"] = motivation
        data["body"] = {
            "type": "TextualBody",
            "bodyValue": "A good description of the topic that bears further investigation",
            "purpose": "describing",
        }
        data["target"] = {
            "source": self.identifier.urn,
            "selector": {
                "type": "oa:FragmentSelector",
                "value": "epubcfi(/6/4[chap01ref]!/4[body01]/10[para05]/3:10)",
            },
        }
        return data

    def test_parse_invalid_json(self):
        annotation = AnnotationParser.parse(self._db, "not json", self.patron)
        assert INVALID_ANNOTATION_FORMAT == annotation

    def test_invalid_identifier(self):
        # If the target source can't be parsed as a URN we send
        # INVALID_ANNOTATION_TARGET
        data = self._sample_jsonld()
        data["target"]["source"] = "not a URN"
        annotation = AnnotationParser.parse(self._db, json.dumps(data), self.patron)
        assert INVALID_ANNOTATION_TARGET == annotation

    def test_null_id(self):
        # A JSON-LD document can have its @id set to null -- it's the
        # same as if the @id wasn't present -- but the jsonld library
        # can't handle this, so we need to test it specially.
        self.pool.loan_to(self.patron)
        data = self._sample_jsonld()
        data["id"] = None
        annotation = AnnotationParser.parse(self._db, json.dumps(data), self.patron)
        assert isinstance(annotation, Annotation)

    def test_parse_expanded_jsonld(self):
        self.pool.loan_to(self.patron)

        data = dict()
        data["@type"] = ["http://www.w3.org/ns/oa#Annotation"]
        data["http://www.w3.org/ns/oa#motivatedBy"] = [{"@id": Annotation.IDLING}]
        data["http://www.w3.org/ns/oa#hasBody"] = [
            {
                "@type": ["http://www.w3.org/ns/oa#TextualBody"],
                "http://www.w3.org/ns/oa#bodyValue": [
                    {
                        "@value": "A good description of the topic that bears further investigation"
                    }
                ],
                "http://www.w3.org/ns/oa#hasPurpose": [
                    {"@id": "http://www.w3.org/ns/oa#describing"}
                ],
            }
        ]
        data["http://www.w3.org/ns/oa#hasTarget"] = [
            {
                "http://www.w3.org/ns/oa#hasSelector": [
                    {
                        "@type": ["http://www.w3.org/ns/oa#FragmentSelector"],
                        "http://www.w3.org/1999/02/22-rdf-syntax-ns#value": [
                            {
                                "@value": "epubcfi(/6/4[chap01ref]!/4[body01]/10[para05]/3:10)"
                            }
                        ],
                    }
                ],
                "http://www.w3.org/ns/oa#hasSource": [{"@id": self.identifier.urn}],
            }
        ]

        data_json = json.dumps(data)

        annotation = AnnotationParser.parse(self._db, data_json, self.patron)
        assert self.patron.id == annotation.patron_id
        assert self.identifier.id == annotation.identifier_id
        assert Annotation.IDLING == annotation.motivation
        assert True == annotation.active
        assert (
            json.dumps(data["http://www.w3.org/ns/oa#hasTarget"][0])
            == annotation.target
        )
        assert (
            json.dumps(data["http://www.w3.org/ns/oa#hasBody"][0]) == annotation.content
        )

    def test_parse_compacted_jsonld(self):
        self.pool.loan_to(self.patron)

        data = dict()
        data["@type"] = "http://www.w3.org/ns/oa#Annotation"
        data["http://www.w3.org/ns/oa#motivatedBy"] = {"@id": Annotation.IDLING}
        data["http://www.w3.org/ns/oa#hasBody"] = {
            "@type": "http://www.w3.org/ns/oa#TextualBody",
            "http://www.w3.org/ns/oa#bodyValue": "A good description of the topic that bears further investigation",
            "http://www.w3.org/ns/oa#hasPurpose": {
                "@id": "http://www.w3.org/ns/oa#describing"
            },
        }
        data["http://www.w3.org/ns/oa#hasTarget"] = {
            "http://www.w3.org/ns/oa#hasSource": {"@id": self.identifier.urn},
            "http://www.w3.org/ns/oa#hasSelector": {
                "@type": "http://www.w3.org/ns/oa#FragmentSelector",
                "http://www.w3.org/1999/02/22-rdf-syntax-ns#value": "epubcfi(/6/4[chap01ref]!/4[body01]/10[para05]/3:10)",
            },
        }

        data_json = json.dumps(data)
        expanded = jsonld.expand(data)[0]

        annotation = AnnotationParser.parse(self._db, data_json, self.patron)
        assert self.patron.id == annotation.patron_id
        assert self.identifier.id == annotation.identifier_id
        assert Annotation.IDLING == annotation.motivation
        assert True == annotation.active
        assert (
            json.dumps(expanded["http://www.w3.org/ns/oa#hasTarget"][0])
            == annotation.target
        )
        assert (
            json.dumps(expanded["http://www.w3.org/ns/oa#hasBody"][0])
            == annotation.content
        )

    def test_parse_jsonld_with_context(self):
        self.pool.loan_to(self.patron)

        data = self._sample_jsonld()
        data_json = json.dumps(data)
        expanded = jsonld.expand(data)[0]

        annotation = AnnotationParser.parse(self._db, data_json, self.patron)

        assert self.patron.id == annotation.patron_id
        assert self.identifier.id == annotation.identifier_id
        assert Annotation.IDLING == annotation.motivation
        assert True == annotation.active
        assert (
            json.dumps(expanded["http://www.w3.org/ns/oa#hasTarget"][0])
            == annotation.target
        )
        assert (
            json.dumps(expanded["http://www.w3.org/ns/oa#hasBody"][0])
            == annotation.content
        )

    def test_parse_jsonld_with_bookmarking_motivation(self):
        """You can create multiple bookmarks in a single book."""
        self.pool.loan_to(self.patron)

        data = self._sample_jsonld(motivation=Annotation.BOOKMARKING)
        data_json = json.dumps(data)
        annotation = AnnotationParser.parse(self._db, data_json, self.patron)
        assert Annotation.BOOKMARKING == annotation.motivation

        # You can't create another bookmark at the exact same location --
        # you just get the same annotation again.
        annotation2 = AnnotationParser.parse(self._db, data_json, self.patron)
        assert annotation == annotation2

        # But unlike with IDLING, you _can_ create multiple bookmarks
        # for the same identifier, so long as the selector value
        # (ie. the location within the book) is different.
        data["target"]["selector"][
            "value"
        ] = "epubcfi(/3/4[chap01ref]!/4[body01]/15[para05]/3:10)"
        data_json = json.dumps(data)
        annotation3 = AnnotationParser.parse(self._db, data_json, self.patron)
        assert annotation3 != annotation
        assert 2 == len(self.patron.annotations)

    def test_parse_jsonld_with_invalid_motivation(self):
        self.pool.loan_to(self.patron)

        data = self._sample_jsonld()
        data["motivation"] = "not-a-valid-motivation"
        data_json = json.dumps(data)

        annotation = AnnotationParser.parse(self._db, data_json, self.patron)

        assert INVALID_ANNOTATION_MOTIVATION == annotation

    def test_parse_jsonld_with_no_loan(self):
        data = self._sample_jsonld()
        data_json = json.dumps(data)

        annotation = AnnotationParser.parse(self._db, data_json, self.patron)

        assert INVALID_ANNOTATION_TARGET == annotation

    def test_parse_jsonld_with_no_target(self):
        data = self._sample_jsonld()
        del data["target"]
        data_json = json.dumps(data)

        annotation = AnnotationParser.parse(self._db, data_json, self.patron)

        assert INVALID_ANNOTATION_TARGET == annotation

    def test_parse_updates_existing_annotation(self):
        self.pool.loan_to(self.patron)

        original_annotation, ignore = create(
            self._db,
            Annotation,
            patron_id=self.patron.id,
            identifier_id=self.identifier.id,
            motivation=Annotation.IDLING,
        )
        original_annotation.active = False
        yesterday = utc_now() - datetime.timedelta(days=1)
        original_annotation.timestamp = yesterday

        data = self._sample_jsonld()
        data = json.dumps(data)

        annotation = AnnotationParser.parse(self._db, data, self.patron)

        assert original_annotation == annotation
        assert True == annotation.active
        assert annotation.timestamp > yesterday

    def test_parse_treats_duplicates_as_interchangeable(self):
        self.pool.loan_to(self.patron)

        # Due to an earlier race condition, two duplicate annotations
        # were put in the database.
        a1, ignore = create(
            self._db,
            Annotation,
            patron_id=self.patron.id,
            identifier_id=self.identifier.id,
            motivation=Annotation.IDLING,
        )

        a2, ignore = create(
            self._db,
            Annotation,
            patron_id=self.patron.id,
            identifier_id=self.identifier.id,
            motivation=Annotation.IDLING,
        )

        assert a1 != a2

        # Parsing the annotation again retrieves one or the other
        # of the annotations rather than crashing or creating a third
        # annotation.
        data = self._sample_jsonld()
        data = json.dumps(data)
        annotation = AnnotationParser.parse(self._db, data, self.patron)
        assert annotation in (a1, a2)

    def test_parse_jsonld_with_patron_opt_out(self):
        self.pool.loan_to(self.patron)
        data = self._sample_jsonld()
        data_json = json.dumps(data)

        self.patron.synchronize_annotations = False
        annotation = AnnotationParser.parse(self._db, data_json, self.patron)
        assert PATRON_NOT_OPTED_IN_TO_ANNOTATION_SYNC == annotation
