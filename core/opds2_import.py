import logging
from contextlib import contextmanager
from datetime import datetime
from io import BytesIO, StringIO
from typing import Callable, Collection, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import sqlalchemy
import webpub_manifest_parser.opds2.ast as opds2_ast
from flask_babel import lazy_gettext as _
from webpub_manifest_parser.core import ManifestParserFactory, ManifestParserResult
from webpub_manifest_parser.core.analyzer import NodeFinder
from webpub_manifest_parser.core.ast import Manifestlike
from webpub_manifest_parser.errors import BaseError
from webpub_manifest_parser.opds2.registry import (
    OPDS2LinkRelationsRegistry,
    OPDS2MediaTypesRegistry,
)
from webpub_manifest_parser.utils import encode, first_or_default

from core.configuration.ignored_identifier import IgnoredIdentifierImporterMixin
from core.mirror import MirrorUploader
from core.model.configuration import (
    ConfigurationFactory,
    ConfigurationGrouping,
    ConfigurationStorage,
    HasExternalIntegration,
)

from .coverage import CoverageFailure
from .importers import BaseImporterConfiguration
from .metadata_layer import (
    CirculationData,
    ContributorData,
    FormatData,
    IdentifierData,
    LinkData,
    Metadata,
    SubjectData,
)
from .model import (
    Contributor,
    DeliveryMechanism,
    Edition,
    ExternalIntegration,
    Hyperlink,
    Identifier,
    LicensePool,
    LinkRelations,
    MediaTypes,
    Representation,
    RightsStatus,
    Subject,
)
from .opds_import import OPDSImporter, OPDSImportMonitor, SimplifiedOPDSLookup
from .util.http import BadResponseException
from .util.opds_writer import OPDSFeed


class RWPMManifestParser(object):
    def __init__(self, manifest_parser_factory: ManifestParserFactory):
        """Initialize a new instance of RWPMManifestParser class.

        :param manifest_parser_factory: Factory creating a new instance
            of a RWPM-compatible parser (RWPM, OPDS 2.x, ODL 2.x, etc.)
        """
        if not isinstance(manifest_parser_factory, ManifestParserFactory):
            raise ValueError(
                "Argument 'manifest_parser_factory' must be an instance of {0}".format(
                    ManifestParserFactory
                )
            )

        self._manifest_parser_factory = manifest_parser_factory

    def parse_manifest(
        self, manifest: Union[str, Dict, Manifestlike]
    ) -> ManifestParserResult:
        """Parse the feed into an RPWM-like AST object.

        :param manifest: RWPM-like manifest
        :return: Parsed RWPM-like manifest
        """
        result = None

        try:
            if isinstance(manifest, bytes):
                input_stream = BytesIO(manifest)
                parser = self._manifest_parser_factory.create()
                result = parser.parse_stream(input_stream)
            elif isinstance(manifest, str):
                input_stream = StringIO(manifest)
                parser = self._manifest_parser_factory.create()
                result = parser.parse_stream(input_stream)
            elif isinstance(manifest, dict):
                parser = self._manifest_parser_factory.create()
                result = parser.parse_json(manifest)
            elif isinstance(manifest, Manifestlike):
                result = ManifestParserResult(manifest)
            else:
                raise ValueError(
                    "Argument 'manifest' must be either a string, a dictionary, or an instance of {0}".format(
                        Manifestlike
                    )
                )
        except BaseError:
            logging.exception("Failed to parse the RWPM-like manifest")

            raise

        return result


class OPDS2ImporterConfiguration(ConfigurationGrouping, BaseImporterConfiguration):
    """Contains configuration settings of OPDS2Importer.
    Currently empty, but maintaining it as a base class for others"""


class OPDS2Importer(
    IgnoredIdentifierImporterMixin, OPDSImporter, HasExternalIntegration
):
    """Imports editions and license pools from an OPDS 2.0 feed."""

    NAME = ExternalIntegration.OPDS2_IMPORT
    DESCRIPTION = _("Import books from a publicly-accessible OPDS 2.0 feed.")
    SETTINGS = OPDSImporter.SETTINGS + OPDS2ImporterConfiguration.to_settings()
    NEXT_LINK_RELATION = "next"

    def __init__(
        self,
        db: sqlalchemy.orm.session.Session,
        collection: Collection,
        parser: RWPMManifestParser,
        data_source_name: str = None,
        identifier_mapping: Dict = None,
        http_get: Callable = None,
        metadata_client: SimplifiedOPDSLookup = None,
        content_modifier: Callable = None,
        map_from_collection: Dict = None,
        mirrors: Dict[str, MirrorUploader] = None,
    ):
        """Initialize a new instance of OPDS2Importer class.

        :param db: Database session

        :param collection: Circulation Manager's collection.
            LicensePools created by this OPDS2Import class will be associated with the given Collection.
            If this is None, no LicensePools will be created -- only Editions.
        :param parser: Feed parser
        :param data_source_name: Name of the source of this OPDS feed.
            All Editions created by this import will be associated with this DataSource.
            If there is no DataSource with this name, one will be created.
            NOTE: If `collection` is provided, its .data_source will take precedence over any value provided here.
            This is only for use when you are importing OPDS metadata without any particular Collection in mind.
        :param identifier_mapping: Dictionary used for mapping external identifiers into a set of internal ones
        :param metadata_client: A SimplifiedOPDSLookup object that is used to fill in missing metadata
        :param content_modifier: A function that may modify-in-place representations (such as images and EPUB documents)
            as they come in from the network.
        :param map_from_collection: Identifier mapping
        :param mirrors: A dictionary of different MirrorUploader objects for different purposes
        """
        super(OPDS2Importer, self).__init__(
            db,
            collection,
            data_source_name,
            identifier_mapping,
            http_get,
            metadata_client,
            content_modifier,
            map_from_collection,
            mirrors,
        )

        if not isinstance(parser, RWPMManifestParser):
            raise ValueError(
                "Argument 'parser' must be an instance of {0}".format(
                    RWPMManifestParser
                )
            )

        self._parser: RWPMManifestParser = parser
        self._logger: logging.Logger = logging.getLogger(__name__)

        self._external_integration_id: int = collection.external_integration.id
        self._configuration_storage: ConfigurationStorage = ConfigurationStorage(self)
        self._configuration_factory: ConfigurationFactory = ConfigurationFactory()

    def _is_identifier_allowed(self, identifier: Identifier) -> bool:
        """Check the identifier and return a boolean value indicating whether CM can import it.

        NOTE: Currently, this method hard codes allowed identifier types.
        The next PR will add an additional configuration setting allowing to override this behaviour
        and configure allowed identifier types in the CM Admin UI.

        :param identifier: Identifier object
        :return: Boolean value indicating whether CM can import the identifier
        """
        configuration = self._get_configuration(self._db)
        with configuration as conf:
            ignored_identifier_types = self._get_ignored_identifier_types(conf)

        return identifier.type not in ignored_identifier_types

    def _extract_subjects(
        self, subjects: List["core_ast.Subject"]
    ) -> List["SubjectMetadata"]:
        """Extract a list of SubjectData objects from the webpub-manifest-parser's subject.

        :param subjects: Parsed subject object
        :return: List of subjects metadata
        """
        self._logger.debug("Started extracting subjects metadata")

        subject_metadata_list = []

        for subject in subjects:
            self._logger.debug(
                "Started extracting subject metadata from {0}".format(encode(subject))
            )

            scheme = subject.scheme

            subject_type = Subject.by_uri.get(scheme)
            if not subject_type:
                # We can't represent this subject because we don't
                # know its scheme. Just treat it as a tag.
                subject_type = Subject.TAG

            subject_metadata = SubjectData(
                type=subject_type, identifier=subject.code, name=subject.name, weight=1
            )

            subject_metadata_list.append(subject_metadata)

            self._logger.debug(
                "Finished extracting subject metadata from {0}: {1}".format(
                    encode(subject), encode(subject_metadata)
                )
            )

        self._logger.debug(
            "Finished extracting subjects metadata: {0}".format(
                encode(subject_metadata_list)
            )
        )

        return subject_metadata_list

    def _extract_contributors(
        self,
        contributors: List["core_ast.Contributor"],
        default_role: Optional[str] = Contributor.AUTHOR_ROLE,
    ) -> List[ContributorData]:
        """Extract a list of ContributorData objects from the webpub-manifest-parser's contributor.

        :param contributors: Parsed contributor object
        :param default_role: Default role
        :return: List of contributors metadata
        """
        self._logger.debug("Started extracting contributors metadata")

        contributor_metadata_list = []

        for contributor in contributors:
            self._logger.debug(
                "Started extracting contributor metadata from {0}".format(
                    encode(contributor)
                )
            )

            contributor_metadata = ContributorData(
                sort_name=contributor.sort_as,
                display_name=contributor.name,
                family_name=None,
                wikipedia_name=None,
                roles=contributor.roles if contributor.roles else default_role,
            )

            self._logger.debug(
                "Finished extracting contributor metadata from {0}: {1}".format(
                    encode(contributor), encode(contributor_metadata)
                )
            )

            contributor_metadata_list.append(contributor_metadata)

        self._logger.debug(
            "Finished extracting contributors metadata: {0}".format(
                encode(contributor_metadata_list)
            )
        )

        return contributor_metadata_list

    def _extract_link(self, link, feed_self_url, default_link_rel=None):
        """Extract a LinkData object from webpub-manifest-parser's link.

        :param link: webpub-manifest-parser's link
        :type link: ast_core.Link

        :param feed_self_url: Feed's self URL
        :type feed_self_url: str

        :param default_link_rel: Default link's relation
        :type default_link_rel: Optional[str]

        :return: Link metadata
        :rtype: LinkData
        """
        self._logger.debug(
            "Started extracting link metadata from {0}".format(encode(link))
        )

        # FIXME: It seems that OPDS 2.0 spec doesn't contain information about rights so we use the default one.
        rights_uri = RightsStatus.rights_uri_from_string("")
        rel = first_or_default(link.rels, default_link_rel)
        media_type = link.type
        href = link.href

        if feed_self_url and not urlparse(href).netloc:
            # This link is relative, so we need to get the absolute url
            href = urljoin(feed_self_url, href)

        link_metadata = LinkData(
            rel=rel,
            href=href,
            media_type=media_type,
            rights_uri=rights_uri,
            content=None,
        )

        self._logger.debug(
            "Finished extracting link metadata from {0}: {1}".format(
                encode(link), encode(link_metadata)
            )
        )

        return link_metadata

    def _extract_description_link(
        self, publication: opds2_ast.OPDS2Publication
    ) -> LinkData:
        """Extract description from the publication object and create a Hyperlink.DESCRIPTION link containing it.

        :param publication: Publication object
        :return: LinkData object containing publication's description
        """
        self._logger.debug(
            "Started extracting a description link from {0}".format(
                encode(publication.metadata.description)
            )
        )

        description_link = None

        if publication.metadata.description:
            description_link = LinkData(
                rel=Hyperlink.DESCRIPTION,
                media_type=MediaTypes.TEXT_PLAIN,
                content=publication.metadata.description,
            )

        self._logger.debug(
            "Finished extracting a description link from {0}: {1}".format(
                encode(publication.metadata.description), encode(description_link)
            )
        )

        return description_link

    def _extract_image_links(
        self, publication: "ast_core.Publication", feed_self_url: str
    ) -> List[LinkData]:
        """Extracts a list of LinkData objects containing information about artwork.

        :param publication: Publication object
        :param feed_self_url: Feed's self URL
        :return: List of links metadata
        """
        self._logger.debug(
            "Started extracting image links from {0}".format(encode(publication.images))
        )

        if not publication.images:
            return []

        # FIXME: This code most likely will not work in general.
        # There's no guarantee that these images have the same media type,
        # or that the second-largest image isn't far too large to use as a thumbnail.
        # Instead of using the second-largest image as a thumbnail,
        # find the image that would make the best thumbnail
        # because of its dimensions, media type, and aspect ratio:
        #       IDEAL_COVER_ASPECT_RATIO = 2.0/3
        #       IDEAL_IMAGE_HEIGHT = 240
        #       IDEAL_IMAGE_WIDTH = 160

        sorted_raw_image_links = list(
            reversed(
                sorted(
                    publication.images.links,
                    key=lambda link: (link.width or 0, link.height or 0),
                )
            )
        )
        image_links = []

        if len(sorted_raw_image_links) > 0:
            cover_link = self._extract_link(
                sorted_raw_image_links[0],
                feed_self_url,
                default_link_rel=Hyperlink.IMAGE,
            )
            image_links.append(cover_link)

        if len(sorted_raw_image_links) > 1:
            cover_link = self._extract_link(
                sorted_raw_image_links[1],
                feed_self_url,
                default_link_rel=Hyperlink.THUMBNAIL_IMAGE,
            )
            image_links.append(cover_link)

        self._logger.debug(
            "Finished extracting image links from {0}: {1}".format(
                encode(publication.images), encode(image_links)
            )
        )

        return image_links

    def _extract_links(
        self, publication: "ast_core.Publication", feed_self_url: str
    ) -> List[LinkData]:
        """Extract a list of LinkData objects from a list of webpub-manifest-parser links.

        :param publication: Publication object
        :param feed_self_url: Feed's self URL
        :return: List of links metadata
        """
        self._logger.debug(
            "Started extracting links from {0}".format(encode(publication.links))
        )

        links = []

        for link in publication.links:
            link_metadata = self._extract_link(link, feed_self_url)
            links.append(link_metadata)

        description_link = self._extract_description_link(publication)
        if description_link:
            links.append(description_link)

        image_links = self._extract_image_links(publication, feed_self_url)
        if image_links:
            links.extend(image_links)

        self._logger.debug(
            "Finished extracting links from {0}: {1}".format(
                encode(publication.links), encode(links)
            )
        )

        return links

    def _extract_media_types_and_drm_scheme_from_link(
        self, link: "ast_core.Link"
    ) -> List[Tuple[str, str]]:
        """Extract information about content's media type and used DRM schema from the link.

        :param link: Link object
        :return: 2-tuple containing information about the content's media type and its DRM schema
        """
        self._logger.debug(
            "Started extracting media types and a DRM scheme from {0}".format(
                encode(link)
            )
        )

        media_types_and_drm_scheme = []

        if link.properties:
            if (
                not link.properties.availability
                or link.properties.availability.state
                == opds2_ast.OPDS2AvailabilityType.AVAILABLE.value
            ):
                for acquisition_object in link.properties.indirect_acquisition:
                    nested_acquisition_object = acquisition_object

                    while nested_acquisition_object.child:
                        nested_acquisition_object = first_or_default(
                            acquisition_object.child
                        )

                    drm_scheme = (
                        acquisition_object.type
                        if acquisition_object.type in DeliveryMechanism.KNOWN_DRM_TYPES
                        else DeliveryMechanism.NO_DRM
                    )

                    media_types_and_drm_scheme.append(
                        (nested_acquisition_object.type, drm_scheme)
                    )
        else:
            if (
                link.type in MediaTypes.BOOK_MEDIA_TYPES
                or link.type in MediaTypes.AUDIOBOOK_MEDIA_TYPES
            ):
                media_types_and_drm_scheme.append((link.type, DeliveryMechanism.NO_DRM))

        self._logger.debug(
            "Finished extracting media types and a DRM scheme from {0}: {1}".format(
                encode(link), encode(media_types_and_drm_scheme)
            )
        )

        return media_types_and_drm_scheme

    def _extract_medium_from_links(self, links: "ast_core.LinkList") -> Optional[str]:
        """Extract the publication's medium from its links.

        :param links: List of links
        :return: Publication's medium
        """
        derived = None

        for link in links:
            if not link.rels or not link.type or not self._is_acquisition_link(link):
                continue

            link_media_type, _ = first_or_default(
                self._extract_media_types_and_drm_scheme_from_link(link),
                default=(None, None),
            )
            derived = Edition.medium_from_media_type(link_media_type)

            if derived:
                break

        return derived

    @staticmethod
    def _extract_medium(
        publication: opds2_ast.OPDS2Publication,
        default_medium: str = Edition.BOOK_MEDIUM,
    ) -> str:
        """Extract the publication's medium from its metadata.

        :param publication: Publication object
        :return: Publication's medium
        """
        medium = default_medium

        if publication.metadata.type:
            medium = Edition.additional_type_to_medium.get(
                publication.metadata.type, default_medium
            )

        return medium

    def _extract_identifier(
        self, publication: opds2_ast.OPDS2Publication
    ) -> Identifier:
        """Extract the publication's identifier from its metadata.

        :param publication: Publication object
        :return: Identifier object
        """
        return self._parse_identifier(publication.metadata.identifier)

    def _extract_publication_metadata(
        self,
        feed: opds2_ast.OPDS2Feed,
        publication: opds2_ast.OPDS2Publication,
        data_source_name: str,
    ) -> Metadata:
        """Extract a Metadata object from webpub-manifest-parser's publication.

        :param publication: Feed object
        :param publication: Publication object
        :param data_source_name: Data source's name
        :return: Publication's metadata
        """
        self._logger.debug(
            "Started extracting metadata from publication {0}".format(
                encode(publication)
            )
        )

        title = publication.metadata.title

        if title == OPDSFeed.NO_TITLE:
            title = None

        subtitle = publication.metadata.subtitle

        languages = first_or_default(publication.metadata.languages)
        derived_medium = self._extract_medium_from_links(publication.links)
        medium = self._extract_medium(publication, derived_medium)

        publisher = first_or_default(publication.metadata.publishers)
        if publisher:
            publisher = publisher.name

        imprint = first_or_default(publication.metadata.imprints)
        if imprint:
            imprint = imprint.name

        published = publication.metadata.published
        subjects = self._extract_subjects(publication.metadata.subjects)
        contributors = (
            self._extract_contributors(
                publication.metadata.authors, Contributor.AUTHOR_ROLE
            )
            + self._extract_contributors(
                publication.metadata.translators, Contributor.TRANSLATOR_ROLE
            )
            + self._extract_contributors(
                publication.metadata.editors, Contributor.EDITOR_ROLE
            )
            + self._extract_contributors(
                publication.metadata.artists, Contributor.ARTIST_ROLE
            )
            + self._extract_contributors(
                publication.metadata.illustrators, Contributor.ILLUSTRATOR_ROLE
            )
            + self._extract_contributors(
                publication.metadata.letterers, Contributor.LETTERER_ROLE
            )
            + self._extract_contributors(
                publication.metadata.pencilers, Contributor.PENCILER_ROLE
            )
            + self._extract_contributors(
                publication.metadata.colorists, Contributor.COLORIST_ROLE
            )
            + self._extract_contributors(
                publication.metadata.inkers, Contributor.INKER_ROLE
            )
            + self._extract_contributors(
                publication.metadata.narrators, Contributor.NARRATOR_ROLE
            )
            + self._extract_contributors(
                publication.metadata.contributors, Contributor.CONTRIBUTOR_ROLE
            )
        )

        feed_self_url = first_or_default(
            feed.links.get_by_rel(OPDS2LinkRelationsRegistry.SELF.key)
        ).href
        links = self._extract_links(publication, feed_self_url)

        last_opds_update = publication.metadata.modified

        identifier = self._extract_identifier(publication)
        identifier_data = IdentifierData(
            type=identifier.type, identifier=identifier.identifier
        )

        # FIXME: There are no measurements in OPDS 2.0
        measurements = []

        # FIXME: There is no series information in OPDS 2.0
        series = None
        series_position = None

        # FIXME: It seems that OPDS 2.0 spec doesn't contain information about rights so we use the default one
        rights_uri = RightsStatus.rights_uri_from_string("")

        circulation_data = CirculationData(
            default_rights_uri=rights_uri,
            data_source=data_source_name,
            primary_identifier=identifier_data,
            links=links,
            licenses_owned=LicensePool.UNLIMITED_ACCESS,
            licenses_available=LicensePool.UNLIMITED_ACCESS,
            licenses_reserved=0,
            patrons_in_hold_queue=0,
            formats=[],
        )

        formats = self._find_formats_in_non_open_access_acquisition_links(
            publication.links, links, rights_uri, circulation_data
        )
        circulation_data.formats.extend(formats)

        metadata = Metadata(
            data_source=data_source_name,
            title=title,
            subtitle=subtitle,
            language=languages,
            medium=medium,
            publisher=publisher,
            published=published,
            imprint=imprint,
            primary_identifier=identifier_data,
            subjects=subjects,
            contributors=contributors,
            measurements=measurements,
            series=series,
            series_position=series_position,
            links=links,
            data_source_last_updated=last_opds_update,
            circulation=circulation_data,
        )

        self._logger.debug(
            "Finished extracting metadata from publication {0}: {1}".format(
                encode(publication), encode(metadata)
            )
        )

        return metadata

    def _find_formats_in_non_open_access_acquisition_links(
        self,
        ast_link_list: List["ast_core.Link"],
        link_data_list: List[LinkData],
        rights_uri: str,
        circulation_data: CirculationData,
    ) -> List[FormatData]:
        """Find circulation formats in non open-access acquisition links.

        :param ast_link_list: List of Link objects
        :param link_data_list: List of LinkData objects
        :param rights_uri: Rights URI
        :param circulation_data: Circulation data
        :return: List of additional circulation formats found in non-open access links
        """
        formats = []

        for ast_link, parsed_link in zip(ast_link_list, link_data_list):
            if not self._is_acquisition_link(ast_link):
                continue
            if self._is_open_access_link_(parsed_link, circulation_data):
                continue

            for (
                content_type,
                drm_scheme,
            ) in self._extract_media_types_and_drm_scheme_from_link(ast_link):
                formats.append(
                    FormatData(
                        content_type=content_type,
                        drm_scheme=drm_scheme,
                        link=parsed_link,
                        rights_uri=rights_uri,
                    )
                )

        return formats

    @contextmanager
    def _get_configuration(
        self, db: sqlalchemy.orm.session.Session
    ) -> OPDS2ImporterConfiguration:
        """Return the configuration object.
        :param db: Database session
        :return: Configuration object
        """
        with self._configuration_factory.create(
            self._configuration_storage, db, OPDS2ImporterConfiguration
        ) as configuration:
            yield configuration

    def external_integration(
        self, db: sqlalchemy.orm.session.Session
    ) -> ExternalIntegration:
        """Return an external integration associated with this object.
        :param db: Database session
        :return: External integration associated with this object
        """
        return self.collection.external_integration

    @staticmethod
    def _get_publications(
        feed: opds2_ast.OPDS2Feed,
    ) -> Iterable[opds2_ast.OPDS2Publication]:
        """Return all the publications in the feed.
        :param feed: OPDS 2.0 feed
        :return: An iterable list of publications containing in the feed
        """
        if feed.publications:
            for publication in feed.publications:
                yield publication

        if feed.groups:
            for group in feed.groups:
                if group.publications:
                    for publication in group.publications:
                        yield publication

    @staticmethod
    def _is_acquisition_link(link: "ast_core.Link") -> bool:
        """Return a boolean value indicating whether a link can be considered an acquisition link.

        :param link: Link object
        :return: Boolean value indicating whether a link can be considered an acquisition link
        """
        return any(
            [rel for rel in link.rels if rel in LinkRelations.CIRCULATION_ALLOWED]
        )

    @staticmethod
    def _is_open_access_link_(
        link_data: LinkData, circulation_data: CirculationData
    ) -> bool:
        """Return a boolean value indicating whether the specified LinkData object describes an open-access link.

        :param link_data: LinkData object
        :param circulation_data: CirculationData object
        """
        open_access_link = (
            link_data.rel == Hyperlink.OPEN_ACCESS_DOWNLOAD and link_data.href
        )

        if open_access_link:
            return True

        # Try to deduce if the ast_link is open-access, even if it doesn't explicitly say it is
        rights_uri = link_data.rights_uri or circulation_data.default_rights_uri
        open_access_rights_link = (
            link_data.media_type in Representation.BOOK_MEDIA_TYPES
            and link_data.href
            and rights_uri in RightsStatus.OPEN_ACCESS
        )

        return open_access_rights_link

    def _record_coverage_failure(
        self,
        failures: Dict[str, List[CoverageFailure]],
        identifier: Identifier,
        error_message: str,
        transient: bool = True,
    ) -> CoverageFailure:
        """Record a new coverage failure.

        :param failures: Dictionary mapping publication identifiers to corresponding CoverageFailure objects
        :param identifier: Publication's identifier
        :param error_message: Message describing the failure
        :param transient: Boolean value indicating whether the failure is final or it can go away in the future
        :return: CoverageFailure object describing the error
        """
        if identifier.identifier is None:
            raise ValueError

        if identifier not in failures:
            failures[identifier.identifier] = []

        failure = CoverageFailure(
            identifier,
            error_message,
            data_source=self.data_source,
            transient=transient,
        )
        failures[identifier.identifier].append(failure)

        return failure

    def _record_publication_unrecognizable_identifier(
        self, publication: opds2_ast.OPDS2Publication
    ) -> None:
        """Record a publication's unrecognizable identifier, i.e. identifier that has an unknown format
            and could not be parsed by CM.

        :param publication: OPDS 2.x publication object
        """
        original_identifier = publication.metadata.identifier
        title = publication.metadata.title

        if original_identifier is None:
            self._logger.warning(f"Publication '{title}' does not have an identifier.")
        else:
            self._logger.warning(
                f"Publication # {original_identifier} ('{title}') has an unrecognizable identifier."
            )

    def extract_next_links(self, feed: Union[str, opds2_ast.OPDS2Feed]) -> List[str]:
        """Extracts "next" links from the feed.

        :param feed: OPDS 2.0 feed
        :return: List of "next" links
        """
        parser_result = self._parser.parse_manifest(feed)
        parsed_feed = parser_result.root

        if not parsed_feed:
            return []

        next_links = parsed_feed.links.get_by_rel(self.NEXT_LINK_RELATION)
        next_links = [next_link.href for next_link in next_links]

        return next_links

    def extract_last_update_dates(
        self, feed: Union[str, opds2_ast.OPDS2Feed]
    ) -> List[Tuple[str, datetime]]:
        """Extract last update date of the feed.

        :param feed: OPDS 2.0 feed
        :return: A list of 2-tuples containing publication's identifiers and their last modified dates
        """
        parser_result = self._parser.parse_manifest(feed)
        parsed_feed = parser_result.root

        if not parsed_feed:
            return []

        dates = [
            (publication.metadata.identifier, publication.metadata.modified)
            for publication in self._get_publications(parsed_feed)
            if publication.metadata.modified
        ]

        return dates

    def extract_feed_data(
        self, feed: Union[str, opds2_ast.OPDS2Feed], feed_url: Optional[str] = None
    ) -> Tuple[Dict, Dict]:
        """Turn an OPDS 2.0 feed into lists of Metadata and CirculationData objects.
        :param feed: OPDS 2.0 feed
        :param feed_url: Feed URL used to resolve relative links
        """
        parser_result = self._parser.parse_manifest(feed)
        feed = parser_result.root
        publication_metadata_dictionary = {}
        failures = {}

        for publication in self._get_publications(feed):
            recognized_identifier = self._extract_identifier(publication)

            if not recognized_identifier or not self._is_identifier_allowed(
                recognized_identifier
            ):
                self._record_publication_unrecognizable_identifier(publication)
                continue

            publication_metadata = self._extract_publication_metadata(
                feed, publication, self.data_source_name
            )

            publication_metadata_dictionary[
                publication_metadata.primary_identifier.identifier
            ] = publication_metadata

        node_finder = NodeFinder()

        for error in parser_result.errors:
            publication = node_finder.find_parent_or_self(
                parser_result.root, error.node, opds2_ast.OPDS2Publication
            )

            if publication:
                recognized_identifier = self._extract_identifier(publication)

                if not recognized_identifier or not self._is_identifier_allowed(
                    recognized_identifier
                ):
                    self._record_publication_unrecognizable_identifier(publication)
                else:
                    self._record_coverage_failure(
                        failures, recognized_identifier, error.error_message
                    )
            else:
                self._logger.warning(f"{error.error_message}")

        return publication_metadata_dictionary, failures


class OPDS2ImportMonitor(OPDSImportMonitor):
    PROTOCOL = ExternalIntegration.OPDS2_IMPORT
    MEDIA_TYPE = OPDS2MediaTypesRegistry.OPDS_FEED.key, "application/json"

    def _verify_media_type(self, url, status_code, headers, feed):
        # Make sure we got an OPDS feed, and not an error page that was
        # sent with a 200 status code.
        media_type = headers.get("content-type")
        if not media_type or not any(x in media_type for x in self.MEDIA_TYPE):
            message = "Expected {0} OPDS 2.0 feed, got {1}".format(
                self.MEDIA_TYPE, media_type
            )

            raise BadResponseException(
                url, message=message, debug_message=feed, status_code=status_code
            )

    def _get_accept_header(self):
        return "{0}, {1};q=0.9, */*;q=0.1".format(
            OPDS2MediaTypesRegistry.OPDS_FEED.key, "application/json"
        )
