import os
import datetime
from StringIO import StringIO
from nose.tools import (
    set_trace,
    eq_,
    assert_raises
)
import feedparser

from lxml import etree
import pkgutil
from psycopg2.extras import NumericRange
from . import (
    DatabaseTest,
)
from config import (
    Configuration,
    temp_config,
    CannotLoadConfiguration
)
from opds_import import (
    SimplifiedOPDSLookup,
    OPDSImporter,
    OPDSImporterWithS3Mirror,
    StatusMessage,
)
from metadata_layer import (
    LinkData
)
from model import (
    Contributor,
    DataSource,
    DeliveryMechanism,
    Hyperlink,
    Identifier,
    Edition,
    Measurement,
    Representation,
    RightsStatus,
    Subject,
)

from s3 import DummyS3Uploader
from testing import DummyHTTPClient

class TestStatusMessage(object):

    def test_constructor(self):

        message = StatusMessage(200, "success")
        eq_(True, message.success)
        eq_(False, message.transient)

        message = StatusMessage(201, "try later")
        eq_(False, message.success)
        eq_(True, message.transient)

        message = StatusMessage(500, "oops")
        eq_(False, message.success)
        eq_(True, message.transient)

        message = StatusMessage(404, "nope")
        eq_(False, message.success)
        eq_(False, message.transient)


class TestSimplifiedOPDSLookup(object):

    def test_authenticates_wrangler_requests(self):
        """Tests that the client_id and client_secret are set for any
        Metadata Wrangler lookups"""

        mw_integration = Configuration.METADATA_WRANGLER_INTEGRATION
        mw_client_id = Configuration.METADATA_WRANGLER_CLIENT_ID
        mw_client_secret = Configuration.METADATA_WRANGLER_CLIENT_SECRET

        with temp_config() as config:
            config['integrations'][mw_integration] = {
                Configuration.URL : "http://localhost",
                mw_client_id : "abc",
                mw_client_secret : "def"
            }
            importer = SimplifiedOPDSLookup.from_config()
            eq_("abc", importer.client_id)
            eq_("def", importer.client_secret)

            # An error is raised if only one value is set.
            del config['integrations'][mw_integration][mw_client_secret]
            assert_raises(CannotLoadConfiguration, SimplifiedOPDSLookup.from_config)

            # The details are None if client configuration isn't set at all.
            del config['integrations'][mw_integration][mw_client_id]
            importer = SimplifiedOPDSLookup.from_config()
            eq_(None, importer.client_id)
            eq_(None, importer.client_secret)

            # For other integrations, the details aren't created at all.
            config['integrations']["Content Server"] = dict(
                url = "http://whatevz"
            )
            importer = SimplifiedOPDSLookup.from_config("Content Server")
            eq_(False, hasattr(importer, "client_id"))
            eq_(False, hasattr(importer, "client_secret"))


class OPDSImporterTest(DatabaseTest):

    def setup(self):
        super(OPDSImporterTest, self).setup()
        base_path = os.path.split(__file__)[0]
        self.resource_path = os.path.join(base_path, "files", "opds")
        self.content_server_feed = open(
            os.path.join(self.resource_path, "content_server.opds")).read()
        self.content_server_mini_feed = open(
            os.path.join(self.resource_path, "content_server_mini.opds")).read()


class TestOPDSImporter(OPDSImporterTest):

    def test_extract_metadata(self):
        importer = OPDSImporter(self._db, DataSource.NYT)
        data, status_messages, next_link = importer.extract_feed_data(
            self.content_server_mini_feed
        )
        m1, m2 = sorted(data, key=lambda x:x.title)
        eq_("The Green Mouse", m2.title)
        eq_("A Tale of Mousy Terror", m2.subtitle)

        eq_(None, m1._license_data_source)
        eq_(DataSource.GUTENBERG, m2._license_data_source)

        [message] = status_messages.values()
        eq_(202, message.status_code)
        eq_(u"I'm working to locate a source for this identifier.", message.message)

        eq_("http://localhost:5000/?after=327&size=100", next_link[0])

    def test_extract_metadata_from_feedparser(self):

        data, status_messages, next_link = OPDSImporter.extract_data_from_feedparser(
            self.content_server_mini_feed
        )        

        metadata = data['urn:librarysimplified.org/terms/id/Gutenberg%20ID/10441']
        eq_("The Green Mouse", metadata['title'])
        eq_("A Tale of Mousy Terror", metadata['subtitle'])
        eq_('en', metadata['language'])
        eq_('Project Gutenberg', metadata['publisher'])
        eq_(DataSource.GUTENBERG, metadata['license_data_source'])

        circulation = metadata['circulation']
        eq_(datetime.datetime(2014, 1, 2, 16, 56, 40), circulation.first_appearance)

        message = status_messages['http://www.gutenberg.org/ebooks/1984']
        eq_(202, message.status_code)
        eq_(u"I'm working to locate a source for this identifier.", message.message)

    def test_extract_metadata_from_elementtree(self):

        data = OPDSImporter.extract_metadata_from_elementtree(
            self.content_server_feed
        )

        # There are 76 entries in the feed, and we got metadata for
        # every one of them.
        eq_(76, len(data))

        # We're going to do spot checks on a book and a periodical.

        # First, the book.
        book_id = 'urn:librarysimplified.org/terms/id/Gutenberg%20ID/1022'
        book = data[book_id]
        eq_(Edition.BOOK_MEDIUM, book['medium'])

        [contributor] = book['contributors']
        eq_("Thoreau, Henry David", contributor.sort_name)
        eq_([Contributor.AUTHOR_ROLE], contributor.roles)

        subjects = book['subjects']
        eq_(['LCSH', 'LCSH', 'LCSH', 'LCC'], [x.type for x in subjects])
        eq_(
            ['Essays', 'Nature', 'Walking', 'PS'],
            [x.identifier for x in subjects]
        )
        eq_(
            [None, None, None, 'American Literature'],
            [x.name for x in book['subjects']]
        )
        eq_(
            [1, 1, 1, 10],
            [x.weight for x in book['subjects']]
        )

        eq_([], book['measurements'])

        [link] = book['links']
        eq_(Hyperlink.OPEN_ACCESS_DOWNLOAD, link.rel)
        eq_("http://www.gutenberg.org/ebooks/1022.epub.noimages", link.href)
        eq_(Representation.EPUB_MEDIA_TYPE, link.media_type)

        # And now, the periodical.
        periodical_id = 'urn:librarysimplified.org/terms/id/Gutenberg%20ID/10441'
        periodical = data[periodical_id]
        eq_(Edition.PERIODICAL_MEDIUM, periodical['medium'])

        subjects = periodical['subjects']
        eq_(
            ['LCSH', 'LCSH', 'LCSH', 'LCSH', 'LCC', 'schema:audience', 'schema:typicalAgeRange'], 
            [x.type for x in subjects]
        )
        eq_(
            ['Courtship -- Fiction', 'New York (N.Y.) -- Fiction', 'Fantasy fiction', 'Magic -- Fiction', 'PZ', 'Children', '7'],
            [x.identifier for x in subjects]
        )
        eq_([1, 1, 1, 1, 1, 100, 100], [x.weight for x in subjects])
        
        r1, r2, r3 = periodical['measurements']

        eq_(Measurement.QUALITY, r1.quantity_measured)
        eq_(0.3333, r1.value)
        eq_(1, r1.weight)

        eq_(Measurement.RATING, r2.quantity_measured)
        eq_(0.6, r2.value)
        eq_(1, r2.weight)

        eq_(Measurement.POPULARITY, r3.quantity_measured)
        eq_(0.25, r3.value)
        eq_(1, r3.weight)

    def test_import(self):
        path = os.path.join(self.resource_path, "content_server_mini.opds")
        feed = open(path).read()
        #imported, messages, next_links = OPDSImporter(self._db).import_from_feed(feed)
        imported_editions, imported_pools, imported_works, error_messages, next_links = (
            OPDSImporter(self._db).import_from_feed(feed)
        )

        [crow, mouse] = sorted(imported, key=lambda x: x.title)

        # By default, this feed is treated as though it came from the
        # metadata wrangler. No Work has been created for the 'crow'
        # book because the metadat wrangler doesn't know who actually
        # provides copies of this book.
        eq_(DataSource.METADATA_WRANGLER, crow.data_source.name)
        eq_(None, crow.work)
        eq_(Edition.BOOK_MEDIUM, crow.medium)

        # But the 'mouse' book is known to come from Project Gutenberg,
        # so a Work has been created for that book.
        assert mouse.license_pool.work is not None
        eq_(Edition.PERIODICAL_MEDIUM, mouse.medium)

        popularity, quality, rating = sorted(
            [x for x in mouse.primary_identifier.measurements
             if x.is_most_recent],
            key=lambda x: x.quantity_measured
        )

        eq_(DataSource.METADATA_WRANGLER, popularity.data_source.name)
        eq_(Measurement.POPULARITY, popularity.quantity_measured)
        eq_(0.25, popularity.value)

        eq_(DataSource.METADATA_WRANGLER, quality.data_source.name)
        eq_(Measurement.QUALITY, quality.quantity_measured)
        eq_(0.3333, quality.value)

        eq_(DataSource.METADATA_WRANGLER, rating.data_source.name)
        eq_(Measurement.RATING, rating.quantity_measured)
        eq_(0.6, rating.value)

        seven, children, courtship, fantasy, pz, magic, new_york = sorted(
            mouse.primary_identifier.classifications,
            key=lambda x: x.subject.name)

        pz_s = pz.subject
        eq_("Juvenile Fiction", pz_s.name)
        eq_("PZ", pz_s.identifier)

        new_york_s = new_york.subject
        eq_("New York (N.Y.) -- Fiction", new_york_s.name)
        eq_("sh2008108377", new_york_s.identifier)

        eq_('7', seven.subject.identifier)
        eq_(100, seven.weight)
        eq_(Subject.AGE_RANGE, seven.subject.type)
        from classifier import Classifier
        classifier = Classifier.classifiers.get(seven.subject.type, None)
        classifier.classify(seven.subject)

        work = mouse.license_pool.work
        work.calculate_presentation()
        eq_(0.4142, round(work.quality, 4))
        eq_(Classifier.AUDIENCE_CHILDREN, work.audience)
        eq_(NumericRange(7,7, '[]'), work.target_age)

        # The other book has no license pool and no work because we
        # could not figure out whether the license source was Project
        # Gutenberg or Project GITenberg.
        eq_(None, crow.work)
        eq_(None, crow.license_pool)

        # Bonus: make sure that delivery mechanisms are set appropriately.
        [mech] = mouse.license_pool.delivery_mechanisms
        eq_(Representation.EPUB_MEDIA_TYPE, mech.delivery_mechanism.content_type)
        eq_(DeliveryMechanism.NO_DRM, mech.delivery_mechanism.drm_scheme)
        eq_('http://www.gutenberg.org/ebooks/10441.epub.images', 
            mech.resource.url)

        # If we import the same file again, we get the same list of Editions.
        imported2, messages, next_links = OPDSImporter(self._db).import_from_feed(feed)
        eq_(imported2, imported)



    def test_import_with_wrangler_data_source(self):
        # Tests that will create Edition, LicensePool, and Work objects, when appropriate.
        # For example, on a Metadata_Wrangler data source, it is only appropriate to create 
        # editions, but not pools or works.  On a lendable data source, should create 
        # pools and works as well as editions.
        # Tests that the number and contents of error messages are appropriate to the task.

        # will create editions, but not license pools or works, because the 
        # metadata wrangler data source is not lendable
        cutoff = datetime.datetime(2016, 1, 2, 16, 56, 40)
        path = os.path.join(self.resource_path, "content_server_mini.opds")
        feed = open(path).read()

        importer_mw = OPDSImporter(self._db, data_source_name=DataSource.METADATA_WRANGLER)
        imported_editions_mw, imported_pools_mw, imported_works_mw, error_messages_mw, next_links_mw = (
            importer_mw.import_from_feed(feed, cutoff_date=cutoff)
        )

        # Despite the cutoff, both books were imported, because they were new.
        eq_(2, len(imported_editions_mw))

        # but pools and works weren't, because we passed the wrong data source
        # 1 error message, because correctly didn't even get to trying to create pools, 
        # so no messages there, but do have that entry stub at end of sample xml file, 
        # which should fail with a message.
        eq_(1, len(error_messages_mw))
        eq_(0, len(imported_pools_mw))
        eq_(0, len(imported_works_mw))

        # try again, with a license pool-acceptable data source
        importer_g = OPDSImporter(self._db, data_source_name=DataSource.GUTENBERG)
        imported_editions_g, imported_pools_g, imported_works_g, error_messages_g, next_links_g = (
            importer_g.import_from_feed(feed, cutoff_date=cutoff)
        )

        # we made new editions, because we're now creating edition per data source, not overwriting
        eq_(2, len(imported_editions_g))
        # TODO: and we also created presentation editions, with author and title set

        # now pools and works are in, too
        eq_(1, len(error_messages_g))
        eq_(2, len(imported_pools_g))
        eq_(2, len(imported_works_g))        

        # assert that bibframe datasource from feed was correctly overwritten
        # with data source I passed into the importer.
        for pool in imported_pools_g:
            eq_(pool.data_source.name, DataSource.GUTENBERG)



    def test_import_with_cutoff(self):
        cutoff = datetime.datetime(2016, 1, 2, 16, 56, 40)
        path = os.path.join(self.resource_path, "content_server_mini.opds")
        feed = open(path).read()
        importer = OPDSImporter(self._db, data_source_name=DataSource.GUTENBERG)
        imported_editions, imported_pools, imported_works, error_messages, next_links = (
            importer.import_from_feed(feed, cutoff_date=cutoff)
        )

        # Despite the cutoff, both books were imported, because they were new.
        eq_(2, len(imported_editions))
        eq_(2, len(imported_pools))
        eq_(2, len(imported_works))        

        # But if we try it again...
        imported_editions, imported_pools, imported_works, error_messages, next_links = (
            importer.import_from_feed(feed, cutoff_date=cutoff)
        )

        # None of the books were imported because they weren't updated
        # after the cutoff.
        eq_(0, len(imported_editions))
        eq_(0, len(imported_pools))
        eq_(0, len(imported_works))

        # And if we change the cutoff...
        # TODO:  we've messed with the cutoff date in import_editions_from_metadata, 
        # and need to fix it before re-activating the assert.
        cutoff = datetime.datetime(2013, 1, 2, 16, 56, 40)
        imported_editions, imported_pools, imported_works, error_messages, next_links = (
            importer.import_from_feed(feed, cutoff_date=cutoff)
        )

        # Both books were imported again.
        eq_(2, len(imported_editions))
        eq_(2, len(imported_pools))
        eq_(2, len(imported_works))

        # TODO: last_checked isn't getting set.  correct behavior or should fix?
        #assert (datetime.datetime.utcnow() - imported_pools[0].last_checked) < datetime.timedelta(seconds=10)


    def test_import_updates_metadata(self):

        path = os.path.join(self.resource_path, "metadata_wrangler_overdrive.opds")
        feed = open(path).read()

        edition, is_new = self._edition(
            DataSource.OVERDRIVE, Identifier.OVERDRIVE_ID,
            with_license_pool=True
        )
        edition.license_pool.calculate_work()
        work = edition.license_pool.work

        old_license_pool = edition.license_pool
        feed = feed.replace("{OVERDRIVE ID}", edition.primary_identifier.identifier)

        imported_editions, imported_pools, imported_works, error_messages, next_links = (
            OPDSImporter(self._db, data_source_name=DataSource.OVERDRIVE).import_from_feed(feed)
        )

        # The edition we created has had its metadata updated.
        eq_(imported_editions[0], edition)
        eq_("The Green Mouse", imported_editions[0].title)

        # But the license pools have not changed.
        eq_(edition.license_pool, old_license_pool)
        eq_(work.license_pools, [old_license_pool])


    def test_import_from_license_source(self):
        # Instead of importing this data as though it came from the
        # metadata wrangler, let's import it as though it came from the
        # open-access content server.
        path = os.path.join(self.resource_path, "content_server_mini.opds")
        feed = open(path).read()
        importer = OPDSImporter(
            self._db, data_source_name=DataSource.OA_CONTENT_SERVER
        )

        imported_editions, imported_pools, imported_works, error_messages, next_links = (
            importer.import_from_feed(feed)
        )

        [crow, mouse] = sorted(imported_editions, key=lambda x: x.title)

        # Because the content server actually tells you how to get a
        # copy of the 'mouse' book, a work and licensepool have been
        # created for it.
        assert mouse.license_pool != None
        assert mouse.license_pool.work != None

        # The OPDS importer knows that the content server aggregates
        # books from elsewhere, so the data source for the 'mouse'
        # Edition is the underlying license source -- Project
        # Gutenberg -- not the content server.
        eq_(DataSource.GUTENBERG, mouse.data_source.name)

        # Since the 'mouse' book came with an open-access link, the license
        # pool has been marked as open access.
        eq_(True, mouse.license_pool.open_access)
        eq_(RightsStatus.GENERIC_OPEN_ACCESS, 
            mouse.license_pool.rights_status.uri)

        # The 'mouse' work has not been marked presentation-ready,
        # because the OPDS importer was not told to make works
        # presentation-ready as they're imported.
        eq_(False, mouse.license_pool.work.presentation_ready)

        # The OPDS feed didn't actually say where the 'crow' book
        # comes from, so no Work or LicensePool have been created for
        # it, and its data source is the open access content server,
        # not Project Gutenberg.
        eq_(None, crow.work)
        eq_(None, crow.license_pool)
        eq_(DataSource.OA_CONTENT_SERVER, crow.data_source.name)

    def test_import_and_make_presentation_ready(self):
        # Now let's tell the OPDS importer to make works presentation-ready
        # as soon as they're imported.
        path = os.path.join(self.resource_path, "content_server_mini.opds")
        feed = open(path).read()
        importer = OPDSImporter(
            self._db, data_source_name=DataSource.OA_CONTENT_SERVER
        )
        #imported, messages, next_links = importer.import_from_feed(
        #    feed, immediately_presentation_ready=True
        #)
        imported_editions, imported_pools, imported_works, error_messages, next_link = (
            importer.import_from_feed(feed, immediately_presentation_ready=True)
        )

        [crow, mouse] = sorted(imported_editions, key=lambda x: x.title)

        # Nothing happens for the 'crow' book.
        eq_(None, crow.work)
        
        # But the 'mouse' book has had a presentation-ready work
        # created for it.
        eq_(True, mouse.license_pool.work.presentation_ready)


    def test_status_and_message(self):
        path = os.path.join(self.resource_path, "unrecognized_identifier.opds")
        feed = open(path).read()
        #imported, messages, next_link = OPDSImporter(self._db).import_from_feed(feed)
        imported_editions, imported_pools, imported_works, error_messages, next_links = (
            OPDSImporter(self._db).import_from_feed(feed)
        )

        [message] = error_messages.values()
        eq_(404, message.status_code)
        eq_("I've never heard of this work.", message.message)


    def test_import_failure_becomes_status_message(self):
        # Make sure that an exception during import stops the import process, 
        # and generates a meaningful error message.

        class DoomedOPDSImporter(OPDSImporter):
            def import_edition_from_metadata(self, metadata, *args):
                if metadata.title == "Johnny Crow's Party":
                    # This import succeeds.
                    return super(DoomedOPDSImporter, self).import_edition_from_metadata(metadata, *args)
                else:
                    # Any other import fails.
                    raise Exception("Utter failure!")
        path = os.path.join(self.resource_path, "content_server_mini.opds")
        feed = open(path).read()
        #imported, messages, next_links = DoomedOPDSImporter(self._db).import_from_feed(feed)
        imported_editions, imported_pools, imported_works, error_messages, next_links = (
            DoomedOPDSImporter(self._db).import_from_feed(feed)
        )

        # No books were imported.
        eq_(0, len(imported_editions))

        # The other failed to import, and became a StatusMessage
        message = error_messages['http://www.gutenberg.org/ebooks/10441']
        eq_(500, message.status_code)
        assert "Utter failure!" in message.message


    def test_consolidate_links(self):

        links = [LinkData(href=self._url, rel=rel, media_type="image/jpeg")
                 for rel in [Hyperlink.OPEN_ACCESS_DOWNLOAD,
                             Hyperlink.IMAGE,
                             Hyperlink.THUMBNAIL_IMAGE,
                             Hyperlink.OPEN_ACCESS_DOWNLOAD]
        ]
        old_link = links[2]
        links = OPDSImporter.consolidate_links(links)
        eq_([Hyperlink.OPEN_ACCESS_DOWNLOAD,
             Hyperlink.IMAGE,
             Hyperlink.OPEN_ACCESS_DOWNLOAD], [x.rel for x in links])
        link = links[1]
        eq_(old_link, link.thumbnail)

        links = [LinkData(href=self._url, rel=rel, media_type="image/jpeg")
                 for rel in [Hyperlink.THUMBNAIL_IMAGE,
                             Hyperlink.IMAGE,
                             Hyperlink.THUMBNAIL_IMAGE,
                             Hyperlink.IMAGE]
        ]
        t1, i1, t2, i2 = links
        links = OPDSImporter.consolidate_links(links)
        eq_([Hyperlink.IMAGE,
             Hyperlink.IMAGE], [x.rel for x in links])
        eq_(t1, i1.thumbnail)
        eq_(t2, i2.thumbnail)

        links = [LinkData(href=self._url, rel=rel, media_type="image/jpeg")
                 for rel in [Hyperlink.THUMBNAIL_IMAGE,
                             Hyperlink.IMAGE,
                             Hyperlink.IMAGE]
        ]
        t1, i1, i2 = links
        links = OPDSImporter.consolidate_links(links)
        eq_([Hyperlink.IMAGE,
             Hyperlink.IMAGE], [x.rel for x in links])
        eq_(t1, i1.thumbnail)
        eq_(None, i2.thumbnail)



class TestOPDSImporterWithS3Mirror(OPDSImporterTest):

    def test_resources_are_mirrored_on_import(self):

        svg = """<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">

<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">
    <ellipse cx="50" cy="25" rx="50" ry="25" style="fill:blue;"/>
</svg>"""

        http = DummyHTTPClient()
        http.queue_response(
            200, content='I am 10557.epub.images',
            media_type=Representation.EPUB_MEDIA_TYPE,
        )
        http.queue_response(
            200, content=svg, media_type=Representation.SVG_MEDIA_TYPE
        )
        http.queue_response(
            200, content='I am 10441.epub.images',
            media_type=Representation.EPUB_MEDIA_TYPE
        )

        s3 = DummyS3Uploader()

        importer = OPDSImporter(
            self._db, data_source_name=DataSource.OA_CONTENT_SERVER,
            mirror=s3, http_get=http.do_get
        )

        #[e1, e2], messages, next_link = importer.import_from_feed(self.content_server_mini_feed)
        imported_editions, imported_pools, imported_works, error_messages, next_links = (
            importer.import_from_feed(self.content_server_mini_feed)
        )
        e1 = imported_editions[1]
        e2 = imported_editions[0]

        # The import process requested each remote resource in the
        # order they appeared in the OPDS feed. The thumbnail
        # image was not requested, since we were going to make our own
        # thumbnail anyway.
        eq_(http.requests, [
            'https://s3.amazonaws.com/book-covers.nypl.org/Gutenberg-Illustrated/10441/cover_10441_9.png', 
            'http://www.gutenberg.org/ebooks/10441.epub.images',
            'http://www.gutenberg.org/ebooks/10557.epub.images',
        ])

        [e1_oa_link] = e1.primary_identifier.links
        [e2_oa_link, e2_image_link, e2_description_link] = sorted(
            e2.primary_identifier.links, key=lambda x: x.rel
        )

        # The two open-access links were mirrored to S3, as was the
        # original SVG image and its PNG thumbnail.
        set_trace()
        eq_(
            [e1_oa_link.resource.representation,
             e2_image_link.resource.representation,
             e2_image_link.resource.representation.thumbnails[0],
             e2_oa_link.resource.representation,
         ],
            s3.uploaded
        )

        # Each resource was 'mirrored' to an Amazon S3 bucket.
        # The first resource has no bibframe provider in OPDS so it uses the importer's data source.
        eq_(
            ['http://s3.amazonaws.com/test.content.bucket/Library%20Simplified%20Open%20Access%20Content%20Server/Gutenberg%20ID/10557/Johnny%20Crow%27s%20Party.epub.images',
             'http://s3.amazonaws.com/test.cover.bucket/Library%20Simplified%20Open%20Access%20Content%20Server/Gutenberg%20ID/10441/cover_10441_9.png', 
             'http://s3.amazonaws.com/test.cover.bucket/scaled/300/Library%20Simplified%20Open%20Access%20Content%20Server/Gutenberg%20ID/10441/cover_10441_9.png', 
             'http://s3.amazonaws.com/test.content.bucket/Library%20Simplified%20Open%20Access%20Content%20Server/Gutenberg%20ID/10441/The%20Green%20Mouse.epub.images'
         ],
            [x.mirror_url for x in s3.uploaded]
        )

        # If we fetch the feed again, and the entries have been updated since the
        # cutoff, but the content of the open access links hasn't changed, we won't mirror
        # them again.
        cutoff = datetime.datetime(2013, 1, 2, 16, 56, 40)

        http.queue_response(
            304, media_type=Representation.EPUB_MEDIA_TYPE
        )

        http.queue_response(
            304, media_type=Representation.SVG_MEDIA_TYPE
        )

        http.queue_response(
            304, media_type=Representation.EPUB_MEDIA_TYPE
        )

        imported, messages, next_link = importer.import_from_feed(self.content_server_mini_feed, cutoff_date=cutoff)

        eq_([e1, e2], imported)
        # Nothing new has been uploaded
        eq_(4, len(s3.uploaded))

        # If the content has changed, it will be mirrored again.
        http.queue_response(
            200, content="I am a new version of 10557.epub.images",
            media_type=Representation.EPUB_MEDIA_TYPE
        )

        http.queue_response(
            200, content=svg,
            media_type=Representation.SVG_MEDIA_TYPE
        )

        http.queue_response(
            200, content="I am a new version of 10441.epub.images",
            media_type=Representation.EPUB_MEDIA_TYPE
        )

        imported, messages, next_link = importer.import_from_feed(self.content_server_mini_feed, cutoff_date=cutoff)

        eq_([e1, e2], imported)
        eq_(8, len(s3.uploaded))
