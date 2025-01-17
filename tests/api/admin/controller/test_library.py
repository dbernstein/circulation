import base64
import datetime
import json
from io import BytesIO

import flask
import pytest
from PIL import Image
from werkzeug.datastructures import MultiDict

from api.admin.announcement_list_validator import AnnouncementListValidator
from api.admin.controller.library_settings import LibrarySettingsController
from api.admin.exceptions import *
from api.admin.geographic_validator import GeographicValidator
from api.announcements import Announcement, Announcements
from api.config import Configuration
from api.testing import AnnouncementTest
from core.facets import FacetConstants
from core.model import (
    AdminRole,
    ConfigurationSetting,
    Library,
    get_one,
    get_one_or_create,
)
from core.util.problem_detail import ProblemDetail

from .test_controller import SettingsControllerTest


class TestLibrarySettings(SettingsControllerTest, AnnouncementTest):
    @pytest.fixture()
    def logo_properties(self):
        image_data_raw = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x01\x03\x00\x00\x00%\xdbV\xca\x00\x00\x00\x06PLTE\xffM\x00\x01\x01\x01\x8e\x1e\xe5\x1b\x00\x00\x00\x01tRNS\xcc\xd24V\xfd\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
        image_data_b64_bytes = base64.b64encode(image_data_raw)
        image_data_b64_unicode = image_data_b64_bytes.decode("utf-8")
        data_url = "data:image/png;base64," + image_data_b64_unicode
        image = Image.open(BytesIO(image_data_raw))
        return {
            "raw_bytes": image_data_raw,
            "base64_bytes": image_data_b64_bytes,
            "base64_unicode": image_data_b64_unicode,
            "data_url": data_url,
            "image": image,
        }

    def library_form(self, library, fields={}):

        defaults = {
            "uuid": library.uuid,
            "name": "The New York Public Library",
            "short_name": library.short_name,
            Configuration.WEBSITE_URL: "https://library.library/",
            Configuration.HELP_EMAIL: "help@example.com",
            Configuration.DEFAULT_NOTIFICATION_EMAIL_ADDRESS: "email@example.com",
        }
        defaults.update(fields)
        form = MultiDict(list(defaults.items()))
        return form

    def test_libraries_get_with_no_libraries(self):
        # Delete any existing library created by the controller test setup.
        library = get_one(self._db, Library)
        if library:
            self._db.delete(library)

        with self.app.test_request_context("/"):
            response = self.manager.admin_library_settings_controller.process_get()
            assert response.get("libraries") == []

    def test_libraries_get_with_geographic_info(self):
        # Delete any existing library created by the controller test setup.
        library = get_one(self._db, Library)
        if library:
            self._db.delete(library)

        test_library = self._library("Library 1", "L1")
        ConfigurationSetting.for_library(
            Configuration.LIBRARY_FOCUS_AREA, test_library
        ).value = '{"CA": ["N3L"], "US": ["11235"]}'
        ConfigurationSetting.for_library(
            Configuration.LIBRARY_SERVICE_AREA, test_library
        ).value = '{"CA": ["J2S"], "US": ["31415"]}'

        with self.request_context_with_admin("/"):
            response = self.manager.admin_library_settings_controller.process_get()
            library_settings = response.get("libraries")[0].get("settings")
            assert library_settings.get("focus_area") == {
                "CA": [{"N3L": "Paris, Ontario"}],
                "US": [{"11235": "Brooklyn, NY"}],
            }
            assert library_settings.get("service_area") == {
                "CA": [{"J2S": "Saint-Hyacinthe Southwest, Quebec"}],
                "US": [{"31415": "Savannah, GA"}],
            }

    def test_libraries_get_with_announcements(self):
        # Delete any existing library created by the controller test setup.
        library = get_one(self._db, Library)
        if library:
            self._db.delete(library)

        # Set some announcements for this library.
        test_library = self._library("Library 1", "L1")
        ConfigurationSetting.for_library(
            Announcements.SETTING_NAME, test_library
        ).value = json.dumps([self.active, self.expired, self.forthcoming])

        # When we request information about this library...
        with self.request_context_with_admin("/"):
            response = self.manager.admin_library_settings_controller.process_get()
            library_settings = response.get("libraries")[0].get("settings")

            # We find out about the library's announcements.
            announcements = library_settings.get(Announcements.SETTING_NAME)
            assert [self.active["id"], self.expired["id"], self.forthcoming["id"]] == [
                x.get("id") for x in json.loads(announcements)
            ]

            # The objects found in `library_settings` aren't exactly
            # the same as what is stored in the database: string dates
            # can be parsed into datetime.date objects.
            for i in json.loads(announcements):
                assert isinstance(
                    datetime.datetime.strptime(i.get("start"), "%Y-%m-%d"),
                    datetime.date,
                )
                assert isinstance(
                    datetime.datetime.strptime(i.get("finish"), "%Y-%m-%d"),
                    datetime.date,
                )

    def test_libraries_get_with_multiple_libraries(self):
        # Delete any existing library created by the controller test setup.
        library = get_one(self._db, Library)
        if library:
            self._db.delete(library)

        l1 = self._library("Library 1", "L1")
        l2 = self._library("Library 2", "L2")
        l3 = self._library("Library 3", "L3")
        # L2 has some additional library-wide settings.
        ConfigurationSetting.for_library(Configuration.FEATURED_LANE_SIZE, l2).value = 5
        ConfigurationSetting.for_library(
            Configuration.DEFAULT_FACET_KEY_PREFIX
            + FacetConstants.ORDER_FACET_GROUP_NAME,
            l2,
        ).value = FacetConstants.ORDER_TITLE
        ConfigurationSetting.for_library(
            Configuration.ENABLED_FACETS_KEY_PREFIX
            + FacetConstants.ORDER_FACET_GROUP_NAME,
            l2,
        ).value = json.dumps([FacetConstants.ORDER_TITLE, FacetConstants.ORDER_AUTHOR])
        ConfigurationSetting.for_library(
            Configuration.LARGE_COLLECTION_LANGUAGES, l2
        ).value = json.dumps(["French"])
        # The admin only has access to L1 and L2.
        self.admin.remove_role(AdminRole.SYSTEM_ADMIN)
        self.admin.add_role(AdminRole.LIBRARIAN, l1)
        self.admin.add_role(AdminRole.LIBRARY_MANAGER, l2)

        with self.request_context_with_admin("/"):
            response = self.manager.admin_library_settings_controller.process_get()
            libraries = response.get("libraries")
            assert 2 == len(libraries)

            assert l1.uuid == libraries[0].get("uuid")
            assert l2.uuid == libraries[1].get("uuid")

            assert l1.name == libraries[0].get("name")
            assert l2.name == libraries[1].get("name")

            assert l1.short_name == libraries[0].get("short_name")
            assert l2.short_name == libraries[1].get("short_name")

            assert {} == libraries[0].get("settings")
            assert 4 == len(libraries[1].get("settings").keys())
            settings = libraries[1].get("settings")
            assert "5" == settings.get(Configuration.FEATURED_LANE_SIZE)
            assert FacetConstants.ORDER_TITLE == settings.get(
                Configuration.DEFAULT_FACET_KEY_PREFIX
                + FacetConstants.ORDER_FACET_GROUP_NAME
            )
            assert [
                FacetConstants.ORDER_TITLE,
                FacetConstants.ORDER_AUTHOR,
            ] == settings.get(
                Configuration.ENABLED_FACETS_KEY_PREFIX
                + FacetConstants.ORDER_FACET_GROUP_NAME
            )
            assert ["French"] == settings.get(Configuration.LARGE_COLLECTION_LANGUAGES)

    def test_libraries_post_errors(self):
        with self.request_context_with_admin("/", method="POST"):
            flask.request.form = MultiDict(
                [
                    ("name", "Brooklyn Public Library"),
                ]
            )
            response = self.manager.admin_library_settings_controller.process_post()
            assert response == MISSING_LIBRARY_SHORT_NAME

        library = self._library()
        with self.request_context_with_admin("/", method="POST"):
            flask.request.form = self.library_form(library, {"uuid": "1234"})
            response = self.manager.admin_library_settings_controller.process_post()
            assert response.uri == LIBRARY_NOT_FOUND.uri

        with self.request_context_with_admin("/", method="POST"):
            flask.request.form = MultiDict(
                [
                    ("name", "Brooklyn Public Library"),
                    ("short_name", library.short_name),
                ]
            )
            response = self.manager.admin_library_settings_controller.process_post()
            assert response == LIBRARY_SHORT_NAME_ALREADY_IN_USE

        bpl, ignore = get_one_or_create(self._db, Library, short_name="bpl")
        with self.request_context_with_admin("/", method="POST"):
            flask.request.form = MultiDict(
                [
                    ("uuid", bpl.uuid),
                    ("name", "Brooklyn Public Library"),
                    ("short_name", library.short_name),
                ]
            )
            response = self.manager.admin_library_settings_controller.process_post()
            assert response == LIBRARY_SHORT_NAME_ALREADY_IN_USE

        with self.request_context_with_admin("/", method="POST"):
            flask.request.form = MultiDict(
                [
                    ("uuid", library.uuid),
                    ("name", "The New York Public Library"),
                    ("short_name", library.short_name),
                ]
            )
            response = self.manager.admin_library_settings_controller.process_post()
            assert response.uri == INCOMPLETE_CONFIGURATION.uri

        # Test a web primary and secondary color that doesn't contrast
        # well on white. Here primary will, secondary should not.
        with self.request_context_with_admin("/", method="POST"):
            flask.request.form = self.library_form(
                library,
                {
                    Configuration.WEB_PRIMARY_COLOR: "#000000",
                    Configuration.WEB_SECONDARY_COLOR: "#e0e0e0",
                },
            )
            response = self.manager.admin_library_settings_controller.process_post()
            assert response.uri == INVALID_CONFIGURATION_OPTION.uri
            assert "contrast-ratio.com/#%23e0e0e0-on-%23ffffff" in response.detail
            assert "contrast-ratio.com/#%23e0e0e0-on-%23ffffff" in response.detail

        # Test a list of web header links and a list of labels that
        # aren't the same length.
        library = self._library()
        with self.request_context_with_admin("/", method="POST"):
            flask.request.form = MultiDict(
                [
                    ("uuid", library.uuid),
                    ("name", "The New York Public Library"),
                    ("short_name", library.short_name),
                    (Configuration.WEBSITE_URL, "https://library.library/"),
                    (
                        Configuration.DEFAULT_NOTIFICATION_EMAIL_ADDRESS,
                        "email@example.com",
                    ),
                    (Configuration.HELP_EMAIL, "help@example.com"),
                    (Configuration.WEB_HEADER_LINKS, "http://library.com/1"),
                    (Configuration.WEB_HEADER_LINKS, "http://library.com/2"),
                    (Configuration.WEB_HEADER_LABELS, "One"),
                ]
            )
            response = self.manager.admin_library_settings_controller.process_post()
            assert response.uri == INVALID_CONFIGURATION_OPTION.uri

    def test__data_url_for_image(self, logo_properties):
        """"""
        image, expected_data_url = [
            logo_properties[key] for key in ("image", "data_url")
        ]
        data_url = LibrarySettingsController._data_url_for_image(image)
        assert expected_data_url == data_url

    def test_libraries_post_create(self, logo_properties):
        class TestFileUpload(BytesIO):
            headers = {"Content-Type": "image/png"}

        # Pull needed properties from logo fixture
        image_data, expected_logo_data_url, image = [
            logo_properties[key] for key in ("raw_bytes", "data_url", "image")
        ]
        # LibrarySettingsController scales down images that are too large,
        # so we fail here if our test fixture image is large enough to cause
        # a mismatch between the expected data URL and the one configured.
        assert max(*image.size) <= Configuration.LOGO_MAX_DIMENSION

        original_geographic_validate = GeographicValidator().validate_geographic_areas

        class MockGeographicValidator(GeographicValidator):
            def __init__(self):
                self.was_called = False

            def validate_geographic_areas(self, values, db):
                self.was_called = True
                return original_geographic_validate(values, db)

        original_announcement_validate = (
            AnnouncementListValidator().validate_announcements
        )

        class MockAnnouncementListValidator(AnnouncementListValidator):
            def __init__(self):
                self.was_called = False

            def validate_announcements(self, values):
                self.was_called = True
                return original_announcement_validate(values)

        with self.request_context_with_admin("/", method="POST"):
            flask.request.form = MultiDict(
                [
                    ("name", "The New York Public Library"),
                    ("short_name", "nypl"),
                    ("library_description", "Short description of library"),
                    (Configuration.WEBSITE_URL, "https://library.library/"),
                    (Configuration.TINY_COLLECTION_LANGUAGES, ["ger"]),
                    (
                        Configuration.LIBRARY_SERVICE_AREA,
                        ["06759", "everywhere", "MD", "Boston, MA"],
                    ),
                    (
                        Configuration.LIBRARY_FOCUS_AREA,
                        ["Manitoba", "Broward County, FL", "QC"],
                    ),
                    (
                        Announcements.SETTING_NAME,
                        json.dumps([self.active, self.forthcoming]),
                    ),
                    (
                        Configuration.DEFAULT_NOTIFICATION_EMAIL_ADDRESS,
                        "email@example.com",
                    ),
                    (Configuration.HELP_EMAIL, "help@example.com"),
                    (Configuration.FEATURED_LANE_SIZE, "5"),
                    (
                        Configuration.DEFAULT_FACET_KEY_PREFIX
                        + FacetConstants.ORDER_FACET_GROUP_NAME,
                        FacetConstants.ORDER_RANDOM,
                    ),
                    (
                        Configuration.ENABLED_FACETS_KEY_PREFIX
                        + FacetConstants.ORDER_FACET_GROUP_NAME
                        + "_"
                        + FacetConstants.ORDER_TITLE,
                        "",
                    ),
                    (
                        Configuration.ENABLED_FACETS_KEY_PREFIX
                        + FacetConstants.ORDER_FACET_GROUP_NAME
                        + "_"
                        + FacetConstants.ORDER_RANDOM,
                        "",
                    ),
                ]
            )
            flask.request.files = MultiDict(
                [
                    (Configuration.LOGO, TestFileUpload(image_data)),
                ]
            )
            geographic_validator = MockGeographicValidator()
            announcement_validator = MockAnnouncementListValidator()
            validators = dict(
                geographic=geographic_validator,
                announcements=announcement_validator,
            )
            response = self.manager.admin_library_settings_controller.process_post(
                validators
            )
            assert response.status_code == 201

        library = get_one(self._db, Library, short_name="nypl")
        assert library.uuid == response.get_data(as_text=True)
        assert library.name == "The New York Public Library"
        assert library.short_name == "nypl"
        assert (
            "5"
            == ConfigurationSetting.for_library(
                Configuration.FEATURED_LANE_SIZE, library
            ).value
        )
        assert (
            FacetConstants.ORDER_RANDOM
            == ConfigurationSetting.for_library(
                Configuration.DEFAULT_FACET_KEY_PREFIX
                + FacetConstants.ORDER_FACET_GROUP_NAME,
                library,
            ).value
        )
        assert (
            json.dumps([FacetConstants.ORDER_TITLE])
            == ConfigurationSetting.for_library(
                Configuration.ENABLED_FACETS_KEY_PREFIX
                + FacetConstants.ORDER_FACET_GROUP_NAME,
                library,
            ).value
        )
        assert (
            expected_logo_data_url
            == ConfigurationSetting.for_library(Configuration.LOGO, library).value
        )
        assert geographic_validator.was_called == True
        assert (
            '{"US": ["06759", "everywhere", "MD", "Boston, MA"], "CA": []}'
            == ConfigurationSetting.for_library(
                Configuration.LIBRARY_SERVICE_AREA, library
            ).value
        )
        assert (
            '{"US": ["Broward County, FL"], "CA": ["Manitoba", "Quebec"]}'
            == ConfigurationSetting.for_library(
                Configuration.LIBRARY_FOCUS_AREA, library
            ).value
        )

        # Announcements were validated.
        assert announcement_validator.was_called == True

        # The validated result was written to the database, such that we can
        # parse it as a list of Announcement objects.
        announcements = Announcements.for_library(library).announcements
        assert [self.active["id"], self.forthcoming["id"]] == [
            x.id for x in announcements
        ]
        assert all(isinstance(x, Announcement) for x in announcements)

        # When the library was created, default lanes were also created
        # according to its language setup. This library has one tiny
        # collection (not a good choice for a real library), so only
        # two lanes were created: "Other Languages" and then "German"
        # underneath it.
        [german, other_languages] = sorted(library.lanes, key=lambda x: x.display_name)
        assert None == other_languages.parent
        assert ["ger"] == other_languages.languages
        assert other_languages == german.parent
        assert ["ger"] == german.languages

    def test_libraries_post_edit(self):
        # A library already exists.
        library = self._library("New York Public Library", "nypl")

        ConfigurationSetting.for_library(
            Configuration.FEATURED_LANE_SIZE, library
        ).value = 5
        ConfigurationSetting.for_library(
            Configuration.DEFAULT_FACET_KEY_PREFIX
            + FacetConstants.ORDER_FACET_GROUP_NAME,
            library,
        ).value = FacetConstants.ORDER_RANDOM
        ConfigurationSetting.for_library(
            Configuration.ENABLED_FACETS_KEY_PREFIX
            + FacetConstants.ORDER_FACET_GROUP_NAME,
            library,
        ).value = json.dumps([FacetConstants.ORDER_TITLE, FacetConstants.ORDER_RANDOM])
        ConfigurationSetting.for_library(
            Configuration.LOGO, library
        ).value = "A tiny image"

        with self.request_context_with_admin("/", method="POST"):
            flask.request.form = MultiDict(
                [
                    ("uuid", library.uuid),
                    ("name", "The New York Public Library"),
                    ("short_name", "nypl"),
                    (Configuration.FEATURED_LANE_SIZE, "20"),
                    (Configuration.MINIMUM_FEATURED_QUALITY, "0.9"),
                    (Configuration.WEBSITE_URL, "https://library.library/"),
                    (
                        Configuration.DEFAULT_NOTIFICATION_EMAIL_ADDRESS,
                        "email@example.com",
                    ),
                    (Configuration.HELP_EMAIL, "help@example.com"),
                    (
                        Configuration.DEFAULT_FACET_KEY_PREFIX
                        + FacetConstants.ORDER_FACET_GROUP_NAME,
                        FacetConstants.ORDER_AUTHOR,
                    ),
                    (
                        Configuration.ENABLED_FACETS_KEY_PREFIX
                        + FacetConstants.ORDER_FACET_GROUP_NAME
                        + "_"
                        + FacetConstants.ORDER_AUTHOR,
                        "",
                    ),
                    (
                        Configuration.ENABLED_FACETS_KEY_PREFIX
                        + FacetConstants.ORDER_FACET_GROUP_NAME
                        + "_"
                        + FacetConstants.ORDER_RANDOM,
                        "",
                    ),
                ]
            )
            flask.request.files = MultiDict([])
            response = self.manager.admin_library_settings_controller.process_post()
            assert response.status_code == 200

        library = get_one(self._db, Library, uuid=library.uuid)

        assert library.uuid == response.get_data(as_text=True)
        assert library.name == "The New York Public Library"
        assert library.short_name == "nypl"

        # The library-wide settings were updated.
        def val(x):
            return ConfigurationSetting.for_library(x, library).value

        assert "https://library.library/" == val(Configuration.WEBSITE_URL)
        assert "email@example.com" == val(
            Configuration.DEFAULT_NOTIFICATION_EMAIL_ADDRESS
        )
        assert "help@example.com" == val(Configuration.HELP_EMAIL)
        assert "20" == val(Configuration.FEATURED_LANE_SIZE)
        assert "0.9" == val(Configuration.MINIMUM_FEATURED_QUALITY)
        assert FacetConstants.ORDER_AUTHOR == val(
            Configuration.DEFAULT_FACET_KEY_PREFIX
            + FacetConstants.ORDER_FACET_GROUP_NAME
        )
        assert json.dumps([FacetConstants.ORDER_AUTHOR]) == val(
            Configuration.ENABLED_FACETS_KEY_PREFIX
            + FacetConstants.ORDER_FACET_GROUP_NAME
        )

        # The library-wide logo was not updated and has been left alone.
        assert (
            "A tiny image"
            == ConfigurationSetting.for_library(Configuration.LOGO, library).value
        )

    def test_library_delete(self):
        library = self._library()

        with self.request_context_with_admin("/", method="DELETE"):
            self.admin.remove_role(AdminRole.SYSTEM_ADMIN)
            pytest.raises(
                AdminNotAuthorized,
                self.manager.admin_library_settings_controller.process_delete,
                library.uuid,
            )

            self.admin.add_role(AdminRole.SYSTEM_ADMIN)
            response = self.manager.admin_library_settings_controller.process_delete(
                library.uuid
            )
            assert response.status_code == 200

        library = get_one(self._db, Library, uuid=library.uuid)
        assert None == library

    def test_library_configuration_settings(self):
        # Verify that library_configuration_settings validates and updates every
        # setting for a library.
        settings = [
            dict(key="setting1", format="format1"),
            dict(key="setting2", format="format2"),
        ]

        # format1 has a custom validation class; format2 does not.
        class MockValidator(object):
            def format_as_string(self, value):
                self.format_as_string_called_with = value
                return value + ", formatted for storage"

        validator1 = MockValidator()
        validators = dict(format1=validator1)

        class MockController(LibrarySettingsController):
            succeed = True
            _validate_setting_calls = []

            def _validate_setting(self, library, setting, validator):
                self._validate_setting_calls.append((library, setting, validator))
                if self.succeed:
                    return "validated %s" % setting["key"]
                else:
                    return INVALID_INPUT.detailed("invalid!")

        # Run library_configuration_settings in a situation where all validations succeed.
        controller = MockController(self.manager)
        library = self._default_library
        result = controller.library_configuration_settings(
            library, validators, settings
        )

        # No problem detail was returned -- the 'request' can continue.
        assert None == result

        # _validate_setting was called twice...
        [c1, c2] = controller._validate_setting_calls

        # ...once for each item in `settings`. One of the settings was
        # of a type with a known validator, so the validator was
        # passed in.
        assert (library, settings[0], validator1) == c1
        assert (library, settings[1], None) == c2

        # The 'validated' value from the MockValidator was then formatted
        # for storage using the format() method.
        assert (
            "validated %s" % settings[0]["key"]
            == validator1.format_as_string_called_with
        )

        # Each (validated and formatted) value was written to the
        # database.
        setting1, setting2 = [library.setting(x["key"]) for x in settings]
        assert "validated %s, formatted for storage" % setting1.key == setting1.value
        assert "validated %s" % setting2.key == setting2.value

        # Try again in a situation where there are validation failures.
        setting1.value = None
        setting2.value = None
        controller.succeed = False
        controller._validate_setting_calls = []
        result = controller.library_configuration_settings(
            self._default_library, validators, settings
        )

        # _validate_setting was only called once.
        assert [
            (library, settings[0], validator1)
        ] == controller._validate_setting_calls

        # When it returned a ProblemDetail, that ProblemDetail
        # was propagated outwards.
        assert isinstance(result, ProblemDetail)
        assert "invalid!" == result.detail

        # No new values were written to the database.
        for x in settings:
            assert None == library.setting(x["key"]).value

    def test__validate_setting(self):
        # Verify the rules for validating different kinds of settings,
        # one simulated setting at a time.

        library = self._default_library

        class MockController(LibrarySettingsController):

            # Mock the functions that pull various values out of the
            # 'current request' or the 'database' so we don't need an
            # actual current request or actual database settings.
            def scalar_setting(self, setting):
                return self.scalar_form_values.get(setting["key"])

            def list_setting(self, setting, json_objects=False):
                value = self.list_form_values.get(setting["key"])
                if json_objects:
                    value = [json.loads(x) for x in value]
                return json.dumps(value)

            def image_setting(self, setting):
                return self.image_form_values.get(setting["key"])

            def current_value(self, setting, _library):
                # While we're here, make sure the right Library
                # object was passed in.
                assert _library == library
                return self.current_values.get(setting["key"])

            # Now insert mock data into the 'form submission' and
            # the 'database'.

            # Simulate list values in a form submission. The geographic values
            # go in as normal strings; the announcements go in as strings that are
            # JSON-encoded data structures.
            announcement_list = [
                {"content": "announcement1"},
                {"content": "announcement2"},
            ]
            list_form_values = dict(
                geographic_setting=["geographic values"],
                announcement_list=[json.dumps(x) for x in announcement_list],
                language_codes=["English", "fr"],
                list_value=["a list"],
            )

            # Simulate scalar values in a form submission.
            scalar_form_values = dict(string_value="a scalar value")

            # Simulate uploaded images in a form submission.
            image_form_values = dict(image_setting="some image data")

            # Simulate values present in the database but not present
            # in the form submission.
            current_values = dict(
                value_not_present_in_request="a database value",
                previously_uploaded_image="an old image",
            )

        # First test some simple cases: scalar values.
        controller = MockController(self.manager)
        m = controller._validate_setting

        # The incoming request has a value for this setting.
        assert "a scalar value" == m(library, dict(key="string_value"))

        # But not for this setting: we end up going to the database
        # instead.
        assert "a database value" == m(
            library, dict(key="value_not_present_in_request")
        )

        # And not for this setting either: there is no database value,
        # so we have to use the default associated with the setting configuration.
        assert "a default value" == m(
            library, dict(key="some_other_value", default="a default value")
        )

        # An uploaded image is (from the perspective of this method) also simple.

        # Here, a new image was uploaded.
        assert "some image data" == m(library, dict(key="image_setting", type="image"))

        # Here, no image was uploaded so we use the currently stored database value.
        assert "an old image" == m(
            library, dict(key="previously_uploaded_image", type="image")
        )

        # There are some lists which are more complex, but a normal list is
        # simple: the return value is the JSON-encoded list.
        assert json.dumps(["a list"]) == m(library, dict(key="list_value", type="list"))

        # Now let's look at the more complex lists.

        # A list of language codes.
        assert json.dumps(["eng", "fre"]) == m(
            library, dict(key="language_codes", format="language-code", type="list")
        )

        # A list of geographic places
        class MockGeographicValidator(object):
            value = "validated value"

            def validate_geographic_areas(self, value, _db):
                self.called_with = (value, _db)
                return self.value

        validator = MockGeographicValidator()

        # The validator was consulted and its response was used as the
        # value.
        assert "validated value" == m(
            library, dict(key="geographic_setting", format="geographic"), validator
        )
        assert (json.dumps(["geographic values"]), self._db) == validator.called_with

        # Just to be explicit, let's also test the case where the 'response' sent from the
        # validator is a ProblemDetail.
        validator.value = INVALID_INPUT
        assert INVALID_INPUT == m(
            library, dict(key="geographic_setting", format="geographic"), validator
        )

        # A list of announcements.
        class MockAnnouncementValidator(object):
            value = "validated value"

            def validate_announcements(self, value):
                self.called_with = value
                return self.value

        validator = MockAnnouncementValidator()

        assert "validated value" == m(
            library, dict(key="announcement_list", type="announcements"), validator
        )
        assert json.dumps(controller.announcement_list) == validator.called_with

    def test__format_validated_value(self):

        m = LibrarySettingsController._format_validated_value

        # When there is no validator, the incoming value is used as the formatted value,
        # unchanged.
        value = object()
        assert value == m(value, validator=None)

        # When there is a validator, its format_as_string method is
        # called, and its return value is used as the formatted value.
        class MockValidator(object):
            def format_as_string(self, value):
                self.called_with = value
                return "formatted value"

        validator = MockValidator()
        assert "formatted value" == m(value, validator=validator)
        assert value == validator.called_with
