from lxml import etree

from api.kansas_patron import KansasAuthenticationAPI
from core.model import ExternalIntegration
from core.testing import DatabaseTest

from . import sample_data


class MockResponse(object):
    def __init__(self, content):
        self.status_code = 200
        self.content = content


class MockAPI(KansasAuthenticationAPI):
    def __init__(self, library_id, integration, analytics=None, base_url=None):
        super(MockAPI, self).__init__(library_id, integration, analytics, base_url)
        self.queue = []

    def sample_data(self, filename):
        return sample_data(filename, "kansas_patron")

    def enqueue(self, filename):
        data = self.sample_data(filename)
        self.queue.append(data)

    def post_request(self, data):
        response = self.queue[0]
        self.queue = self.queue[1:]
        return MockResponse(response)


class TestKansasPatronAPI(DatabaseTest):
    def setup_method(self):
        super(TestKansasPatronAPI, self).setup_method()
        self.integration = self._external_integration(
            ExternalIntegration.PATRON_AUTH_GOAL
        )
        self.api = MockAPI(
            self._default_library, self.integration, base_url="http://test.com"
        )

    def test_request(self):
        request = KansasAuthenticationAPI.create_authorize_request("12345", "6666")
        mock_request = sample_data("authorize_request.xml", "kansas_patron")
        parser = etree.XMLParser(remove_blank_text=True)
        mock_request = etree.tostring(etree.fromstring(mock_request, parser=parser))
        assert request == mock_request

    def test_parse_response(self):
        response = sample_data("authorization_response_good.xml", "kansas_patron")
        authorized, patron_name, library_identifier = self.api.parse_authorize_response(
            response
        )
        assert authorized == True
        assert patron_name == "Montgomery Burns"
        assert library_identifier == "-2"

        response = sample_data("authorization_response_bad.xml", "kansas_patron")
        authorized, patron_name, library_identifier = self.api.parse_authorize_response(
            response
        )
        assert authorized == False
        assert patron_name == "Jay Gee"
        assert library_identifier == "12"

        response = sample_data("authorization_response_no_status.xml", "kansas_patron")
        authorized, patron_name, library_identifier = self.api.parse_authorize_response(
            response
        )
        assert authorized == False
        assert patron_name == "Simpson"
        assert library_identifier == "test"

        response = sample_data("authorization_response_no_id.xml", "kansas_patron")
        authorized, patron_name, library_identifier = self.api.parse_authorize_response(
            response
        )
        assert authorized == True
        assert patron_name == "Gee"
        assert library_identifier == None

        response = sample_data("authorization_response_empty_tag.xml", "kansas_patron")
        authorized, patron_name, library_identifier = self.api.parse_authorize_response(
            response
        )
        assert authorized == False
        assert patron_name == None
        assert library_identifier == "0"

    def test_remote_authenticate(self):
        self.api.enqueue("authorization_response_good.xml")
        patrondata = self.api.remote_authenticate("1234", "4321")
        assert patrondata.authorization_identifier == "1234"
        assert patrondata.permanent_id == "1234"
        assert patrondata.library_identifier == "-2"
        assert patrondata.personal_name == "Montgomery Burns"

        self.api.enqueue("authorization_response_bad.xml")
        patrondata = self.api.remote_authenticate("1234", "4321")
        assert patrondata == False

        self.api.enqueue("authorization_response_no_status.xml")
        patrondata = self.api.remote_authenticate("1234", "4321")
        assert patrondata == False

        self.api.enqueue("authorization_response_no_id.xml")
        patrondata = self.api.remote_authenticate("1234", "4321")
        assert patrondata.authorization_identifier == "1234"
        assert patrondata.permanent_id == "1234"
        assert patrondata.library_identifier == None
        assert patrondata.personal_name == "Gee"

        self.api.enqueue("authorization_response_empty_tag.xml")
        patrondata = self.api.remote_authenticate("1234", "4321")
        assert patrondata == False

        self.api.enqueue("authorization_response_malformed.xml")
        patrondata = self.api.remote_authenticate("1234", "4321")
        assert patrondata == False
