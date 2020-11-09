import unittest
import logging
import socket
from datetime import datetime
from src.lib.rltrace.trace import Trace
from src.lib.settings import Settings
from src.lib.filestream import FileStream
from src.lib.transformer import Transformer


class TestSettings(unittest.TestCase):
    _id = 1

    class DummyStream:
        pass

    class NoneStream:
        def __call__(self, *args, **kwargs):
            return None

    @classmethod
    def setUpClass(cls):
        Trace()

    def setUp(self) -> None:
        logging.info("\n\n- - - - - - C A S E  {} - - - - - -\n\n".format(TestSettings._id))
        TestSettings._id += 1
        return

    def tearDown(self) -> None:
        return

    def test_bad_stream(self):
        with self.assertRaises(ValueError):
            _ = Settings(None)
        return

    def test_none_stream(self):
        with self.assertRaises(ValueError):
            _ = Settings(TestSettings.NoneStream())
        return

    def test_not_callable(self):
        with self.assertRaises(ValueError):
            _ = Settings(TestSettings.DummyStream())
        return

    def test_bad_yaml(self):
        with self.assertRaises(Settings.BadYamlError):
            _ = Settings(FileStream("bad_settings.yml"))
        return

    def test_empty_yaml(self):
        with self.assertRaises(Settings.BadYamlError):
            _ = Settings(FileStream("empty_settings.yml"))
        return

    def test_wrong_tags_yaml(self):
        with self.assertRaises(Settings.BadYamlError):
            _ = Settings(FileStream("wrong_tag_settings.yml"))
        return

    def test_settings(self):
        """
        Simple test to verify that all fields are pulled back like for like
        """
        cases = [None,
                 dict(),
                 {'kafka': ['host', 'port', 'msg_map_url'],
                  'zookeeper': ['host', 'zport']}]
        for sections in cases:
            settings = Settings(settings_yaml_stream=FileStream("settings_test.yml"),
                                sections=sections)
            self.assertEqual(settings.description, "Test Settings")
            self.assertEqual(settings.version, "1.2.3")
            self.assertEqual(settings.date, datetime.strptime("06 Jun 2020", "%d %b %Y"))
            host, port, url = settings.kafka()  # Func is dynamically added
            self.assertEqual(host, "kafka-host-name")
            self.assertEqual(port, "3142")
            self.assertEqual(url, "https://url/file.yml")
            self.assertEqual(settings.kafka_host, "kafka-host-name")
            self.assertEqual(settings.kafka_port, "3142")
            self.assertEqual(settings.kafka_msg_map_url, "https://url/file.yml")
            host, zport, = settings.zookeeper()  # Func is dynamically added
            self.assertEqual(host, "zookeeper-host-name")
            self.assertEqual(zport, "6284")
            self.assertEqual(settings.zookeeper_host, "zookeeper-host-name")
            self.assertEqual(settings.zookeeper_zport, "6284")
        return

    def test_settings_current_host(self):
        """
        Test the <current-host> capability where the special marker <current-host> is replaced with the IP
        of the host on which the process is running
        """
        sections = {'kafka': ['host', 'port', 'msg_map_url']}
        settings = Settings(settings_yaml_stream=FileStream("settings_current_host.yml"),
                            sections=sections)
        host, _, _ = settings.kafka()
        current_host = socket.gethostbyname(socket.gethostname())
        self.assertEqual(host, current_host)
        return

    @staticmethod
    def _replace_git_branch(s: str) -> str:
        return s.replace('<git-branch>', 'master', 1)

    def test_setting_with_bespoke_transforms(self):
        """
        Test the host host swap + an additional transformer that finds and replaces <git-branch> marker
        with 'master'
        """
        transformer = Transformer.Transform(regular_expression=".*<git-branch>.*",
                                            transform=TestSettings._replace_git_branch)
        sections = {'kafka': ['host', 'port', 'msg_map_url']}
        settings = Settings(settings_yaml_stream=FileStream("settings_transform.yml"),
                            sections=sections,
                            bespoke_transforms=[transformer])
        host, _, url = settings.kafka()
        current_host = socket.gethostbyname(socket.gethostname())
        self.assertEqual(host, current_host)
        self.assertEqual("https://url/master/file.yml", url)
        return


if __name__ == "__main__":
    unittest.main()
