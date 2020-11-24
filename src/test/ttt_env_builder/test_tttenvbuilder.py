import unittest
from elasticsearch import Elasticsearch
from src.interface.envbuilder import EnvBuilder
from src.lib.envboot.env import Env
from src.lib.rltrace.trace import Trace
from src.lib.settings import Settings
from src.lib.streams.webstream import WebStream
from src.lib.envboot.runspec import RunSpec
from src.lib.elastic.esutil import ESUtil
from src.tictactoe.experiment.experiment_env_builder import ExperimentEnvBuilder


class TestTTTEnvBuilder(unittest.TestCase):
    _run: int
    _trace: Trace
    _env: Env
    _settings: Settings
    _run_spec: RunSpec
    _es: Elasticsearch

    _run = int(0)
    _trace = None
    _env = None

    def __init__(self, *args, **kwargs):
        super(TestTTTEnvBuilder, self).__init__(*args, **kwargs)
        return

    @classmethod
    def setUpClass(cls):
        cls._env = Env()
        cls._trace = cls._env.get_trace()
        cls._run_spec = cls._env.get_context()[EnvBuilder.RunSpecificationContext]
        cls._es = cls._env.get_context()[EnvBuilder.ElasticDbConnectionContext]
        cls._settings = Settings(settings_yaml_stream=WebStream(cls._run_spec.ttt_settings_yaml()),
                                 bespoke_transforms=cls._run_spec.setting_transformers())
        return

    def setUp(self) -> None:
        TestTTTEnvBuilder._run += 1
        self._trace.log().info("- - - - - - C A S E {} Start - - - - - -".format(TestTTTEnvBuilder._run))
        # ToDo create an ES instance for each test cycle and tear down after.
        return

    def tearDown(self) -> None:
        self._trace.log().info("- - - - - - C A S E {} Passed - - - - - -".format(TestTTTEnvBuilder._run))
        self._transformer = None
        return

    def test_1(self):
        """
        Test ttt_event index is created when TTT Env is bootstrapped.
        :return:
        """
        self._trace.log().info("Test 1")
        ttteb = ExperimentEnvBuilder(TestTTTEnvBuilder._env.get_context())
        ttteb.execute(purge=False)
        self.assertEqual(True,
                         ESUtil.index_exists(TestTTTEnvBuilder._es, TestTTTEnvBuilder._settings.ttt_event_index_name))
        return


if __name__ == "__main__":
    unittest.main()
