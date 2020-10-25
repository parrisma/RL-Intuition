import time
import unittest
from elasticsearch import Elasticsearch
from src.interface.envbuilder import EnvBuilder
from src.lib.uniqueref import UniqueRef
from src.lib.envboot.env import Env
from src.lib.rltrace.trace import Trace
from src.lib.settings import Settings
from src.lib.webstream import WebStream
from src.lib.envboot.runspec import RunSpec
from src.lib.elastic.esutil import ESUtil
from src.tictactoe.TicTacToeEventStream import TicTacToeEventStream
from src.test.state.dummy_state import DummyState


class TestTTTEventStream(unittest.TestCase):
    _run: int
    _trace: Trace
    _env: Env
    _settings: Settings
    _run_spec: RunSpec
    _es: Elasticsearch

    _run = int(0)
    _trace = None
    _env = None

    SESSION_UUID_Q = '{"query":{"term":{ "session_uuid":"<arg0>"}}}'

    def __init__(self, *args, **kwargs):
        super(TestTTTEventStream, self).__init__(*args, **kwargs)
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
        TestTTTEventStream._run += 1
        self._trace.log().info("- - - - - - C A S E {} Start - - - - - -".format(TestTTTEventStream._run))
        # ToDo create an ES instance for each test cycle and tear down after.
        return

    def tearDown(self) -> None:
        self._trace.log().info("- - - - - - C A S E {} Passed - - - - - -".format(TestTTTEventStream._run))
        self._transformer = None
        return

    def test_1(self):
        """
        Test write of single dummy TTT event
        :return:
        """
        sess_uuid = UniqueRef().ref
        tttes = TicTacToeEventStream(es=TestTTTEventStream._es,
                                     es_index=TestTTTEventStream._settings.ttt_event_index_name,
                                     session_uuid=sess_uuid)
        tttes.record_event(episode_uuid=UniqueRef().ref,
                           episode_step=1,
                           state=DummyState(),
                           action="1",
                           reward=3.142,
                           episode_end=True)
        time.sleep(1)
        cnt = ESUtil.run_count(es=TestTTTEventStream._es,
                               idx_name=TestTTTEventStream._settings.ttt_event_index_name,
                               json_query=TestTTTEventStream.SESSION_UUID_Q,
                               arg0=sess_uuid)
        self.assertEqual(1, cnt)
        return


if __name__ == "__main__":
    unittest.main()
