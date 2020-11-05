import unittest
import time
import numpy as np
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
from src.test.ttt_event_stream.DummyStateFactory import DummyStateFactory
from src.tictactoe.tictacttoe_event import TicTacToeEvent


class TestTTTEventStream(unittest.TestCase):
    _run: int
    _trace: Trace
    _env: Env
    _settings: Settings
    _run_spec: RunSpec
    _es: Elasticsearch
    _sess_uuid: str
    _state_factory: DummyStateFactory

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
        cls._sess_uuid = UniqueRef().ref
        cls._state_factory = DummyStateFactory()

        return

    def setUp(self) -> None:
        TestTTTEventStream._run += 1
        self._trace.log().info("- - - - - - C A S E {} Start - - - - - -".format(TestTTTEventStream._run))
        # ToDo create an ES instance for each test cycle and tear down after.
        return

    def tearDown(self) -> None:
        """
        Delete all documents added for the test session uuid
        """
        self._trace.log().info("- - - - - - C A S E {} Passed - - - - - -".format(TestTTTEventStream._run))

        # Delete all documents created by this test session uuid
        ESUtil.delete_documents(es=TestTTTEventStream._es,
                                idx_name=TestTTTEventStream._settings.ttt_event_index_name,
                                json_query=TestTTTEventStream.SESSION_UUID_Q,
                                arg0=TestTTTEventStream._sess_uuid)
        return

    def test_1(self):
        """
        Test write of single dummy TTT event
        :return:
        """
        tttes = TicTacToeEventStream(es=TestTTTEventStream._es,
                                     es_index=TestTTTEventStream._settings.ttt_event_index_name,
                                     state_factory=self._state_factory,
                                     session_uuid=TestTTTEventStream._sess_uuid)
        tttes.record_event(episode_uuid=UniqueRef().ref,
                           episode_step=1,
                           state=DummyState(),
                           action="1",
                           reward=3.142,
                           episode_end=True,
                           episode_outcome=TicTacToeEvent.STEP)

        TestTTTEventStream._es.indices.flush(index=TestTTTEventStream._settings.ttt_event_index_name)
        time.sleep(1)

        cnt = ESUtil.run_count(es=TestTTTEventStream._es,
                               idx_name=TestTTTEventStream._settings.ttt_event_index_name,
                               json_query=TestTTTEventStream.SESSION_UUID_Q,
                               arg0=TestTTTEventStream._sess_uuid)
        self.assertEqual(1, cnt)
        return

    def test_2(self):
        """
        Test an episode can be created and written back as event objects
        """
        tttes = TicTacToeEventStream(es=TestTTTEventStream._es,
                                     es_index=TestTTTEventStream._settings.ttt_event_index_name,
                                     state_factory=self._state_factory,
                                     session_uuid=TestTTTEventStream._sess_uuid)
        episode_id = UniqueRef().ref
        evnts = list()
        num_to_test = 10
        outcomes = [TicTacToeEvent.X_WIN,
                    TicTacToeEvent.O_WIN,
                    TicTacToeEvent.DRAW]

        for i in range(0, num_to_test, 1):
            eend = (i == num_to_test - 1)
            if eend:
                outc = outcomes[np.random.randint(3)]
            else:
                outc = TicTacToeEvent.STEP

            ttt_e = TicTacToeEvent(episode_uuid=episode_id,
                                   episode_step=i,
                                   state=DummyState(),
                                   action=str(np.random.randint(100)),
                                   reward=np.random.random(),
                                   episode_end=eend,
                                   episode_outcome=outc)
            tttes.record_event(episode_uuid=ttt_e.episode_uuid,
                               episode_step=ttt_e.episode_step,
                               state=ttt_e.state,
                               action=ttt_e.action,
                               reward=ttt_e.reward,
                               episode_end=ttt_e.episode_end,
                               episode_outcome=ttt_e.episode_outcome)
            evnts.append(ttt_e)

        TestTTTEventStream._es.indices.flush(index=TestTTTEventStream._settings.ttt_event_index_name)
        time.sleep(1)

        res = tttes.get_session(session_uuid=tttes.session_uuid)

        self.assertEqual(len(evnts), len(res))

        evnts = sorted(evnts, key=lambda x: x.episode_step)

        for expected, actual in zip(evnts, res):
            self.assertEqual(True, expected == actual)

        return


if __name__ == "__main__":
    unittest.main()