import unittest
import numpy as np
from src.lib.envboot.env import Env
from src.lib.rltrace.trace import Trace
from src.tictactoe.TicTacToe import TicTacToe
from src.tictactoe.TicTacToeState import TicTacToeState
from src.test.ttt_test.TestAgent import TestAgent
from src.interface.envbuilder import EnvBuilder
from src.lib.settings import Settings
from src.lib.webstream import WebStream
from src.tictactoe.TicTacToeEventStream import TicTacToeEventStream
from src.lib.uniqueref import UniqueRef
from src.tictactoe.TicTacToeStateFactory import TicTacToeStateFactory


#
# Unit Test Suite for the TicTacToe concrete implementation of an Environment.
#


class TestTicTacToe(unittest.TestCase):
    _run: int
    _env: Env
    _trace: Trace
    _state_factory: TicTacToeStateFactory

    _run = int(0)
    _trace = None
    _env = None

    def __init__(self, *args, **kwargs):
        super(TestTicTacToe, self).__init__(*args, **kwargs)
        return

    @classmethod
    def setUpClass(cls):
        cls._env = Env()
        cls._trace = cls._env.get_trace()
        cls._run_spec = cls._env.get_context()[EnvBuilder.RunSpecificationContext]
        cls._es = cls._env.get_context()[EnvBuilder.ElasticDbConnectionContext]
        cls._settings = Settings(settings_yaml_stream=WebStream(cls._run_spec.ttt_settings_yaml()),
                                 bespoke_transforms=cls._run_spec.setting_transformers())
        cls._state_factory = TicTacToeStateFactory(x_agent=TestAgent(-1, "X"), o_agent=TestAgent(1, "O"))
        cls.__ttt_event_stream = TicTacToeEventStream(es=cls._es,
                                                      es_index=cls._settings.ttt_event_index_name,
                                                      state_factory=cls._state_factory,
                                                      session_uuid=UniqueRef().ref)
        return

    def setUp(self) -> None:
        TestTicTacToe._run += 1
        self._trace.log().info("- - - - - - C A S E {} Start - - - - - -".format(TestTicTacToe._run))
        return

    def tearDown(self) -> None:
        self._trace.log().info("- - - - - - C A S E {} Passed - - - - - -".format(TestTicTacToe._run))
        return

    def test_1(self):
        test_cases = [("", False, False, False, TicTacToe.no_agent()),
                      ("1:0", False, False, False, 1),
                      ("1:0~1:1~1:2~-1:3~-1:4~-1:8", True, True, False, -1),
                      ("1:0~-1:1~1:2~1:3~-1:4~-1:5~-1:6~1:7~-1:8", True, False, True, -1),
                      ("1:0~-1:1~1:2~-1:3~1:4~-1:5~1:6~-1:7~1:8", True, True, False, 1)]

        agent_o = TestAgent(1, "O")
        agent_x = TestAgent(-1, "X")
        for test_case, expected1, expected2, expected3, expected4 in test_cases:
            ttt = TicTacToe(env=self._env,
                            ttt_event_stream=self.__ttt_event_stream,
                            x=agent_x,
                            o=agent_o)
            ttt.import_state(test_case)
            self.assertEqual(ttt.episode_complete(), expected1)
            # verify same but where state is supplied.
            tts = ttt.state()
            self.assertEqual(ttt.episode_complete(tts), expected1)

            episode_summary = ttt.attributes()
            self.assertEqual(episode_summary[TicTacToe.attribute_won[0]], expected2)
            self.assertEqual(episode_summary[TicTacToe.attribute_draw[0]], expected3)
            if episode_summary[TicTacToe.attribute_agent[0]] is not None:
                self.assertEqual(episode_summary[TicTacToe.attribute_agent[0]].id(), expected4)
            else:
                self.assertEqual(episode_summary[TicTacToe.attribute_agent[0]], expected4)
            del ttt
        return

    def test_2(self):
        self._trace.log().info("Test case for TicTacToe environment import / export")
        test_cases = ("", "1:0", "-1:0", "-1:1",
                      "-1:0~1:2~-1:4~1:6~-1:8",
                      "1:0~-1:1~1:2~-1:3~1:4~-1:5~1:6~-1:7~1:8")
        agent_o = TestAgent(1, "O")
        agent_x = TestAgent(-1, "X")
        for test_case in test_cases:
            ttt = TicTacToe(env=self._env,
                            ttt_event_stream=self.__ttt_event_stream,
                            x=agent_x,
                            o=agent_o)
            ttt.import_state(test_case)
            self.assertEqual(ttt.export_state(), test_case)
            del ttt
        return

    def test_scripted_win(self):
        """
        The TestAgents have scripted actions that result in a Win for X and Win for O
        """
        cases = [
            [TestAgent(-1, "X", [0, 2, 6, 3]), TestAgent(1, "O", [8, 1, 4]),
             TicTacToeEventStream.TicTacToeEvent.X_WIN, True],
            [TestAgent(-1, "X", [8, 1, 4]), TestAgent(1, "O", [0, 2, 6, 3]),
             TicTacToeEventStream.TicTacToeEvent.O_WIN, False]
        ]
        for case in cases:
            agent_x, agent_o, res, x_to_start = case
            ttt = TicTacToe(env=self._env,
                            ttt_event_stream=self.__ttt_event_stream,
                            x=agent_x,
                            o=agent_o,
                            x_to_start=x_to_start)
            episodes = ttt.run(num_episodes=1)
            self.assertEqual(1, len(episodes))
            events = self.__ttt_event_stream.get_episode(episode_uuid=episodes[0])
            expected_results = [
                [1, '000000000', 0, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [2, '-100000000', 8, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [3, '-100000001', 2, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [4, '-10-1000001', 1, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [5, '-11-1000001', 6, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [6, '-11-1000-101', 4, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [7, '-11-1010-101', 3, TicTacToe.win_reward, True, res]]
            for event, expected in zip(events, expected_results):
                step, state_as_str, action, reward, end, outcome = expected
                self.assertEqual(episodes[0], event.episode_uuid)
                self.assertEqual(step, event.episode_step)
                if not x_to_start:
                    st = event.state.invert_player_perspective()
                else:
                    st = event.state
                self.assertEqual(state_as_str, st.state_as_string())
                self.assertEqual(str(action), event.action)
                self.assertEqual(reward, event.reward)
                self.assertEqual(end, event.episode_end)
                self.assertEqual(outcome, event.episode_outcome)
            del ttt
        return

    def test_scripted_draw(self):
        """
        The TestAgents have scripted actions that result in a Draw with both open from X and O
        """
        cases = [
            [TestAgent(-1, "X", [0, 2, 3, 7, 8]), TestAgent(1, "O", [1, 4, 5, 6]),
             TicTacToeEventStream.TicTacToeEvent.DRAW, True],
            [TestAgent(-1, "X", [1, 4, 5, 6]), TestAgent(1, "O", [0, 2, 3, 7, 8]),
             TicTacToeEventStream.TicTacToeEvent.DRAW, False]
        ]
        for case in cases:
            agent_x, agent_o, res, x_to_start = case
            ttt = TicTacToe(env=self._env,
                            ttt_event_stream=self.__ttt_event_stream,
                            x=agent_x,
                            o=agent_o,
                            x_to_start=x_to_start)
            episodes = ttt.run(num_episodes=1)
            self.assertEqual(1, len(episodes))
            events = self.__ttt_event_stream.get_episode(episode_uuid=episodes[0])
            expected_results = [
                [1, '000000000', 0, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [2, '-100000000', 1, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [3, '-110000000', 2, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [4, '-11-1000000', 4, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [5, '-11-1010000', 3, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [6, '-11-1-110000', 5, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [7, '-11-1-111000', 7, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [8, '-11-1-1110-10', 6, TicTacToe.play_reward, False, TicTacToeEventStream.TicTacToeEvent.STEP],
                [9, '-11-1-1111-10', 8, TicTacToe.draw_reward, True, res]]
            for event, expected in zip(events, expected_results):
                step, state_as_str, action, reward, end, outcome = expected
                self.assertEqual(episodes[0], event.episode_uuid)
                self.assertEqual(step, event.episode_step)
                if not x_to_start:
                    st = event.state.invert_player_perspective()
                else:
                    st = event.state
                self.assertEqual(state_as_str, st.state_as_string())
                self.assertEqual(str(action), event.action)
                self.assertEqual(reward, event.reward)
                self.assertEqual(end, event.episode_end)
                self.assertEqual(outcome, event.episode_outcome)
            del ttt
        return

    def test_tic_tac_toe_state(self):
        ao = TestAgent(1, "O")
        ax = TestAgent(-1, "X")
        id_o = ao.id()
        id_x = ax.id()
        test_cases = [(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])),
                      (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                      (np.array([id_o, id_o, id_o, id_o, id_o, id_o, id_o, id_o, id_o])),
                      (np.array([id_x, id_x, id_x, id_x, id_x, id_x, id_x, id_x, id_x])),
                      (np.array([id_x, id_o, id_x, id_o, id_x, id_o, id_x, id_o, id_x])),
                      (np.array([id_o, id_x, id_o, id_x, id_o, id_x, id_o, id_x, id_o])),
                      (np.array([id_x, 0, id_o, id_x, 0, id_x, np.inf, np.nan, id_o])),
                      (np.array([[id_x, 0, id_o], [id_x, 0, id_x], [np.inf, np.nan, id_o]]))
                      ]
        expected_inv = [(np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])),
                        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])),
                        (np.array([id_x, id_x, id_x, id_x, id_x, id_x, id_x, id_x, id_x])),
                        (np.array([id_o, id_o, id_o, id_o, id_o, id_o, id_o, id_o, id_o])),
                        (np.array([id_o, id_x, id_o, id_x, id_o, id_x, id_o, id_x, id_o])),
                        (np.array([id_x, id_o, id_x, id_o, id_x, id_o, id_x, id_o, id_x])),
                        (np.array([id_o, 0, id_x, id_o, 0, id_o, np.inf, np.nan, id_x])),
                        (np.array([[id_o, 0, id_x], [id_o, 0, id_o], [np.inf, np.nan, id_x]]))
                        ]
        for case, expected1 in zip(test_cases, expected_inv):
            tts = TicTacToeState(board=case, agent_o=ao, agent_x=ax)
            self.assertTrue(tts.state() is not case)  # we expect State to deep-copy the input
            self.assertTrue(self.__np_eq(tts.state(), case))
            self.assertTrue(self.__np_eq(tts.invert_player_perspective().state(), expected1))
        return

    #
    # Are the given arrays equal shape and element by element content. We allow nan = nan as equal.
    #
    @classmethod
    def __np_eq(cls,
                npa1: np.array,
                npa2: np.array) -> bool:
        if np.shape(npa1) != np.shape(npa2):
            return False
        v1 = np.reshape(npa1, np.size(npa1))
        v2 = np.reshape(npa2, np.size(npa2))

        for vl, vr in np.stack((v1, v2), axis=-1):
            if np.isnan(vl):
                if np.isnan(vl) != np.isnan(vr):
                    return False
            else:
                if vl != vr:
                    return False
        return True


if __name__ == "__main__":
    unittest.main()
