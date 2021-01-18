import unittest
from src.lib.envboot.env import Env
from src.lib.rltrace.trace import Trace
from src.interface.envbuilder import EnvBuilder
from src.lib.settings import Settings
from src.lib.streams.webstream import WebStream
from src.tictactoe.experiment2.ex2_cmd import Ex2Cmd
from src.tictactoe.experiment2.ex2_cmd_map import Ex2CmdMap
from src.test.nav.dummy_nav import DummyNav


class TestNav(unittest.TestCase):
    _run: int
    _env: Env
    _trace: Trace

    _run = int(0)
    _trace = None
    _env = None

    def __init__(self, *args, **kwargs):
        super(TestNav, self).__init__(*args, **kwargs)
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
        TestNav._run += 1
        self._trace.log().info("- - - - - - C A S E {} Start - - - - - -".format(TestNav._run))
        return

    def tearDown(self) -> None:
        self._trace.log().info("- - - - - - C A S E {} Passed - - - - - -".format(TestNav._run))
        return

    def test_action_nav(self):
        test_cases = [[Ex2Cmd.Ex2Commands.cmd_0, Ex2Cmd.Ex2Actions.action_0, None],
                      [Ex2Cmd.Ex2Commands.cmd_1, Ex2Cmd.Ex2Actions.action_1, None],
                      [Ex2Cmd.Ex2Commands.cmd_2, Ex2Cmd.Ex2Actions.action_2, None],
                      [Ex2Cmd.Ex2Commands.cmd_3, Ex2Cmd.Ex2Actions.action_3, None],
                      [Ex2Cmd.Ex2Commands.cmd_4, Ex2Cmd.Ex2Actions.action_4, None],
                      [Ex2Cmd.Ex2Commands.cmd_5, Ex2Cmd.Ex2Actions.action_5, None],
                      [Ex2Cmd.Ex2Commands.cmd_6, Ex2Cmd.Ex2Actions.action_6, None],
                      [Ex2Cmd.Ex2Commands.cmd_7, Ex2Cmd.Ex2Actions.action_7, None],
                      [Ex2Cmd.Ex2Commands.cmd_8, Ex2Cmd.Ex2Actions.action_8, None],
                      [Ex2Cmd.Ex2Commands.cmd_9, Ex2Cmd.Ex2Actions.action_9, None],
                      [Ex2Cmd.Ex2Commands.cmd_back, Ex2Cmd.Ex2Actions.back, None],
                      [Ex2Cmd.Ex2Commands.cmd_home, Ex2Cmd.Ex2Actions.home, None],
                      [Ex2Cmd.Ex2Commands.cmd_switch, Ex2Cmd.Ex2Actions.switch, None],
                      [Ex2Cmd.Ex2Commands.cmd_load, Ex2Cmd.Ex2Actions.load, "session_uuid"]]
        dummy_nav = DummyNav(trace=self._trace)
        nav_cmd = Ex2CmdMap(dummy_nav)
        for case in test_cases:
            cmd, expected, arg = case
            if arg is not None:
                cmd_line = "{} {}".format(cmd.value, arg)
            else:
                cmd_line = cmd.value
            nav_cmd.onecmd(cmd_line)
            self.assertEqual(expected, dummy_nav.last_action)
        return

    def test_level_nav(self):
        return


if __name__ == "__main__":
    unittest.main()
