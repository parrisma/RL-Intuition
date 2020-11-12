import unittest
from src.lib.envboot.env import Env
from src.lib.rltrace.trace import Trace
from src.interface.envbuilder import EnvBuilder
from src.lib.settings import Settings
from src.lib.webstream import WebStream
from src.tictactoe.interface.nav import Nav
from src.tictactoe.nav_cmd import NavCmd
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

    def test_actions(self):
        test_cases = [[Nav.ActionCmd.action_0_cmd, Nav.Action.action_0, None],
                      [Nav.ActionCmd.action_1_cmd, Nav.Action.action_1, None],
                      [Nav.ActionCmd.action_2_cmd, Nav.Action.action_2, None],
                      [Nav.ActionCmd.action_3_cmd, Nav.Action.action_3, None],
                      [Nav.ActionCmd.action_4_cmd, Nav.Action.action_4, None],
                      [Nav.ActionCmd.action_5_cmd, Nav.Action.action_5, None],
                      [Nav.ActionCmd.action_6_cmd, Nav.Action.action_6, None],
                      [Nav.ActionCmd.action_7_cmd, Nav.Action.action_7, None],
                      [Nav.ActionCmd.action_8_cmd, Nav.Action.action_8, None],
                      [Nav.ActionCmd.action_9_cmd, Nav.Action.action_9, None],
                      [Nav.ActionCmd.action_back_cmd, Nav.Action.action_back, None],
                      [Nav.ActionCmd.action_home_cmd, Nav.Action.action_home, None],
                      [Nav.ActionCmd.action_switch_cmd, Nav.Action.action_switch, None],
                      [Nav.ActionCmd.action_load_cmd, Nav.Action.action_load, "session_uuid"]]
        dummy_nav = DummyNav(trace=self._trace)
        nav_cmd = NavCmd(dummy_nav)
        for case in test_cases:
            cmd, expected, arg = case
            if arg is not None:
                cmd_line = "{} {}".format(cmd.value, arg)
            else:
                cmd_line = cmd.value
            nav_cmd.onecmd(cmd_line)
            self.assertEqual(expected, dummy_nav.last_action)
        return


if __name__ == "__main__":
    unittest.main()