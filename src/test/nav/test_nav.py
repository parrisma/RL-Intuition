import unittest
from src.lib.envboot.env import Env
from src.lib.rltrace.trace import Trace
from src.interface.envbuilder import EnvBuilder
from src.lib.settings import Settings
from src.lib.streams.webstream import WebStream
from src.tictactoe.interface.actionnav import ActionNav
from src.tictactoe.experiment.action_nav_cmd import ActionNavCmd
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
        test_cases = [[ActionNav.ActionCmd.cmd_0, ActionNav.Action.action_0, None],
                      [ActionNav.ActionCmd.cmd_1, ActionNav.Action.action_1, None],
                      [ActionNav.ActionCmd.cmd_2, ActionNav.Action.action_2, None],
                      [ActionNav.ActionCmd.cmd_3, ActionNav.Action.action_3, None],
                      [ActionNav.ActionCmd.cmd_4, ActionNav.Action.action_4, None],
                      [ActionNav.ActionCmd.cmd_5, ActionNav.Action.action_5, None],
                      [ActionNav.ActionCmd.cmd_6, ActionNav.Action.action_6, None],
                      [ActionNav.ActionCmd.cmd_7, ActionNav.Action.action_7, None],
                      [ActionNav.ActionCmd.cmd_8, ActionNav.Action.action_8, None],
                      [ActionNav.ActionCmd.cmd_9, ActionNav.Action.action_9, None],
                      [ActionNav.ActionCmd.cmd_back, ActionNav.Action.back, None],
                      [ActionNav.ActionCmd.cmd_home, ActionNav.Action.home, None],
                      [ActionNav.ActionCmd.cmd_switch, ActionNav.Action.switch, None],
                      [ActionNav.ActionCmd.cmd_load, ActionNav.Action.load, "session_uuid"]]
        dummy_nav = DummyNav(trace=self._trace)
        nav_cmd = ActionNavCmd(dummy_nav)
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
