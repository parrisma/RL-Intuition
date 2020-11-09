import unittest
import logging
import subprocess
from src.lib.rltrace.trace import Trace
from src.lib.envboot.runspec import RunSpec


class TestRunSpec(unittest.TestCase):
    _trace: Trace
    _id: int
    _expected_git_branch: str

    @classmethod
    def setUpClass(cls):
        cls._trace = Trace()
        cls._id = 1
        cls._expected_git_branch = cls._git_current_branch()
        return

    def setUp(self) -> None:
        self._trace.log().info("- - - - - - C A S E  {} Started - - - - - -".format(TestRunSpec._id))
        return

    def tearDown(self) -> None:
        self._trace.log().info("- - - - - - C A S E {} Passed - - - - - -".format(TestRunSpec._id))
        TestRunSpec._id += 1
        return

    def test_bootstrap(self):
        # RunSpec is bootstrapped in the module __init__.py
        run_spec = RunSpec("./specs.yaml")
        current_spec = run_spec.get_spec()
        branch = run_spec.branch()
        self.assertEqual(RunSpec.DEFAULT, current_spec)
        self.assertTrue(len(branch) > 0)
        self.assertTrue(branch == self._expected_git_branch)
        return

    def test_elastic(self):
        run_spec = RunSpec("./specs.yaml")
        elastic_settings_yaml = run_spec.elastic_settings_yaml()
        self.assertTrue(len(elastic_settings_yaml) > 0)
        return

    def test_trace(self):
        run_spec = RunSpec("./specs.yaml")
        trace_settings_yaml = run_spec.trace_settings_yaml()
        self.assertTrue(len(trace_settings_yaml) > 0)
        return

    def test_ttt(self):
        run_spec = RunSpec("./specs.yaml")
        ttt_settings_yaml = run_spec.ttt_settings_yaml()
        self.assertTrue(len(ttt_settings_yaml) > 0)
        return

    #
    # Utility funcs.
    #
    @classmethod
    def _git_current_branch(cls) -> str:
        res = subprocess.check_output("git rev-parse --abbrev-ref HEAD").decode('utf-8')
        if res is None or len(res) == 0:
            res = "Warning cannot establish current git branch"
        else:
            res = cls._chomp(res)
        return res

    @classmethod
    def _chomp(cls,
               s: str) -> str:
        """
        Remove all line breaks
        :param s: The string to remove line breaks from
        :return: string without line breaks
        """
        return s.replace("\r", "").replace("\n", "")


if __name__ == "__main__":
    unittest.main()
