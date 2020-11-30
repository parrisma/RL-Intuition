import numpy as np
import unittest
from src.lib.rltrace.trace import Trace
from src.tictactoe.q_val.q_vals import QVals
from src.tictactoe.q_val.q_calc import QCalc


class TestQVal(unittest.TestCase):
    _trace: Trace
    _id: int
    _expected_git_branch: str

    @classmethod
    def setUpClass(cls):
        cls._trace = Trace()
        cls._id = 1
        return

    def setUp(self) -> None:
        self._trace.log().info("- - - - - - C A S E  {} Started - - - - - -".format(TestQVal._id))
        return

    def tearDown(self) -> None:
        self._trace.log().info("- - - - - - C A S E {} Passed - - - - - -".format(TestQVal._id))
        TestQVal._id += 1
        return

    def test_normalise(self):
        """
        Test the normalise function
        """
        cases = [
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1]],
            [[0, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 1]],
            [[3.142, 3.142, 3.142, 3.142, 3.142, 3.142, 3.142, 3.142, 3.142], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1]],
            [[-2, -2, -2, -2, -2, -2, -2, -2, -2], [-1, -1, -1, -1, -1, -1, -1, -1, -1]],
            [[-1, 0, 1], [-1, 0, 1]],
            [[-.5, 0, 1], [-.5, 0, 1]],
            [[-.5, 0, 2], [-.25, 0, 1]],
            [[-2, -1, -0.5, 0, 0.5, 1, 2], [-1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0]],
            [[-.732, 0, 4.3], [-.732 / 4.3, 0, 1]]
        ]
        for case in cases:
            a, expected = case
            self.assertTrue(np.array_equal(QCalc.normalize(np.asarray(a=a)), (np.asarray(expected))))
        return


if __name__ == "__main__":
    unittest.main()
