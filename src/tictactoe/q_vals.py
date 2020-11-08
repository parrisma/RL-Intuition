import numpy as np


class QVals:
    """
    Representation of a board state as Q-Values
        Simple utility class so no getter/setter - members access directly.
    """
    state: str
    q_vals: np.ndarray
    x_as_str: str
    o_as_str: str
    _X = 'X'
    _O = 'O'
    _b = ' '
    _fmt = "[{}][{}][{}] [{:12.6f}][{:12.6f}][{:12.6f}] [{:7.3f}][{:7.3f}][{:7.3f}]"

    def __init__(self,
                 state: str,
                 x_as_str: str = '-1',
                 o_as_str: str = '1'):
        self.state = state
        self.q_vals = np.zeros((9))
        self.x_as_str = x_as_str
        self.o_as_str = o_as_str
        return

    def _norm(self,
              v: np.ndarray,
              scale: float = 100) -> np.ndarray:
        """
        Normalise values and scale
        :param a: The numpy array to normalise
        :return: The normalised array in range [0.0 to 1.0] * scale
        """
        a = v.copy()
        a_min = np.min(self.q_vals)
        a_max = np.max(self.q_vals)
        df = (a_max - a_min)
        if df != 0:
            a /= df
        a -= a_min
        a *= scale
        return a

    def __str__(self):
        st = self.state.replace(self.x_as_str, self._X).replace(self.o_as_str, self._O).replace("0", self._b)
        nq = self._norm(v=self.q_vals)
        s1 = self._fmt.format(st[0], st[1], st[2], nq[0], nq[1], nq[2], self.q_vals[0], self.q_vals[1], self.q_vals[2])
        s2 = self._fmt.format(st[3], st[4], st[5], nq[3], nq[4], nq[5], self.q_vals[3], self.q_vals[4], self.q_vals[5])
        s3 = self._fmt.format(st[6], st[7], st[8], nq[6], nq[7], nq[8], self.q_vals[6], self.q_vals[7], self.q_vals[8])
        return "{}\n{}\n{}\n".format(s1, s2, s3)

    def __repr__(self):
        return self.__str__()
