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
    _fmt = "[{}][{}][{}]  [{}][{}][{}]  [{:12.6f}][{:12.6f}][{:12.6f}]  [{:5.2f}][{:5.2f}][{:5.2f}] "
    _sep = "_______________________________________________________________________________________"
    _title = " Actions     Board                Raw Q Values                     Softmax     "
    def __init__(self,
                 state: str,
                 x_as_str: str = '-1',
                 o_as_str: str = '1'):
        self.state = state
        self.q_vals = np.zeros((9))
        self.x_as_str = x_as_str
        self.o_as_str = o_as_str
        return

    def _softmax(self,
                 v: np.ndarray) -> np.ndarray:
        """
        Calculate the softmax
        :param v: The numpy array to normalise
        :return: The array of softmax probabilities
        """
        # Make more numerically stable by nomalizing first to ensure we dont take exp of a v large number.
        a = v.copy()
        a_min = np.min(self.q_vals)
        a_max = np.max(self.q_vals)
        df = (a_max - a_min)
        if df != 0:
            a /= df
        a -= a_min
        # now, return the softmax
        return np.exp(a) / np.sum(np.exp(a), axis=0)

    def __str__(self):
        st = self.state.replace(self.x_as_str, self._X).replace(self.o_as_str, self._O).replace("0", self._b)
        nq = self._softmax(v=self.q_vals) * 100
        s1 = self._fmt.format(0, 1, 2,
                              st[0], st[1], st[2],
                              self.q_vals[0], self.q_vals[1],
                              self.q_vals[2], nq[0], nq[1], nq[2])
        s2 = self._fmt.format(3, 4, 5,
                              st[3], st[4], st[5],
                              self.q_vals[3], self.q_vals[4],
                              self.q_vals[5], nq[3], nq[4], nq[5])
        s3 = self._fmt.format(6, 7, 8,
                              st[6], st[7], st[8],
                              self.q_vals[6], self.q_vals[7],
                              self.q_vals[8], nq[6], nq[7], nq[8])
        return "\n{}\n\n{}\n{}\n{}\n{}\n{}\n".format(self._sep, self._title, s1, s2, s3, self._sep)

    def __repr__(self):
        return self.__str__()
