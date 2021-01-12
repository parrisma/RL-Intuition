from typing import List
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
    _fmt = "[{}][{}][{}]  [{}][{}][{}]  [{}][{}][{}]  [{:6.2f}][{:6.2f}][{:6.2f}] "
    _fmt_qvs = "[{:6.2f}][{:6.2f}][{:6.2f}] [{:6.2f}][{:6.2f}][{:6.2f}] [{:6.2f}][{:6.2f}][{:6.2f}]"
    _qv_fmt = '{:12.6f}'
    _qv_nan = '     --     '
    _sep = "_______________________________________________________________________________________"
    _title = " Actions     Board                   Raw Q Values                     Softmax     "

    def __init__(self,
                 state: str,
                 x_as_str: str = '-1',
                 o_as_str: str = '1'):
        self.state = state
        self.q_vals = self._init_q_vals()
        self.x_as_str = x_as_str
        self.o_as_str = o_as_str
        return

    def _init_q_vals(self) -> np.ndarray:
        """
        Set the initial state of teh Q Values. By default all Q Values are initially NaN to indicate that there
        is no know value.
        :return: Initial Q Vales as ndarray
        """
        return np.full((9), np.nan)

    def _softmax(self,
                 v: np.ndarray) -> np.ndarray:
        """
        Calculate the softmax ignoring any action qvalues that have not been set (np.nan)
        :param v: The numpy array to normalise
        :return: The array of softmax probabilities
        """
        # Make more numerically stable by normalizing first to ensure we don't take exponential of a v large number.
        a = v.copy()
        df = 0
        if not np.isnan(self.q_vals).all():
            a_min = np.nanmin(self.q_vals)
            a_max = np.nanmax(self.q_vals)
            df = (a_max - a_min)
        smax = None
        if not np.isnan(df):
            if df == 0:
                a = np.zeros(np.size(v))
                non_nan_elements = np.argwhere(~np.isnan(self.q_vals))
                num_non_nan = len(non_nan_elements)
                for i in range(num_non_nan):
                    a[non_nan_elements[i][0]] = 1 / num_non_nan
                smax = a
            else:
                for i in range(np.size(v)):
                    if not np.isnan(a[i]):
                        a[i] /= df
                        a[i] -= a_min
                        # now, return the softmax
                smax = np.exp(a) / np.nansum(np.exp(a))
                for nan_idx in np.argwhere(np.isnan(smax)):
                    smax[nan_idx[0]] = float(0)
        else:
            smax = np.zeros(np.size(self.q_vals))
        return smax

    def _q_vals_to_str(self,
                       q_vals: np.ndarray) -> List:
        """
        Convert the given q-values to a list of strings with nan replaces by '-'
        :param q_vals: The q-value numpy array
        :return: Q Values as string with nan replaced by -
        """
        qvs = list()
        for i in range(len(q_vals)):
            if np.isnan(q_vals[i]):
                qvs.append(self._qv_nan)
            else:
                qvs.append(self._qv_fmt.format(float(q_vals[i])))
        return qvs

    def __str__(self):
        st = self.state.replace(self.x_as_str, self._X).replace(self.o_as_str, self._O).replace("0", self._b)
        nq = self._softmax(v=self.q_vals) * 100
        qvs = self._q_vals_to_str(self.q_vals)

        s1 = self._fmt.format(0, 1, 2,
                              st[0], st[1], st[2],
                              qvs[0], qvs[1], qvs[2],
                              nq[0], nq[1], nq[2])
        s2 = self._fmt.format(3, 4, 5,
                              st[3], st[4], st[5],
                              qvs[3], qvs[4], qvs[5],
                              nq[3], nq[4], nq[5])
        s3 = self._fmt.format(6, 7, 8,
                              st[6], st[7], st[8],
                              qvs[6], qvs[7], qvs[8],
                              nq[6], nq[7], nq[8])
        return "\n{}\n\n{}\n{}\n{}\n{}\n{}\n".format(self._sep, self._title, s1, s2, s3, self._sep)

    def __repr__(self):
        return self._fmt_qvs.format(self.q_vals[0], self.q_vals[1], self.q_vals[2],
                                    self.q_vals[3], self.q_vals[4], self.q_vals[5],
                                    self.q_vals[6], self.q_vals[7], self.q_vals[8])
