from typing import Dict, List
from src.tictactoe.interface.nav import Nav
from src.lib.rltrace.trace import Trace
from src.tictactoe.TicTacToe import TicTacToe
from src.tictactoe.q_vals import QVals


class QNav(Nav):
    """
    Navigate a dictionary of Q-Values. The given action moves from current state to the state resulting from
    the action.
    """
    _ttt: TicTacToe
    _q_vals: Dict[str, QVals]
    _last: List
    _trace: Trace
    _agent_name: str

    def __init__(self,
                 ttt: TicTacToe,
                 q_vals: Dict[str, QVals],
                 trace: Trace):
        self._ttt = ttt
        self._q_vals = q_vals
        self._trace = trace
        self._reset()
        return

    def _reset(self):
        """
        Reset to initial state
        """
        self._ttt.episode_start()  # ensure we are in a known state
        self._last = list()
        self._agent_name = self._ttt.x_agent_name()
        self._last.append([self._ttt.state_as_str(), "{}".format(self._agent_name)])
        if self._ttt.state().state_as_string() not in self._q_vals:
            err = "Initial state [{}] not found in given Q Values".format(self._ttt.state_as_str())
            self._trace.log().critical(err)
            raise RuntimeError(err)
        return

    def do_action(self,
                  action: int) -> None:
        """
        If the given action is a valid action in current state more to that state and show the Q Values use
        report an illegal move and do nothing
        :param action: The action to take from the current state
        """
        self._trace.log().debug("- - - - - - Nav Action {} Invoked".format(str(action)))
        next_agent = self._ttt.do_action(agent_id=self._agent_name, action=action)
        if next_agent is not None:
            if self._ttt.state().state_as_string() not in self._q_vals:
                self._trace.log().debug("- - - - - - [{}] not found in given Q Values".format(self._ttt.state_as_str()))
                st, agnt = self._last[-1]
                self._ttt.import_state(st)
                self._agent_name = agnt
            else:
                self._last.append([self._ttt.state_as_str(), "{}".format(self._agent_name)])
                self._agent_name = next_agent
                self._trace.log().debug("\n{}".format(self._q_vals[self._ttt.state().state_as_string()]))
        return

    def do_back(self) -> None:
        """
        Wind back to previous state if there was one
        """
        if len(self._last) > 1:
            self._last.pop()
            st, agnt = self._last[-1]
            self._ttt.import_state(st)
            self._agent_name = agnt
            self._trace.log().debug("\n{}".format(self._q_vals[self._ttt.state().state_as_string()]))
        else:
            self._trace.log().debug("Cannot go back, already at initial state")
        return

    def do_home(self) -> None:
        """
        Navigate back to the initial state
        """
        self._ttt.episode_start()
        self._last = [self._ttt.state_as_str(), self._ttt.x_agent_name()]
        self._trace.log().debug("\n{}".format(self._q_vals[self._ttt.state().state_as_string()]))
        return
