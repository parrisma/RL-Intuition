from typing import Dict, List
import numpy as np
from src.tictactoe.interface.nav import Nav
from src.lib.rltrace.trace import Trace
from src.tictactoe.TicTacToe import TicTacToe
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.q_val.q_vals import QVals
from src.tictactoe.q_val.q_calc import QCalc


class QNav(Nav):
    """
    Navigate a dictionary of Q-Values. The given action moves from current state to the state resulting from
    the action.
    """
    _ttt: TicTacToe
    _q_vals: Dict[str, QVals]
    _last: List
    _trace: Trace
    _ttt_event_stream: TicTacToeEventStream
    _agent_name: str

    _sep: str = "_____________________________________________________________________"

    def __init__(self,
                 ttt: TicTacToe,
                 ttt_event_stream: TicTacToeEventStream,
                 trace: Trace):
        self._trace = trace
        self._ttt_event_stream = ttt_event_stream
        self._ttt = ttt
        self._q_vals = dict()
        self._reset()
        self._show(state=self._ttt.state().state_as_string())
        return

    def _greedy_action(self,
                       q_vals: np.ndarray) -> int:
        """
        Return the valid action associated with the highest reward
        :return: The action to maximise reward
        """
        if self._ttt.episode_complete():
            greedy_actn = -1
        else:
            res = list()
            valid_actions = self._ttt.actions(self._ttt.state())
            if len(valid_actions) > 0:
                for actn in valid_actions:
                    res.append([actn, q_vals[actn]])
                res = sorted(res, key=lambda x: x[1])
                greedy_actn = res[-1][0]
        return greedy_actn

    def _show(self,
              state: str) -> None:
        """
        Show the Q Values for the current state
        :param state: The State (as string) to show Q Values for
        """
        if len(self._q_vals) > 0:
            q_vals = self._q_vals[state]
            self._trace.log().debug("\n{}".format(q_vals))
            self._trace.log().debug("Greedy Action [{}]".format(self._greedy_action(q_vals=q_vals.q_vals)))
        else:
            self._trace.log().info("Use the load <session_uuid> command to get started")
        return

    def _reset(self):
        """
        Reset to initial state
        """
        self._ttt.episode_start()  # ensure we are in a known state
        self._last = list()
        self._agent_name = self._ttt.x_agent_name()
        self._last.append([self._ttt.state_as_str(), "{}".format(self._agent_name)])
        return

    def do_action(self,
                  action: int) -> None:
        """
        If the given action is a valid action in current state more to that state and show the Q Values use
        report an illegal move and do nothing
        :param action: The action to take from the current state
        """
        self._trace.log().info("- - - - - - Nav Action {} Invoked".format(str(action)))
        if not self._ttt.episode_complete():
            next_agent = self._ttt.do_action(agent_id=self._agent_name, action=action)
            if next_agent is not None:
                if self._ttt.state().state_as_string() not in self._q_vals:
                    self._trace.log().info("- - - - - - [{}] not found in given Q Values".
                                           format(self._ttt.state_as_str()))
                    st, agnt = self._last[-1]
                    self._ttt.import_state(st)
                    self._agent_name = agnt
                else:
                    self._last.append([self._ttt.state_as_str(), "{}".format(self._agent_name)])
                    self._agent_name = next_agent
                    self._show(state=self._ttt.state().state_as_string())

        if self._ttt.episode_complete():
            self._trace.log().info("This is the end of this episode sequence, use [home] command to reset")
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
            self._show(state=self._ttt.state().state_as_string())
        else:
            self._trace.log().debug("Cannot go back because we are at initial (root) state")
        return

    def do_home(self) -> None:
        """
        Navigate back to the initial state
        """
        self._ttt.episode_start()
        self._last.append([self._ttt.state_as_str(), "{}".format(self._agent_name)])
        self._show(state=self._ttt.state().state_as_string())
        return

    def do_switch(self) -> None:
        """
        Switch Player perspective
        """
        if self._agent_name == self._ttt.x_agent_name():
            self._agent_name = self._ttt.o_agent_name()
        else:
            self._agent_name = self._ttt.x_agent_name()
        self._show(state=self._ttt.state().state_as_string())
        return

    def do_load(self,
                arg) -> None:
        """
        Load the session uuid given as arg
        """
        self._q_vals = QCalc(trace=self._trace,
                             ttt_event_stream=self._ttt_event_stream).calc_q(session_uuid=arg)
        self._ttt.episode_start()
        self._last = [self._ttt.state_as_str(), self._ttt.x_agent_name()]
        self.do_home()
        return
