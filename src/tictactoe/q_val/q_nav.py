from typing import Dict, List
import numpy as np
from src.tictactoe.interface.actionnav import ActionNav
from src.lib.rltrace.trace import Trace
from src.tictactoe.tictactoe import TicTacToe
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.q_val.q_vals import QVals
from src.tictactoe.q_val.q_calc import QCalc
from src.tictactoe.explore.explore import Explore


class QNav(ActionNav):
    """
    Navigate a dictionary of Q-Values. The given action moves from current state to the state resulting from
    the action.
    """
    _ttt: TicTacToe
    _q_vals: Dict[str, Dict[str, QVals]]
    _last: List
    _trace: Trace
    _ttt_event_stream: TicTacToeEventStream
    _dir_to_use: str
    _q_calc: QCalc
    _exploration: Explore

    _Q_HIST_FMT = "{} : {} : Action: [{}] Reward [{:12.6f}] Ns Max [{:12.6f}] Old [{:12.6f}] Update [{:12.6f}] New [{:12.6f}]"
    _FMT = "[{}][{}][{}]  [{}][{}][{}]"
    _SEP = "_______________________________________________________________________________________"
    _TITLE = "       X-Visits               O-Visits"
    _X = 'X'
    _O = 'O'
    _B = '-'

    def __init__(self,
                 ttt: TicTacToe,
                 ttt_event_stream: TicTacToeEventStream,
                 trace: Trace,
                 dir_to_use: str):
        self._dir_to_use = dir_to_use
        self._trace = trace
        self._ttt_event_stream = ttt_event_stream
        self._ttt = ttt
        self._q_vals = dict()
        self._reset()
        self._show(agent=self._ttt.get_current_agent().name(), state=self._ttt.state().state_as_string())
        self._exploration = None  # noqa
        self._q_calc = None  # noqa
        return

    def _ready(self) -> bool:
        """
        Is the navigation loaded and ready to explore
        :return: True if navigation is loaded & ready to explore
        """
        return len(self._q_vals) > 0 and self._exploration is not None

    def _greedy_action(self,
                       q_vals: np.ndarray) -> int:
        """
        Return the valid action associated with the highest reward
        :return: The action to maximise reward
        """
        greedy_actn = None
        if self._ttt.episode_complete():
            greedy_actn = -1
        else:
            res = list()
            valid_actions = self._ttt.actions(self._ttt.state())
            if len(valid_actions) > 0:
                for actn in valid_actions:
                    if not np.isnan(q_vals[actn]):
                        res.append([actn, q_vals[actn]])
                res = sorted(res, key=lambda x: x[1])
                if len(res) >= 1:
                    greedy_actn = res[-1][0]
        return greedy_actn

    def _show_state_visit_data(self) -> None:
        """
        Show the visit data for the current state of the TTT env.
        This view presents the number of passes that have been seen for the actions that transition to the states
        possible from this state
        """
        if self._exploration.get_visited() is None:
            return

        if not self._ttt.episode_complete():
            prev_state = self._ttt.state_action_str()
            prev_agnt = self._ttt.get_current_agent()
            o_vis = ['     '] * 9
            x_vis = ['     '] * 9
            valid_actions = self._ttt.actions(self._ttt.state())
            if len(valid_actions) > 0:
                for actn in valid_actions:
                    for agnt, visit in [[self._ttt.get_o_agent().id(), o_vis],
                                        [self._ttt.get_x_agent().id(), x_vis]]:
                        self._ttt.do_action(action=actn, agent_id=agnt)
                        if self._ttt.state().state_as_string() in self._exploration.get_visited():
                            visit[actn] = "{:5d}".format(
                                self._exploration.get_visited()[self._ttt.state().state_as_string()])
                        else:
                            if self._ttt.legal_board_state():
                                visit[actn] = "  -  "
                            else:
                                visit[actn] = "  I  "
                        self._ttt.import_state(prev_state)
                        self._ttt.set_current_agent(agent=prev_agnt)
            self._ttt.import_state(prev_state)
            self._ttt.set_current_agent(agent=prev_agnt)
            s = "\n\n{}\n{}\n{}\n{}\n{}\n{}\n". \
                format(self._SEP,
                       self._TITLE,
                       self._FMT.format(x_vis[0], x_vis[1], x_vis[2], o_vis[0], o_vis[1], o_vis[2]),
                       self._FMT.format(x_vis[3], x_vis[4], x_vis[5], o_vis[3], o_vis[4], o_vis[5]),
                       self._FMT.format(x_vis[6], x_vis[7], x_vis[8], o_vis[6], o_vis[7], o_vis[8]),
                       self._SEP)
            self._trace.log().info(s)
        else:
            self._trace.log().info("No visit analysis for a complete episode")
        return

    def _show(self,
              agent: str,
              state: str) -> None:
        """
        Show the Q Values for the current state
        :param agent: The agent to show q values for
        :param state: The State (as string) to show Q Values for
        """
        if len(self._q_vals) > 0:
            q_vals = self._q_vals[agent][state]
            self._trace.log().info("\n{}".format(q_vals))
            self._trace.log().info("Greedy Action [{}]".format(self._greedy_action(q_vals=q_vals.q_vals)))
            self._show_state_visit_data()
        else:
            self._trace.log().info("Use the load <session_uuid> command to get started")
        return

    def _last_reset(self) -> List:
        """
        Initial state for teh last state tracking
        :return: The initial state of teh last state list
        """
        return [[self._ttt.state_action_str(), "{}".format(self._ttt.get_current_agent().name())]]

    def _reset(self):
        """
        Reset to initial state
        """
        self._ttt.episode_start()  # ensure we are in a known state
        self._last = self._last_reset()
        return

    def _prompt(self) -> str:
        """
        The navigation prompt based on the current player X or O
        :return: The Player specific prompt
        """
        if self._ready():
            p = "(Agent-{})".format(self._ttt.get_current_agent().name())
        else:
            p = "(Use [list] or [load] to start or [help list] / [help load]"
        return p

    def do_hist(self,
                arg) -> str:
        """
        Show the history of Q Value updates for the action in the current state
        :param arg: The action to show Q Value history for
        :return: The Nav prompt as string
        """
        args = self.parse(arg)
        if len(args) == 1:
            action = int(args[0])
            if self._ready():
                if not self._ttt.episode_complete():
                    hst = self._q_calc.get_q_hist(agent=self._ttt.get_current_agent().name(),
                                                  state=self._ttt.state().state_as_string(),
                                                  action=action)
                    xs = str(self._ttt.get_x_agent().id())
                    os = str(self._ttt.get_o_agent().id())
                    if hst is not None:
                        i = 1
                        for entry in hst:
                            st = entry[0].replace(xs, self._X).replace(os, self._O).replace("0", self._B)
                            self._trace.log().info(
                                self._Q_HIST_FMT.format(i, st, entry[1], entry[2], entry[3], entry[4], entry[5],
                                                        entry[6]))
                            i += 1
                    else:
                        self._trace.log().info("No Q Val history for {} - {}".
                                               format(self._ttt.state().state_as_string(), action))
        else:
            self._trace.log().info("Hist command requires the action as in hist <action 0 or 1 or .. 8>")
        return self._prompt()

    def do_action(self,
                  action: int) -> str:
        """
        If the given action is a valid action in current state more to that state and show the Q Values use
        report an illegal move and do nothing
        :param action: The action to take from the current state
        """
        agent = self._ttt.get_current_agent().name()
        if self._ready():
            self._trace.log().info("- - - - - - Nav Action {} Invoked".format(str(action)))
            if not self._ttt.episode_complete():
                next_agent = self._ttt.do_action(agent_id=agent, action=action)
                self._ttt.set_current_agent(next_agent)
                if next_agent is not None:
                    if next_agent not in self._q_vals:
                        self._trace.log().info("- - - - - - Agent [{}] has no Q Values".format(next_agent))
                    else:
                        if self._ttt.state().state_as_string() not in self._q_vals[next_agent]:
                            self._trace.log().info("- - - - - - [{}] not found in given Q Values".
                                                   format(self._ttt.state_action_str()))
                            st, agnt = self._last[-1]
                            self._ttt.import_state(st)
                            self._ttt.set_current_agent(agnt)
                        else:
                            self._last.append([self._ttt.state_action_str(), "{}".
                                              format(agent)])
                            self._show(agent=self._ttt.get_current_agent().name(),
                                       state=self._ttt.state().state_as_string())
            if self._ttt.episode_complete():
                self._trace.log().info("This is the end of this episode sequence, use [home] command to reset")
        return self._prompt()

    def do_back(self) -> str:
        """
        Wind back to previous state if there was one
        """
        if self._ready():
            if len(self._last) > 1:
                self._last.pop()
                st, agnt = self._last[-1]
                self._ttt.import_state(st)
                self._ttt.set_current_agent(agnt)
                self._show(agent=self._ttt.get_current_agent().name(),
                           state=self._ttt.state().state_as_string())
            else:
                self._trace.log().info("Cannot go back because we are at initial (root) state")
        return self._prompt()

    def do_home(self) -> str:
        """
        Navigate back to the initial state
        """
        if self._ready():
            self._ttt.episode_start()
            self._last = self._last_reset()
            self._show(agent=self._ttt.get_current_agent().name(),
                       state=self._ttt.state().state_as_string())
        return self._prompt()

    def do_switch(self) -> str:
        """
        Switch Player perspective
        """
        if self._ready():
            self._ttt.set_current_agent(self._ttt.get_next_agent())
            self._show(agent=self._ttt.get_current_agent().name(),
                       state=self._ttt.state().state_as_string())
        return self._prompt()

    @staticmethod
    def parse(arg):
        """
        Convert a series of zero or more numbers to an argument tuple
        """
        return tuple(map(str, arg.split()))

    def do_load(self,
                arg) -> str:
        """
        Load the session uuid given as arg
        """
        reprocess_count = 1
        args = self.parse(arg)
        if len(args) == 1:
            uuid = args[0]
        elif len(args) == 2:
            uuid = args[0]
            reprocess_count = int(args[1])
        else:
            self._trace.log().info("Load command requires at least a session UUID with an optional reprocess count")
            uuid = None
        if uuid is not None:
            self._trace.log().info("Initiating Q-Val Cals")
            self._q_calc = QCalc(trace=self._trace,
                                 ttt_event_stream=self._ttt_event_stream,
                                 session_uuid=uuid)
            self._q_vals = self._q_calc.calc_q(reprocess_count=reprocess_count)
            self._exploration = Explore(ttt=self._ttt,
                                        trace=self._trace,
                                        ttt_event_stream=self._ttt_event_stream,
                                        visited=self._q_calc.get_visits(),
                                        graph=self._q_calc.get_graph())
            self._ttt.episode_start()
            self._last = self._last_reset()
            self._trace.log().info("Done Q-Val Cals")
            self.do_home()
        return self._prompt()

    def do_list(self) -> None:
        """
        List all the session uuids to select from
        """
        sessions = self._ttt_event_stream.list_of_available_sessions()
        self._trace.log().info("Available Sessions")
        for session in sessions:
            uuid, cnt = session
            self._trace.log().info("{} with {} events".format(uuid, cnt))
        self._trace.log().info("Use the [load <uuid>] command to load and calc one of these session uuid's")
        return
