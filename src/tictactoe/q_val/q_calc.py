import numpy as np
import networkx as nx
from typing import Dict, List, Callable, Tuple
from src.lib.rltrace.trace import Trace
from src.tictactoe.tictactoe import TicTacToe
from src.tictactoe.simulation_agent import SimulationAgent
from src.tictactoe.event.tictacttoe_event import TicTacToeEvent
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.q_val.q_vals import QVals


class QCalc:
    """
    Load and calculate Q Values for a TicTacToe event session

    This is only possible as TicTacToe has a very low number of discrete states (order 6'000) so we can hold the
    entire Q-Value <-> Action map in memory.

    By doing this we will be able to see how Q-Value works such that when we replace the in memory Q Value map with
    a Neural Network we will be able to validate that the NN is learning as expected.

    Q Value Update:
        Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])
    """
    _trace: Trace
    _ttt_event_stream: TicTacToeEventStream
    _num_actions: int
    _gamma: float  # How much do we value future rewards
    _learning_rate: float  # Min (initial) size of Q Value update
    _q_values: Dict[str, Dict[str, QVals]]  # State str : Key -> actions q values & #num times reward seen
    _q_hist: Dict[str, Dict[str, List[Tuple[str, int, float, float, float, float, float]]]]
    _events: List[TicTacToeEvent]
    _visits: Dict[str, int]
    _graph: nx.DiGraph
    _session_uuid: str
    _ttt: TicTacToe
    _max_func_idx: int
    _max_funcs: List[Callable[[str, str, Dict], np.float]]

    NAIVE_MAX = 0
    ONE_STEP_MAX = 1

    END_STATE = "End"

    HIST_FMT = "{}:{}"

    def __init__(self,
                 trace: Trace,
                 ttt_event_stream: TicTacToeEventStream,
                 session_uuid: str = None):
        self._ttt_event_stream = ttt_event_stream
        if session_uuid is None:
            self._session_uuid = self._ttt_event_stream.session_uuid()
        else:
            self._session_uuid = session_uuid
        self._trace = trace
        self._num_actions = 9  # max number of actions in any given board state
        self._gamma = 0.9
        self._learning_rate = 0.1
        self._q_values = dict()
        self._q_hist = dict()
        self._events = None  # noqa
        self._visits = None  # noqa
        self._graph = None  # noqa
        self._load_session(session_uuid=self._session_uuid)

        agent_factory = SimulationAgent.SimulationAgentFactory()
        self._ttt = TicTacToe(trace=self._trace,
                              ttt_event_stream=self._ttt_event_stream,
                              x=agent_factory.new_x_agent(),
                              o=agent_factory.new_o_agent())

        self._max_funcs = [self._naive_max,  # NAIVE_MAX
                           self._one_step_max]  # ONE_STEP_MAX
        self._max_func_idx = self.ONE_STEP_MAX

        return

    def _get_state_q_values(self,
                            agent: str,
                            state: str) -> np.ndarray:
        """
        Get the Q values of the given agent in the given state.
        :param agent: The Agent to get Q Values for
        :param state: The state to get Q Values for
        :return: The Q Values for given agent in the given state as numpy array of float
        """
        if agent not in self._q_values:
            self._q_values[agent] = dict()
        if state not in self._q_values[agent]:
            self._q_values[agent][state] = QVals(state=state)
        return self._q_values[agent][state].q_vals

    def _set_state_q_values(self,
                            agent: str,
                            state: str,
                            q_values: np.ndarray) -> None:
        """
        Set the Q values of the given agent in the given state.
        :param agent: The agent to set q values for
        :param state: The state to set Q Values for
        :param q_values: The Q Values to set for the given agent in the given state
        """
        if agent not in self._q_values:
            self._q_values[agent] = dict()
        if state not in self._q_values[agent]:
            self._q_values[agent][state] = QVals(state=state)
        self._q_values[agent][state].q_vals = q_values
        return

    @staticmethod
    def normalize(a: np.ndarray) -> np.ndarray:
        """
        Normalise the values in the range -1.0 to 1.0
        :param a: The numpy array to normalise
        :return: The normalised numpy array with all values in range 0.0 to 1.0
        """
        ac = a.copy()
        ac_min = np.min(ac)
        ac_max = np.max(ac)
        if ac_max == ac_min:
            if ac_min != 0:
                dv = ac_min
                if ac_min < 0:
                    dv = -dv
                ac = np.divide(ac, dv)
        else:
            # no div zero check as we know max != min
            if ac_min < 0:
                dv = max(-ac_min, ac_max)
                ac = np.divide(ac, dv)
            else:
                dv = ac_max - ac_min
                ac = np.divide(ac - ac_min, dv)
        return ac

    def _one_step_max(self,
                      agent: str,
                      next_state: str,
                      **kwargs) -> np.float:
        """
        Determine the max by looking at the next state given and all possible states the follow the next_state.This
        is simple done by looking at all of the out going edges from the next_state given. So only states that
        manifested as part of the loaded simulation will be considered.

        As such this method does not require a model of the game (environment) it only needs to know the states
        that formed part of the simulation.

        :param agent: The agent to get the max for
        :param next_state: The state to estimate the max for
        :param kwargs: Optional arguments
        :return: The estimated max reward from the next_state + all (known) states at step + 1
        """
        os_max = np.nan
        if not next_state == self.END_STATE and next_state is not None:
            states_to_check = list()
            for u, v in self._graph.out_edges(next_state):  # States of next player
                for uu, vv in self._graph.out_edges(v):  # Next states of current player
                    if vv not in states_to_check:
                        states_to_check.append(vv)
            for state in states_to_check:
                os_max = np.nanmax(np.array([os_max, np.nanmax(self._get_state_q_values(agent, state))]))
                self._trace.log().debug("1 Step : {} - {} - {}".format(next_state,
                                                                       state,
                                                                       os_max))
        return os_max

    def _naive_max(self,
                   agent: str,
                   next_state: str,
                   **kwargs) -> np.float:
        """
        Estimate the max reward available from the next state by simply taking the max Q-Val from the
        next state.
        :param agent: The Agent to get the max for
        :param next_state: The next state to inspect
        :return: The max (estimated) reward available at the next state for the given agent
        """
        n_max = np.nan
        if not next_state == self.END_STATE:
            q_next_state = self._get_state_q_values(agent, next_state)
            if not np.isnan(q_next_state).all():
                n_max = np.nanmax(q_next_state)
        return n_max

    def _record_q_hist(self,
                       agent: str,
                       state: str,
                       next_state: str,
                       ns_max: float,
                       action: int,
                       reward: float,
                       old_q: float,
                       update: float,
                       new_q) -> None:
        """
        Record the update to Q for the given state
        :param agent: The agent to record the history for
        :param state: The State the Q update relates to
        :param action: The action within the state the Q Value update relates to
        :param reward: The reward for the action taken
        :param old_q: The Previous value of Q
        :param update: The Update made to Q
        :param new_q: The new value of Q
        :return:
        """
        if agent not in self._q_hist:
            self._q_hist[agent] = dict()
        hist_key = self.HIST_FMT.format(state, action)
        if hist_key not in self._q_hist:
            self._q_hist[agent][hist_key] = list()
        self._q_hist[agent][hist_key].append((next_state, action, reward, ns_max, old_q, update, new_q))
        return

    def get_q_hist(self,
                   agent: str,
                   state: str,
                   action: int) -> List[Tuple[str, int, float, float, float, float, float]]:
        """
        Get the history of updates for a given agent in teh given state - action Q Value
        :param agent: The agent to get the history for.
        :param state: The state of interest
        :param action: The action within the state to get history for
        :return: List of Q Value Updates in order where an update = [next_state, next_state_max, old, update, new]
        """
        hist_key = self.HIST_FMT.format(state, action)
        res = None
        if agent in self._q_hist:
            if hist_key in self._q_hist[agent]:
                res = self._q_hist[agent][hist_key]
        return res

    def _update_q(self,
                  state: str,
                  next_state: str,
                  action: int,
                  agent: str,
                  reward: float,
                  max_func: Callable[[str, str, Dict], np.float]) -> None:
        """
        Update internal Q Values with the given State/Action/Reward update
        :param state: The current state
        :param next_state: The state after the given action is taken
        :param action: The action taken in the given state
        :param agent: The Agent (player) that took the given action
        :param reward: The reward on arrival in the next_state
        """
        if state == '-1-10-11111-1':
            print('x')
        if next_state != self.END_STATE:
            upd = [[1, agent], [-1, self._ttt.get_other_agent(agent).name()]]
            for rfac, agnt in upd:
                q_state = self._get_state_q_values(agnt, state)
                q_val_prev = q_state[action]
                next_state_max = max_func(agent=agnt, next_state=next_state)  # noqa
                # Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])
                if not np.isnan(next_state_max):
                    q_update = self._learning_rate * (
                            (reward * rfac) + self._gamma * next_state_max - self.zero_if_nan(q_val_prev))
                else:
                    q_update = self._learning_rate * ((reward * rfac) - self.zero_if_nan(q_val_prev))
                # Advantage
                q_state[action] = self.zero_if_nan(q_state[action]) + q_update
                # self._trace.log().info("{} -> {} @ {} - {:7.3f}".format(state, next_state, reward, q_state[action]))
                self._set_state_q_values(agent, state, q_state)
                self._record_q_hist(agent, state, next_state, next_state_max, action, reward, q_val_prev, q_update,
                                    q_state[action])
        return

    def _load_session(self,
                      session_uuid: str) -> None:
        """
        Load the given session UUID
        :param session_uuid: An existing session UUID to load
        """
        self._trace.log().info("Start loading events for session {}".format(session_uuid))
        try:
            self._events = self._ttt_event_stream.get_session(session_uuid=session_uuid)
            if self._events is None or len(self._events) == 0:
                raise RuntimeError("No events for session uuid [{}]".format(session_uuid))
            self._trace.log().info("Loaded [{}] evenhomets for session {}".format(len(self._events), session_uuid))

            self._visits = self._ttt_event_stream.load_visits_from_yaml(session_uuid=session_uuid)
            if self._visits is None or len(self._visits) == 0:
                raise RuntimeError("No Visit data for session uuid [{}]".format(session_uuid))
            self._trace.log().info("Loaded [{}] visits for session {}".format(len(self._visits), session_uuid))

            self._graph = self._ttt_event_stream.load_graph_from_yaml(session_uuid=session_uuid)
            if self._graph is None or len(self._graph) == 0:
                raise RuntimeError("No graph for session uuid [{}]".format(session_uuid))
            self._trace.log().info("Loaded [{}] graph nodes for session {}".format(len(self._graph), session_uuid))
        except RuntimeError as _:
            self._trace.log().error("Failed to load events for session [{}]".format(session_uuid))
            self._events = None  # noqa
            self._graph = None  # noqa
            self._visits = None  # noqa
        return

    def calc_q(self,
               reprocess_count: int = 1) -> Dict[str, Dict[str, QVals]]:
        """
        For yhe already loaded session calculate and update q values for event by episode.
        :param reprocess_count: The number of times to process the events to allow for q value propagation
        """
        if reprocess_count < 0:
            reprocess_count = 1
        self._q_values = dict()
        if len(self._events) > 0:
            for i in range(reprocess_count):
                self._trace.log().info("Starting Q Value Calc")
                event_to_process = 0
                rept = max(1, int(len(self._events) / 10))
                while event_to_process < len(self._events):
                    if not self._events[event_to_process].episode_end:
                        self._update_q(state=self._events[event_to_process].state.state_as_string(),
                                       next_state=self._events[event_to_process + 1].state.state_as_string(),
                                       action=int(self._events[event_to_process].action),
                                       agent=self._events[event_to_process].agent,
                                       reward=self._events[event_to_process].reward,
                                       max_func=self._max_funcs[self._max_func_idx])
                    else:
                        self._update_q(state=self._events[event_to_process].state.state_as_string(),
                                       next_state=self.END_STATE,
                                       action=int(self._events[event_to_process].action),
                                       agent=self._events[event_to_process].agent,
                                       reward=self._events[event_to_process].reward,
                                       max_func=self._max_funcs[self._max_func_idx])
                    event_to_process += 1
                    if event_to_process % rept == 0:
                        self._trace.log().info(
                            "Processed {:3.0f}% of events".format((event_to_process / len(self._events)) * 100))
            self._trace.log().info("Done Q Value Calc")
        else:
            self._trace.log().error("No TicTacToe event data for session [{}]".format(self._session_uuid))
        return self._q_values

    def get_visits(self) -> Dict[str, int]:
        """
        Get the visits for this calc set and session
        :return: The visits loaded for the session uuid associated with this calc.
        """
        return self._visits

    def get_graph(self) -> nx.DiGraph:
        """
        Get the Graph for this calc set and session
        :return: The visits loaded for the session uuid associated with this calc.
        """
        return self._graph

    def get_events(self) -> List[TicTacToeEvent]:
        """
        Get the raw events for this calc set and session
        :return: The raw events
        """
        return self._events

    @staticmethod
    def zero_if_nan(v: float) -> float:
        """
        Return zero if the value is NaN else return the value
        :param v: The values to check
        :return: Max ignoring NaN or 0 if all NaN
        """
        r = v
        if np.isnan(r):
            r = float(0)
        return r
