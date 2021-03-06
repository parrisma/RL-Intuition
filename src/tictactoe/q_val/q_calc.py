import numpy as np
import networkx as nx
from typing import Dict, List, Callable, Tuple
from src.lib.rltrace.trace import Trace
from src.tictactoe.ttt.tictactoe import TicTacToe
from src.tictactoe.agent.simulation_agent import SimulationAgent
from src.tictactoe.event.tictacttoe_event import TicTacToeEvent
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.q_val.q_vals import QVals


class QCalc:
    class Player:
        """
        Elements that are specific to a given player
        """
        q_values: Dict[str, Dict[str, QVals]]  # State str : Key -> actions q values & #num times reward seen
        q_hist: Dict[str, Dict[str, List[Tuple[str, int, float, float, float, float, float]]]]

        def __init__(self):
            self.q_values = dict()
            self.q_hist = dict()
            return

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
    _events: List[TicTacToeEvent]
    _visits: Dict[str, int]
    _graph: nx.DiGraph
    _session_uuid: str
    _ttt: TicTacToe
    _optimal_func_idx: int
    _optimal_funcs: List[Callable[[str, str, str, Dict], np.float]]
    _players: Dict[str, Player]

    NAIVE_OPTIMAL = 0
    OPTIMAL = 1

    END_STATE = "End"

    HIST_FMT = "{}:{}={}"

    # Q History element indexes
    QH_NEXT_STATE = 0
    QH_ACTION = 1
    QH_REWARD = 2
    QH_NS_MAX = 3
    QH_OLD_Q = 4
    QH_UPDATE = 5
    QH_NEW_Q = 6

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
        self._events = None  # noqa
        self._visits = None  # noqa
        self._graph = None  # noqa
        self._load_session(session_uuid=self._session_uuid)

        agent_factory = SimulationAgent.SimulationAgentFactory()
        self._ttt = TicTacToe(trace=self._trace,
                              ttt_event_stream=self._ttt_event_stream,
                              x=agent_factory.new_x_agent(),
                              o=agent_factory.new_o_agent())

        self._optimal_funcs = [self._naive_optimal,
                               self._optimal]
        self._optimal_func_idx = self.OPTIMAL

        self._players = dict()

        return

    def _init_q(self,
                player: str,
                agent: str,
                state: str) -> None:
        """
        Create empty q-values if they do not exist for the given player, agent, state
        :param player: The Player name
        :param agent: The Agent name
        :param state: The State as string
        """
        if player not in self._players:
            self._players[player] = QCalc.Player()
        if agent not in self._players[player].q_values:
            self._players[player].q_values[agent] = dict()
        if state not in self._players[player].q_values[agent]:
            self._players[player].q_values[agent][state] = QVals(state=state)
        return

    def _get_state_q_values(self,
                            player: str,
                            agent: str,
                            state: str) -> np.ndarray:
        """
        Get the Q values of the given agent in the given state.
        :param player: The player name
        :param agent: The Agent to get Q Values for
        :param state: The state to get Q Values for
        :return: The Q Values for given agent in the given state as numpy array of float
        """
        self._init_q(player, agent, state)
        return self._players[player].q_values[agent][state].q_vals

    def _set_state_q_values(self,
                            player: str,
                            agent: str,
                            state: str,
                            q_values: np.ndarray) -> None:
        """
        Set the Q values of the given agent in the given state.
        :param player: The player name
        :param agent: The agent to set q values for
        :param state: The state to set Q Values for
        :param q_values: The Q Values to set for the given agent in the given state
        """
        self._init_q(player, agent, state)
        self._players[player].q_values[agent][state].q_vals = q_values
        return

    def _optimal(self,
                 player: str,
                 agent: str,
                 next_state: str,
                 **kwargs) -> np.float:
        """
        The optimal action is to maximise the gain or minimise the loss. So if the absolute loss is greater
        than the gain return the loss else the gain.

        :param player: The player name
        :param agent: The agent to get the max for
        :param next_state: The state to estimate the max for
        :param kwargs: Optional arguments
        :return: The estimated optimal reward from the next_state
        """
        os_max = np.nan
        if not next_state == self.END_STATE and next_state is not None:
            qv = self._get_state_q_values(player, agent, next_state)
            if not np.isnan(qv).all():
                os_max = np.nanmax(qv)
                os_min = np.nanmin(qv)
                if np.abs(os_min) > np.abs(os_max):
                    os_max = os_min
        return os_max

    def _naive_optimal(self,
                       player: str,
                       agent: str,
                       next_state: str,
                       **kwargs) -> np.float:
        """
        Estimate the max reward available from the next state by simply taking the max Q-Val from the
        next state.
        :param player: The player name
        :param agent: The Agent to get the max for
        :param next_state: The next state to inspect
        :return: The max (estimated) reward available at the next state for the given agent
        """
        os_max = np.nan
        if not next_state == self.END_STATE and next_state is not None:
            qv = self._get_state_q_values(player, agent, next_state)
            if not np.isnan(qv).all():
                os_max = np.nanmax(qv)
        return os_max

    def _init_hist(self,
                   player: str,
                   agent: str,
                   state: str,
                   action: int) -> str:
        """
        Create empty q-history if it do not exist for the given player, agent, state
        :param player: The Player name
        :param agent: The Agent name
        :param state: The State as string
        :param action: The action played by the agent
        :return: The history key for the given state, agent & action
        """
        if player not in self._players:
            self._players[player] = QCalc.Player()
        if agent not in self._players[player].q_hist:
            self._players[player].q_hist[agent] = dict()
        hist_key = self.HIST_FMT.format(state, agent, action)
        if hist_key not in self._players[player].q_hist[agent]:
            self._players[player].q_hist[agent][hist_key] = list()
        return hist_key

    def _record_q_hist(self,
                       player: str,
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
        hist_key = self._init_hist(player, agent, state, action)
        self._players[player].q_hist[agent][hist_key].append((next_state, action, reward, ns_max, old_q, update, new_q))
        return

    def get_q_hist(self,
                   player: str,
                   agent: str,
                   state: str,
                   action: int) -> List[Tuple[str, int, float, float, float, float, float]]:
        """
        Get the history of updates for a given agent in teh given state - action Q Value
        :param player: The player name
        :param agent: The agent to get the history for.
        :param state: The state of interest
        :param action: The action within the state to get history for
        :return: List of Q Value Updates in order where an update = [next_state, next_state_max, old, update, new]
        """
        hist_key = self._init_hist(player, agent, state, action)
        res = None
        if agent in self._players[player].q_hist:
            if hist_key in self._players[player].q_hist[agent]:
                res = self._players[player].q_hist[agent][hist_key]
        return res

    def _update_q_vals(self,
                       state: str,
                       next_state: str,
                       action: int,
                       agent: str,
                       reward: float,
                       max_func: Callable[[str, str, str, Dict], np.float]) -> None:
        """
        Update internal Q Values with the given State/Action/Reward update for the given player.

        So update =

        1. Agent + Reward
        2. Other Agent +  Reward from other Agent perspective

        :param state: The current state
        :param next_state: The state after the given action is taken
        :param action: The action taken in the given state
        :param agent: The Agent (player) that took the given action and for whom the reward was awarded
        :param reward: The reward on arrival in the next_state
        """
        other_agent = self._ttt.get_other_agent(agent).name()
        other_reward = self._ttt.get_other_reward(reward)
        # ,
        #                         [other_agent, [[agent, other_reward], [other_agent, reward]]]
        perspectives = [[agent, [[agent, reward]]],
                        [other_agent, [[agent, other_reward]]]
                        ]
        # Both players must see all actions
        for player, view in perspectives:
            # Each player updates q from both perspectives
            for a, r in view:
                self._update_player_q_for_agent_perspective(player=player,
                                                            state=state,
                                                            next_state=next_state,
                                                            action=action,
                                                            agent=a,
                                                            reward=r,
                                                            max_func=max_func)
        return

    def _update_player_q_for_agent_perspective(self,
                                               player: str,
                                               state: str,
                                               next_state: str,
                                               action: int,
                                               agent: str,
                                               reward: float,
                                               max_func: Callable[[str, str, str, Dict], np.float]) -> None:
        """
        Update internal Q Values with the given State/Action/Reward update for the given agent

        Where: Q[state, action] = Q[state, action]
                                  + learning_rate * (reward + gamma * agent_max(Q[new_state, :]) — Q[state, action])
        and: agent_max = the max (optimum) reward possible after moving to next_state

        :param player: The name of the player to update
        :param state: The current state
        :param next_state: The state after the given action is taken
        :param action: The action taken in the given state
        :param agent: The Agent (player) that took the given action
        :param reward: The reward on arrival in the next_state
        """
        if next_state != self.END_STATE:
            q_state = self._get_state_q_values(player, agent, state)  # Current Q Values for entire state
            q_val_prev = q_state[action]  # Current Q Value for specific action
            oa = self._ttt.get_other_agent(agent).name()
            next_state_max = self.zero_if_nan(max_func(player=player, agent=oa, next_state=next_state))  # noqa
            q_update = self._learning_rate * (reward + self._gamma * next_state_max - self.zero_if_nan(q_val_prev))
            q_state[action] = self.zero_if_nan(q_val_prev) + q_update
            self._set_state_q_values(player, agent, state, q_state)
            self._record_q_hist(player, agent, state, next_state, next_state_max, action, reward, q_val_prev, q_update,
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
            self._trace.log().info("Loaded [{}] events for session {}".format(len(self._events), session_uuid))

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
               reprocess_count: int = 1) -> Dict[str, Player]:
        """
        For the already loaded session calculate and update q values for event by episode.
        :param reprocess_count: The number of times to process the events to allow for q value propagation
        """
        if reprocess_count < 0:
            reprocess_count = 1
        self._players = dict()

        if len(self._events) > 0:
            for i in range(reprocess_count):
                self._trace.log().info("Starting Q Value Calc - iteration [{}] of [{}]".format(i, reprocess_count))
                event_to_process = 0
                rept = max(1, int(len(self._events) / 10))
                while event_to_process < len(self._events):
                    if not self._events[event_to_process].episode_end:
                        self._update_q_vals(state=self._events[event_to_process].state.state_as_string(),
                                            next_state=self._events[
                                                event_to_process + 1].state.state_as_string(),
                                            action=int(self._events[event_to_process].action),
                                            agent=self._events[event_to_process].agent,
                                            reward=self._events[event_to_process].reward,
                                            max_func=self._optimal_funcs[self._optimal_func_idx])
                    else:
                        self._update_q_vals(state=self._events[event_to_process].state.state_as_string(),
                                            next_state=self.END_STATE,
                                            action=int(self._events[event_to_process].action),
                                            agent=self._events[event_to_process].agent,
                                            reward=self._events[event_to_process].reward,
                                            max_func=self._optimal_funcs[self._optimal_func_idx])
                    event_to_process += 1
                    if event_to_process % rept == 0:
                        self._trace.log().info(
                            "Processed {:3.0f}% of events".format((event_to_process / len(self._events)) * 100))
            self._trace.log().info("Done Q Value Calc")
        else:
            self._trace.log().error("No TicTacToe event data for session [{}]".format(self._session_uuid))
        return self._players

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

    def q_vals_as_simple(self) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """
        Render the Q Values as simple data structures so they can be encoded as JSON
            Dict[player, Dict[agent, Dict[state, List[q-values]]]]
        :return: Q Values for as Dict[str, Dict[str, Dict[str, List[float]]]]
        """
        simple = dict()
        for k1, v1 in self._players.items():
            simple[k1] = dict()
            for k2, v2 in v1.q_values.items():
                simple[k1][k2] = dict()
                for k3, v3 in v2.items():
                    simple[k1][k2][k3] = v3.q_vals.tolist()

        return simple

    def q_convergence_by_level(self) -> List[List[float]]:
        """
        Return the averaged Q Value history by level
        :return: A list with an entry for every level present with the average history for that level.
        """
        level_buckets = dict()
        level_min_size = dict()
        for player in [self._ttt.x_agent_name(), self._ttt.o_agent_name()]:
            for state, q_val_hist in self._players[player].q_hist[player].items():
                state = state[0: state.find(":")]
                level = state.count("1")
                if level not in level_buckets:
                    level_buckets[level] = list()
                    level_min_size[level] = list()
                level_buckets[level].append(q_val_hist)
                level_min_size[level].append(len(q_val_hist))

        for level in level_min_size:
            szs = np.array(level_min_size[level])
            cutoff = max(0, int(szs.mean() - (szs.std() * 2)))
            level_min_size[level] = cutoff

        for level in level_min_size:
            cropped = list()
            for hst in level_buckets[level]:
                if len(hst) >= max(level_min_size[level], 10):
                    cropped.append(hst)
            level_buckets[level] = cropped

        res = list()
        avg = None
        for level in range(0, 9):
            if level in level_buckets and len(level_buckets[level]) > 0:
                sz = level_min_size[level]
                avg = np.zeros(sz)
                i = 0
                for q_hist in level_buckets[level]:
                    for hst in q_hist:
                        if i < sz:
                            avg[i] += np.abs(np.nan_to_num(hst[self.QH_NEW_Q]))  # 6th element is updated q val
                        i += 1
                    i = 0
                avg /= len(level_buckets[level])
                res.append(avg.tolist())
            else:
                self._trace.log().info(
                    "Level [{}] has insufficient history for meaningful convergence analysis".format(level))
                res.append(np.nan)
        return res

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
