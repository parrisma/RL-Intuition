import numpy as np
from typing import Dict, List
from src.lib.rltrace.trace import Trace
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
    gamma: float  # How much do we value future rewards
    learning_rate: float  # Min (initial) size of Q Value update
    num_actions: int
    q_values: Dict[str, QVals]  # State str : Key -> actions q values & #num times reward seen

    def __init__(self,
                 trace: Trace,
                 ttt_event_stream: TicTacToeEventStream):
        self._trace = trace
        self._ttt_event_stream = ttt_event_stream
        self.num_actions = 9  # max number of actions in any given board state
        self.gamma = 0.9
        self.learning_rate = 0.1
        self.q_values = dict()
        return

    def _get_state_q_values(self,
                            state: str) -> np.ndarray:
        """
        Get the Q values of the given state.
        :param state: The state to get Q Values for
        :return: The Q Values for given state as numpy array of float
        """
        if state not in self.q_values:
            self.q_values[state] = QVals(state=state)
        return self.q_values[state].q_vals

    def _set_state_q_values(self,
                            state: str,
                            q_values: np.ndarray) -> None:
        """
        Set the Q values of the given state.
        :param state: The state to set Q Values for
        :param q_values: The Q Values to set for the given state
        """
        if state not in self.q_values:
            self.q_values[state] = QVals(state=state)
        self.q_values[state].q_vals = q_values
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

    def _update_q(self,
                  state: str,
                  next_state: str,
                  action: int,
                  reward: float) -> None:
        """
        Update internal Q Values with the given State/Action/Reward update
        :param state: The current state
        :param next_state: The state after the given action is taken
        :param action: The action taken in the given state
        :param reward: The reward on arrival in the next_state
        """
        q_state = self._get_state_q_values(state)
        q_next_state = self._get_state_q_values(next_state)
        q_update = self.learning_rate * (reward + (self.gamma * (np.max(q_next_state) - q_state[action])))
        q_state[action] = q_state[action] + q_update
        q_state = QCalc.normalize(q_state)
        self._set_state_q_values(state, q_state)
        return

    def _load_session(self,
                      session_uuid: str) -> List['TicTacToeEvent']:
        """
        Load the given session UUID
        :param session_uuid: An existing session UUID to load
        """
        self._trace.log().info("Start loading events for session {}".format(session_uuid))
        try:
            events = self._ttt_event_stream.get_session(session_uuid=session_uuid)
            self._trace.log().info("Loaded [{}] events for session {}".format(len(events), session_uuid))
        except RuntimeError as _:
            self._trace.log().error("Failed to load events for session [{}]".format(session_uuid))
            events = None
        return events

    def calc_q(self,
               session_uuid: str) -> Dict[str, QVals]:
        """
        Load all episodes for given session and iteratively update q values for event by episode.
        :param session_uuid: The session uuid to lead events for.
        """
        self.q_values = dict()
        events = self._load_session(session_uuid=session_uuid)
        if len(events) > 0:
            for i in range(0, 50):
                self._trace.log().info("Starting Q Value Calc")
                event_to_process = 0
                rept = max(1, int(len(events) / 10))
                while event_to_process < len(events):
                    if not events[event_to_process].episode_end:
                        self._update_q(state=events[event_to_process].state.state_as_string(),
                                       next_state=events[event_to_process + 1].state.state_as_string(),
                                       action=int(events[event_to_process].action),
                                       reward=events[event_to_process].reward)
                    else:
                        self._update_q(state=events[event_to_process].state.state_as_string(),
                                       next_state="End",
                                       action=int(events[event_to_process].action),
                                       reward=events[event_to_process].reward)
                    event_to_process += 1
                    if event_to_process % rept == 0:
                        self._trace.log().info(
                            "Processed {:3.0f}% of events".format((event_to_process / len(events)) * 100))
            self._trace.log().info("Done Q Value Calc")
        else:
            self._trace.log().error("No TicTacToe event found for session [{}]".format(session_uuid))
        return self.q_values
