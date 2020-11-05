from typing import Dict
import numpy as np
from src.tictactoe.experiment_base import ExperimentBase
from src.tictactoe.random_play_agent import RandomPlayAgent


class Experiment3(ExperimentBase):
    """
    This Experiment does not play games in the TicTacToe environment, instead it loads game data from experiment 1
    and does a brute force calculation of Q Values by state.

    This is only possible as TicTacToe has a very low number of discrete states (order 6'000) so we can hold the
    entire Q-Value <-> Action map in memory.

    By doing this we will be able to see how Q-Value works such that when we replace the in memory Q Value map with
    a Neural Network we will be able to validate that the NN is learning as expected.

    Q Value Update:
        Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])

    There is no Epsilon (discovery) factor as we are reading 'canned' random game data back from ttt_events, where
    every action was selected at random by the agents.
    """

    gamma: float  # How much do we value future rewards
    learning_rate: float  # Min (initial) size of Q Value update
    num_actions: int
    q_values: Dict[str, np.ndarray]  # State str : Key -> actions q values & #num times reward seen

    def __init__(self):
        # Needed to bootstrap the base environment but not used in this experiment so just use
        # random agent arbitrarily.
        super().__init__(RandomPlayAgent.RandomAgentFactory())
        self.num_actions = 9  # max number of actions in any given board state
        self.gamma = 0.9
        self.learning_rate = 0.1
        self.q_values = dict()
        return

    def get_state_q_values(self,
                           state: str) -> np.ndarray:
        """
        Get the Q values of the given state.
        :param state: The state to get Q Values for
        :return: The Q Values for given state as numpy array of float
        """
        if state not in self.q_values:
            self.q_values[state] = np.zeros((self.num_actions))
        return self.q_values[state]

    def set_state_q_values(self,
                           state: str,
                           q_values: np.ndarray) -> None:
        """
        Set the Q values of the given state.
        :param state: The state to set Q Values for
        :param q_values: The Q Values to set for the given state
        """
        if state not in self.q_values:
            self.q_values[state] = np.zeros((self.num_actions))
        self.q_values[state] = q_values
        return

    def update_q(self,
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
        q_state = self.get_state_q_values(state)
        q_next_state = self.get_state_q_values(next_state)
        q_update = self.learning_rate * (reward + (self.gamma * (np.max(q_next_state) - q_state[action])))
        q_state[action] = q_state[action] + q_update
        self.set_state_q_values(state, q_state)
        return

    def calc_q(self,
               session_uuid: str) -> None:
        """
        Load all episodes for given session and iteratively update q values for event by episode.
        :param session_uuid: The session uuid to lead events for.
        """
        events = self._ttt_event_stream.get_session(session_uuid=session_uuid)
        event_to_process = 0
        while event_to_process < len(events):
            if not events[event_to_process].episode_end:
                self.update_q(state=events[event_to_process].state.state_as_string(),
                              next_state=events[event_to_process + 1].state.state_as_string(),
                              action=int(events[event_to_process].action),
                              reward=events[event_to_process].reward)
            else:
                self.update_q(state=events[event_to_process].state.state_as_string(),
                              next_state="End",
                              action=int(events[event_to_process].action),
                              reward=events[event_to_process].reward)
            event_to_process += 1
        return

    def run(self) -> None:
        """
        Run the experiment where two random agents play against each other
        """
        self._trace.log().info("Experiment {} Started".format(self.__class__.__name__))
        self.calc_q(session_uuid="16bb709a30414471b0b7cc226a25c172")
        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment3().run()
