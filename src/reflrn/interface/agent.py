import abc

from src.reflrn.interface.state import State


class Agent(metaclass=abc.ABCMeta):
    """
    An Agent that can participate in a reinforcement learning environment
    """
    X_ID: int
    O_ID: int
    X_NAME: str
    O_NAME: str

    X_ID = -1
    O_ID = 1
    X_NAME = "X"
    O_NAME = "O"

    @abc.abstractmethod
    def id(self) -> int:
        """
        The immutable numerical id of teh agent
        :return: The id of the agent as int
        """
        pass

    @abc.abstractmethod
    def name(self) -> str:
        """
        The immutable name of the agent
        :return: The name of teh agent as string
        """
        pass

    @abc.abstractmethod
    def session_init(self, actions: dict) -> None:
        """
        Called by the environment *once* at the start of the session and the action set is given as dictionary
        :param actions: The actions the agent will be able to play in the environment it is being attached to
        """
        pass

    @abc.abstractmethod
    def terminate(self,
                  save_on_terminate: bool = False) -> None:
        """
        Called by the environment *one* when environment shuts down
        :param save_on_terminate:
        """
        pass

    @abc.abstractmethod
    def episode_init(self, state: State) -> None:
        """
        Called by the environment at the start of every episode
        :param state: The state as at the episode start
        """
        pass

    @abc.abstractmethod
    def episode_complete(self, state: State) -> None:
        """
        Called by the environment at the episode end (no more viable actions to play in given state)
        :param state: The state as at the episode end
        """
        pass

    @abc.abstractmethod
    def choose_action(self, state: State,
                      possible_actions: [int]) -> int:
        """
        Called by the environment to ask for the action the agent will play in the given state
        :param state: The state the agent is to play an action in
        :param possible_actions: The allowable actions in the given state
        :return: The action from the list if possible_actions the agent has chosen to play in the given state
        """
        pass

    @abc.abstractmethod
    def reward(self,
               state: State,
               next_state: State,
               action: int,
               reward_for_play: float,
               episode_complete: bool) -> None:
        """
        The callback via which the environment informs the agent of a reward as a result of an action.
        :param state: The curr_coords *before* the action is taken : S
        :param next_state: The State after the action is taken : S'
        :param action: The action that transitioned S to S'
        :param reward_for_play: The reward for playing action in curr_coords S
        :param episode_complete: If environment is episodic, then true if the reward relates to the last reward in an episode.
        """
        pass
