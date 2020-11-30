import numpy as np
from src.lib.envboot.env import Env
from src.lib.rltrace.trace import Trace
from src.reflrn.interface.agent import Agent
from src.reflrn.interface.agent_factory import AgentFactory
from src.reflrn.interface.state import State
from src.tictactoe.explore.explore import Explore


class RandomPlayAgent(Agent):
    class RandomAgentFactory(AgentFactory):

        def new_x_agent(self) -> Agent:
            return RandomPlayAgent(agent_id=Agent.X_ID, agent_name=Agent.X_NAME)

        def new_o_agent(self) -> Agent:
            return RandomPlayAgent(agent_id=Agent.O_ID, agent_name=Agent.O_NAME)

    _env: Env
    _trace: Trace
    _id: int
    _name: str
    _explore: Explore
    _prev_state: State

    def __init__(self,
                 agent_id: int,
                 agent_name: str):
        self._env = Env()
        self._trace = self._env.get_trace()
        self._id = agent_id
        self._name = agent_name
        self._explore = None
        self._prev_state = None
        self._trace.log().debug("Agent created => {}:{}".format(self._id, self._name))
        return

    def id(self):
        """
        The unique id of the Agent
        :return: The unique id of the Agent as string
        """
        return self._id

    def name(self):
        """
        The name of the Agent
        :return: The name of the Agent as string
        """
        return self._name

    def terminate(self,
                  save_on_terminate: bool = False) -> None:
        """
        Callback for agent to process notification of termination
        :param save_on_terminate: If True agent should save its state on exit
        """
        self._trace.log().debug("Agent notified of termination => {}:{}".format(self._id, self._name))
        return

    def episode_init(self, state: State) -> None:
        """
        Callback for agent to process notification of a new episode
        :param state: The opening state of the episode
        """
        self._prev_state = None
        self._trace.log().debug("Agent notified of episode start => {}:{}".format(self._id, self._name))
        return

    def episode_complete(self, state: State) -> None:
        """
        Callback for agent to process notification of episode completion
        :param state: The state at as episode completion
        """
        self._trace.log().debug("Agent notified of episode completion => {}:{}".format(self._id, self._name))
        return

    def choose_action(self, state: State, possible_actions: [int]) -> int:
        """
        Request for Agent to select an action.
        :param state: The current state of the environment
        :param possible_actions: The possible actions left to play
        :return: The action to play as an int
        """
        return possible_actions[np.random.randint(len(possible_actions))]

    def reward(self,
               state: State,
               next_state: State,
               action: int,
               reward_for_play: float,
               episode_complete: bool) -> None:
        """
        Callback for agent to process the allocation of an award
        :param state: The current state of the environment
        :param next_state: The state of the environment after the given action
        :param action: The action that will transition from state to next_state
        :param reward_for_play: The reward given to teh agent for playing action in state
        :param episode_complete: True if the next_state represents a terminal state
        """
        #
        # If attached to an exploration record this event to visit and network (graph) so that it
        # can be saved for later investigation: Will create visit & graph YAML files.
        #
        if self._explore is not None:
            if self._prev_state is None:
                self._prev_state = state

            step = np.nansum(np.abs(state.state_model_input()))
            prev_state = self._explore.other(self).get_prev_state()
            if prev_state is not None:
                prev_state = prev_state.state_as_string()
            self._explore.record(prev_state=prev_state,
                                 curr_state=state.state_as_string(),
                                 curr_state_is_episode_end=episode_complete,
                                 step=step)
            self._prev_state = state
        return

    def session_init(self,
                     actions: dict) -> None:
        """
        Callback to allow agent to process the initialisation of a session
        :param actions: The actions that will be supported by the session
        """
        return

    def __str__(self):
        """
        Render Agent as string
        :return: Agent as string
        """
        return "{}:{}".format(str(self._id), self._name)

    def __repr__(self):
        """
        Render Agent as human readable
        :return: Agent as human readable
        """
        return self.__str__()

    @property
    def explain(self) -> bool:
        """
        TBC
        :return: TBC
        """
        raise NotImplementedError()

    def attach_to_explore(self,
                          explore: Explore) -> None:
        """
        Attach the agent to an exploration such that it's activity can be tracked
        :param explore: The Exploration to attach to
        """
        self._explore = explore
        return

    def get_prev_state(self) -> State:
        """
        Get the previous state seen by this agent
        :return: The previous state that was seen
        """
        return self._prev_state
