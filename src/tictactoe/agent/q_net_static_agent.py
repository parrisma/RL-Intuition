from typing import Dict, List
import numpy as np
from src.lib.envboot.env import Env
from src.lib.rltrace.trace import Trace
from src.reflrn.interface.agent import Agent
from src.reflrn.interface.agent_factory import AgentFactory
from src.reflrn.interface.state import State
from src.tictactoe.explore.explore import Explore
from src.tictactoe.util.vals_json import ValsJson


class QNetStaticAgent(Agent):
    class QAgentFactory(AgentFactory):

        def new_x_agent(self, q_model_save_location: str) -> Agent:
            """
            Create an X Agent from the given saved Q NN Model
            :param q_model_save_location: The file loca
            :return: QAgent as Player X
            """
            return QNetStaticAgent(agent_id=Agent.X_ID, agent_name=Agent.X_NAME, filename=q_model_save_location)

        def new_o_agent(self, q_vals_filename: str) -> Agent:
            """
            Create an X Agent from the given saved Q NN Model
            :param q_vals_filename: The filename of the q values as JSON
            :return: QAgent as Player O
            """
            return QNetStaticAgent(agent_id=Agent.O_ID, agent_name=Agent.O_NAME, filename=q_vals_filename)

    _env: Env
    _trace: Trace
    _id: int
    _name: str
    _explore: Explore
    _prev_state: State
    #  Dict[Player(X|O), Dict[AgentPerspective(X|O), Dict[StateAsStr('000000000'), List[of nine q values as float]]]]
    _q_values: Dict[str, Dict[str, Dict[str, List[float]]]]

    def __init__(self,
                 agent_id: int,
                 agent_name: str,
                 filename: str):
        """
        Bootstrap a Q Value Agent
        :param agent_id: The numerical id of the Agent
        :param agent_name: The name of the Agent
        :param filename: The existing filename from which Q Values should be loaded
        """
        self._env = Env()
        self._trace = self._env.get_trace()
        self._id = agent_id
        self._name = agent_name
        self._explore = None  # noqa
        self._prev_state = None  # noqa
        self._trace.log().debug("Agent created => {}:{}".format(self._id, self._name))
        try:
            self._q_values = ValsJson.load_values_from_json(filename=filename)
        except Exception as e:
            self._q_values = None  # noqa
            raise  # last exception
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
        action = None
        try:
            q_vals = self._q_values[self._name][self._name][state.state_as_string()]
            action = self._greedy_action(possible_actions=possible_actions,
                                         q_vals=np.asarray(q_vals, dtype=np.float))
        except Exception as _:
            self._trace.log().info("Q Agent failed to predict using Q - Random action selected for state [{}]".format(
                state.state_as_string()))
            action = possible_actions[np.random.randint(len(possible_actions))]
        return action

    @staticmethod
    def _greedy_action(possible_actions: [int],
                       q_vals: np.ndarray) -> int:
        """
        Return the valid action associated with the highest reward
        :return: The optimal action for the agent to take
        """
        greedy_actn = None
        res = list()
        if len(possible_actions) > 0:
            for actn in possible_actions:
                if not np.isnan(q_vals[actn]):
                    res.append([actn, q_vals[actn]])
            res = sorted(res, key=lambda x: x[1])
            if len(res) >= 1:
                greedy_actn = res[-1][0]
        return greedy_actn

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
            step = np.nansum(np.abs(state.state_model_input()))
            self._explore.record(prev_state=state.state_as_string(),
                                 curr_state=next_state.state_as_string(),
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
