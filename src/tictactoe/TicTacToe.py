from typing import Tuple, Dict, List
from random import randint

import numpy as np

from src.tictactoe.TicTacToeState import TicTacToeState
from src.reflrn.interface.agent import Agent
from src.reflrn.interface.environment import Environment
from src.reflrn.interface.state import State
from src.lib.envboot.env import Env
from src.tictactoe.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.tictacttoe_event import TicTacToeEvent
from src.lib.rltrace.trace import Trace
from src.lib.uniqueref import UniqueRef


class TicTacToe(Environment):
    # There are 5812 legal board states that can be reached before there is a winner
    # http://brianshourd.com/posts/2012-11-06-tilt-number-of-tic-tac-toe-boards.html
    __env: Env
    __trace: Trace
    __ttt_event_stream: TicTacToeEventStream
    __session_uuid: str
    __episode_uuid: str
    __episode_step: int
    __board: np.array
    __last_board: np.array
    __agent: Agent
    __last_agent: Agent
    __x_agent: Agent
    __o_agent: Agent
    __next_agent: Dict
    __agents = Dict

    play_reward = float(-1)  # reward for playing an action
    draw_reward = float(-10)  # reward for playing to end but no one wins
    win_reward = float(100)  # reward for winning a game
    __no_agent = None
    __win_mask = np.full((1, 3), 3, np.int8)
    __actions = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2), 6: (2, 0), 7: (2, 1), 8: (2, 2)}
    __drawn = "draw"
    __games = "games"
    __states = "states"
    empty_cell = np.nan  # value of a free action space on board
    asStr = True
    attribute_draw = ("Draw", "True if episode ended in a drawn curr_coords", bool)
    attribute_won = ("Won", "True if episode ended in a win curr_coords", bool)
    attribute_complete = ("Complete", "True if the environment is in a complete curr_coords for any reason", bool)
    attribute_agent = ("agent", "The last agent to make a move", Agent)
    attribute_board = (
        "board", "The game board as a numpy array (3,3), np.nan => no move else the id of the agent", np.array)
    __episode = 'episode number'

    #
    # Constructor has no arguments as it just sets the game
    # to an initial up-played set-up
    #
    def __init__(self,
                 env: Env,
                 ttt_event_stream: TicTacToeEventStream,
                 x: Agent,
                 o: Agent,
                 x_to_start: bool = None):
        """
        Establish TicTacToe environment with both Agents ready to play
        :param env: The Environment to attach to
        :param x: The Agent to play the X role
        :param o: The Agent to play to O role
        """
        self.__session_uuid = None
        self.__episode_uuid = None
        self.__episode_step = None

        self.__env = env
        self.__ttt_event_stream = ttt_event_stream
        self.__trace = self.__env.get_trace()

        self.__board = None
        self.__last_board = None
        self.__agent = None
        self.__last_agent = None

        self.__x_agent = x
        self.__o_agent = o
        self.__next_agent = {x.name(): o, o.name(): x}
        self.__x_agent.session_init(self.actions())
        self.__o_agent.session_init(self.actions())
        self.__agents = dict()
        self.__agents[self.__o_agent.id()] = self.__o_agent
        self.__agents[self.__x_agent.id()] = self.__x_agent

        self._x_to_start = x_to_start

        self.__session_start()
        return

    def __del__(self):
        """
        Ensure episode and sessions are closed before exit
        """
        self.__episode_end()
        self.__session_end()
        return

    def episode_start(self) -> Tuple[State, Agent, str]:
        """
        Establish internal state to be that as at start of new Episode. If an existing Episode is in progress that
        Episode will be formally closed out.
        :return: The board state, the first agent to play and the UUID of this episode
        """
        if self.__session_uuid is None:
            raise RuntimeError("Episode cannot be started outside of an active session")
        if self.__episode_uuid is not None:
            self.__episode_end()

        self.__episode_uuid = UniqueRef().ref
        self.__trace.log().info("Start Episode [{}]".format(self.__episode_uuid))
        self.__episode_step = 0
        self.__board = TicTacToe.__empty_board()
        self.__last_board = None
        self.__agent = TicTacToe.__no_agent
        self.__last_agent = TicTacToe.__no_agent
        state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)
        self.__x_agent.episode_init(state)
        self.__o_agent.episode_init(state)
        if self._x_to_start is None:
            agent = (self.__x_agent, self.__o_agent)[randint(0, 1)]
        else:
            if self._x_to_start:
                agent = self.__x_agent
            else:
                agent = self.__o_agent
        return state, agent, self.__episode_uuid

    def __episode_end(self) -> None:
        """
        End the current episode.
        """
        if self.__episode_uuid is not None:
            self.__trace.log().info("End Episode [{}]".format(self.__episode_uuid))
            self.__episode_uuid = None
            self.__episode_step = 0
        return

    def __session_end(self) -> None:
        """
        End the current session
        """
        if self.__session_uuid is not None:
            self.__trace.log().info("End Session [{}]".format(self.__session_uuid))
            self.__episode_end()
            self.__session_uuid = None
        return

    def __session_start(self) -> None:
        """
        Start a new session
        """
        self.__session_end()
        self.__session_uuid = self.__ttt_event_stream.session_uuid  # inherit same session uuid as event stream
        self.__trace.log().info("Start Session [{}]".format(self.__session_uuid))
        return

    def run(self, num_episodes: int) -> List[str]:
        """
        Play the given number of episodes with the Agents supplied as part of class Init. Select a random Agent
        to go at the start of each episode.
        :param num_episodes: The number of episodes to play
        """
        episodes = list()
        i = 0
        while len(episodes) < num_episodes:
            state, agent, episode_uuid = self.episode_start()
            episodes.append(episode_uuid)
            while not self.episode_complete():
                agent = self.__play_action(agent)
                i = len(episodes)
                if i % 500 == 0:
                    self.__trace.log().info("Iteration: " + str(i))

            state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)
            self.__x_agent.episode_complete(state)
            self.__o_agent.episode_complete(state)
        self.__x_agent.terminate()
        self.__o_agent.terminate()
        return episodes

    @classmethod
    def __empty_board(cls):
        """
        Return an empty board
        :return: an empty bpard as numpy array
        """
        return np.full((3, 3), np.nan)

    @classmethod
    def no_agent(cls):
        return cls.__no_agent

    @classmethod
    def actions(cls,
                state: State = None) -> dict:
        """
        Return the actions as a list of integers. If no state is given return the list of all
        action else return the list of actions valid in this state.
        :param state: The (optional) state to return actions for, if not given use current internal state
        :return: Dictionary of actions & action values.
        """
        if state is None:
            return TicTacToe.__actions
        else:
            return np.array(list(TicTacToe.__actions.keys()))[np.isnan(state.state()).reshape(9)]

    def __take_action(self, action: int, agent: Agent) -> None:
        """
        Assume the play_action has been validated by play_action method
        Make a deep_copy of board before play_action is made and the last player
        :param action: The action to play
        :param agent: The Agent playing teh action
        """
        self.__episode_step += 1
        self.__last_board = np.copy(self.__board)
        self.__last_agent = self.__agent
        self.__agent = agent
        self.__board[self.__actions[action]] = self.__agent.id()
        return

    def __assign_reward(self,
                        action: int,
                        state: State,
                        next_state: State,
                        agent: Agent,
                        other_agent: Agent) -> None:
        """
        Assign a reward to the agent for the given action that was just played
        :param action: The action just played
        :param state: The state before the action
        :param next_state: The state after the action
        :param agent: The agent that played the action
        :param other_agent: The agent the action was played against
        """
        episode_end = False
        reward = self.play_reward
        episode_outcome = TicTacToeEvent.STEP
        if self.episode_complete():
            episode_end = True
            attributes = self.attributes()
            if attributes[self.attribute_won[0]]:
                reward = self.win_reward
                if agent == self.__x_agent:
                    episode_outcome = TicTacToeEvent.X_WIN
                else:
                    episode_outcome = TicTacToeEvent.O_WIN
            if attributes[self.attribute_draw[0]]:
                reward = self.draw_reward
                episode_outcome = TicTacToeEvent.DRAW

        agent.reward(state, next_state, action, reward, episode_end)
        self.__ttt_event_stream.record_event(episode_uuid=self.__episode_uuid,
                                             episode_step=self.__episode_step,
                                             state=state,
                                             action="{}".format(action),
                                             reward=reward,
                                             episode_end=episode_end,
                                             episode_outcome=episode_outcome)
        return

    def __play_action(self, agent: Agent) -> Agent:
        """
        Make the play chosen by the given agent. If it is a valid play
        confer reward and switch play to other agent. If invalid play
        i.e. play in a cell where there is already a marker confer
        penalty and leave play with the same agent.

        :param agent: The Agent to play the next move
        :return: The Agent that will play next
        """

        other_agent = self.__next_agent[agent.name()]
        state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)

        # Make the play on the board.
        action = agent.choose_action(state, self.__actions_ids_left_to_take())
        if action not in self.__actions_ids_left_to_take():
            raise TicTacToe.IllegalActorAction("Actor Proposed Illegal action in current state :" + str(action))
        self.__take_action(action, agent)
        next_state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)

        self.__assign_reward(action=action,
                             state=state,
                             next_state=next_state,
                             agent=agent,
                             other_agent=other_agent)
        return other_agent  # play moves to next agent

    def attributes(self):
        """
        The attributes of the current game state
        :return: Dictionary of attributes and their current values.
        """
        attr_dict = dict()
        attr_dict[TicTacToe.attribute_won[0]] = self.__episode_won()
        attr_dict[TicTacToe.attribute_draw[0]] = False
        if not attr_dict[TicTacToe.attribute_won[0]]:
            attr_dict[TicTacToe.attribute_draw[0]] = not self.__actions_left_to_take()
        attr_dict[TicTacToe.attribute_complete[0]] = \
            attr_dict[TicTacToe.attribute_draw[0]] or attr_dict[TicTacToe.attribute_won[0]]
        attr_dict[TicTacToe.attribute_agent[0]] = self.__agent
        attr_dict[TicTacToe.attribute_board[0]] = np.copy(self.__board)
        return attr_dict

    def __episode_won(self,
                      board=None) -> bool:
        """
        Return True if the current board has a winning row on it.
        :param board: If supplied return based on given board else use the internal board state
        :return: True if board has a winning move on it.
        """
        if board is None:
            board = self.__board
        rows = np.abs(np.sum(board, axis=1))
        cols = np.abs(np.sum(board, axis=0))
        diag_lr = np.abs(np.sum(board.diagonal()))
        diag_rl = np.abs(np.sum(np.rot90(board).diagonal()))

        if np.sum(rows == 3) > 0:
            return True
        if np.sum(cols == 3) > 0:
            return True
        if not np.isnan(diag_lr):
            if ((np.mod(diag_lr, 3)) == 0) and diag_lr > 0:
                return True
        if not np.isnan(diag_rl):
            if ((np.mod(diag_rl, 3)) == 0) and diag_rl > 0:
                return True
        return False

    def __actions_left_to_take(self,
                               board=None) -> bool:
        """
        Return True if there are any actions left to take given the current board state
        :param board: If supplied return based on given board else use the internal board state
        :return: True if actions remain on board
        """
        if board is None:
            board = self.__board
        return board[np.isnan(board)].size > 0

    def __actions_ids_left_to_take(self,
                                   board=None) -> np.array:
        """
        The possible game actions remaining given the board state
        :param board: If given the actions for the given board else return actions for internal board state
        :return: The actions as a Numpy array of int
        """
        if board is None:
            board = self.__board
        alt = np.reshape(board, board.size)
        alt = np.fromiter(self.actions().keys(), int)[np.isnan(alt)]
        return alt

    def episode_complete(self,
                         state: State = None) -> bool:
        """
        Return True if the given game state represents a terminal game state
        :param state:
        :return:
        """
        board = None
        if state is not None:
            board = state.state()

        if self.__episode_won(board) or not self.__actions_left_to_take(board):
            return True
        return False

    def __string_to_internal_state(self, moves_as_str) -> None:
        """
        Establish current game state to match that of the given structured text string. The steps are not simulated
        but just loaded directly as a board state.
        :param moves_as_str: The game state as structured text to load internally
        """
        mvs = moves_as_str.split('~')
        if moves_as_str is not None:
            for mv in mvs:
                if len(mv) > 0:
                    pl, ps = mv.split(":")
                    self.__take_action(int(ps), self.__agents[int(pl)])
        return

    def __internal_state_to_string(self) -> str:
        """
        Render the current game state as a structured text string
        :return: Current game state as structures text
        """
        mvs = ""
        bd = np.reshape(self.__board, self.__board.size)
        cell_num = 0
        for actor in bd:
            if not np.isnan(actor):
                mvs += str(int(actor)) + ":" + str(int(cell_num)) + "~"
            cell_num += 1
        if len(mvs) > 0:
            mvs = mvs[:-1]
        return mvs

    def state_as_str(self) -> str:
        """
        The current TicTacToe game state as a string
        :return: Game state as string
        """
        return self.__internal_state_to_string()

    def export_state(self) -> str:
        """
        The current TicTacToe game state as structured text that can be parsed
        :return: Game state as structured test string
        """
        return self.__internal_state_to_string()

    def import_state(self,
                     state_as_string: str):
        """
        Set game state to that of the structured text string
        :param state_as_string: The structured text from which to establish the game state
        :return:
        """
        # Must be an open episode
        if self.__episode_uuid is None:
            self.episode_start()
        self.__string_to_internal_state(state_as_string)
        return

    def state(self) -> State:
        """
        The current game state as a State object
        :return: The current game state as a State object
        """
        return TicTacToeState(self.__board,
                              self.__x_agent,
                              self.__o_agent)

    class IllegalActorAction(Exception):
        """
        Given actor was not in correct state to participate in a game
        """

        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)
