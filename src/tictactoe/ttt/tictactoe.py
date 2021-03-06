from random import randint
from typing import Tuple, Dict, List
import numpy as np
from src.lib.rltrace.trace import Trace
from src.lib.uniqueref import UniqueRef
from src.reflrn.interface.agent import Agent
from src.reflrn.interface.environment import Environment
from src.reflrn.interface.state import State
from src.tictactoe.ttt.tictactoe_playerid import TicTacToePlayerId
from src.tictactoe.ttt.tictactoe_state import TicTacToeState
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.event.tictacttoe_event import TicTacToeEvent
from src.tictactoe.ttt.tictactoe_board_state import TicTacToeBoardState


class TicTacToe(Environment):
    # There are 8534 legal board states (see experiment 0)
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
    __next_agent: Dict[str, Agent]
    __agents = Dict[int, Agent]
    __reward_map = Dict[float, float]

    step_reward = float(0)  # reward for playing an action
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
    _X = 0
    _O = 1
    _D = 2
    _STAT_FMT = "Iter {:5} :- X Win [{:5}] O Win [{:5}] Draw [{:5}] as % X [{:6.2f}], O [{:6.2f}] D[{:6.2f}]"

    #
    # Constructor has no arguments as it just sets the game
    # to an initial up-played set-up
    #
    def __init__(self,
                 trace: Trace,
                 ttt_event_stream: TicTacToeEventStream,
                 x: Agent,
                 o: Agent,
                 x_to_start: bool = None):
        """
        Establish TicTacToe environment with both Agents ready to play
        :param trace: Trace handle
        :param ttt_event_stream: THe event stream to write logs to
        :param x: The Agent to play the X role
        :param o: The Agent to play the O role
        :param x_to_start: (optional) X Always starts if true
        """
        self.__reward_map = {self.step_reward: self.step_reward,
                             self.draw_reward: self.draw_reward,
                             self.win_reward: -self.win_reward}

        self.__session_uuid = None  # noqa
        self.__episode_uuid = None  # noqa
        self.__episode_step = None  # noqa

        self.__ttt_event_stream = ttt_event_stream
        self.__trace = trace

        self.__board = None
        self.__last_board = None

        self.__x_agent = x
        self.__o_agent = o
        self.__next_agent = {x.name(): o, o.name(): x}
        self.__agents = dict()
        self.__agents[self.__o_agent.id()] = self.__o_agent
        self.__agents[self.__x_agent.id()] = self.__x_agent
        self._x_to_start = x_to_start
        self.__agent = self._reset_agent()
        self.__last_agent = None  # noqa

        self.__x_agent.session_init(self.actions())
        self.__o_agent.session_init(self.actions())

        self.__session_start()
        return

    def __del__(self):
        """
        Ensure episode and sessions are closed before exit
        """
        self.__episode_end()
        self.__session_end()
        return

    def _reset_agent(self) -> Agent:
        """
        Set the initial agent
        :return: The selected Agent to take the next move
        """
        if self._x_to_start is None:
            agent = [self.__x_agent, self.__o_agent][randint(0, 1)]
        else:
            if self._x_to_start:
                agent = self.__x_agent
            else:
                agent = self.__o_agent
        return agent

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
        self.__trace.log().debug("Start Episode [{}]".format(self.__episode_uuid))
        self.__episode_step = 0
        self.__board = TicTacToe.__empty_board()
        self.__last_board = None
        self.__agent = self._reset_agent()
        self.__last_agent = TicTacToe.__no_agent
        state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)
        self.__x_agent.episode_init(state)
        self.__o_agent.episode_init(state)
        agent = self._reset_agent()
        return state, agent, self.__episode_uuid

    def __episode_end(self) -> None:
        """
        End the current episode.
        """
        if self.__episode_uuid is not None:
            self.__trace.log().debug("End Episode [{}]".format(self.__episode_uuid))
            self.__episode_uuid = None  # noqa
            self.__episode_step = 0
        return

    def __session_end(self) -> None:
        """
        End the current session
        """
        if self.__session_uuid is not None:
            self.__trace.log().debug("End Session [{}]".format(self.__session_uuid))
            self.__episode_end()
            self.__session_uuid = None  # noqa
        return

    def __session_start(self) -> None:
        """
        Start a new session
        """
        self.__session_end()
        self.__session_uuid = self.__ttt_event_stream.session_uuid  # noqa inherit same session uuid as event stream
        self.__trace.log().debug("Start Session [{}]".format(self.__session_uuid))
        return

    def run(self,
            num_episodes: int,
            simple_stats: bool = False) -> List[str]:
        """
        Play the given number of episodes with the Agents supplied as part of class Init. Select a random Agent
        to go at the start of each episode.
        :param num_episodes: The number of episodes to play
        :param simple_stats: If True log the win/lose/draw result of each game
        :return: The list of episode UUIDs or if simple_stats enabled with W/L/D status of each game
        """
        episodes = list()
        stats = np.zeros(3, dtype=np.int)
        while len(episodes) < num_episodes:
            state, agent, episode_uuid = self.episode_start()
            while not self.episode_complete():
                agent = self.play_action(agent)
            self.__stats(stats, episodes, episode_uuid, simple_stats, num_episodes)
            state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)
            self.__x_agent.episode_complete(state)
            self.__o_agent.episode_complete(state)
        self.__x_agent.terminate()
        self.__o_agent.terminate()
        return episodes

    def __stats(self,
                stats: np.ndarray,
                episodes: List[str],
                episode_uuid: str,
                simple_stats: bool,
                num_episodes: int) -> None:
        """
        Record game stats
        :param stats: Running total of x win, o win, draw
        :param episodes: The list of running stats to update
        :param episode_uuid: The UUID of the current episode
        :param simple_stats: If True log the win/lose/draw result of each game
        :param num_episodes: The total number of episodes being run
        """
        if simple_stats:
            episodes.append(self.__episode_state().as_str())
            if episodes[-1] == TicTacToeBoardState.x_win.as_str():
                stats[self._X] += 1
            elif episodes[-1] == TicTacToeBoardState.o_win.as_str():
                stats[self._O] += 1
            elif episodes[-1] == TicTacToeBoardState.draw.as_str():
                stats[self._D] += 1
            else:
                pass
        else:
            episodes.append(episode_uuid)
        i = len(episodes)
        if i % max(1, int(num_episodes / 10)) == 0:
            if simple_stats:
                pct = (stats.copy() * 1.0)
                pct /= np.sum(pct)
                pct *= 100
                self.__trace.log().info(self._STAT_FMT.format(
                    i,
                    stats[self._X], stats[self._O], stats[self._D],
                    pct[self._X], pct[self._O], pct[self._D]))
            else:
                self.__trace.log().info("Iteration: " + str(i))
        return

    @classmethod
    def __empty_board(cls):
        """
        Return an empty board
        :return: an empty board as numpy array
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
        if self.__episode_step is None:
            self.__episode_step = 1
        else:
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
                        agent: Agent) -> None:
        """
        Assign a reward to the agent for the given action that was just played
        :param action: The action just played
        :param state: The state before the action
        :param next_state: The state after the action
        :param agent: The agent that played the action
        """
        # ToDo re write using __episode_state directly
        episode_end = False
        reward = self.step_reward
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

        # Communicate the reward for taking the action to teh Agent
        agent.reward(state, next_state, action, reward, episode_end)

        # Persist the reward in event stream
        self.__ttt_event_stream.record_event(episode_uuid=self.__episode_uuid,
                                             episode_step=self.__episode_step,
                                             state=state,
                                             action="{}".format(action),
                                             agent=agent.name(),
                                             reward=reward,
                                             episode_end=False,
                                             episode_outcome=TicTacToeEvent.STEP)
        if episode_end:
            self.__ttt_event_stream.record_event(episode_uuid=self.__episode_uuid,
                                                 episode_step=self.__episode_step + 1,
                                                 state=next_state,
                                                 action="-1",
                                                 agent=agent.name(),
                                                 reward=0,
                                                 episode_end=episode_end,
                                                 episode_outcome=episode_outcome)
        return

    def id_to_agent(self,
                    agent_id,
                    other: bool = False) -> Agent:
        """
        Take the given id and return the matching agent.
        Where the agent_id can be
        The string name of an agent associated with this TTT instance
        The numeric id of an agent associated with this TTT instance
        An instance of an Agent
        :param agent_id: The numerical Id or name of the agent
        :param other: If true return details of the agent
        :return: The matching Agent or none if no match
        """
        agent = None
        if type(agent_id) == int:
            if self.__x_agent.id() == agent_id:
                agent = self.__x_agent
            elif self.__o_agent.id() == agent_id:
                agent = self.__o_agent
            else:
                raise RuntimeError("[{}] is not a valid numeric Id of an Agent".format(agent_id))
        elif type(agent_id) == str:
            if self.__x_agent.name() == agent_id:
                agent = self.__x_agent
            elif self.__o_agent.name() == agent_id:
                agent = self.__o_agent
            else:
                raise RuntimeError("[{}] is not a valid name of an Agent".format(agent_id))
        elif isinstance(agent_id, Agent):
            if self.__x_agent == agent_id:
                agent = agent_id
            elif self.__o_agent == agent_id:
                agent = agent_id
            else:
                raise RuntimeError("[{}] is not an Agent associated with this instance of TTT".format(str(agent_id)))
        else:
            raise RuntimeError("[{}] of Type [{}] Cannot be converted to an Agent".
                               format(str(agent_id), type(agent_id)))
        if other:
            if agent == self.__x_agent:
                agent = self.__o_agent
            else:
                agent = self.__x_agent
        return agent

    def do_action(self,
                  agent_id,
                  action: int) -> str:
        """
        Update the internal state to reflect the given action. This only effects internal state it does not
        notify the agents; for this __play_action() should be used.

        This is used for debug and is not part of the core game algorithm

        :param agent_id: The Id of the agent to play the action as
        :param action: The action to play
        :return: The name of the next agent to play or None if the Agent/Action combo was illegal
        """
        next_agent = None
        agent = self.id_to_agent(agent_id)
        if agent is None:
            self.__trace.log().debug("do_action ignored unknown agent [{}]".
                                     format(agent_id))
        elif not self.legal_board_state():
            self.__trace.log().debug("do_action ignored for action [{}] as state [{}] is illegal".
                                     format(action,
                                            self.state().state_as_string()))
        elif action not in self.actions_ids_left_to_take():
            self.__trace.log().debug("do_action ignored illegal action [{}] in state [{}]".
                                     format(action,
                                            self.state().state_as_string()))
        else:
            st = self.state_action_str()
            if len(st) > 0:
                new_st = "{}~{}:{}".format(st, agent.id(), action)
            else:
                new_st = "{}:{}".format(agent.id(), action)
            self.__string_to_internal_state(new_st)
            if not self.legal_board_state():
                self.__trace.log().debug("do_action ignored for action [{}] as state [{}] is illegal".
                                         format(action,
                                                self.state().state_as_string()))
            else:
                next_agent = (self.__next_agent[agent.name()]).name()
                self.__last_agent = agent_id
        return next_agent

    def play_action(self,
                    agent: Agent,
                    action_override: int = None) -> Agent:
        """
        Make the play chosen by the given agent. If it is a valid play
        confer reward and switch play to other agent. If invalid play
        i.e. play in a cell where there is already a marker confer
        penalty and leave play with the same agent.

        :param agent: The Agent to play the next move
        :param action_override: Optional action, if supplied this action is played in place of the action selected by
               the agent
        :return: The Agent that will play next
        """

        other_agent = self.__next_agent[agent.name()]
        state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)

        # Make the play on the board.
        if action_override is not None:
            action = action_override
        else:
            action = agent.choose_action(state, self.actions_ids_left_to_take())
        if action not in self.actions_ids_left_to_take():
            raise TicTacToe.IllegalActorAction("Actor Proposed Illegal action in current state :" + str(action))
        self.__take_action(action, agent)
        next_state = TicTacToeState(self.__board, self.__x_agent, self.__o_agent)

        self.__assign_reward(action=action,
                             state=state,
                             next_state=next_state,
                             agent=agent)
        return other_agent  # play moves to next agent

    def attributes(self) -> Dict:
        """
        The attributes of the current game state
        :return: Dictionary of attributes and their current values.
        """
        attr_dict = dict()
        e_state = self.__episode_state()
        attr_dict[
            TicTacToe.attribute_won[0]] = e_state == TicTacToeBoardState.o_win or e_state == TicTacToeBoardState.x_win
        attr_dict[TicTacToe.attribute_draw[0]] = e_state == TicTacToeBoardState.draw
        attr_dict[TicTacToe.attribute_complete[0]] = \
            attr_dict[TicTacToe.attribute_draw[0]] or attr_dict[TicTacToe.attribute_won[0]]
        attr_dict[TicTacToe.attribute_agent[0]] = self.__agent
        attr_dict[TicTacToe.attribute_board[0]] = np.copy(self.__board)
        return attr_dict

    def __episode_state(self,
                        board=None) -> TicTacToeBoardState:
        """
        Return the state of the episode X-WIN, O-WIN, DRAW, STEP
        :param board: 3 by 3 numpy array - if supplied return based on given board else use the internal board state
        :return: Return the state of the episode X-WIN, O-WIN, DRAW, STEP
        """
        if board is None:
            board = self.__board
        brd = list()
        brd.extend(np.nansum(board, axis=1).tolist())  # Rows
        brd.extend(np.nansum(board, axis=0).tolist())  # Cols
        brd.append(np.nansum(board.diagonal()))
        brd.append(np.nansum(np.rot90(board).diagonal()).tolist())
        o_win = 3 * self.__o_agent.id()
        x_win = 3 * self.__x_agent.id()

        for v in brd:
            if v == o_win:
                return TicTacToeBoardState.o_win
            elif v == x_win:
                return TicTacToeBoardState.x_win

        if np.nansum(np.abs(board)) == 9:
            return TicTacToeBoardState.draw

        return TicTacToeBoardState.step

    def __episode_won(self,
                      board=None) -> bool:
        """
        Return True if the current board has a winning row, col or diagonal on it.
        :param board: If supplied return based on given board else use the internal board state
        :return: True if board has a winning move on it.
        """
        e_state = self.__episode_state(board=board)
        return e_state == TicTacToeBoardState.x_win or e_state == TicTacToeBoardState.o_win

    def game_step(self,
                  board: np.ndarray = None) -> int:
        """
        The step number of the game 0 to 9
        0 = no plays have been made
        ...
        8 = all plays have been made
        :param board: If supplied return based on given board else use the internal board state
        :return: The step number as integer in range 0 to 8
        """
        if board is None:
            board = self.__board
        return 8 - np.sum(np.isnan(board) * 1)

    def __actions_left_to_take(self,
                               board: np.ndarray = None) -> bool:
        """
        Return True if there are any actions left to take given the current board state
        :param board: If supplied return based on given board else use the internal board state
        :return: True if actions remain on board
        """
        if board is None:
            board = self.__board
        return board[np.isnan(board)].size > 0

    def actions_ids_left_to_take(self,
                                 board: np.ndarray = None) -> np.array:
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
        :param state: Optional game state to test episode complete on, else use current internal board state
        :return: True if episode is complete for requested state or internal board state
        """
        if state is not None:
            board = state.state()
        else:
            board = self.__board

        return self.__episode_state(board=board) != TicTacToeBoardState.step

    def __string_to_internal_state(self,
                                   moves_as_str: str) -> None:
        """
        Establish current game state to match that of the given structured text string. The steps are not simulated
        but just loaded directly as a board state.
        :param moves_as_str: The game state as structured text to load internally
        """
        self.__board = TicTacToe.__empty_board()
        if len(moves_as_str) > 0:
            mvs = moves_as_str.split('~')
            if moves_as_str is not None:
                for mv in mvs:
                    if len(mv) > 0:
                        pl, ps = mv.split(":")
                        self.__take_action(int(ps), self.__agents[int(pl)])
        return

    def game_state(self) -> TicTacToeBoardState:
        """
        What is the current state of the board
        STEP - In progress
        DRAW - Over and drawn
        X-WIN - Over with X as winner
        O-WIN - Over with O as winner
        :return: BoardState state : STEP, DRAW, O-WIN, X-WIN
        """
        return self.__episode_state()

    def board_as_string_to_internal_state(self,
                                          board_as_str: str) -> str:
        """
        Take the board as sequence of x-id, o-id & 0 (not played) and convert to and internal state
        :param board_as_str: The board state as string of x-id, o-id, 0's
        :return: TicTacToeEvent state as string STEP, DRAW, O-WIN, X-WIN
        """

        self.episode_start()
        new_board = np.zeros(9)
        x_id_as_str = str(self.__x_agent.id())
        o_id_as_str = str(self.__o_agent.id())
        all_chars = "{}{}{}".format(x_id_as_str, TicTacToePlayerId.none.as_str(), o_id_as_str)
        pos = 0
        st = ""
        for c in board_as_str:
            if pos >= 9:
                raise RuntimeError("Board state must be exactly 9 characters long, but given [{}]".format(board_as_str))
            if c in all_chars:
                st += c
            else:
                raise RuntimeError("Illegal Id in cell should be [{},{},{}] but given [{}]".
                                   format(x_id_as_str,
                                          o_id_as_str,
                                          TicTacToePlayerId.none.as_str(),
                                          c))
            if st == x_id_as_str:
                new_board[pos] = self.__x_agent.id()
                st = ""
                pos += 1
            elif st == o_id_as_str:
                new_board[pos] = self.__o_agent.id()
                st = ""
                pos += 1
            elif st == TicTacToePlayerId.none.as_str():
                new_board[pos] = self.empty_cell
                st = ""
                pos += 1
        self.__board = new_board.reshape((3, 3))
        if pos != 9:
            raise RuntimeError("Board state must be exactly 9 characters long, but given [{}]".format(board_as_str))
        if not self.legal_board_state():
            raise RuntimeError("Board state is not a legal board state [{}]".format(board_as_str))

        return self.game_state().as_str()

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

    def state_action_str(self) -> str:
        """
        The current TicTacToe game state as structured text that can be parsed
        :return: Game state as structured test string
        """
        return self.__internal_state_to_string()

    def export_state(self) -> str:
        """
        The current TicTacToe game state as structured text that can be parsed
        :return: Game state as structured test string
        """
        return self.__internal_state_to_string()

    def import_state(self,
                     state_as_string: str,
                     same_episode: bool = True):
        """
        Set game state to that of the structured text string
        :param state_as_string: The structured text from which to establish the game state
        :param same_episode: If True then load state into the current episode else start a new episode
        :return:
        """
        if not same_episode:
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

    def x_agent_name(self) -> str:
        """
        Get the name of the X agent
        :return: The name of the X Agent as string
        """
        return self.__x_agent.name()

    def o_agent_name(self) -> str:
        """
        Get the name of the O agent
        :return: The name of the O Agent as string
        """
        return self.__o_agent.name()

    def get_x_agent(self) -> Agent:
        """
        Get the X agent
        :return: The X Agent
        """
        return self.__x_agent

    def get_o_agent(self) -> Agent:
        """
        Get the O agent
        :return: The O Agent
        """
        return self.__o_agent

    def get_current_agent(self) -> Agent:
        """
        Return the agent that is to up to take the next move
        :return: The Agent to take the next move
        """
        return self.__agent

    def get_next_agent(self) -> Agent:
        """
        Return the agent that is next  up to take a move (after current agent)
        :return: The Agent to take the move after the current agent
        """
        if self.__agent == self.__x_agent:
            return self.__o_agent
        return self.__x_agent

    def get_other_reward(self,
                         reward: float
                         ) -> float:
        """
        Return the reward from the perspective of the other play
        Win to one agent is -Win (loss) to the other agent
        Draw is the same to both agents as they both draw
        Step is the same to both agents as they both 'pay' the same to play a move
        :param reward: The reward to find the opposite
        :return: The reward from the other perspective
        """
        if reward not in self.__reward_map:
            raise RuntimeError("[{}] Unknown reward value, cannot find reward from other perspective".format(reward))
        return self.__reward_map[reward]

    def get_other_agent(self,
                        agent) -> Agent:
        """
        Get the agent that is not the given agent
        :param agent: Can be an Agent object, Agent name or Agent Id. If name or id given it must be the id or name of
        an agent mapped as X or O agent.
        :return: The Agent that is not the given Agent
        """
        other_agent = None
        if isinstance(agent, Agent):
            if self.__agent == self.__x_agent:
                other_agent = self.__o_agent
            else:
                other_agent = self.__x_agent
        elif type(agent) == str:
            if agent == self.__x_agent.name():
                other_agent = self.__o_agent
            elif agent == self.__o_agent.name():
                other_agent = self.__x_agent
            else:
                self.__trace.log().error("Cannot get other agent, [{}] is not valid agent name".format(agent))
        elif type(agent) == int:
            if agent == self.__x_agent.id():
                other_agent = self.__o_agent
            elif agent == self.__o_agent.id():
                other_agent = self.__x_agent
            else:
                self.__trace.log().error("Cannot get other agent, [{}] is not valid agent name".format(str(agent)))
        else:
            self.__trace.log().error("Cannot get other agent given [{}]".format(str(agent)))
        return other_agent

    def set_current_agent(self,
                          agent) -> None:
        """
        Override the next agent to play with the given Agent.
        :param agent: Can be an Agent object, Agent name or Agent Id. If name or id given it must be the id or name of
        an agent mapped as X or O agent.
        """
        if isinstance(agent, Agent):
            self.__agent = agent
        elif type(agent) == str:
            if agent == self.__x_agent.name():
                self.__agent = self.__x_agent
            elif agent == self.__o_agent.name():
                self.__agent = self.__o_agent
            else:
                self.__trace.log().error("Cannot set agent, [{}] is not valid agent name".format(agent))
        elif type(agent) == int:
            if agent == self.__x_agent.id():
                self.__agent = self.__x_agent
            elif agent == self.__o_agent.id():
                self.__agent = self.__o_agent
            else:
                self.__trace.log().error("Cannot set agent, [{}] is not valid agent name".format(str(agent)))
        else:
            self.__trace.log().error("Cannot set agent given [{}]".format(str(agent)))
        return

    def legal_board_state(self) -> bool:
        """
        True if the current board state is legal else False.
        Illegal board states can result from direct state imports where agents are not taking turns
        :return: True if board state is legal else False.
        """
        num_x = np.sum(self.__board == self.__x_agent.id())
        num_o = np.sum(self.__board == self.__o_agent.id())

        # Number of X and O plays are always equal or 1 ahead
        diff = np.abs(num_x - num_o)
        return diff == 0 or diff == 1

    class IllegalActorAction(Exception):
        """
        Given actor was not in correct state to participate in a game
        """

        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)
