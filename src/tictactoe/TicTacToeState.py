import numpy as np
from typing import Dict
from src.reflrn.interface.agent import Agent
from src.reflrn.interface.state import State
from src.tictactoe.event.tictacttoe_event import TicTacToeEvent


class TicTacToeState(State):
    """
    Concrete implementation of the State object for TicTacToe
    """
    board: np.ndarray
    x_id: int
    o_id: int
    x_name: str
    o_name: str
    agents: Dict
    unused: float
    agent_x: Agent
    agent_o: Agent

    def __init__(self,
                 board: np.array,
                 agent_x: Agent,
                 agent_o: Agent):
        """
        Create a TicTacToe state object
        :param board: The board state to initialise the State to
        :param agent_x: The Agent playing as X
        :param agent_o: The Agent playing as O
        """
        self.board = None
        if board is not None:
            self.board = np.copy(board)  # State must be immutable
        self.x_id = agent_x.id()
        self.x_name = agent_x.name()
        self.o_id = agent_o.id()
        self.o_name = agent_o.name()
        self.agents = dict()
        self.agents[self.x_id] = self.x_name
        self.agents[self.o_id] = self.o_name
        self.unused = (self.o_id + self.x_id) / 2.0
        self.agent_x = agent_x
        self.agent_o = agent_o

    def state(self) -> object:
        """
        The state object
        :return: A Copy of the state object.
        """
        return np.copy(self.board)

    def __other(self,
                agent_id: int) -> int:
        """
        The id of the next player given the current player
        :param agent_id: The current player id
        :return: The next player id
        """
        if agent_id == self.x_id:
            return self.o_id
        elif agent_id == self.o_id:
            return self.x_id
        return agent_id

    def invert_player_perspective(self) -> State:
        """
        The board state with all player positions inverted
        :return: The board with all player positions inverted
        """
        brd = np.copy(self.board)
        shp = brd.shape
        brd = np.reshape(brd, np.size(brd))
        brd = np.array([self.__other(x) for x in brd])
        brd = np.reshape(brd, shp)
        return TicTacToeState(board=brd,
                              agent_x=self.agent_x,
                              agent_o=self.agent_o)

    def state_as_string(self) -> str:
        """
        The State rendered as string
        :return: A string representation of the current state
        """
        st = ""
        for cell in np.reshape(self.board, self.board.size):
            if np.isnan(cell):
                st += State.POSITION_NOT_PLAYED
            else:
                st += str(int(cell))
        return st

    def state_as_visualisation(self) -> str:
        """
        An easy to read visualisation of the state object for print and debug
        :return: An easy to read string form of the current state
        """
        s = ""
        for i in range(0, 3):
            rbd = ""
            for j in range(0, 3):
                rbd += "["
                if np.isnan(self.board[i][j]):
                    rbd += " "
                else:
                    rbd += self.agents[self.board[i][j]]
                rbd += "]"
            s += rbd + " "
        return s

    def __str__(self):
        """
        Render state as string
        :return: State as string
        """
        return self.state_as_string()

    def __repr__(self):
        """
        Render state as human readable
        :return: Human readable state
        """
        return self.state_as_visualisation()

    def init_from_string(self, state_as_str) -> None:
        """
        Set the board state from the string. Where the string is 9 chars long as a sequence of id's for actors
        with [.] for an un played cell. There is no validation applied to the given board state
        :param state_as_str: State in string from to inti from.
        """
        # ToDo this only works if agent id's are 1 & -1
        new_board = np.zeros(9)
        i = 0
        f = 1
        for c in state_as_str:
            if c == '.':
                new_board[i] = np.nan
            elif c == "-":
                f = -1
                i -= 1
            else:
                new_board[i] = int(c) * f
                f = 1
            i += 1
        self.board = new_board.reshape((3, 3))
        return

    def state_model_input(self) -> np.ndarray:
        """
        The State object rendered in numerical numpy.ndarray form compatable with the X input of a neural network.
        Each board position is the numerical player id or zero if the position is empty
        :return: The state as (1,9) numpy.ndarray
        """
        st = np.copy(self.board)
        st = st.reshape([1, 9])
        return st
