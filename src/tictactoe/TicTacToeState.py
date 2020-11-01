import numpy as np

from lib.reflrn.interface.Agent import Agent
from lib.reflrn.interface.State import State


class TicTacToeState(State):

    #
    # Constructor.
    #
    def __init__(self,
                 board: np.array,
                 agent_x: Agent,
                 agent_o: Agent):
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

    #
    # An environment specific representation for Env. State
    #
    def state(self) -> object:
        return np.copy(self.board)

    #
    # Return the id of the other agent
    #
    def __other(self,
                agent_id: int) -> int:
        if agent_id == self.x_id:
            return self.o_id
        elif agent_id == self.o_id:
            return self.x_id
        return agent_id

    #
    # Return a new state with an inverted player perspective of the board.
    #
    def invert_player_perspective(self) -> State:
        brd = np.copy(self.board)
        shp = brd.shape
        brd = np.reshape(brd, np.size(brd))
        brd = np.array([self.__other(x) for x in brd])
        brd = np.reshape(brd, shp)
        return TicTacToeState(board=brd,
                              agent_x=self.agent_x,
                              agent_o=self.agent_o)

    #
    # An string representation of the environment state
    #
    def state_as_string(self) -> str:
        st = ""
        for cell in np.reshape(self.board, self.board.size):
            if np.isnan(cell):
                st += "0"
            else:
                st += str(int(cell))
        return st

    #
    # Render the board as human readable
    #
    def state_as_visualisation(self) -> str:
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
            s += rbd + "\n"
        s += "\n"
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

    #
    # State encoded as a numpy array that can be passed as the X (input) into
    # a Neural Net.
    #
    def state_as_array(self) -> np.ndarray:
        bc = np.copy(self.board)
        bc[np.isnan(bc)] = self.unused
        return bc

    def init_from_string(self, state_as_str) -> None:
        """
        Set the board state from the string. Where the string is 9 chars long as a sequence of id's for actors
        with [.] for an un played cell. There is no validation applied to the given board state
        :param state_as_str: State in string from to inti from.
        """
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
        Convert the state to the tensor form needed for model inout
        :param state: State object
        :return: state as numpy array
        """
        st = np.copy(self.board)
        st = st.reshape([1, 9])
        return st
