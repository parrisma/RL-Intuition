import numpy as np
from src.reflrn.interface.agent import Agent
from src.reflrn.interface.state import State
from src.reflrn.interface.state_factory import StateFactory
from src.tictactoe.ttt.tictactoe_state import TicTacToeState


class TicTacToeStateFactory(StateFactory):
    """
    Factory class to create TicTacToeState Objects
    """
    __x_agent: Agent
    __o_agent: Agent

    def __init__(self,
                 x_agent: Agent,
                 o_agent: Agent):
        """
        TicTacToe state objects need a reference to the Agent objects
        """
        self.__x_agent = x_agent
        self.__o_agent = o_agent
        return

    def new_state(self, state_as_str: str = None) -> State:
        """
        Create a new TicTacToe state object from the given structured text.
        :param state_as_str: Structured text to construct from
        :return: A TicTacToe state object constructed from given structured text.
        """
        st = TicTacToeState(board=np.zeros((3, 3)), agent_x=self.__x_agent, agent_o=self.__o_agent)
        st.init_from_string(state_as_str=state_as_str)
        return st
