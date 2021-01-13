import abc
from src.reflrn.interface.agent import Agent


class AgentFactory(metaclass=abc.ABCMeta):
    """
    Factory class to create Agent objects
    """

    @abc.abstractmethod
    def new_x_agent(self, *args, **kwargs) -> Agent:
        """
        Create a new X Agent
        :return: An agent to play as X
        """
        pass

    @abc.abstractmethod
    def new_o_agent(self, *args, **kwargs) -> Agent:
        """
        Create a new O Agent
        :return: An agent to play as O
        """
        pass
