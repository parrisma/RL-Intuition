import abc
from enum import Enum, IntEnum, unique


class PlayNav(metaclass=abc.ABCMeta):
    """
    A Navigation Interface for classes that support playing TicTacToe with different AI Agents
    """

    @unique
    class PlayCmd(Enum):
        cmd_hist = "x"
        cmd_show = "o"
        cmd_dump = "play"

    @unique
    class Action(IntEnum):
        x = 0
        o = 1
        play = 2

        def __init__(self,
                     action: int):
            super().__init__()
            self._action = action
            return

        def do(self,
               nav: 'PlayNav',
               args=None):
            if self.value == PlayNav.Action.x:
                return nav.do_x(args)
            if self.value == PlayNav.Action.o:
                return nav.do_x(args)
            if self.value == PlayNav.Action.play:
                return nav.do_play(args)
            return

    @abc.abstractmethod
    def do_x(self,
             args) -> str:
        """
        Create Agent X of type given
        :param args: The parameters that describe the creation of an X Agent
        :return: The Nav prompt as string
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_o(self,
             args) -> str:
        """
        Create Agent O of type given
        :param args: The parameters that describe the creation of an O Agent
        :return: The Nav prompt as string
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_play(self,
                args) -> str:
        """
        Play games between X and O as described by the given parameters
        :param args: The parameters that describe the games to play between X and O
        :return: The Nav prompt as string
        """
        raise NotImplementedError()
