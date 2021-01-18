import abc
from enum import Enum, IntEnum, unique


class Ex3Cmd(metaclass=abc.ABCMeta):
    """
    A Navigation Interface for classes that support playing TicTacToe with different AI Agents
    """

    @unique
    class Ex3Commands(Enum):
        cmd_hist = "x"
        cmd_show = "o"
        cmd_dump = "play"
        cmd_exit = "exit"
        cmd_list = "list"

    @unique
    class Ex3Action(IntEnum):
        x = 0
        o = 1
        play = 2
        exit = 3
        list = 4

        def __init__(self,
                     action: int):
            super().__init__()
            self._action = action
            return

        def do(self,
               nav: 'Ex3Cmd',
               args=None):
            if self.value == Ex3Cmd.Ex3Action.x:
                return nav.do_x(args)
            if self.value == Ex3Cmd.Ex3Action.o:
                return nav.do_o(args)
            if self.value == Ex3Cmd.Ex3Action.play:
                return nav.do_play(args)
            if self.value == Ex3Cmd.Ex3Action.exit:
                return nav.do_exit()
            if self.value == Ex3Cmd.Ex3Action.list:
                return nav.do_list(args)
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

    @abc.abstractmethod
    def do_list(self,
                args) -> str:
        """
        List the types of data that can be used to create Agents e.g. Q Values as JSON
        :param args: Which type of data to list
        :return: The Nav prompt as string
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_exit(self) -> None:
        """Terminate the command session"""
        raise NotImplementedError()
