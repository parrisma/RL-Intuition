import abc
from enum import Enum, IntEnum, unique


class Ex1Cmd(metaclass=abc.ABCMeta):
    """
    A Navigation Interface for classes that support navigation of TicTacToe game play
    """

    @unique
    class Ex1Commands(Enum):
        cmd_run = "run"
        cmd_list = "list"
        cmd_head = "head"
        cmd_set = "set"

    @unique
    class Ex1Actions(IntEnum):
        run = -1
        list = -2
        head = -3
        set = -4

        def __init__(self,
                     action: int):
            super().__init__()
            self._action = action
            return

        def do(self,
               nav: 'Ex1Cmd',
               args=None):
            if self.value == Ex1Cmd.Ex1Actions.run:
                return nav.do_run(args)
            if self.value == Ex1Cmd.Ex1Actions.set:
                return nav.do_set(args)
            if self.value == Ex1Cmd.Ex1Actions.head:
                return nav.do_head(args)
            return nav.do_list()

    @abc.abstractmethod
    def do_run(self,
               args) -> None:
        """
        Execute a run of TicTacToe env for a specified number of games
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_set(self,
               args) -> None:
        """
        Execute a set game based on a specific set of supplied moves
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_list(self) -> None:
        """
        List the sessions UUID's in the database that hold TTT events
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_head(self,
                args) -> None:
        """
        Show the first 10 game events for the given session uuid
        """
        raise NotImplementedError()
