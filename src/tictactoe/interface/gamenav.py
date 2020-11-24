import abc
from enum import Enum, IntEnum, unique


class GameNav(metaclass=abc.ABCMeta):
    """
    A Navigation Interface for classes that support navigation of TicTacToe game play
    """

    @unique
    class StateCmd(Enum):
        cmd_run = "run"
        cmd_list = "list"
        cmd_head = "head"

    @unique
    class Action(IntEnum):
        run = -1
        list = -2
        head = -3

        def __init__(self,
                     action: int):
            super().__init__()
            self._action = action
            return

        def do(self,
               nav: 'GameNav',
               args=None):
            if self.value == GameNav.Action.run:
                return nav.do_run(args)
            if self.value == GameNav.Action.head:
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
