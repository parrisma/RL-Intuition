import abc
from enum import Enum, IntEnum, unique


class Ex0Cmd(metaclass=abc.ABCMeta):
    """
    A Navigation Interface for classes that support navigation of TicTacToe data structures by state
    """

    @unique
    class Ex0Commands(Enum):
        cmd_0 = "0"
        cmd_1 = "1"
        cmd_2 = "2"
        cmd_3 = "3"
        cmd_4 = "4"
        cmd_5 = "5"
        cmd_6 = "6"
        cmd_7 = "7"
        cmd_8 = "8"
        cmd_load = "load"
        cmd_list = "list"
        cmd_summary = "summary"
        cmd_explore = "explore"
        cmd_exit = "exit"

    @unique
    class ExplorationFile(Enum):
        visit = "visit"
        graph = "graph"

    @unique
    class Exploration(Enum):
        all = "all"
        random = "random"

    @unique
    class Ex0Actions(IntEnum):
        level_0 = 0
        level_1 = 1
        level_2 = 2
        level_3 = 3
        level_4 = 4
        level_5 = 5
        level_6 = 6
        level_7 = 7
        level_8 = 8
        level_9 = 9
        load = -1
        list = -2
        summary = -3
        explore = -4
        exit = -5

        def __init__(self,
                     action: int):
            super().__init__()
            self._action = action
            return

        def do(self,
               nav: 'Ex0Cmd',
               args=None):
            if self.value == Ex0Cmd.Ex0Actions.load:
                return nav.do_load(args)
            if self.value == Ex0Cmd.Ex0Actions.list:
                return nav.do_list(args)
            if self.value == Ex0Cmd.Ex0Actions.summary:
                return nav.do_summary()
            if self.value == Ex0Cmd.Ex0Actions.explore:
                return nav.do_explore(args)
            if self.value == Ex0Cmd.Ex0Actions.exit:
                nav.do_exit(args)
            return nav.do_action(self._action)

    @abc.abstractmethod
    def do_action(self,
                  action: int) -> None:
        """
        Navigate the structure by following the given action
        :param action: The action to navigate by
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_load(self,
                session_uuid: str) -> None:
        """
        Load the given session UUID
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_list(self,
                args) -> None:
        """
        List all the session uuids to select from for loading
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_summary(self) -> None:
        """
        Produce a summary of currently loaded game stats (visit or graph)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_explore(self,
                   args) -> None:
        """
        Run a simulation of every TicTacToe game and save as a list of states and as an
        action graph
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_exit(self,
                args) -> None:
        """
        Terminate the session
        """
        raise NotImplementedError()
