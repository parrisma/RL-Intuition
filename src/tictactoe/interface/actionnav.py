import abc
from enum import Enum, IntEnum, unique


class ActionNav(metaclass=abc.ABCMeta):
    """
    A Navigation Interface for classes that support navigation of TicTacToe data structures by action
    """

    @unique
    class ActionCmd(Enum):
        cmd_0 = "0"
        cmd_1 = "1"
        cmd_2 = "2"
        cmd_3 = "3"
        cmd_4 = "4"
        cmd_5 = "5"
        cmd_6 = "6"
        cmd_7 = "7"
        cmd_8 = "8"
        cmd_9 = "9"
        cmd_back = "back"
        cmd_home = "home"
        cmd_load = "load"
        cmd_switch = "switch"
        cmd_list = "list"

    @unique
    class Action(IntEnum):
        action_0 = 0
        action_1 = 1
        action_2 = 2
        action_3 = 3
        action_4 = 4
        action_5 = 5
        action_6 = 6
        action_7 = 7
        action_8 = 8
        action_9 = 9
        back = -1
        home = -2
        load = -3
        switch = -4
        list = -5

        def __init__(self,
                     action: int):
            super().__init__()
            self._action = action
            return

        def do(self,
               nav: 'ActionNav',
               args=None):
            if self.value == ActionNav.Action.back:
                return nav.do_back()
            if self.value == ActionNav.Action.home:
                return nav.do_home()
            if self.value == ActionNav.Action.load:
                return nav.do_load(args)
            if self.value == ActionNav.Action.switch:
                return nav.do_switch()
            if self.value == ActionNav.Action.list:
                return nav.do_list()
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
    def do_back(self) -> None:
        """
        Navigate back to the previous state is there was one
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_home(self) -> None:
        """
        Navigate to the initial state
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
    def do_switch(self) -> None:
        """
        Switch perspective to other player e.g. O -> X or X -> O
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_list(self) -> None:
        """
        List all the session uuids to select from
        """
        raise NotImplementedError()
