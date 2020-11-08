import abc
from enum import Enum, IntEnum, unique


class Nav(metaclass=abc.ABCMeta):
    """
    A Navigation Interface for classes that support navigation of TicTacToe data structures by action
    """

    @unique
    class ActionCmd(Enum):
        action_0_cmd = "0"
        action_1_cmd = "1"
        action_2_cmd = "2"
        action_3_cmd = "3"
        action_4_cmd = "4"
        action_5_cmd = "5"
        action_6_cmd = "6"
        action_7_cmd = "7"
        action_8_cmd = "8"
        action_9_cmd = "9"
        action_back_cmd = "back"

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
        action_back = -1

        def __init__(self,
                     action: int):
            super().__init__()
            self._action = action
            return

        def do(self,
               nav: 'Nav'):
            return nav.do_action(self._action)

    @abc.abstractmethod
    def do_action(self,
                  action: int) -> None:
        """
        Navigate the structure by following the given action
        :param action: The action to navigate by
        """
        raise NotImplementedError()
