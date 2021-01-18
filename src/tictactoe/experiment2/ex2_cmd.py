import abc
from enum import Enum, IntEnum, unique


class Ex2Cmd(metaclass=abc.ABCMeta):
    """
    A Navigation Interface for classes that support navigation of TicTacToe data structures by action
    """

    @unique
    class Ex2Commands(Enum):
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
        cmd_hist = "hist"
        cmd_show = "show"
        cmd_dump = "dump"

    @unique
    class Ex2Actions(IntEnum):
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
        hist = -6
        show = -7
        dump = -8

        def __init__(self,
                     action: int):
            super().__init__()
            self._action = action
            return

        def do(self,
               nav: 'Ex2Cmd',
               args=None):
            if self.value == Ex2Cmd.Ex2Actions.back:
                return nav.do_back()
            if self.value == Ex2Cmd.Ex2Actions.home:
                return nav.do_home()
            if self.value == Ex2Cmd.Ex2Actions.load:
                return nav.do_load(args)
            if self.value == Ex2Cmd.Ex2Actions.switch:
                return nav.do_switch()
            if self.value == Ex2Cmd.Ex2Actions.list:
                return nav.do_list()
            if self.value == Ex2Cmd.Ex2Actions.hist:
                return nav.do_hist(args)
            if self.value == Ex2Cmd.Ex2Actions.show:
                return nav.do_show()
            if self.value == Ex2Cmd.Ex2Actions.dump:
                return nav.do_dump(args)
            return nav.do_action(self._action)

    @abc.abstractmethod
    def do_action(self,
                  action: int) -> str:
        """
        Navigate the structure by following the given action
        :param action: The action to navigate by
        :return: The Nav prompt as string
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_hist(self,
                action: int) -> str:
        """
        Show the Q VAlue history for the given action in the current state
        :param action: The action to show Q Value history for
        :return: The Nav prompt as string
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_back(self) -> str:
        """
        Navigate back to the previous state is there was one
        :return: The Nav prompt as string
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_home(self) -> str:
        """
        Navigate to the initial state
        :return: The Nav prompt as string
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
    def do_switch(self) -> str:
        """
        Switch perspective to other player e.g. O -> X or X -> O
        :return: The Nav prompt as string
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_list(self) -> None:
        """
        List all the session uuids to select from
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_show(self) -> str:
        """
        Show (or re-show) the details of the current game position
        :return: The Nav prompt as string
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def do_dump(self,
                data_type_to_dump: str) -> str:
        """
        Dump the nominated set of values to local file in JSON format
        :return: The Nav prompt as string
        """
        raise NotImplementedError()
