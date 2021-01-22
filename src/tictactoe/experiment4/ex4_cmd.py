import abc
from enum import Enum, IntEnum, unique
from src.tictactoe.experiment.cmd_base import CmdBase


class Ex4Cmd(CmdBase):
    """
    A Navigation Interface for training Neural Network Agents
    """

    @unique
    class Ex4Commands(Enum):
        cmd_net = "net"
        cmd_exit = "exit"

    @unique
    class Ex4Action(IntEnum):
        net = 1
        exit = -99

        def __init__(self,
                     action: int):
            super().__init__()
            self._action = action
            return

        def do(self,
               nav: 'Ex4Cmd',
               args=None):
            if self.value == Ex4Cmd.Ex4Action.net:
                return nav.do_net(args=args)
            if self.value == Ex4Cmd.Ex4Action.exit:
                nav.do_exit(args=args)
            return

    @abc.abstractmethod
    def do_net(self,
               args) -> str:
        """Create a Neural Network of given type"""
        raise NotImplementedError()
