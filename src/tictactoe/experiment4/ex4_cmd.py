import abc
from enum import Enum, IntEnum, unique
from src.tictactoe.experiment.cmd_base import CmdBase


class Ex4Cmd(CmdBase):
    """
    A Navigation Interface for training Neural Network Agents
    """

    @unique
    class Ex4Commands(Enum):
        cmd_build = "build"
        cmd_train = "train"
        cmd_predict = "predict"
        cmd_exit = "exit"
        cmd_list = "list"

    @unique
    class Ex4Action(IntEnum):
        build = 1
        train = 2
        predict = 3
        list = 4
        exit = -99

        def __init__(self,
                     action: int):
            super().__init__()
            self._action = action
            return

        def do(self,
               nav: 'Ex4Cmd',
               args=None):
            if self.value == Ex4Cmd.Ex4Action.build:
                return nav.do_build(args=args)
            if self.value == Ex4Cmd.Ex4Action.train:
                return nav.do_train(args=args)
            if self.value == Ex4Cmd.Ex4Action.predict:
                return nav.do_predict(args=args)
            if self.value == Ex4Cmd.Ex4Action.list:
                return nav.do_list(args=args)
            if self.value == Ex4Cmd.Ex4Action.exit:
                nav.do_exit(args=args)
            return

    @abc.abstractmethod
    def do_build(self,
                 args):
        """Create a Neural Network of given type"""
        raise NotImplementedError()

    @abc.abstractmethod
    def do_train(self,
                 args):
        """Train the Neural Network created by net command"""
        raise NotImplementedError()

    @abc.abstractmethod
    def do_predict(self,
                   args):
        """Test the Neural Network created by net command"""
        raise NotImplementedError()

    @abc.abstractmethod
    def do_list(self,
                args):
        """list local files that match given pattern"""
        raise NotImplementedError()
