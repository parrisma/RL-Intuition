import abc
from typing import Tuple, Dict


class CmdBase(metaclass=abc.ABCMeta):
    """
    Common capabilities for all command interfaces
    """

    @abc.abstractmethod
    def do_exit(self,
                args) -> None:
        """Terminate the command session"""
        raise NotImplementedError()

    @staticmethod
    def _parse(arg) -> Tuple:
        """
        Convert a series of zero or more numbers to an argument tuple
        """
        res = None
        if arg is not None:
            res = tuple(map(str, arg.split()))
        return res
