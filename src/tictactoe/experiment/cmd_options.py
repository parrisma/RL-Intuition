from typing import Tuple, Dict, List, Callable
from src.lib.rltrace.trace import Trace


class CmdOptions:
    """
    Handle command sub options
    """

    class Option:
        _aliases = List[str]
        _function_name: Callable[[Tuple[str]], None]
        _description: str

        def __init__(self,
                     aliases: List[str],
                     function: Callable[[Tuple[str]], None],
                     description: str):
            self._aliases = aliases
            self._function = function
            self._description = description
            return

        @property
        def func(self) -> Callable[[Tuple[str]], None]:
            return self._function

        @property
        def desc(self):
            return self._description

    _options: Dict[str, Option]
    _trace: Trace
    _help: str

    def __init__(self,
                 trace: Trace):
        self._options = dict()
        self._trace = trace
        self._help = "No options defined"
        return

    def add_option(self,
                   aliases: List[str],
                   function: Callable,
                   description: str) -> None:
        """
        Add the given option mapping it to the list of given aliases
        :param aliases: List of string aliases for teh option
        :param function: The function to invoke to process the option
        :param description: The help description of the option
        """
        for alias in aliases:
            self._options[alias] = self.Option(aliases, function, description)
        self._help = self._option_help(self._options)
        return

    def do_option(self,
                  option_args: str) -> None:
        """
        Invoke the given option with the given args
        :param option_args: The option & args to pass to the option
        """
        args = self._parse(option_args)
        if len(args) > 0:
            option = option_args[0].upper()
            if option in self._options:
                try:
                    self._options[option].func(args[1:])
                except Exception as e:
                    self._trace.log().info("Option request failed [{}]".format(str(e)))
            else:
                self._trace.log().info("You must specify a valid option from {}".format(self._help))
        else:
            self._trace.log().info("You must specify a valid option from {}".format(self._help))
        return

    @staticmethod
    def _option_help(option_types: Dict[str, Option]) -> str:
        """
        Return a help string for a list of command options
        :return: Help String
        """
        option_help = ""
        for k, v in option_types.items():
            option_help = "{}, {} = {}".format(option_help, k, v.desc)
        return option_help

    @staticmethod
    def _parse(arg) -> Tuple[str]:
        """
        Split out the argument string where arguments are separated by single space
        """
        if arg is not None:
            res = tuple(map(str, arg.split()))
        else:
            res = tuple()
        return res
