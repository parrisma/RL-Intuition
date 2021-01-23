import glob
from typing import Tuple, Dict, List, Callable, Any
from enum import Enum, unique
from src.lib.rltrace.trace import Trace
from src.tictactoe.experiment.cmd_option import Option


class CmdOptions:
    """
    Handle command sub options
    """
    _options: Dict[str, Option]
    _trace: Trace
    _dir_to_use: str
    _help: str

    @unique
    class Settings(str, Enum):
        pattern = 'pattern'

    def __init__(self,
                 trace: Trace,
                 dir_to_use: str):
        self._options = dict()
        self._trace = trace
        self._dir_to_use = dir_to_use
        self._help = "No options defined"
        return

    def add_option(self,
                   aliases: List[str],
                   function: Callable[[Dict, Tuple[str]], object],  # Func(Settings,Args) -> None
                   description: str,
                   settings: Dict = None) -> None:
        """
        Add the given option mapping it to the list of given aliases
        :param aliases: List of string aliases for teh option
        :param function: The function to invoke to process the option : Callable[[Dict, Tuple[str]], None]
        :param description: The help description of the option
        :param settings: Optional settings that adapt option behaviour
        """
        for alias in aliases:
            self._options[alias] = Option(aliases, function, description, self._trace, settings)
        self._help = self._option_help(self._options)
        return

    def do_option(self,
                  option_args: str,
                  default_return: Any = None) -> Any:
        """
        Invoke the given option with the given args
        :param option_args: The option & args to pass to the option
        :param default_return: The value to return if the option fails.
        """
        res = default_return
        args = self._parse(option_args)
        if len(args) > 0:
            option = option_args[0].upper()
            if option in self._options:
                try:
                    res = self._options[option].func(self._options[option].settings, args[1:])
                except Exception as e:
                    res = default_return
                    self._trace.log().info("Option request failed [{}]".format(str(e)))
            else:
                self._trace.log().info("You must specify a valid option from {}".format(self._help))
        else:
            self._trace.log().info("You must specify a valid option from {}".format(self._help))
        return res

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

    def ls(self,
           settings: Dict,
           __: Tuple[str]) -> List[str]:
        """
        List of files that match the pattern passed in settings
        :return: List of full file and path that match the given pattern
        """
        patt = settings[self.Settings.pattern]
        self._trace.log().info("Searching for files in [{}]".format(self._dir_to_use))
        file_pattern = "{}\\{}".format(self._dir_to_use, patt)
        file_list = glob.glob(file_pattern)
        if file_list is None or len(file_list) == 0:
            self._trace.log().info("No matching files found that match [{}]".format(patt))
            file_list = list()
        return file_list

    def pattern_to_fully_qualified_filename(self,
                                            pattern: str) -> str:
        """
        Take the given pattern and find teh single fully qualified file name that matches the
        pattern.
        :param pattern: The pattern to convert to single file name
        :return: The full path and file name or None if no match or multiple matches
        """
        file_name = None
        file_names = self.ls({self.Settings.pattern: pattern}, ('',))
        if len(file_names) > 1:
            self._trace.log().info(
                "Be more specific multiple files match [{}] expected just one to match".format(pattern))
            self._trace.log().info("Found:")
            for f in file_names:
                self._trace.log().info(f)
            file_name = None
        elif len(file_names) == 1:
            if len(file_names[0]) > 0:
                file_name = file_names[0]
        else:
            file_name = None
        return file_name
