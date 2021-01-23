from typing import List, Tuple, Dict, Callable, Any
from src.lib.rltrace.trace import Trace


class Option:
    """
    Class to manage a specific option of a main command
    """
    _aliases = List[str]
    _function_name: Callable[[Dict, Tuple[str]], Any]
    _description: str
    _trace: Trace
    _settings: Dict

    def __init__(self,
                 aliases: List[str],
                 function: Callable[[Dict, Tuple[str]], Any],
                 description: str,
                 trace: Trace,
                 settings: Dict = None):
        self._aliases = aliases
        self._function = function
        self._trace = trace
        self._description = description
        if settings is None:
            settings = dict()
        self._settings = settings
        return

    @property
    def func(self) -> Callable[[Dict, Tuple[str]], Any]:
        return self._function

    @property
    def desc(self) -> str:
        return self._description

    @property
    def settings(self) -> Dict:
        return self._settings
