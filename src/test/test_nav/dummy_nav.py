from src.tictactoe.interface.nav import Nav
from src.lib.rltrace.trace import Trace


class DummyNav(Nav):
    """
    Dummy Navigation class for testing Nav class.
    """
    last_action: int
    _trace: Trace

    def __init__(self,
                 trace: Trace):
        self.last_action = None
        self._trace = trace
        return

    def do_action(self,
                  action: int) -> None:
        self._trace.log().debug("- - - - - - Nav Action {} Invoked".format(str(action)))
        self.last_action = action
        return

    def do_home(self) -> None:
        self._trace.log().debug("- - - - - - Nav Action {} Invoked action Home")
        self.last_action = Nav.Action.action_home.value
        return

    def do_back(self) -> None:
        self._trace.log().debug("- - - - - - Nav Action {} Invoked action Back")
        self.last_action = Nav.Action.action_back.value
        return
