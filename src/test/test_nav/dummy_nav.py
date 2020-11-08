from src.tictactoe.nav import Nav
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

    def do_action(self, action: int) -> None:
        self._trace.log().debug("- - - - - - Nav Action {} Invoked".format(str(action)))
        self.last_action = action
        return
