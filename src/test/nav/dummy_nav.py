from src.tictactoe.experiment2.ex2_cmd import Ex2Cmd
from src.lib.rltrace.trace import Trace


class DummyNav(Ex2Cmd):
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
        self.last_action = Ex2Cmd.Ex2Actions.home.value
        return

    def do_back(self) -> None:
        self._trace.log().debug("- - - - - - Nav Action {} Invoked action Back")
        self.last_action = Ex2Cmd.Ex2Actions.back.value
        return

    def do_load(self,
                session_uuid: str) -> None:
        self._trace.log().debug("- - - - - - Nav Action {} Invoked action Load")
        self.last_action = Ex2Cmd.Ex2Actions.load.value
        return

    def do_switch(self) -> None:
        self._trace.log().debug("- - - - - - Nav Action {} Invoked action Switch")
        self.last_action = Ex2Cmd.Ex2Actions.switch.value
        return
