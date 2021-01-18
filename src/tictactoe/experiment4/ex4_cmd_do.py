import sys
import os
from src.tictactoe.experiment4.ex4_cmd import Ex4Cmd
from src.lib.rltrace.trace import Trace
from src.tictactoe.ttt.tictactoe import TicTacToe
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream


class Ex4CmdDo(Ex4Cmd):
    """
    Actions for NN Agent training
    """
    _ttt: TicTacToe
    _trace: Trace
    _ttt_event_stream: TicTacToeEventStream
    _dir_to_use: str

    def __init__(self,
                 ttt: TicTacToe,
                 ttt_event_stream: TicTacToeEventStream,
                 trace: Trace,
                 dir_to_use: str):
        self._dir_to_use = os.path.abspath(dir_to_use)
        self._trace = trace
        self._ttt_event_stream = ttt_event_stream
        self._ttt = ttt
        return

    def do_exit(self,
                args) -> None:
        """
        Terminate the session
        """
        sys.exit(0)
