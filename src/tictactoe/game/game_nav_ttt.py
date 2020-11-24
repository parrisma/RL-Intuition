from src.lib.rltrace.trace import Trace
from src.tictactoe.interface.gamenav import GameNav
from src.tictactoe.tictactoe import TicTacToe
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream


class TTTGameNav(GameNav):
    """
    Run TTT Games
    """
    _dir_to_use: str
    _trace: Trace
    _ttt_event_stream: TicTacToeEventStream
    _ttt: TicTacToe
    _session_uuid: str

    RUN_FMT = "command is <run num_episodes>"

    def __init__(self,
                 ttt: TicTacToe,
                 ttt_event_stream: TicTacToeEventStream,
                 trace: Trace,
                 session_uuid: str,
                 dir_to_use: str = '.'):
        self._dir_to_use = dir_to_use
        self._trace = trace
        self._ttt_event_stream = ttt_event_stream
        self._ttt = ttt
        self._session_uuid = session_uuid
        return

    @staticmethod
    def parse(arg):
        """
        Convert a series of zero or more numbers to an argument tuple
        """
        return tuple(map(str, arg.split()))

    def do_run(self,
               args) -> None:
        """
        Run the given number of TTT episodes
        """
        if args is None or len(args) == 0:
            self._trace.log().info(self.RUN_FMT)
            return
        parsed_args = self.parse(args)
        if len(parsed_args) != 1:
            self._trace.log().info(self.RUN_FMT)
        else:
            try:
                num_episodes = int(parsed_args[0])
                self._ttt.run(num_episodes=num_episodes)
            except Exception as _:
                self._trace.log().info(self.RUN_FMT)
        return
