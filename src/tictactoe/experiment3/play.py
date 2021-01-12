from src.tictactoe.experiment3.playnav import PlayNav
from src.lib.rltrace.trace import Trace
from src.tictactoe.tictactoe import TicTacToe
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream


class Play(PlayNav):
    """
    All creation of different types of AI Agent and play between them
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
        self._dir_to_use = dir_to_use
        self._trace = trace
        self._ttt_event_stream = ttt_event_stream
        self._ttt = ttt
        return

    def _ready(self) -> bool:
        """
        True if the agents are created and ready to play
        :return: True if both agents are created and ready to play
        """
        return False

    def do_x(self,
             arg) -> str:
        """
        Create an X Agent to play
        :param arg: The parameters needed to create the X Agent
        :return: The Nav prompt as string
        """
        _ = self.parse(arg)
        return self._prompt()

    def do_o(self,
             arg) -> str:
        """
        Create an O Agent to play
        :param arg: The parameters needed to create the O Agent
        :return: The Nav prompt as string
        """
        _ = self.parse(arg)
        return self._prompt()

    def do_play(self,
                arg) -> str:
        """
        Run games between both X and O if they are created and ready to go
        :param arg: The parameters needed to create the O Agent
        :return: The Nav prompt as string
        """
        _ = self.parse(arg)
        return self._prompt()

    def _prompt(self) -> str:
        """
        The navigation prompt
        :return: The game prompt
        """
        if self._ready():
            p = "(play)".format(self._ttt.get_current_agent().name())
        else:
            p = "(Use [X] and [O] to create agents to play"
        return p

    @staticmethod
    def parse(arg):
        """
        Convert a series of zero or more numbers to an argument tuple
        """
        res = None
        if arg is not None:
            res = tuple(map(str, arg.split()))
        return res
