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
    DONE_FMT = "Done running episodes, [{}] events saved to elastic db with session_uuid {}"
    HEAD_FMT = "command is <head session_uuid>"
    EVENT_FMT = "Episode UUID [{}] State [{}] Outcome [{}] Reward [{}]"

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
                self._trace.log().info("Running {} TTT episodes".format(num_episodes))
                self._ttt.run(num_episodes=num_episodes)
                cnt = self._ttt_event_stream.count_session(session_uuid=self._session_uuid)
                self._trace.log().info(self.DONE_FMT.format(cnt, self._session_uuid))
            except Exception as _:
                self._trace.log().info(self.RUN_FMT)
        return

    def do_list(self) -> None:
        """
        List all the session uuids to select from
        """
        sessions = self._ttt_event_stream.list_of_available_sessions()
        self._trace.log().info("Available Sessions with events")
        for session in sessions:
            uuid, cnt = session
            self._trace.log().info("{} with {} events".format(uuid, cnt))
        self._trace.log().info("Use the [head <session_uuid>] command to see the events")
        return

    def do_head(self,
                args) -> None:
        """
        Show the first 10 Game events for teh given session
        """
        if args is None or len(args) == 0:
            self._trace.log().info(self.HEAD_FMT)
            return
        parsed_args = self.parse(args)
        if len(parsed_args) != 1:
            self._trace.log().info(self.HEAD_FMT)
        else:
            try:
                head_session_uuid = parsed_args[0]
                self._trace.log().info("Showing first 10 events for session_uuid [{}]".format(head_session_uuid))
                events = self._ttt_event_stream.get_session(session_uuid=head_session_uuid)
                for event in events[:10]:
                    self._trace.log().info(self.EVENT_FMT.format(event.episode_uuid,
                                                                 event.state,
                                                                 event.episode_outcome,
                                                                 event.reward))
                self._trace.log().info("Done")
            except Exception as _:
                self._trace.log().info(self.HEAD_FMT)
        return
