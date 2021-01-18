import os
from typing import Tuple, List
from src.lib.rltrace.trace import Trace
from src.tictactoe.experiment1.ex1_cmd import Ex1Cmd
from src.tictactoe.tictactoe import TicTacToe
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.explore.explore import Explore


class Ex1CmdDo(Ex1Cmd):
    """
    Run TTT Games
    """
    _dir_to_use: str
    _trace: Trace
    _ttt_event_stream: TicTacToeEventStream
    _ttt: TicTacToe
    _session_uuid: str
    _exploration: Explore

    RUN_FMT = "command is <run num_episodes>"
    SET_FMT = "command is <run O|X a1 a2 a3 .. an> where an is an integer action in range 0 to 8"
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
        self._exploration = Explore(ttt=ttt, trace=trace, ttt_event_stream=ttt_event_stream)
        self._ttt.get_o_agent().attach_to_explore(self._exploration)  # ToDo make attach part of agent interface
        self._ttt.get_x_agent().attach_to_explore(self._exploration)
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
                self._exploration.save(session_uuid=self._session_uuid, dir_to_use=self._dir_to_use)
                cnt = self._ttt_event_stream.count_session(session_uuid=self._session_uuid)
                self._trace.log().info(self.DONE_FMT.format(cnt, self._session_uuid))
            except Exception as _:
                self._trace.log().info(self.RUN_FMT)
        return

    def _do_run_set_game(self,
                         moves: Tuple[str]) -> None:
        """
        Run a set of defined moves
        :param moves: Moves of form <Agent to play first move> <action-1> .. <action-n> e.g. X 0 6 2 8 1
                      Moves must be valid and define a complete game.
        """
        if len(moves) < 6:
            self._trace.log().info("SET: Must be a complete game of agent + at least 5 moves")
        else:
            agent = self._ttt.id_to_agent(moves[0])
            self._trace.log().info("Running set game with {} to play first action".format(agent.name()))
            self._ttt.episode_start()
            for action in moves[1:]:
                if not action.isnumeric():
                    self._trace.log().info("Action {} skipped as it is not numeric".format(action))
                else:
                    if self._ttt.episode_complete():
                        self._trace.log().info("Skipped action {} as game is over".format(action))
                    else:
                        self._trace.log().info("Agent {} plays action {}".format(agent.name(), action))
                        agent = self._ttt.play_action(agent=agent, action_override=int(action))
        return

    def _do_set_parse(self,
                      args) -> List[Tuple[str]]:
        """
        Parse the given args. If args[0] is a valid file name assume the file is a list of args
        for teh set command and open it and return contents
        :param args: The args to parse
        :return: List of parsed args
        """
        arg_list = list()
        parsed_args = self.parse(args)
        if os.path.exists(parsed_args[0]):
            self._trace.log().info("Loading set games from [{}]".format(parsed_args[0]))
            with open(parsed_args[0]) as fp:
                for line in fp:
                    parsed_line = self.parse(line)
                    if len(parsed_line) > 0:
                        arg_list.append(parsed_line)
            self._trace.log().info("Loaded {} set games".format(len(arg_list)))
        else:
            arg_list.append((parsed_args))  # noqa - we need to add a the tuple *not* the elements of the tuple
        return arg_list

    def do_set(self,
               args) -> None:
        """
        Run the set moves of a complete game given in the form set O|X a1 a2 a3 .. an
        """
        if args is None or len(args) == 0:
            self._trace.log().info(self.SET_FMT)
        else:
            arg_list = self._do_set_parse(args)
            try:
                for parsed_args in arg_list:
                    self._do_run_set_game(parsed_args)
                    if self._ttt.episode_complete():
                        self._exploration.save(session_uuid=self._session_uuid, dir_to_use=self._dir_to_use)
                        cnt = self._ttt_event_stream.count_session(session_uuid=self._session_uuid)
                        self._trace.log().info(self.DONE_FMT.format(cnt, self._session_uuid))
                    else:
                        self._trace.log().info("Cannot save events as action set did not complete a game")
            except Exception as _:
                self._trace.log().info(self.SET_FMT)
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
        Show the first 10 Game events for the given session
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
