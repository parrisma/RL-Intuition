from typing import Tuple
from enum import Enum, unique
from src.tictactoe.experiment3.playnav import PlayNav
from src.lib.rltrace.trace import Trace
from src.tictactoe.tictactoe import TicTacToe
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.agent.random_play_agent import RandomPlayAgent
from src.tictactoe.agent.q_play_agent import QPlayAgent
from src.tictactoe.agent.human_agent import HumanAgent


class Play(PlayNav):
    """
    All creation of different types of AI Agent and play between them
    """
    _ttt: TicTacToe
    _trace: Trace
    _ttt_event_stream: TicTacToeEventStream
    _dir_to_use: str

    _FUNC = 0
    _DESC = 1
    _AGENT_TYPES = {"R": ["_random_agent", "Random Agent"],
                    "Q": ["_q_agent", "Q Agent"],
                    "H": ["_human_agent", "Human Agent"]}

    _RUN_FMT = "command is <play num_episodes>"
    _RUN_FAIL = "command <play> failed with error [{}]"

    @unique
    class Player(Enum):
        player_x = "X"
        player_o = "O"

    def __init__(self,
                 ttt: TicTacToe,
                 ttt_event_stream: TicTacToeEventStream,
                 trace: Trace,
                 dir_to_use: str):
        self._dir_to_use = dir_to_use
        self._trace = trace
        self._ttt_event_stream = ttt_event_stream
        self._ttt = ttt
        self._x_agent = None
        self._o_agent = None
        return

    def _ready(self) -> bool:
        """
        True if the agents are created and ready to play
        :return: True if both agents are created and ready to play
        """
        return self._x_agent is not None and self._o_agent is not None

    def do_bye(self) -> None:
        """
        Terminate the session
        """
        quit(0)
        return

    def do_x(self,
             arg) -> str:
        """
        Create an X Agent to play
        :param arg: The parameters needed to create the X Agent
        :return: The Nav prompt as string
        """
        self._create_agent(self.Player.player_x, arg)
        return self._prompt()

    def do_o(self,
             arg) -> str:
        """
        Create an O Agent to play
        :param arg: The parameters needed to create the O Agent
        :return: The Nav prompt as string
        """
        self._create_agent(self.Player.player_o, arg)
        return self._prompt()

    def _create_agent(self,
                      player: Player,
                      arg) -> None:
        """
        Create a TicTacToe agent of the requested type
        :param player: The Player (X|O) to create the agent as
        :param arg: The parameters required to create / define the agent
        :return:
        """
        args = self.parse(arg)
        if args is not None:
            data_type = args[0].upper()
            if data_type in self._AGENT_TYPES:
                try:
                    # lookup & call the agent create function
                    getattr(self, self._AGENT_TYPES[data_type][self._FUNC])(player, args[1:])
                except Exception as e:
                    self._trace.log().info(str(e))
                    self._trace.log().info("Agent not created for player [{}]".format(str(player)))
            else:
                self._trace.log().info("You must specify a valid agent type from {}".format(self._agent_type_help()))
        else:
            self._trace.log().info("You must specify a valid agent type from {}".format(self._agent_type_help()))
        return

    def _random_agent(self,
                      player: Player,
                      _: Tuple) -> None:
        """
        Create an agent that selects plays at random
        :param player: The Player to create the random agent for.
        :param _: Requires no parameters to construct a random agent
        :return: A random play Agent
        """
        if player == self.Player.player_o:
            self._o_agent = RandomPlayAgent.RandomAgentFactory().new_o_agent()
            self._trace.log().info("Agent O created as Random Agent")
        else:
            self._x_agent = RandomPlayAgent.RandomAgentFactory().new_x_agent()
            self._trace.log().info("Agent X created as Random Agent")
        return

    def _human_agent(self,
                     player: Player,
                     _: Tuple) -> None:
        """
        Create an agent that allows user to input actions
        :param player: The Player to create the human agent for.
        :param _: Requires no parameters to construct a human agent
        :return: A Human Agent
        """
        if player == self.Player.player_o:
            self._o_agent = HumanAgent.HumanAgentFactory().new_o_agent()
            self._trace.log().info("Agent O created as Human Agent")
        else:
            self._x_agent = HumanAgent.HumanAgentFactory().new_x_agent()
            self._trace.log().info("Agent X created as Human Agent")
        return

    def _q_agent(self,
                 player: Player,
                 args: Tuple) -> None:
        """
        Create an agent that selects plays based on greedy action defined by q learning
        :param player: The Player to create the Q agent for.
        :param args: Expect first argument in Tuple to be a Q Value JSON filename available in current data path
        :return: A Q Agent
        """
        if args is not None and len(args) >= 1:
            try:
                q_file = "{}\\{}".format(self._dir_to_use, args[0])
                if player == self.Player.player_o:
                    self._o_agent = QPlayAgent.QAgentFactory().new_o_agent(q_file)
                    self._trace.log().info("Agent O created as Q Agent")
                else:
                    self._x_agent = QPlayAgent.QAgentFactory().new_x_agent(q_file)
                    self._trace.log().info("Agent X created as Q Agent")
            except Exception as e:
                self._trace.log().info(str(e))
                raise UserWarning("Q Agent expects valid JSON file name in dir [{}] but [{}] given".format(
                    args[0],
                    self._dir_to_use))
        else:
            raise UserWarning("Q Agent expects valid JSON file name in dir [{}] but no filename given".format(
                self._dir_to_use))
        return

    def _agent_type_help(self) -> str:
        """
        Help string listing the supported agent types
        :return: List of valid agent types
        """
        agent_help = ""
        for k, v in self._AGENT_TYPES.items():
            agent_help = "{}, {} - {}".format(agent_help, k, v[self._DESC])
        return agent_help

    def do_play(self,
                arg) -> str:
        """
        Run games between both X and O if they are created and ready to go
        :param arg: The parameters needed to create the O Agent
        :return: The Nav prompt as string
        """
        if self._ready():
            if arg is None or len(arg) == 0:
                self._trace.log().info(self._RUN_FMT)
                return self._prompt()
            parsed_args = self.parse(arg)
            if len(parsed_args) != 1:
                self._trace.log().info(self._RUN_FMT)
            else:
                try:
                    self._ttt = TicTacToe(trace=self._trace,
                                          ttt_event_stream=self._ttt_event_stream,
                                          x=self._x_agent,
                                          o=self._o_agent)
                    num_episodes = int(parsed_args[0])
                    self._trace.log().info("Running {} TTT episodes".format(num_episodes))
                    e = self._ttt.run(num_episodes=num_episodes, simple_stats=True)
                except Exception as e:
                    self._trace.log().info(self._RUN_FAIL.format(str(e)))
        else:
            self._trace.log().info("Use commands [x] and [o] to create agents before <play>")
        return self._prompt()

    def _prompt(self) -> str:
        """
        The navigation prompt
        :return: The game prompt
        """
        if self._ready():
            p = "(play)".format(self._ttt.get_current_agent().name())
        else:
            o = ""
            x = ""
            if self._o_agent is None:
                o = " [O]"
            if self._x_agent is None:
                x = " [X]"
            p = "(Use{}{} to create agents to play)".format(x, o)
        return p

    @staticmethod
    def parse(arg) -> Tuple:
        """
        Convert a series of zero or more numbers to an argument tuple
        """
        res = None
        if arg is not None:
            res = tuple(map(str, arg.split()))
        return res
