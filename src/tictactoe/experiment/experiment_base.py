import argparse
from src.interface.envbuilder import EnvBuilder
from src.lib.envboot.env import Env
from src.lib.rltrace.trace import Trace
from src.lib.settings import Settings
from src.lib.streams.webstream import WebStream
from src.lib.uniqueref import UniqueRef
from src.lib.envboot.log_level import LogLevel
from src.reflrn.interface.experiment import Experiment
from src.reflrn.interface.agent import Agent
from src.reflrn.interface.agent_factory import AgentFactory
from src.tictactoe.tictactoe_state_factory import TicTacToeStateFactory
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.tictactoe import TicTacToe


class ExperimentBase(Experiment):
    _run: int
    _env: Env
    _trace: Trace
    _log_level: LogLevel
    _state_factory: TicTacToeStateFactory
    _agent_factory: AgentFactory
    _agent_x: Agent
    _agent_o: Agent
    _ttt_event_stream: TicTacToeEventStream
    _ttt: TicTacToe
    _session_uuid: str

    _trace = None
    _env = None

    def __init__(self,
                 agent_factory: AgentFactory):
        """
        Establish a connection to the core environment
        """
        args = self._args()
        self._log_level = LogLevel.new(args.debug_level)
        self._env = Env(log_level=self._log_level)
        self._trace = self._env.get_trace()
        self._session_uuid = self._get_session_uuid(args.session_uuid)
        self._run_spec = self._env.get_context()[EnvBuilder.RunSpecificationContext]
        self._es = self._env.get_context()[EnvBuilder.ElasticDbConnectionContext]
        self._settings = Settings(settings_yaml_stream=WebStream(self._run_spec.ttt_settings_yaml()),
                                  bespoke_transforms=self._run_spec.setting_transformers())
        self._agent_factory = agent_factory
        self._agent_o = self._agent_factory.new_o_agent()
        self._agent_x = self._agent_factory.new_x_agent()
        self._state_factory = TicTacToeStateFactory(x_agent=self._agent_x,
                                                    o_agent=self._agent_o)
        self._ttt_event_stream = TicTacToeEventStream(trace=self._trace,
                                                      es=self._es,
                                                      es_index=self._settings.ttt_event_index_name,
                                                      state_factory=self._state_factory,
                                                      session_uuid=self._session_uuid,
                                                      dir_to_use=self._settings.ttt_event_dir_to_use)
        self._ttt = TicTacToe(trace=self._trace,
                              ttt_event_stream=self._ttt_event_stream,
                              x=self._agent_x,
                              o=self._agent_o)
        return

    @staticmethod
    def _get_session_uuid(session_uuid: str = ''):
        """
        Get the session_uuid - if none given then a new UUID is generated
        :param session_uuid: The session uuid
        :return:
        """
        if len(session_uuid) == 0:
            return UniqueRef().ref
        return session_uuid

    @classmethod
    def _args(cls):
        """
        Extract command line arguments
        :return: Parsed command line arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--session_uuid", help="The session uuid to use", default='')
        parser.add_argument("--debug_level", help="The debug level to use debug, info, warn, error", default=None)
        return parser.parse_args()

    @property
    def dir_to_use(self) -> str:
        """
        Local directory to use for data files and local persistence
        :return: The path of local directory to use.
        """
        return "..\\data"

    def run(self) -> None:
        """
        Run the experiment
        """
        super.run()
        return
