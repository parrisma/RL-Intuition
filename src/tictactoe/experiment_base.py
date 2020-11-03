from src.interface.envbuilder import EnvBuilder
from src.lib.envboot.env import Env
from src.lib.rltrace.trace import Trace
from src.lib.settings import Settings
from src.lib.webstream import WebStream
from src.lib.uniqueref import UniqueRef
from src.reflrn.interface.experiment import Experiment
from src.reflrn.interface.agent import Agent
from src.reflrn.interface.agent_factory import AgentFactory
from src.tictactoe.TicTacToeStateFactory import TicTacToeStateFactory
from src.tictactoe.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.TicTacToe import TicTacToe


class ExperimentBase(Experiment):
    _run: int
    _env: Env
    _trace: Trace
    _state_factory: TicTacToeStateFactory
    _agent_factory: AgentFactory
    _agent_x: Agent
    _agent_o: Agent
    _ttt_event_stream: TicTacToeEventStream
    _ttt: TicTacToe
    _session_uuid:str

    _trace = None
    _env = None

    def __init__(self,
                 agent_factory: AgentFactory):
        """
        Establish a connection to the core environment
        """
        self._env = Env()
        self._trace = self._env.get_trace()
        self._run_spec = self._env.get_context()[EnvBuilder.RunSpecificationContext]
        self._es = self._env.get_context()[EnvBuilder.ElasticDbConnectionContext]
        self._settings = Settings(settings_yaml_stream=WebStream(self._run_spec.ttt_settings_yaml()),
                                  bespoke_transforms=self._run_spec.setting_transformers())
        self._agent_factory = agent_factory
        self._agent_o = self._agent_factory.new_o_agent()
        self._agent_x = self._agent_factory.new_x_agent()
        self._state_factory = TicTacToeStateFactory(x_agent=self._agent_x,
                                                    o_agent=self._agent_o)
        self._session_uuid = UniqueRef().ref
        self._ttt_event_stream = TicTacToeEventStream(es=self._es,
                                                      es_index=self._settings.ttt_event_index_name,
                                                      state_factory=self._state_factory,
                                                      session_uuid=self._session_uuid)
        self._ttt = TicTacToe(env=self._env,
                              ttt_event_stream=self._ttt_event_stream,
                              x=self._agent_x,
                              o=self._agent_o)
        return

    def run(self) -> None:
        """
        Run the experiment
        """
        super.run()
        return
