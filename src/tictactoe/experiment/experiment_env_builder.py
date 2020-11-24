from typing import Dict
from elasticsearch import Elasticsearch
from src.interface.envbuilder import EnvBuilder
from src.lib.settings import Settings
from src.lib.envboot.runspec import RunSpec
from src.lib.rltrace.trace import Trace
from src.lib.elastic.esutil import ESUtil
from src.lib.streams.webstream import WebStream


class ExperimentEnvBuilder(EnvBuilder):
    _context: Dict
    _settings: Settings
    _context: Dict
    _run_spec: RunSpec
    _trace: Trace
    _es: Elasticsearch

    def __init__(self,
                 context: Dict):
        self._context = context
        self._trace = context[EnvBuilder.TraceContext]
        self._run_spec = self._context[EnvBuilder.RunSpecificationContext]
        self._es = context[EnvBuilder.ElasticDbConnectionContext]
        self._trace.log().info("Invoked : {}".format(str(self)))
        self._settings = Settings(settings_yaml_stream=WebStream(self._run_spec.ttt_settings_yaml()),
                                  bespoke_transforms=self._run_spec.setting_transformers())
        return

    def execute(self,
                purge: bool) -> None:
        """
        Execute actions to build the TicTacToe environment.
        :return: None: Implementation should throw and exception to indicate failure
        """
        ESUtil.create_index_from_json(self._es,
                                      self._settings.ttt_event_index_name,
                                      WebStream(self._settings.ttt_event_index_json))
        return

    def uuid(self) -> str:
        """
        The immutable UUID of this build phase. This should be fixed at the time of coding.
        :return: immutable UUID
        """
        return "e22bda6ccb6646279cbf8b46ac29b08d"

    def __str__(self) -> str:
        return "TicTacToe Auxiliary Environment Builder - Id: {}".format(self.uuid())

    def __repr__(self):
        return self.__str__()
