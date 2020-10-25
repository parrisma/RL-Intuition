from typing import Dict
from elasticsearch import Elasticsearch
from src.interface.envbuilder import EnvBuilder
from src.lib.webstream import WebStream
from src.lib.settings import Settings
from src.lib.elastic.esutil import ESUtil
from src.lib.envboot.runspec import RunSpec
from src.lib.rltrace.trace import Trace


class ElasticEnvBuilder(EnvBuilder):
    _es: Elasticsearch
    _settings: Settings
    _context: Dict
    _run_spec: RunSpec
    _trace: Trace

    def __init__(self,
                 context: Dict):
        self._context = context
        self._trace = self._context[EnvBuilder.TraceContext]
        self._run_spec = self._context[EnvBuilder.RunSpecificationContext]
        self._es = None
        self._trace.log().info("Invoked : {}".format(str(self)))
        self._settings = Settings(settings_yaml_stream=WebStream(self._run_spec.elastic_settings_yaml()),
                                  bespoke_transforms=self._run_spec.setting_transformers())
        pass

    def execute(self,
                purge: bool) -> None:
        """
        Execute actions to establish the elastic environment.
        Get the environment specific settings for elastic host and port, open a connection and save into the bootstrap
        context
        :param purge: If true eliminate any existing data and context
        :return: None: Implementation should throw and exception to indicate failure
        """
        self._trace.log().info("Initiating connection to Elastic DB")
        hostname, port_id = self._settings.default()
        self._es = ESUtil.get_connection(hostname=hostname, port_id=port_id)
        self._context[EnvBuilder.ElasticDbConnectionContext] = self._es
        self._trace.log().info("Connected to {}".format(str(self._es)))
        return

    def uuid(self) -> str:
        """
        The immutable UUID of this build phase. This should be fixed at the time of coding as it is
        used in the environment factory settings to sequence build stages
        :return: immutable UUID
        """
        return "55cd885be0004c6d84857c9cd260e417"

    def __str__(self) -> str:
        return "Elastic Environment Builder - Id: {}".format(self.uuid())

    def __repr__(self):
        return self.__str__()
