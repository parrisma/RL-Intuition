import logging
import sys
from src.lib.uniqueref import UniqueRef
from src.lib.rltrace.elasticformatter import ElasticFormatter
from src.lib.rltrace.elastichandler import ElasticHandler


class Trace:
    # Annotation
    _CONSOLE_FORMATTER: logging.Formatter
    _logger: logging.Logger
    _console_handler: logging.Handler
    _elastic_handler: logging.Handler
    _session_uuid: str

    _CONSOLE_FORMATTER = logging.Formatter("%(asctime)s — %(name)s - %(levelname)s — %(message)s",
                                           datefmt='%Y-%m-%dT%H:%M:%S%z')
    _ELASTIC_FORMATTER = ElasticFormatter()

    def __init__(self,
                 session_uuid: str = None):
        if session_uuid is None or len(session_uuid) == 0:
            session_uuid = UniqueRef().ref
        self._session_uuid = session_uuid
        self._elastic_handler = None
        self._console_handler = None
        self._logger = None
        self._bootstrap(session_uuid)
        return

    def _bootstrap(self,
                   session_uuid: str) -> None:
        """
        Create a logger and enable the default console logger
        :param session_uuid: The session uuid to use as the name of the logger
        """
        if self._logger is None:
            self._logger = logging.getLogger(session_uuid)
            self._logger.setLevel(logging.DEBUG)
            self._logger.propagate = False  # Ensure only added handlers log i.e. disable parent logging handler
            self.enable_console_handler()
        return

    def enable_console_handler(self) -> None:
        """
        Create the console handler and add it as a handler to the current logger
        """
        if self._console_handler is None:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.name = "{}-ConsoleHandler".format(self._logger.name)
            self._console_handler.setLevel(level=self._logger.level)
            self._console_handler.setFormatter(Trace._CONSOLE_FORMATTER)
            self._logger.addHandler(self._console_handler)
        return

    def enable_elastic_handler(self,
                               elastic_handler: ElasticHandler) -> None:
        """
        Create the elastic handler and add it as a handler to the current logger
        Note: elastic_handler contains the open connection to Elastic DB
        """
        if self._elastic_handler is None:
            self._elastic_handler = elastic_handler
            self._elastic_handler.name = "{}-ElasticHandler".format(self._logger.name)
            self._elastic_handler.setLevel(level=self._logger.level)
            self._elastic_handler.setFormatter(Trace._ELASTIC_FORMATTER)
            self._logger.addHandler(self._elastic_handler)
        return

    def log(self) -> logging.Logger:
        """
        Current logger
        :return: Session Logger
        """
        return self._logger