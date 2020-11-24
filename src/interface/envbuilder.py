from abc import ABC, abstractmethod
from src.lib.abc.purevirtual import purevirtual


class EnvBuilder(ABC):
    #
    # These are in effect global variables - so minimise at all cost adding general variables or state to
    # the context. The intention is just to hold environment specific objects that are shared state of
    # the entire project - i.e. Singleton Class Instances.
    #
    EnvName = 'EnvName'
    EnvSessionUUID = 'EnvSessionUUID'
    TraceContext = "Trace"
    TraceReport = "TraceReporter"
    RunSpecificationContext = 'RunSpecification'
    ElasticDbConnectionContext = 'ElasticConnection'
    Purge = 'Purge'
    LogLevel = 'LogLevel'

    @abstractmethod
    @purevirtual
    def execute(self,
                purge: bool) -> None:
        """
        Execute actions to build the element of the environment owned by this builder
        :param purge: If true eliminate any existing context and data in the environment
        :return: None: Implementation should throw and exception to indicate failure
        """
        pass

    @abstractmethod
    @purevirtual
    def uuid(self) -> str:
        """
        The immutable UUID of this build phase. This should be fixed at the time of coding as it is
        used in the environment factory settings to sequence build stages 
        :return: immutable UUID
        """
        pass

    @abstractmethod
    @purevirtual
    def __str__(self) -> str:
        """
        String representation of the Builder
        :return: Class rendered as a string
        """
        pass

    @abstractmethod
    @purevirtual
    def __repr__(self) -> str:
        """
        String representation of the Builder
        :return: Class rendered as a string
        """
        pass
