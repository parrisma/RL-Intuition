from typing import Dict
from src.interface.envbuilder import EnvBuilder
from src.lib.uniqueref import UniqueRef
from src.lib.namegen.namegen import NameGen
from src.lib.rltrace.trace import Trace
from src.lib.rltrace.traceenvbuilder import TraceEnvBuilder
from src.lib.envboot.runspecenvbuilder import RunSpecEnvBuilder
from src.lib.elastic.elasticenvbuilder import ElasticEnvBuilder
from src.lib.rltrace.traceelasticenvbuilder import TraceElasticEnvBuilder
from src.lib.envboot.log_level import LogLevel


class Env:
    # Annotation
    _name: str
    _env_id: str
    _context: Dict[str, object]
    _trace: Trace

    _name = None
    _env_id = None
    _context = None
    _trace = None  # Can only log via trace once the Trace Env Builder has run.

    @classmethod
    def __init__(cls,
                 log_level: LogLevel = LogLevel.new(),
                 purge: bool = False,
                 name: str = None):
        if cls._context is None:
            cls._context = dict()
            cls._env_id = UniqueRef().ref
            cls._context[EnvBuilder.EnvSessionUUID] = cls._env_id
            cls._context[EnvBuilder.Purge] = purge
            if name is None or len(name) == 0:
                cls._name = NameGen.generate_random_name()
            cls._context[EnvBuilder.EnvName] = cls._name
            cls._context[EnvBuilder.LogLevel] = log_level
            cls._bootstrap(bool(cls._context[EnvBuilder.Purge]))
        else:
            if name is not None:
                raise RuntimeError("Environment {} is running cannot be renamed or reset".format(cls.__str__()))
        return

    @classmethod
    def get_context(cls) -> Dict:
        if cls._context is None:
            raise RuntimeError("Environment is not running cannot get context")
        return cls._context

    @classmethod
    def _bootstrap(cls,
                   purge: bool) -> None:
        """
        Execute the environment builders.
        Note: At the moment this is hard coded but will move to a YAML settings file.
        :param purge: If true eliminate any existing context and data
        """
        TraceEnvBuilder(cls._context).execute(purge)
        cls._trace = cls.get_trace()
        cls._trace.log().info("Starting {}".format(cls.__str__()))

        RunSpecEnvBuilder(cls._context).execute(purge)
        ElasticEnvBuilder(cls._context).execute(purge)
        TraceElasticEnvBuilder(cls._context).execute(purge)

        cls._trace.log().info("Started {}".format(cls.__str__()))
        return

    @classmethod
    def augment(cls,
                aux_builder: EnvBuilder) -> None:
        """
        Execute the given auxiliary builder in the current Env context. If the Env is not bootstrapped raise
        a runtime exception
        :param aux_builder: The auxiliary builder to execute.
        """
        if cls._context is None:
            raise RuntimeError("Environment is not running cannot get context")
        else:
            aux_builder.execute(purge=bool(cls._context[EnvBuilder.Purge]))
        return

    @classmethod
    def get_trace(cls) -> Trace:
        if EnvBuilder.TraceContext not in cls.get_context():
            raise ValueError("Env context does not (yet) contain a Trace Context")
        # noinspection PyTypeChecker
        return cls._context[EnvBuilder.TraceContext]

    @classmethod
    def __str__(cls):
        return "Env: {} - Id: {}".format(cls._name, cls._env_id)

    @classmethod
    def __repr__(cls):
        return cls.__str__()
