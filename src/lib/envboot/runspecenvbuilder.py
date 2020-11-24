from typing import Dict
from src.interface.envbuilder import EnvBuilder
from src.lib.rltrace.trace import Trace
from src.lib.envboot.runspec import RunSpec
import os


class RunSpecEnvBuilder(EnvBuilder):
    """
    Bootstrap environment by extracting details from environment variables and using those to
    load settings YAML
    """
    _run_spec: RunSpec
    _context: Dict
    _trace: Trace

    def __init__(self,
                 context: Dict):
        self._context = context
        self._trace = self._context[EnvBuilder.TraceContext]
        self._trace.log().info("Invoked : {}".format(str(self)))
        return

    def execute(self,
                purge: bool) -> None:
        """
        Execute actions to build the element of the environment owned by this builder
        :return: None: Implementation should throw and exception to indicate failure
        """
        self._trace.log().info("Initiating Run Specification")
        run_spec_path = os.getenv(RunSpec.RUN_SPEC_PATH_ENV_VAR, None)
        if run_spec_path is None:
            raise ValueError(
                "Environment variable {} must be defined & point at directory where [{}] can be found".format(
                    RunSpec.RUN_SPEC_PATH_ENV_VAR, RunSpec.SPECS_FILE))
        if not os.path.isdir(run_spec_path):
            raise ValueError("{} is not a valid directory please update the environment variable {}".format(
                run_spec_path,
                RunSpec.RUN_SPEC_PATH_ENV_VAR))
        specs_file = "{}/{}".format(run_spec_path, RunSpec.SPECS_FILE)
        if not os.path.exists(specs_file):
            raise ValueError("{} run spec file does not exist, please update the environment variable {}".format(
                specs_file,
                RunSpec.RUN_SPEC_PATH_ENV_VAR))
        self._trace.log().info("Loading run specification from [{}]".format(specs_file))
        self._run_spec = RunSpec(specs_file)

        run_spec_to_use = os.getenv(RunSpec.SPEC_TO_USE_ENV_VAR, None)
        if run_spec_to_use is not None:
            self._run_spec.set_spec(run_spec_to_use)  # Default run specification

        self._context[EnvBuilder.RunSpecificationContext] = self._run_spec
        self._trace.log().info("Run specification loaded")
        return

    def uuid(self) -> str:
        """
        The immutable UUID of this build phase. This should be fixed at the time of coding as it is
        used in the environment factory settings to sequence build stages
        :return: immutable UUID
        """
        return "31906ac5c81c49f1a6b267d5c8c4c6d7"

    def __str__(self) -> str:
        return "Run Specification Builder - Id: {}".format(self.uuid())

    def __repr__(self):
        return self.__str__()
