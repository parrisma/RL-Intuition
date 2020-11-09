#
# Settings specific to a build/test context
#
from typing import List
import subprocess
import os
from src.lib.transformer import Transformer
from src.lib.settings import Settings
from src.lib.filestream import FileStream


# todo: bootstrap from a YML file.

class RunSpec:
    # Annotation
    _settings: Settings
    _spec_file: str
    _spec: str

    SPECS_FILE = "specs.yaml"
    DEFAULT = "default"
    RUN_SPEC_PATH_ENV_VAR = "RUN_SPEC_PATH"
    SPEC_TO_USE_ENV_VAR = "RUN_SPEC_TO_USE"
    M_BRANCH = "_branch"
    M_CURR_BRANCH = "_current_git_branch"

    def __init__(self,
                 bootstrap_yaml_filename: str):
        if not os.path.exists(bootstrap_yaml_filename):
            raise ValueError("Run spec bootstrap file does not exist {}".format(bootstrap_yaml_filename))

        self._git_current_branch()
        self._settings = Settings(settings_yaml_stream=FileStream(bootstrap_yaml_filename),
                                  bespoke_transforms=[self.current_branch_transformer()])
        self._spec = self.DEFAULT
        return

    def branch(self) -> str:
        return getattr(self._settings, "{}{}".format(self._spec, self.M_BRANCH))

    def current_branch(self) -> str:
        return getattr(self, "{}".format(self.M_CURR_BRANCH))

    def elastic_settings_yaml(self) -> str:
        return "{}/{}/{}".format(getattr(self._settings, "{}_git_root".format(self._spec)),
                                 getattr(self._settings, "{}_branch".format(self._spec)),
                                 getattr(self._settings, "{}_elastic_yml".format(self._spec)))

    def trace_settings_yaml(self) -> str:
        return "{}/{}/{}".format(getattr(self._settings, "{}_git_root".format(self._spec)),
                                 getattr(self._settings, "{}_branch".format(self._spec)),
                                 getattr(self._settings, "{}_trace_yml".format(self._spec)))

    def ttt_settings_yaml(self) -> str:
        return "{}/{}/{}".format(getattr(self._settings, "{}_git_root".format(self._spec)),
                                 getattr(self._settings, "{}_branch".format(self._spec)),
                                 getattr(self._settings, "{}_ttt_yml".format(self._spec)))

    def get_spec(self) -> str:
        return self._spec

    def set_spec(self,
                 spec: str) -> None:
        if hasattr(self._settings, spec):
            if callable(getattr(self._settings, self._spec)):
                self._spec = spec
            else:
                raise ValueError("No such run spec {} has been loaded from the yaml config".format(spec))
        else:
            raise ValueError("No such run spec {} has been loaded from the yaml config".format(spec))
        return

    def _git_current_branch(self):
        res = subprocess.check_output("git rev-parse --abbrev-ref HEAD").decode('utf-8')
        if res is None or len(res) == 0:
            res = "Warning cannot establish current git branch"
        else:
            res = self._chomp(res)
        setattr(self, self.M_CURR_BRANCH, res)
        return

    def branch_transformer(self) -> Transformer.Transform:
        return Transformer.Transform(regular_expression='.*<git-branch>.*',
                                     transform=lambda s: s.replace('<git-branch>', self.branch(), 1))

    def current_branch_transformer(self) -> Transformer.Transform:
        return Transformer.Transform(regular_expression='.*<current-git-branch>.*',
                                     transform=lambda s: s.replace('<current-git-branch>', self.current_branch(), 1))

    def setting_transformers(self) -> List[Transformer.Transform]:
        return [self.branch_transformer(),
                self.current_branch_transformer()]

    @staticmethod
    def _chomp(s: str) -> str:
        """
        Remove all line breaks
        :param s: The string to remove line breaks from
        :return: string without line breaks
        """
        return s.replace("\r", "").replace("\n", "")
