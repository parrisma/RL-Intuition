import numpy as np
from src.test.gibberish.gibberish import Gibberish

from src.reflrn.interface.state import State


class DummyState(State):
    _st: str

    def __init__(self):
        self._st = Gibberish.word_gibber()

    def state(self) -> object:
        raise RuntimeError("state_as_array not implemented for {}".format(self.__class__.__name__))

    def state_as_string(self) -> str:
        return self._st

    def state_as_array(self) -> np.ndarray:
        raise RuntimeError("state_as_array not implemented for {}".format(self.__class__.__name__))

    def state_from_string(self, state_as_string: str) -> None:
        self._st = state_as_string
        return
