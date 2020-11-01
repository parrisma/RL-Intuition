import abc
from src.reflrn.interface.state import State


class StateFactory(metaclass=abc.ABCMeta):
    """
    Factory class to create State objects
    """

    @abc.abstractmethod
    def new_state(self,
                  state_as_str: str = None) -> State:
        """
        Create a new State object
        :param state_as_str: (Optional) state in string form to construct from
        :return:
        """
        pass
