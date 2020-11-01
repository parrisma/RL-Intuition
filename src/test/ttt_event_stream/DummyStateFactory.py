from src.reflrn.interface.state import State
from src.reflrn.interface.state_factory import StateFactory
from src.test.state.dummy_state import DummyState


class DummyStateFactory(StateFactory):

    def new_state(self,
                  state_as_str: str = None) -> State:
        """
        Create a new DummyState object from given structured text
        :param state_as_str: DummyState object in structured text form
        :return: DummyState Object re-constructed from given structured text
        """
        st = DummyState()
        st.state_from_string(state_as_string=state_as_str)
        return st
