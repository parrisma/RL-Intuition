from src.lib.namegen.verbs import Verbs
from src.lib.namegen.names import Names


class NameGen:
    """
    Generate random proper-nouns+verb identifiers
    """
    @staticmethod
    def generate_random_name() -> str:
        """
        A name composed randomly from arbitrary proper-nouns and verbs
        :return:
        """
        return "{}-{}".format(Verbs.random_verb(), Names.random_name())
