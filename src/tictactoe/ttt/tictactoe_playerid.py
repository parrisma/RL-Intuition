from enum import IntEnum, unique


@unique
class TicTacToePlayerId(IntEnum):
    """
    Numerical Id's of Players (Agents) in TicTacToe
    """
    x_id = -1
    o_id = 1
    none = 0

    def as_str(self) -> str:
        """
        String name of numerical Id
        :return: String name of numerical Id
        """
        if self.value == self.x_id:
            return 'X'
        elif self.value == self.o_id:
            return 'Y'
        elif self.value == self.none:
            return '0'
        else:
            raise RuntimeError("No name defined for PlayerId {}".format(self.value))
