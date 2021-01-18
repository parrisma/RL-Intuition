from enum import IntEnum, unique


@unique
class TicTacToeBoardState(IntEnum):
    """
    The possible game states of a TicTacToe board
    """
    x_win = 1
    o_win = 2
    draw = 3
    step = 4

    def as_str(self) -> str:
        """
        The name (as string) of a board state
        :return: The Board state name as string
        """
        if self.value == self.x_win:
            return 'x-win'
        elif self.value == self.o_win:
            return 'o-win'
        elif self.value == self.draw:
            return 'draw'
        elif self.value == self.step:
            return 'step'
        else:
            raise RuntimeError("No name defined for BoardState {}".format(self.value))
