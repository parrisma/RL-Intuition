from src.tictactoe.BoardState import BoardState
from src.tictactoe.TicTacToeState import TicTacToeState


class TicTacToeEvent:
    """
    Object to contain and manage a single TicTacToe event.
    """
    episode_uuid: str
    episode_step: int
    state: TicTacToeState
    action: str
    reward: float
    episode_end: bool
    episode_outcome: str

    X_WIN = BoardState.x_win.as_str()
    O_WIN = BoardState.o_win.as_str()
    DRAW = BoardState.draw.as_str()
    STEP = BoardState.step.as_str()

    def __init__(self,
                 episode_uuid: str,
                 episode_step: int,
                 state: TicTacToeState,
                 action: str,
                 reward: float,
                 episode_end: bool,
                 episode_outcome):
        self.episode_uuid = episode_uuid
        self.episode_step = episode_step
        self.state = state
        self.action = action
        self.reward = reward
        self.episode_end = episode_end
        self.episode_outcome = episode_outcome
        return

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.episode_uuid == other.episode_uuid and
                self.episode_end == other.episode_end and
                self.state == other.state and
                self.action == other.action and
                self.reward == other.reward and
                self.episode_end == other.episode_end and
                self.episode_outcome == other.episode_outcome
        )

    def __hash__(self):
        return hash(self)

    def __str__(self):
        return "Episode:{} Step:{} Action:{} Reward:{:+.3f} State{}".format(self.episode_uuid,
                                                                            self.episode_step,
                                                                            self.action,
                                                                            self.reward,
                                                                            self.state)

    def __repr__(self):
        return self.__str__()
