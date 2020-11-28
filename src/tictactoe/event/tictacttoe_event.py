from src.tictactoe.tictactoe_board_state import TicTacToeBoardState
from src.tictactoe.tictactoe_state import TicTacToeState


class TicTacToeEvent:
    """
    Object to contain and manage a single TicTacToe event.
    """
    episode_uuid: str
    episode_step: int
    state: TicTacToeState
    action: str
    agent: str
    reward: float
    episode_end: bool
    episode_outcome: str

    X_WIN = TicTacToeBoardState.x_win.as_str()
    O_WIN = TicTacToeBoardState.o_win.as_str()
    DRAW = TicTacToeBoardState.draw.as_str()
    STEP = TicTacToeBoardState.step.as_str()

    def __init__(self,
                 episode_uuid: str,
                 episode_step: int,
                 state: TicTacToeState,
                 action: str,
                 agent: str,
                 reward: float,
                 episode_end: bool,
                 episode_outcome):
        self.episode_uuid = episode_uuid
        self.episode_step = episode_step
        self.state = state
        self.action = action
        self.agent = agent
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
                self.agent == other.agent and
                self.reward == other.reward and
                self.episode_end == other.episode_end and
                self.episode_outcome == other.episode_outcome
        )

    def __hash__(self):
        return hash(self)

    def __str__(self):
        return "Episode:{} Step:{} Action:{} Agent {} Reward:{:+.3f} State{}".format(self.episode_uuid,
                                                                                     self.episode_step,
                                                                                     self.action,
                                                                                     self.agent,
                                                                                     self.reward,
                                                                                     self.state)

    def __repr__(self):
        return self.__str__()
