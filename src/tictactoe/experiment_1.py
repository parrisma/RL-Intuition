from src.tictactoe.experiment_base import ExperimentBase
from src.tictactoe.random_play_agent import RandomPlayAgent


class Experiment1(ExperimentBase):
    """
    This Experiment is simply for two 'random' agents to play each other in the TicTacToe environment. Each agent
    simply selects on of the allowed actions at random. So over a large number of episodes we will see what undirected
    play yields in terms of Wins & Draws.

    It will also create a body of game events State / Action / Reward in the Elastic ttt_events table that we can
    use in the next experiments as training data.
    """

    def __init__(self):
        super().__init__(RandomPlayAgent.RandomAgentFactory())
        return

    def run(self) -> None:
        """
        Run the experiment where two random agents play against each other
        """
        self._trace.log().info("Experiment {} Started".format(self.__class__.__name__))
        self._ttt.run(num_episodes=25000)
        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment1().run()
