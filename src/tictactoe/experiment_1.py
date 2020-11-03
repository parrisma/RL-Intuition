from src.tictactoe.experiment_base import ExperimentBase
from src.tictactoe.random_play_agent import RandomPlayAgent


class Experiment1(ExperimentBase):

    def __init__(self):
        super().__init__(RandomPlayAgent.RandomAgentFactory())
        return

    def run(self) -> None:
        """
        Run the experiment where two random agents play against each other
        """
        self._trace.log().info("Experiment 1 Started")
        self._ttt.run(num_episodes=9999)
        self._trace.log().info("Experiment 1 Finished")
        return


if __name__ == "__main__":
    Experiment1().run()
