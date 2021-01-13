from src.tictactoe.experiment.experiment_base import ExperimentBase
from src.tictactoe.agent.random_play_agent import RandomPlayAgent
from src.tictactoe.experiment1.ttt_nav import TTTNav
from src.tictactoe.experiment1.gamenav_cmd import GameNavCmd


class Experiment1(ExperimentBase):
    """
    This Experiment is simply for two 'random' agents to play each other in the TicTacToe environment. Each agent
    simply selects on of the allowed actions at random. So over a large number of episodes we will see what undirected
    play yields in terms of Wins & Draws.

    It will also create a body of game events State / Action / Reward in the Elastic ttt_events table that we can
    use in the next experiments as training data.
    """

    def __init__(self):
        # This experiment uses RandomPlay Agents only
        super().__init__(RandomPlayAgent.RandomAgentFactory())
        return

    def run(self) -> None:
        """
        Run the experiment where two random agents play against each other
        """
        self._trace.log().info("Experiment {} Started".format(self.__class__.__name__))
        GameNavCmd(nav=
                   TTTNav(ttt=self._ttt,
                          ttt_event_stream=self._ttt_event_stream,
                          trace=self._trace,
                          session_uuid=self._session_uuid,
                          dir_to_use="..\data")).cmdloop()
        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment1().run()
