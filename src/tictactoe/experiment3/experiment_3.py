from src.tictactoe.experiment.experiment_base import ExperimentBase
from src.tictactoe.agent.random_play_agent import RandomPlayAgent
from src.tictactoe.experiment3.ex3_cmd_map import Ex3CmdMap
from src.tictactoe.experiment3.ex3_cmd_do import Ex3CmdDo


class Experiment3(ExperimentBase):
    """
    This Experiment is to create different types of X and O AI agents and play them against each other
    """

    def __init__(self):
        # Needed to bootstrap the base environment but not used in this experiment so just use
        # random agent arbitrarily.
        super().__init__(RandomPlayAgent.RandomAgentFactory())
        return

    def run(self) -> None:
        """
        Allow command lined based creation and interplay of various AI TicTacToe Agents
        """
        self._trace.log().info("Experiment {} Started".format(self.__class__.__name__))
        Ex3CmdMap(Ex3CmdDo(ttt=self._ttt,
                           ttt_event_stream=self._ttt_event_stream,
                           trace=self._trace,
                           dir_to_use=self.dir_to_use)).cmdloop()
        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment3().run()
