import tensorflow as tf
from src.tictactoe.experiment.experiment_base import ExperimentBase
from src.tictactoe.agent.random_play_agent import RandomPlayAgent
from src.tictactoe.experiment4.ex4_cmd_map import Ex4CmdMap
from src.tictactoe.experiment4.ex4_cmd_do import Ex4CmdDo


class Experiment4(ExperimentBase):
    """
    This Experiment is for training of Neural Net Agents
    """

    def __init__(self):
        # Needed to bootstrap the base environment but not used in this experiment so just use
        # random agent arbitrarily.
        super().__init__(RandomPlayAgent.RandomAgentFactory())
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self._trace.enable_tf_capture(tf.get_logger())
        return

    def run(self) -> None:
        """
        Allow command lined based training of Neural Net Agents
        """
        self._trace.log().info("Experiment {} Started".format(self.__class__.__name__))
        Ex4CmdMap(Ex4CmdDo(ttt=self._ttt,
                           ttt_event_stream=self._ttt_event_stream,
                           trace=self._trace,
                           dir_to_use=self.dir_to_use)).cmdloop()
        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment4().run()
