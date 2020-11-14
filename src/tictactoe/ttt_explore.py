from typing import Dict, List
from src.lib.rltrace.trace import Trace
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.TicTacToe import TicTacToe


class Explore:
    """
    This class explores all possible game states and records them as TTT Events.
    """
    visited: Dict
    trace: Trace
    ttt_event_stream: TicTacToeEventStream
    ttt: TicTacToe

    def __init__(self,
                 ttt: TicTacToe,
                 trace: Trace,
                 ttt_event_stream: TicTacToeEventStream):
        self.visited = dict()
        self.trace = trace
        self.ttt_event_stream = ttt_event_stream
        self.ttt = ttt
        ttt.episode_start()
        return

    def _other(self,
               agent_name: str) -> str:
        if agent_name == self.ttt.x_agent_name():
            return self.ttt.o_agent_name()
        else:
            return self.ttt.x_agent_name()

    def explore(self,
                agent_id: str = None,
                state_actions_as_str: str = "",
                depth: int = 0) -> None:
        """
        Recursive routine to visit all possible game states and record them as TTT Events
        """
        if agent_id is None:
            agent_id = self.ttt.x_agent_name()
        self.ttt.import_state(state_as_string=state_actions_as_str)
        actions_to_explore = self.ttt.actions(self.ttt.state())
        for action in actions_to_explore:  # [actions_to_explore[0]]:
            prev_state = self.ttt.state_action_str()
            next_agent_id = self.ttt.do_action(agent_id=agent_id, action=action)
            if self.ttt.state().state_as_string() not in self.visited and next_agent_id is not None:
                self.visited[self.ttt.state().state_as_string()] = True
                self.trace.log().info("Visited {} depth [{}] total found {}".
                                      format(self.ttt.state().state_as_visualisation(),
                                             depth,
                                             len(self.visited)))
                if not self.ttt.episode_complete():
                    self.explore(agent_id=next_agent_id,
                                 state_actions_as_str=self.ttt.state_action_str(),
                                 depth=depth + 1)
                else:
                    self.explore(agent_id=next_agent_id,
                                 state_actions_as_str=prev_state,
                                 depth=depth)
            else:
                self.trace.log().debug("Skipped {} depth [{}]".
                                       format(self.ttt.state().state_as_visualisation(),
                                              depth))
            self.ttt.import_state(prev_state)
        return
