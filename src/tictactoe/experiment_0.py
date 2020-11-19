from typing import Dict
import networkx as nx
import logging
from src.tictactoe.experiment_base import ExperimentBase
from src.tictactoe.random_play_agent import RandomPlayAgent
from src.tictactoe.ttt_explore import Explore
from src.tictactoe.event.tictacttoe_event import TicTacToeEvent


class Experiment0(ExperimentBase):
    """
    This experiment is to map all the possible valid game states such that we can build an
    intuition for how to learn about the environment
    """

    def __init__(self):
        # We drive the TTT Environment via it's debug api setting raw states rather than allowing the
        # agents to play so we just map a random agent. However the agents are not activated
        super().__init__(RandomPlayAgent.RandomAgentFactory())
        return

    @staticmethod
    def add_to_attr(attrs: Dict,
                    level: int,
                    attr_name: str,
                    attr_v) -> None:
        """
        Add attr_v to the current value of 'attr_name'
        :param attrs: The attributes to be updated
        :param level: The level the attr_name relates to
        :param attr_name: The name of the attribute to be updated
        :param attr_v: The value to increment the attribute value by
        """
        if level not in attrs:
            attrs[level] = dict()
        if attr_name not in attrs[level]:
            attrs[level][attr_name] = 0
        attrs[level][attr_name] += attr_v
        return

    def state_transitions(self,
                          graph: nx.DiGraph,
                          attrs: Dict,
                          level: int = 0,
                          state_id: str = '000000000') -> int:
        """
        The number of state transitions below the given state_id (based on recursive descent)
        :param graph: The graph to explore, must be a Directed Graph without loops or recursion will fail.
        :param attrs: A dictionary to hold all the summary attributes by level of the exploration

        Do not pass used by recursive calls only
        :param level: The depth of the exploration
        :param state_id: The state_id to start_with
        """
        edges = graph.edges(state_id, data=True)
        transitions = len(edges)
        tot = transitions
        indent = '  ' * (level + 1)
        self._trace.log().info("{}Level [{}] - State [{}] transitions [{}]".format(indent, level, state_id, tot))
        if tot > 0:
            for u, v, a in edges:
                if level < 2:
                    tot = tot + self.state_transitions(graph, attrs, level + 1, v)
                    Experiment0.add_to_attr(attrs, 10, u, 1)
                    Experiment0.add_to_attr(attrs, 10, v, 1)
                    if Explore.EDGE_ATTR_WON in a:
                        if a[Explore.EDGE_ATTR_WON]:
                            Experiment0.add_to_attr(attrs, level, 'games', 1)

        self._trace.log().info(
            "{}Level [{}] - State Grand Total [{}]".format(indent, level, tot))
        Experiment0.add_to_attr(attrs, level, 'states', transitions)
        return tot

    def run(self) -> None:
        """
        Allow command lined based load and exploration of all TicTacToe states
        """
        self._trace.log().info("Experiment {} Started".format(self.__class__.__name__))
        e = Explore(ttt=self._ttt,
                    trace=self._trace,
                    ttt_event_stream=self._ttt_event_stream)

        load_test = True
        if load_test:
            self._trace.log().setLevel(logging.INFO)
            analysis = dict()
            v = e.load_visits_from_yaml(session_uuid="fb5c87fb96d244249b9b453f5a059438", dir_to_use="./scratch")
            for s in v:
                self._ttt.board_as_string_to_internal_state(s)
                game_step = self._ttt.game_step()
                if game_step not in analysis:
                    analysis[game_step] = [0, 0, 0, 0]
                analysis[game_step][0] += 1
                game_state = self._ttt.game_state()
                self._trace.log().info("State {} Step {} GStat {}".format(s, game_step, game_state))
                if game_state == TicTacToeEvent.O_WIN:
                    analysis[game_step][1] += 1
                elif game_state == TicTacToeEvent.X_WIN:
                    analysis[game_step][2] += 1
                elif game_state == TicTacToeEvent.DRAW:
                    analysis[game_step][3] += 1

            g = e.load_graph_from_yaml(session_uuid="fb5c87fb96d244249b9b453f5a059438", dir_to_use="./scratch")
            # g = e.load_graph_from_yaml(session_uuid="all_ttt")
            attrs = dict()
            self.state_transitions(graph=g, attrs=attrs)
            for level in range(9):
                if level in attrs:
                    games = "-"
                    states = "-"
                    if 'games' in attrs[level]:
                        games = attrs[level]['games']
                    if 'states' in attrs[level]:
                        states = attrs[level]['states']
                    self._trace.log().info("Level {} : Transitions: {} : Games {}".format(level, states, games))
        else:
            e.explore()
            e.save(session_uuid=self._session_uuid, dir_to_use="./scratch")

        self._trace.log().info("Experiment {} Finished".format(self.__class__.__name__))
        return


if __name__ == "__main__":
    Experiment0().run()
