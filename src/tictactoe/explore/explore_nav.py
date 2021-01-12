from typing import Dict
import networkx as nx
from src.lib.rltrace.trace import Trace
from src.lib.uniqueref import UniqueRef
from src.tictactoe.experiment0.statenav import StateNav
from src.tictactoe.tictactoe import TicTacToe
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.explore.explore import Explore


class ExploreNav(StateNav):
    """
    Create a list of all possible game state and a graph of game states and allow exploration
    """
    _dir_to_use: str
    _trace: Trace
    _ttt_event_stream: TicTacToeEventStream
    _ttt: TicTacToe
    _session_uuid: str
    _explore: Explore
    _visits: Dict
    _visit_summary = Dict
    _graph: nx.DiGraph

    LOAD_FMT = "command is <load {} session_id> or <load {} session_id>"
    LIST_FMT = "command is <list {}> or <list {}>"
    LVL_FMT = "Level [{:1.0f}] with [{:4.0f}] states, [{:4.0f}] Won, [{:4.0f}] Lost [{:4.0f}] drawn: Visits - Min [{:4.0f}] Max [{:4.0f}] Mean [{:7.2f}] StdDev [{:7.3f}] CVar [{:5.3f}]"

    _sep: str = "_____________________________________________________________________"

    def __init__(self,
                 ttt: TicTacToe,
                 ttt_event_stream: TicTacToeEventStream,
                 trace: Trace,
                 session_uuid: str,
                 dir_to_use: str = None):
        self._dir_to_use = dir_to_use
        self._trace = trace
        self._ttt_event_stream = ttt_event_stream
        self._ttt = ttt
        self._session_uuid = session_uuid
        self._explore = Explore(ttt=self._ttt,
                                trace=self._trace,
                                ttt_event_stream=self._ttt_event_stream)
        self._visits = None
        self._visit_summary = None
        self._graph = None
        return

    def _reset(self) -> None:
        """
        Return class to intial state
        """
        self._visits = None
        self._visit_summary = None
        self._graph = None
        return

    @staticmethod
    def parse(arg):
        """
        Convert a series of zero or more numbers to an argument tuple
        """
        return tuple(map(str, arg.split()))

    def do_visit_action(self,
                        level: int) -> None:
        """
        Show the summary of the given level, where level = action
        Number of states and number of Win/Lose/Draw
        :param level: The level to show summary for
        """
        coefficient_of_variation = self._visit_summary[level][Explore.VisitSummary.level_stdev] / \
                                   self._visit_summary[level][Explore.VisitSummary.level_mean]
        if level in self._visit_summary:
            self._trace.log().info(self.LVL_FMT.format(
                level,
                self._visit_summary[level][Explore.VisitSummary.game_states.value],
                self._visit_summary[level][Explore.VisitSummary.game_won.value],
                self._visit_summary[level][Explore.VisitSummary.game_lost.value],
                self._visit_summary[level][Explore.VisitSummary.game_drawn.value],
                self._visit_summary[level][Explore.VisitSummary.level_min.value],
                self._visit_summary[level][Explore.VisitSummary.level_max.value],
                self._visit_summary[level][Explore.VisitSummary.level_mean.value],
                self._visit_summary[level][Explore.VisitSummary.level_stdev.value],
                coefficient_of_variation))
        else:
            self._trace.log().info("Level [{}] is not in the visit summary".format(level))
        return

    def do_visit_summary(self) -> None:
        """
        Produce a summary of loaded visit
        """
        levels = self._visit_summary.keys()
        won = 0
        lost = 0
        draw = 0
        num_stat = 0
        ave_game_len = list()
        for level in levels:
            num_stat += self._visit_summary[level][Explore.VisitSummary.game_states.value]
            w = self._visit_summary[level][Explore.VisitSummary.game_won.value]
            l = self._visit_summary[level][Explore.VisitSummary.game_lost.value]
            d = self._visit_summary[level][Explore.VisitSummary.game_drawn.value]
            ave_game_len.extend([(w * (level + 1)), (l * (level + 1)), (d * (level + 1))])
            won += w
            lost += l
            draw += d
            self.do_visit_action(level)
        self._trace.log().info("Total states {} @ {:3.0f}% of total state space".format(num_stat, num_stat / 85.32))
        self._trace.log().info("Total wins {} @ {:3.0f}% of total win space".format(won, won / 9.42))
        self._trace.log().info("Total losses {} @ {:3.0f}% of total loss space".format(lost, lost / 9.42))
        self._trace.log().info("Total draws {} @ {:3.0f}% of total draw space".format(draw, draw / .32))
        self._trace.log().info("Average game length {:.3f}".format(sum(ave_game_len) / (won + lost + draw)))
        self._trace.log().info("End states as % of tot states {:3.0f}%".format(((won + lost + draw) / num_stat) * 100))
        return

    def do_graph_summary(self) -> None:
        """
        Produce a summary of a loaded graph
        :return:
        """
        num_nodes = len(self._graph.nodes)
        num_edges = len(self._graph.edges)
        self._trace.log().info("Total states {} @ {:3.0f}% of total state space".format(num_nodes, num_nodes / 85.32))
        self._trace.log().info("Total edges {}".format(num_edges))
        return

    def do_action(self,
                  action: int) -> None:
        """
        Navigate the structure by following the given action
        :param action: The action to navigate by
        """
        if self._visit_summary is not None:
            self.do_visit_action(level=action)
        elif self._graph is not None:
            self._trace.log().info("Level action not implemented for graph")
        else:
            self._trace.log().info("No visits or graphs loaded, use the <load> command")
        return

    def do_load(self,
                args) -> None:
        """
        Load a visit or a graph file from YAML so it can be explored
        """
        if args is None or len(args) == 0:
            self._trace.log().info(self.LOAD_FMT.format(StateNav.ExplorationFile.visit.value,
                                                        StateNav.ExplorationFile.graph.value))
            return
        parsed_args = self.parse(args)
        if len(parsed_args) != 2:
            self._trace.log().info(self.LOAD_FMT.format(StateNav.ExplorationFile.visit.value,
                                                        StateNav.ExplorationFile.graph.value))
        else:
            if parsed_args[0] == StateNav.ExplorationFile.visit.value:
                self._reset()
                self._visits = self._explore.load_visits_from_yaml(session_uuid=parsed_args[1],
                                                                   dir_to_use=self._dir_to_use)
                if self._visits is not None and len(self._visits) > 0:
                    self._visit_summary = self._explore.generate_visit_summary(self._visits)
            elif parsed_args[0] == StateNav.ExplorationFile.graph.value:
                self._reset()
                self._graph = self._explore.load_graph_from_yaml(session_uuid=parsed_args[1],
                                                                 dir_to_use=self._dir_to_use)
            else:
                self._trace.log().info(self.LOAD_FMT.format(StateNav.ExplorationFile.visit.value,
                                                            StateNav.ExplorationFile.graph.value))
        return

    def do_list(self,
                args) -> None:
        """
        List all matching visit or graph files
        """
        if args is None or len(args) == 0:
            self._trace.log().info(self.LIST_FMT.format(StateNav.ExplorationFile.visit.value,
                                                        StateNav.ExplorationFile.graph.value))
            return
        parsed_args = self.parse(args)
        if len(parsed_args) != 1:
            self._trace.log().info(self.LIST_FMT.format(StateNav.ExplorationFile.visit.value,
                                                        StateNav.ExplorationFile.graph.value))
        else:
            files = None
            if parsed_args[0] == StateNav.ExplorationFile.visit.value:
                files = self._explore.list_visit_files(dir_to_use=self._dir_to_use)
            elif parsed_args[0] == StateNav.ExplorationFile.graph.value:
                files = self._explore.list_graph_files(dir_to_use=self._dir_to_use)
            else:
                self._trace.log().info(self.LIST_FMT.format(StateNav.ExplorationFile.visit.value,
                                                            StateNav.ExplorationFile.graph.value))
            if files is not None:
                if len(files) > 0:
                    self._trace.log().info(
                        "[{}] {} files found in {}".format(len(files), parsed_args[0], self._dir_to_use))
                    for f in files:
                        self._trace.log().info("[{}]".format(f))
                    self._trace.log().info("Type <load file_prefix> to explore more")
                else:
                    self._trace.log().info("No {} files found in {}".format(parsed_args[0], self._dir_to_use))

        return

    def do_summary(self) -> None:
        """
        Produce a summary of currently loaded game stats
        """
        if self._visit_summary is not None:
            self.do_visit_summary()
        elif self._graph is not None:
            self.do_graph_summary()
        else:
            self._trace.log().info("No visits or graphs loaded, use the <load> command")
        return

    def do_explore(self,
                   args) -> None:
        """
        Invoke an exploration of TicTacToe state space.
        """
        explore_session_id = UniqueRef().ref
        if args is None or len(args) == 0:
            self._trace.log().info("Command format <explore all> or <explore random num_games>")
            return
        parsed_args = self.parse(args)
        if parsed_args[0] == StateNav.Exploration.all.value:
            self._trace.log().info("Running simulation of every TicTacToe game - takes a while !")
            self._explore.explore_all()
            self._trace.log().info("Done running simulation of game")
            self._trace.log().info("Saving results under session {}".format(self._session_uuid))
            self._explore.save(session_uuid=explore_session_id, dir_to_use=self._dir_to_use)
            self._trace.log().info("Saved results, use <load visit {}> or <load graph {}> to explore".format(
                self._session_uuid,
                self._session_uuid))
        elif parsed_args[0] == StateNav.Exploration.random.value:
            if len(parsed_args) != 2:
                self._trace.log().info("Command format <explore random num_games>, num games not specified")
            else:
                try:
                    num_games = int(parsed_args[1])
                    if num_games >= 0:
                        self._trace.log().info("Running random simulation of {} games".format(num_games))
                        self._explore.explore_random(num_games=num_games)
                        self._trace.log().info("Done running random simulation")
                        self._trace.log().info("Saving results under session {}".format(self._session_uuid))
                        self._explore.save(session_uuid=explore_session_id, dir_to_use=self._dir_to_use)
                        self._trace.log().info(
                            "Saved results, use <load visit {}> or <load graph {}> to explore".format(
                                self._session_uuid,
                                self._session_uuid))
                    else:
                        self._trace.log().info("Command format <explore random num_games>, num games must be >= 0")
                except ValueError:
                    self._trace.log().info("Command format <explore random num_games>, num games is not an integer")
                except Exception as e:
                    self._trace.log().info("Command format <explore random num_games> failed with error [{}]".
                                           format(str(e)))
        else:
            self._trace.log().info("Command format <explore all> or <explore random num_games>")
        self.do_load("{} {}".format(StateNav.ExplorationFile.visit.value, explore_session_id))
        self.do_summary()
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
                if level < 2:  # This is a debug to limit decent - set to < 9 for full depth - and wait 12 hours!!
                    tot = tot + self.state_transitions(graph, attrs, level + 1, v)
                    self.add_to_attr(attrs, 10, u, 1)
                    self.add_to_attr(attrs, 10, v, 1)
                    if Explore.EDGE_ATTR_WON in a:
                        if a[Explore.EDGE_ATTR_WON]:
                            self.add_to_attr(attrs, level, 'games', 1)

        self._trace.log().info(
            "{}Level [{}] - State Grand Total [{}]".format(indent, level, tot))
        self.add_to_attr(attrs, level, 'states', transitions)
        return tot
