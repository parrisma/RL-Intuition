from typing import List
import networkx as nx
import yaml
from typing import Dict
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
    graph: nx.DiGraph

    VISITS_FILE = "{}/{}_visits.yaml"
    GRAPH_FILE = "{}/{}_networkx_graph.yaml"
    STATES_YAML_KEY = 'states'
    EDGE_ATTR_WON = 'won'

    def __init__(self,
                 ttt: TicTacToe,
                 trace: Trace,
                 ttt_event_stream: TicTacToeEventStream):
        self.trace = trace
        self.ttt_event_stream = ttt_event_stream
        self.ttt = ttt
        self._reset()
        return

    def _reset(self) -> None:
        """
        Reset the internal state
        """
        self.visited = dict()
        self.graph = nx.DiGraph()
        self.ttt.episode_start()
        return

    def _other(self,
               agent_name: str) -> str:
        """
        Return the name of the 'other' agent given on of the agents X or O
        :param agent_name: The name to give the other agent for
        :return: The other agent
        """
        if agent_name == self.ttt.x_agent_name():
            return self.ttt.o_agent_name()
        else:
            return self.ttt.x_agent_name()

    def save_visits(self,
                    filename: str) -> None:
        """
        Save the unique list of states visited as yaml
        :param filename: The filename to save the visits as yaml as
        """
        visited_states = list()
        for visit in self.visited:
            visited_states.append('{}'.format(visit))
        visit_as_dict = dict()
        visit_as_dict[self.STATES_YAML_KEY] = visited_states
        try:
            with open(filename, "w") as file:
                yaml.dump(visit_as_dict, file)
            self.trace.log().debug("Saved visits as YAML [{}]".format(filename))
        except Exception as e:
            self.trace.log().error("Failed to save states to file [{}] with error [{}]"
                                   .format(filename, str(e)))

    def save_graph(self,
                   filename: str) -> None:
        """
        Save the network-x graph as YAML
        :param filename: The filename to save the graph as
        """
        try:
            nx.write_yaml(self.graph, filename)
            self.trace.log().debug("Saved networkx graph as YAML [{}]".format(filename))
        except Exception as e:
            self.trace.log().error("Failed to save networkx graph to file [{}] with error [{}]"
                                   .format(filename, str(e)))
        return

    def save(self,
             session_uuid: str,
             dir_to_use: str = ".") -> None:
        """
        Save the exploration
        1. The session UUID <dir_to_use>/states_<session_uuid>.yaml
        2. The state graph as yam <dir_to_use>/graph_<session_uuid>.yaml
        :param session_uuid: The current session_uuid (to include in the filename)
        :param dir_to_use: The dir to save files in default is "." current working directory
        """
        self.save_visits(self.VISITS_FILE.format(dir_to_use, session_uuid))
        self.save_graph(self.GRAPH_FILE.format(dir_to_use, session_uuid))
        return

    def load_visits_from_yaml(self,
                              session_uuid: str,
                              dir_to_use: str = ".") -> Dict:
        """
        Load the visits in the yaml file from the given dir with the given session uuid
        :param session_uuid: The current session_uuid (to include in the filename)
        :param dir_to_use: The dir to save files in default is "." current working directory
        """
        filename = str()
        res = None
        try:
            filename = self.VISITS_FILE.format(dir_to_use, session_uuid)
            with open(filename, 'r') as stream:
                res = dict()
                visits = yaml.safe_load(stream)[self.STATES_YAML_KEY]
                for visit in visits:
                    res[visit] = True
        except Exception as e:
            self.trace.log().error("Failed to save states to file [{}] with error [{}]"
                                   .format(filename, str(e)))
        return res

    def load_graph_from_yaml(self,
                             session_uuid: str,
                             dir_to_use: str = ".") -> nx.DiGraph:
        """
        Load the graph in the yaml file from the given dir with the given session uuid
        :param session_uuid: The current session_uuid (to include in the filename)
        :param dir_to_use: The dir to save files in default is "." current working directory
        """
        filename = str()
        res = None
        try:
            filename = self.GRAPH_FILE.format(dir_to_use, session_uuid)
            self.trace.log().debug("Start loading [{}]".format(filename))
            res = nx.read_yaml(filename)
            self.trace.log().debug("Finished loading [{}]".format(filename))
        except Exception as e:
            self.trace.log().error("Failed to load states to file [{}] with error [{}]"
                                   .format(filename, str(e)))
            res = None
        return res

    def record_visit(self,
                     state: str,
                     curr_state_is_episode_end: bool) -> None:
        """
        Add the state to the list of all discovered state
        :param state: The state to add
        :param curr_state_is_episode_end: True if the current state is a terminal state (Win/Draw) else False
        """
        if state not in self.visited:
            self.visited[state] = True
        return

    def record_network(self,
                       prev_state: str,
                       curr_state,
                       curr_state_is_episode_end: bool) -> None:
        """
        Add the prev to current state relationship to the network
        :param prev_state: The previous game state as string
        :param curr_state: The current game state as string
        :param curr_state_is_episode_end: True if the current state is a terminal state (Win/Draw) else False
        """
        if not self.graph.has_edge(prev_state, curr_state):
            self.graph.add_edge(u_of_edge=prev_state,
                                v_of_edge=curr_state)
            self.graph[prev_state][curr_state]['weight'] = 0
        weight = self.graph[prev_state][curr_state]['weight']
        self.graph[prev_state][curr_state]['weight'] = weight + 1
        self.graph[prev_state][curr_state][self.EDGE_ATTR_WON] = curr_state_is_episode_end
        return

    def already_seen(self,
                     prev_state: str,
                     curr_state) -> bool:
        """
        True if the exploration has seen a play that goes from prev_state to current state
        :param prev_state: The previous game state as string
        :param curr_state: The current game state as string
        :return: True if transition has been seen else False.
        """
        return self.graph.has_edge(prev_state, curr_state)

    def record(self,
               prev_state: str,
               curr_state,
               step: int,
               curr_state_is_episode_end: bool) -> None:
        """
        Record the discovery of a new game state
        :param prev_state: The previous game state as string
        :param curr_state: The current game state as string
        :param step: The step in the episode
        :param curr_state_is_episode_end: True if the current state is a terminal state (Win/Draw) else False
        """
        self.trace.log().info("Visited {} depth [{}] total found {}".
                              format(self.ttt.state().state_as_visualisation(),
                                     step,
                                     len(self.visited)))
        self.record_visit(curr_state, curr_state_is_episode_end)
        self.record_network(prev_state, curr_state, curr_state_is_episode_end)
        return

    def explore(self,
                agent_id: str = None,
                state_actions_as_str: str = "",
                depth: int = 0) -> None:
        """
        Recursive routine to visit all possible game states and
        """
        if depth > 100:
            return
        if agent_id is None:
            self._reset()
            agents = [self.ttt.x_agent_name(), self.ttt.o_agent_name()]
        else:
            agents = [agent_id]
        for active_agent_id in agents:
            self.ttt.import_state(state_as_string=state_actions_as_str)
            actions_to_explore = self.ttt.actions(self.ttt.state())
            # for action in [actions_to_explore[0]]:
            for action in actions_to_explore:
                prev_state = self.ttt.state_action_str()
                prev_state_s = self.ttt.state().state_as_string()
                next_agent_id = self.ttt.do_action(agent_id=active_agent_id, action=action)
                if not self.already_seen(prev_state=prev_state_s,
                                         curr_state=self.ttt.state().state_as_string()) and next_agent_id is not None:
                    self.record(prev_state=prev_state_s,
                                curr_state=self.ttt.state().state_as_string(),
                                step=depth,
                                curr_state_is_episode_end=self.ttt.episode_complete())
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
