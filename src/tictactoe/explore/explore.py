from enum import IntEnum, unique
from typing import List
import glob
import os
import networkx as nx
import yaml
import numpy as np
from typing import Dict
from src.lib.rltrace.trace import Trace
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.tictactoe import TicTacToe
from src.tictactoe.tictactoe_board_state import TicTacToeBoardState


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
    GRAPHS_FILE = "{}/{}_networkx_graph.yaml"
    STATES_YAML_KEY = 'states'
    EDGE_ATTR_WON = 'won'

    @unique
    class VisitSummary(IntEnum):
        game_states = 0
        game_won = 1
        game_lost = 2
        game_drawn = 3
        level_min = 4
        level_max = 5
        level_total = 6
        level_mean = 7
        level_stdev = 8

    NUM_ATTR = VisitSummary.level_stdev.value + 1

    def __init__(self,
                 ttt: TicTacToe,
                 trace: Trace,
                 ttt_event_stream: TicTacToeEventStream):
        self.trace = trace
        self.ttt_event_stream = ttt_event_stream
        self.ttt = ttt
        self.visited = None
        self.graph = None
        self._reset()
        return

    def list_visit_files(self,
                         dir_to_use: str = '.') -> List[str]:
        """
        List all visits files in the given directory
        :param dir_to_use:
        :return: List of matching visit file names: The oldest + 10 newest
        """
        visit_files = list()
        for f in sorted(glob.glob(self.VISITS_FILE.format(dir_to_use, "*")), key=os.path.getmtime, reverse=True):
            visit_files.append(f)
        return [*visit_files[:10], visit_files[-1]]

    def list_graph_files(self,
                         dir_to_use: str = '.') -> List[str]:
        """
        List all graph files in the given directory
        :param dir_to_use:
        :return: List of matching graph file names
        """
        graph_files = list()
        for f in sorted(glob.glob(self.GRAPHS_FILE.format(dir_to_use, "*")), key=os.path.getmtime, reverse=True):
            graph_files.append(f)
        return graph_files

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
            # ToDo change state so it can be used directly as YAML key with no special chars in it
            visited_states.append('{}: {}'.format(visit, self.visited[visit]))
        visit_as_dict = dict()
        visit_as_dict[self.STATES_YAML_KEY] = visited_states
        try:
            with open(filename, "w") as file:
                yaml.dump(visit_as_dict, file)
            self.trace.log().info("Saved visits as YAML [{}]".format(filename))
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
            self.trace.log().info("Saved networkx graph as YAML [{}]".format(filename))
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
        self.save_graph(self.GRAPHS_FILE.format(dir_to_use, session_uuid))
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
            self.trace.log().info("Loading visits from file [{}]".format(filename))
            with open(filename, 'r') as stream:
                res = dict()
                visits = yaml.safe_load(stream)[self.STATES_YAML_KEY]
                for visit in visits:
                    state, count = visit.split(':')
                    res[state] = int(count)
            self.trace.log().info("Loaded [{}] visits".format(len(res)))
        except Exception as e:
            self.trace.log().error("Failed to load visits from file [{}] with error [{}]"
                                   .format(filename, str(e)))
        return res

    def generate_visit_summary(self,
                               visits: Dict) -> Dict:
        """
        Generate a level by level summary of all visits passed
        For each level calculate the total number of states & games won, lost & drawn
        :param visits: The visits as a dictionary as returned by load_visits_from_yaml
        :return: Level by level analysis as dictionary where the primary key is level 0 - 8
        """
        analysis = dict()
        num_visit = len(visits)
        visit_counts = dict()
        self.trace.log().info("Starting visit analysis {} visits to process".format(num_visit))
        i = 0
        for s in visits:
            if i % max(1, int(num_visit / 10)) == 0:
                self.trace.log().info("[{:3.0f}]% of visits processed]".format((i / num_visit) * 100))
            i += 1
            self.ttt.board_as_string_to_internal_state(s)
            game_step = self.ttt.game_step()
            if game_step not in analysis:
                analysis[game_step] = [0.0] * self.NUM_ATTR
            analysis[game_step][self.VisitSummary.game_states.value] += 1
            if game_step not in visit_counts:
                visit_counts[game_step] = list()
            visit_counts[game_step].append(visits[s])
            game_state = self.ttt.game_state()
            self.trace.log().debug("State {} Step {} GStat {}".format(s, game_step, game_state))
            if game_state == TicTacToeBoardState.o_win:
                analysis[game_step][self.VisitSummary.game_won.value] += 1
            elif game_state == TicTacToeBoardState.x_win:
                analysis[game_step][self.VisitSummary.game_lost.value] += 1
            elif game_state == TicTacToeBoardState.draw:
                analysis[game_step][self.VisitSummary.game_drawn.value] += 1
        for game_step in visit_counts.keys():
            a = np.array(visit_counts[game_step], dtype=np.int)
            analysis[game_step][self.VisitSummary.level_total] = int(np.sum(a))
            analysis[game_step][self.VisitSummary.level_min] = int(np.min(a))
            analysis[game_step][self.VisitSummary.level_max] = int(np.max(a))
            analysis[game_step][self.VisitSummary.level_mean] = float(np.mean(a))
            analysis[game_step][self.VisitSummary.level_stdev] = float(np.std(a))

        self.trace.log().info("Finished visit analysis")
        return analysis

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
            filename = self.GRAPHS_FILE.format(dir_to_use, session_uuid)
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
            self.visited[state] = 0
        self.visited[state] += 1
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
        self.trace.log().debug("Visited {} depth [{}] total found {}".
                               format(self.ttt.state().state_as_visualisation(),
                                      step,
                                      len(self.visited)))
        self.record_visit(curr_state, curr_state_is_episode_end)
        self.record_network(prev_state, curr_state, curr_state_is_episode_end)
        return

    def explore_random(self,
                       num_games: int) -> None:
        """
        Run random games
        :param num_games: The number of random games to run
        """
        self._reset()
        agents = [self.ttt.x_agent_name(), self.ttt.o_agent_name()]
        games_played = 0
        while games_played < num_games:
            active_agent_id = agents[np.random.randint(2)]
            self.ttt.episode_start()
            depth = 0
            while not self.ttt.episode_complete():
                prev_state_s = self.ttt.state().state_as_string()
                actions_to_explore = self.ttt.actions(self.ttt.state())
                random_legal_action = actions_to_explore[np.random.randint(len(actions_to_explore))]
                next_agent_id = self.ttt.do_action(agent_id=active_agent_id, action=random_legal_action)
                self.record(prev_state=prev_state_s,
                            curr_state=self.ttt.state().state_as_string(),
                            step=depth,
                            curr_state_is_episode_end=self.ttt.episode_complete())
                active_agent_id = next_agent_id
                depth += 1
            games_played += 1
            if games_played % max(1, int(num_games / 10)) == 0:
                self.trace.log().info("{:3.0f}% complete".format((games_played / num_games) * 100))
        return

    def explore_all(self,
                    agent_id: str = None,
                    state_actions_as_str: str = "",
                    depth: int = 0) -> None:
        """
        Recursive routine to visit all possible game states
        """
        if agent_id is None:
            self._reset()
            agents = [self.ttt.x_agent_name(), self.ttt.o_agent_name()]
        else:
            agents = [agent_id]
        for active_agent_id in agents:
            self.ttt.import_state(state_as_string=state_actions_as_str)
            actions_to_explore = self.ttt.actions(self.ttt.state())
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
                        self.explore_all(agent_id=next_agent_id,
                                         state_actions_as_str=self.ttt.state_action_str(),
                                         depth=depth + 1)
                    else:
                        self.explore_all(agent_id=next_agent_id,
                                         state_actions_as_str=prev_state,
                                         depth=depth)
                else:
                    self.trace.log().debug("Skipped {} depth [{}]".
                                           format(self.ttt.state().state_as_visualisation(),
                                                  depth))
                self.ttt.import_state(prev_state)
        return
