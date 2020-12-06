from enum import IntEnum, unique
from typing import List, Union
import networkx as nx
import numpy as np
from typing import Dict
from src.lib.rltrace.trace import Trace
from src.reflrn.interface.agent import Agent
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.tictactoe import TicTacToe
from src.tictactoe.tictactoe_board_state import TicTacToeBoardState


class Explore:
    """
    This class explores all possible game states and records them as TTT Events.
    """
    _trace: Trace
    _ttt_event_stream: TicTacToeEventStream
    _ttt: TicTacToe
    _visited: Dict[str, int]
    _graph: nx.DiGraph

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
                 ttt_event_stream: TicTacToeEventStream,
                 visited: Dict[str, int] = None,
                 graph: nx.DiGraph = None):
        self._trace = trace
        self._ttt_event_stream = ttt_event_stream
        self._ttt = ttt
        self._visited = visited
        self._graph = graph
        self._reset()
        return

    def list_visit_files(self,
                         dir_to_use: str = '.') -> List[str]:
        """
        List all visits files in the given directory
        :param dir_to_use:
        :return: List of matching visit file names: The oldest + 10 newest
        """
        return self._ttt_event_stream.list_visit_files(dir_to_use=dir_to_use)

    def list_graph_files(self,
                         dir_to_use: str = '.') -> List[str]:
        """
        List all graph files in the given directory
        :param dir_to_use:
        :return: List of matching graph file names
        """
        return self._ttt_event_stream.list_graph_files(dir_to_use=dir_to_use)

    def _reset(self) -> None:
        """
        Reset the internal state
        """
        if self._visited is None:
            self._visited = dict()
        if self._graph is None:
            self._graph = nx.DiGraph()
        self._ttt.episode_start()
        return

    def other(self,
              agent: str) -> Union[int, str, Agent]:
        """
        Return the name, id or Agent object of the 'other' agent given on of the agents X or O
        :param agent: The name, id or Agent object to give the other agent for
        :return: The other agent name, id or object; where return type matches type of request id, name, agent
        """
        if agent is not None:
            if isinstance(agent, str):
                if agent == self._ttt.get_x_agent().name():
                    res = self._ttt.get_o_agent().name()
                else:
                    res = self._ttt.get_x_agent().name()
            elif isinstance(agent, int):
                if agent == self._ttt.get_x_agent().id():
                    res = self._ttt.get_o_agent().id()
                else:
                    res = self._ttt.get_x_agent().id()
            elif isinstance(agent, Agent):
                if agent.id() == self._ttt.get_x_agent().id():
                    res = self._ttt.get_o_agent()
                else:
                    res = self._ttt.get_x_agent()
            else:
                raise RuntimeError("Must be Agent name (str), Id (int) or Object (Agent) - {} was passed".
                                   format(type(agent)))
        else:
            raise RuntimeError("agent my not be None")
        return res

    def save(self,
             session_uuid: str = None,
             dir_to_use: str = None) -> None:
        """
        Save the exploration
        1. The session UUID <dir_to_use>/states_<session_uuid>.yaml
        2. The state graph as yam <dir_to_use>/graph_<session_uuid>.yaml
        :param session_uuid: The current session_uuid (to include in the filename)
        :param dir_to_use: The dir to save files in default is "." current working directory
        """
        self._ttt_event_stream.save_visits(visited=self._visited, session_uuid=session_uuid, dir_to_use=dir_to_use)
        self._ttt_event_stream.save_graph(graph=self._graph, session_uuid=session_uuid, dir_to_use=dir_to_use)
        return

    def load_graph_from_yaml(self,
                             session_uuid: str = None,
                             dir_to_use: str = None) -> nx.DiGraph:
        """
        Load the graph in the yaml file from the given dir with the given session uuid
        :param session_uuid: The current session_uuid (to include in the filename)
        :param dir_to_use: The dir to save files in default is "." current working directory
        """
        return self._ttt_event_stream.load_graph_from_yaml(session_uuid=session_uuid, dir_to_use=dir_to_use)

    def load_visits_from_yaml(self,
                              session_uuid: str,
                              dir_to_use: str = ".") -> Dict[str, int]:
        """
        Load the visits in the yaml file from the given dir with the given session uuid
        :param session_uuid: The current session_uuid (to include in the filename)
        :param dir_to_use: The dir to save files in default is "." current working directory
        """
        return self._ttt_event_stream.load_visits_from_yaml(session_uuid=session_uuid,
                                                            dir_to_use=dir_to_use)

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
        self._trace.log().info("Starting visit analysis {} visits to process".format(num_visit))
        i = 0
        for s in visits:
            if i % max(1, int(num_visit / 10)) == 0:
                self._trace.log().info("[{:3.0f}]% of visits processed]".format((i / num_visit) * 100))
            i += 1
            self._ttt.board_as_string_to_internal_state(s)
            game_step = self._ttt.game_step()
            if game_step not in analysis:
                analysis[game_step] = [0.0] * self.NUM_ATTR
            analysis[game_step][self.VisitSummary.game_states.value] += 1
            if game_step not in visit_counts:
                visit_counts[game_step] = list()
            visit_counts[game_step].append(visits[s])
            game_state = self._ttt.game_state()
            self._trace.log().debug("State {} Step {} GStat {}".format(s, game_step, game_state))
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

        self._trace.log().info("Finished visit analysis")
        return analysis

    def record_visit(self,
                     state: str,
                     curr_state_is_episode_end: bool) -> None:
        """
        Add the state to the list of all discovered state
        :param state: The state to add
        :param curr_state_is_episode_end: True if the current state is a terminal state (Win/Draw) else False
        """
        if state not in self._visited:
            self._visited[state] = 0
        self._visited[state] += 1
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
        if not self._graph.has_edge(prev_state, curr_state):
            self._graph.add_edge(u_of_edge=prev_state,
                                 v_of_edge=curr_state)
            self._graph[prev_state][curr_state]['weight'] = 0
        weight = self._graph[prev_state][curr_state]['weight']
        self._graph[prev_state][curr_state]['weight'] = weight + 1
        self._graph[prev_state][curr_state][self.EDGE_ATTR_WON] = curr_state_is_episode_end
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
        return self._graph.has_edge(prev_state, curr_state)

    def record(self,
               prev_state: str,
               curr_state: str,
               step: int,
               curr_state_is_episode_end: bool) -> None:
        """
        Record the discovery of a new game state
        :param prev_state: The previous game state as string
        :param curr_state: The current game state as string
        :param step: The step in the episode
        :param curr_state_is_episode_end: True if the current state is a terminal state (Win/Draw) else False
        """
        self._trace.log().debug("Visited {} depth [{}] total found {}".
                                format(self._ttt.state().state_as_visualisation(),
                                       step,
                                       len(self._visited)))
        self.record_visit(curr_state, curr_state_is_episode_end)
        if prev_state is not None and curr_state != prev_state:
            self.record_network(prev_state, curr_state, curr_state_is_episode_end)
        return

    def explore_random(self,
                       num_games: int) -> None:
        """
        Run random games
        :param num_games: The number of random games to run
        """
        self._reset()
        agents = [self._ttt.x_agent_name(), self._ttt.o_agent_name()]
        games_played = 0
        while games_played < num_games:
            active_agent_id = agents[np.random.randint(2)]
            self._ttt.episode_start()
            depth = 0
            while not self._ttt.episode_complete():
                prev_state_s = self._ttt.state().state_as_string()
                actions_to_explore = self._ttt.actions(self._ttt.state())
                random_legal_action = actions_to_explore[np.random.randint(len(actions_to_explore))]
                next_agent_id = self._ttt.do_action(agent_id=active_agent_id, action=random_legal_action)
                self.record(prev_state=prev_state_s,
                            curr_state=self._ttt.state().state_as_string(),
                            step=depth,
                            curr_state_is_episode_end=self._ttt.episode_complete())
                active_agent_id = next_agent_id
                depth += 1
            games_played += 1
            if games_played % max(1, int(num_games / 10)) == 0:
                self._trace.log().info("{:3.0f}% complete".format((games_played / num_games) * 100))
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
            agents = [self._ttt.x_agent_name(), self._ttt.o_agent_name()]
        else:
            agents = [agent_id]
        for active_agent_id in agents:
            self._ttt.import_state(state_as_string=state_actions_as_str)
            actions_to_explore = self._ttt.actions(self._ttt.state())
            for action in actions_to_explore:
                prev_state = self._ttt.state_action_str()
                prev_state_s = self._ttt.state().state_as_string()
                next_agent_id = self._ttt.do_action(agent_id=active_agent_id, action=action)
                if not self.already_seen(prev_state=prev_state_s,
                                         curr_state=self._ttt.state().state_as_string()) and next_agent_id is not None:
                    self.record(prev_state=prev_state_s,
                                curr_state=self._ttt.state().state_as_string(),
                                step=depth,
                                curr_state_is_episode_end=self._ttt.episode_complete())
                    if not self._ttt.episode_complete():
                        self.explore_all(agent_id=next_agent_id,
                                         state_actions_as_str=self._ttt.state_action_str(),
                                         depth=depth + 1)
                    else:
                        self.explore_all(agent_id=next_agent_id,
                                         state_actions_as_str=prev_state,
                                         depth=depth)
                else:
                    self._trace.log().debug("Skipped {} depth [{}]".
                                            format(self._ttt.state().state_as_visualisation(),
                                                   depth))
                self._ttt.import_state(prev_state)
        return

    def get_visited(self) -> Dict[str, int]:
        """
        Get the dict of visted states
        :return: Dictionary of visted states and associated visit count
        """
        return self._visited
