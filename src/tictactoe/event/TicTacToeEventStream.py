from typing import List, Dict
from datetime import datetime
import os
import glob
import yaml
import networkx as nx
from elasticsearch import Elasticsearch
from src.lib.rltrace.trace import Trace
from src.lib.elastic.esutil import ESUtil
from src.reflrn.interface.state import State
from src.reflrn.interface.state_factory import StateFactory
from src.tictactoe.event.tictacttoe_event import TicTacToeEvent


class TicTacToeEventStream:
    """
    Manage all event related data

    1. Raw Events are persisted directly to Elastic keyed with session_uuid
    2. Visit Data (State, num_visits) stored in YAML with file name based on session_uuid
    3. Graph Data (Nodes & Edges) stored in YAML (as persisted by network nx lib) with file names based on session_uuid

    """
    _trace: Trace
    _es: Elasticsearch
    _session_uuid: str
    _state_factory: StateFactory
    _jflds: List
    _dir_to_use: str

    _jflds = ["timestamp",
              "session_uuid",
              "episode_uuid",
              "episode_step",
              "state",
              "action",
              "agent",
              "reward",
              "episode_end",
              "episode_outcome"]
    TIMESTAMP = 0
    SESSION_UUID = 1
    EPISODE_UUID = 2
    EPISODE_STEP = 3
    STATE = 4
    ACTION = 5
    AGENT = 6
    REWARD = 7
    EPISODE_END = 8
    EPISODE_OUTCOME = 9

    JSON_TRUE = "true"

    SESSION_UUID_Q = '{"query":{"match":{ "session_uuid":"<arg0>"}}}'
    EPISODE_UUID_Q = '{"query":{"match":{ "episode_uuid":"<arg0>"}}}'
    SESSION_LIST_Q = '{"size": 0,"aggs": {"session_uuid_list": {"terms": {"field": "session_uuid.keyword", "order": {"ts": "desc"}},"aggs": {"ts": {"max": {"field": "timestamp"}}}}}}'
    SESSION_COUNT_Q = '{"query":{"match": {"session_uuid":"<arg0>"}}}'

    VISITS_FILE = "{}/{}_visits.yaml"
    GRAPHS_FILE = "{}/{}_networkx_graph.yaml"

    STATES_YAML_KEY = 'states'

    def __init__(self,
                 trace: Trace,
                 es: Elasticsearch,
                 es_index: str,
                 state_factory: StateFactory,
                 session_uuid: str,
                 dir_to_use: str):
        self._dir_to_use = self._verify_dir(dir_to_verify=dir_to_use)
        self._trace = trace
        self._es = es
        if ESUtil.index_exists(es=es, idx_name=es_index):
            self._es_index = es_index
        else:
            raise RuntimeError("Cannot create TicTacToeEventStream as index {} does not exist".format(es_index))
        self._state_factory = state_factory
        self._session_uuid = session_uuid
        self._fmt = '{{{{"{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}"}}}}'
        self._fmt = self._fmt.format(*self._jflds)
        self._date_formatter = ESUtil.DefaultElasticDateFormatter()
        return

    def _verify_dir(self,
                    dir_to_verify: str) -> str:
        """
        Raise Runtime error if the given dir does not exists
        :param dir_to_verify: The direct to verify as exists
        :return: The dir if it exists
        """
        if not os.path.exists(dir_to_verify):
            raise RuntimeError("[{}] passed to {} does not exist".format(dir_to_verify, self.__class__.__name__))
        return dir_to_verify

    @property
    def session_uuid(self) -> str:
        """
        Get the session uuid of the event stream
        :return: The session uuid of the event stream
        """
        return self._session_uuid

    def record_event(self,
                     episode_uuid: str,
                     episode_step: int,
                     state: State,
                     agent: str,
                     action: str,
                     reward: float,
                     episode_end: bool,
                     episode_outcome: str) -> None:
        """
        Persist the given event in the ttt_event index. Assume the given elastic instance has the index already
        created.
        :param episode_uuid: the UUID of the episode
        :param episode_step: the step number of the episode
        :param state: the current state of the environment
        :param agent: the name of the agent performing the action
        :param action: the action taken that gave rise to the given state
        :param reward: the reward given to the agent on entering this state
        :param episode_end: true if the given state is a terminal state for the episode
        :param episode_outcome: The outcome of the episode X-Win, O-Win, Draw, Step
        """
        try:
            now_timestamp = self._date_formatter.format(datetime.now())
            json_msg = self._fmt.format(now_timestamp,
                                        self._session_uuid,
                                        episode_uuid,
                                        episode_step,
                                        state.state_as_string(),
                                        action,
                                        agent,
                                        str(reward),
                                        ESUtil.bool_as_es_value(episode_end),
                                        episode_outcome)
            ESUtil.write_doc_to_index(es=self._es,
                                      idx_name=self._es_index,
                                      document_as_json=json_msg)
        except Exception as e:
            raise RuntimeError(
                "Record event failed to index [{}] with exception [{}]".format(self._es_index, str(e)))

        return

    def get_session(self,
                    session_uuid: str) -> List['TicTacToeEvent']:
        """
        return all events for the given session_uuid
        :param session_uuid: The uuid of the session to get
        :return: A list of TicTacToe Events for the given session_uuid
        """
        try:
            ttt_events_as_json = ESUtil.run_search(es=self._es,
                                                   idx_name=self._es_index,
                                                   json_query=self.SESSION_UUID_Q,
                                                   arg0=session_uuid,
                                                   refresh=self.JSON_TRUE)
            res = list()
            for jer in ttt_events_as_json:
                event_as_json = jer['_source']
                st = self._state_factory.new_state(event_as_json[self._jflds[self.STATE]])
                ttt_e = TicTacToeEvent(episode_uuid=event_as_json[self._jflds[self.EPISODE_UUID]],
                                       episode_step=int(event_as_json[self._jflds[self.EPISODE_STEP]]),
                                       state=st,
                                       action=event_as_json[self._jflds[self.ACTION]],
                                       agent=event_as_json[self._jflds[self.AGENT]],
                                       reward=float(event_as_json[self._jflds[self.REWARD]]),
                                       episode_end=bool(self.JSON_TRUE == event_as_json[self._jflds[self.EPISODE_END]]),
                                       episode_outcome=event_as_json[self._jflds[self.EPISODE_OUTCOME]]
                                       )
                res.append(ttt_e)
        except Exception as e:
            raise RuntimeError(
                "Get session failed with exception [{}]".format(str(e)))

        # Always return sorted in uuid & step order.
        res = sorted(res, key=lambda x: "{}{}".format(x.episode_uuid, x.episode_step))
        return res

    def list_of_available_sessions(self) -> List[List]:
        """
        Return a list of all the session_uuid for which there are ttt events
        :return: List of session_uuid's
        """
        try:
            ttt_session_uuids = ESUtil.run_search_agg(es=self._es,
                                                      idx_name=self._es_index,
                                                      json_query=self.SESSION_LIST_Q,
                                                      arg0='session_uuid_list',
                                                      refresh=self.JSON_TRUE)
        except Exception as e:
            raise RuntimeError(
                "Get session uuids failed with exception [{}]".format(str(e)))
        return ttt_session_uuids

    def count_session(self,
                      session_uuid: str) -> int:
        """
        Return the number of events held for the give session uuid
        :param session_uuid: The session UUID to count
        :return: The number of events in the session UUID or None if no such session UUID
        """
        try:
            ttt_session_count = ESUtil.run_count(es=self._es,
                                                 idx_name=self._es_index,
                                                 json_query=self.SESSION_COUNT_Q,
                                                 arg0=session_uuid)
        except Exception as e:
            raise RuntimeError(
                "Get session count failed with exception [{}]".format(str(e)))
        return ttt_session_count

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

    def save_visits(self,
                    visited: Dict[str, int],
                    session_uuid: str = None,
                    dir_to_use: str = None) -> None:
        """
        Persist the given visits where a visit is a State as str + number of hits in that state
        :param visited: The visits to persist in a Dictionary where a single visit = State as Str, Num Hits (Visits)
        :param session_uuid: The current session_uuid (to include in the filename)
        :param dir_to_use: The dir to save files in default is "." current working directory
        """
        filename = self.VISITS_FILE.format(self._default_if_none(dir_to_use, self._dir_to_use),
                                           self._default_if_none(session_uuid, self._session_uuid))
        visited_states = list()
        if visited is None or len(visited) == 0:
            self._trace.log().info("No visits passed to persist")
        else:
            for visit in visited:
                # ToDo change state so it can be used directly as YAML key with no special chars in it
                visited_states.append('{}: {}'.format(visit, visited[visit]))
            visit_as_dict = dict()
            visit_as_dict[self.STATES_YAML_KEY] = visited_states
            try:
                with open(filename, "w") as file:
                    yaml.dump(visit_as_dict, file)
                self._trace.log().info("Saved [{}] visits as YAML [{}]".format(len(visited), filename))
            except Exception as e:
                self._trace.log().error("Failed to save states to file [{}] with error [{}]"
                                        .format(filename, str(e)))
        return

    def load_visits_from_yaml(self,
                              session_uuid: str = None,
                              dir_to_use: str = None) -> Dict[str, int]:
        """
        Load the visits in the yaml file from the given dir with the given session uuid
        :param session_uuid: The current session_uuid (to include in the filename)
        :param dir_to_use: The dir to save files in default is "." current working directory
        """

        filename = self.VISITS_FILE.format(self._default_if_none(dir_to_use, self._dir_to_use),
                                           self._default_if_none(session_uuid, self._session_uuid))
        res = None
        try:
            filename = self.VISITS_FILE.format(dir_to_use, session_uuid)
            self._trace.log().info("Loading visits from file [{}]".format(filename))
            with open(filename, 'r') as stream:
                res = dict()
                visits = yaml.safe_load(stream)[self.STATES_YAML_KEY]
                for visit in visits:
                    state, count = visit.split(':')
                    res[state] = int(count)
            self._trace.log().info("Loaded [{}] visits".format(len(res)))
        except Exception as e:
            self._trace.log().error("Failed to load visits from file [{}] with error [{}]"
                                    .format(filename, str(e)))
        return res

    def list_graph_files(self,
                         dir_to_use: str = None) -> List[str]:
        """
        List all graph files in the given directory
        :param dir_to_use:
        :return: List of matching graph file names
        """
        dtu = self._default_if_none(dir_to_use, self._dir_to_use)
        graph_files = list()
        for f in sorted(glob.glob(self.GRAPHS_FILE.format(dtu, "*")), key=os.path.getmtime, reverse=True):
            graph_files.append(f)
        return graph_files

    def save_graph(self,
                   graph: nx.DiGraph,
                   session_uuid: str = None,
                   dir_to_use: str = None) -> None:
        """
        Save the given graph with in the given dir referenced against the given session_uuid
        :param graph: The graph to sabe
        :param session_uuid: The current session_uuid (to include in the filename)
        :param dir_to_use: The dir to save files in default is "." current working directory
        """
        if graph is None or len(graph) == 0:
            self._trace.log().error("Given graph is empty, nothing saved")
        else:
            filename = self.GRAPHS_FILE.format(self._default_if_none(dir_to_use, self._dir_to_use),
                                               self._default_if_none(session_uuid, self._session_uuid))
            try:
                nx.write_yaml(graph, filename)
                self._trace.log().info("Saved [{}] node networkx graph as YAML [{}]".format(len(graph.nodes),
                                                                                            filename))
            except Exception as e:
                self._trace.log().error("Failed to save networkx graph to file [{}] with error [{}]"
                                        .format(filename, str(e)))
        return

    def load_graph_from_yaml(self,
                             session_uuid: str = None,
                             dir_to_use: str = None) -> nx.DiGraph:
        """
        Load the graph in the yaml file from the given dir with the given session uuid
        :param session_uuid: The current session_uuid (to include in the filename)
        :param dir_to_use: The dir to save files in default is "." current working directory
        """
        filename = self.GRAPHS_FILE.format(self._default_if_none(dir_to_use, self._dir_to_use),
                                           self._default_if_none(session_uuid, self._session_uuid))
        res = None
        try:
            self._trace.log().debug("Start loading [{}]".format(filename))
            res = nx.read_yaml(filename)
            self._trace.log().debug("Finished loading [{}]".format(filename))
        except Exception as e:
            self._trace.log().error("Failed to load states to file [{}] with error [{}]"
                                    .format(filename, str(e)))
            res = None
        return res

    @staticmethod
    def _default_if_none(v,
                         default):
        """
        If v is None return default else return v
        :param v: The Value to inspect
        :param default: The default to return if v is None
        :return: v if not None else default
        """
        if v is None:
            return default
        return v
