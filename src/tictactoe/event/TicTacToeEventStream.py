from typing import List
from datetime import datetime
from elasticsearch import Elasticsearch
from src.reflrn.interface.state import State
from src.reflrn.interface.state_factory import StateFactory
from src.lib.elastic.esutil import ESUtil
from src.tictactoe.event.tictacttoe_event import TicTacToeEvent


class TicTacToeEventStream:
    # --- TicTacToeEventStream --
    _es: Elasticsearch
    _session_uuid: str
    _state_factory: StateFactory
    _jflds: List

    _jflds = ["timestamp",
              "session_uuid",
              "episode_uuid",
              "episode_step",
              "state",
              "action",
              "reward",
              "episode_end",
              "episode_outcome"]

    SESSION_UUID_Q = '{"query":{"term":{ "session_uuid":"<arg0>"}}}'
    EPISODE_UUID_Q = '{"query":{"term":{ "episode_uuid":"<arg0>"}}}'

    def __init__(self,
                 es: Elasticsearch,
                 es_index: str,
                 state_factory: StateFactory,
                 session_uuid: str):
        self._es = es
        if ESUtil.index_exists(es=es, idx_name=es_index):
            self._es_index = es_index
        else:
            raise RuntimeError("Cannot create TicTacToeEventStream as index {} does not exist".format(es_index))
        self._state_factory = state_factory
        self._session_uuid = session_uuid
        self._fmt = '{{{{"{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}"}}}}'
        self._fmt = self._fmt.format(*self._jflds)
        self._date_formatter = ESUtil.DefaultElasticDateFormatter()
        return

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
                                                   arg0=session_uuid)
            res = list()
            for jer in ttt_events_as_json:
                event_as_json = jer['_source']
                st = self._state_factory.new_state(event_as_json['state'])
                ttt_e = TicTacToeEvent(episode_uuid=event_as_json['episode_uuid'],
                                       episode_step=int(event_as_json['episode_step']),
                                       state=st,
                                       action=event_as_json['action'],
                                       reward=float(event_as_json['reward']),
                                       episode_end=bool('true' == event_as_json['episode_end']),
                                       episode_outcome=event_as_json['episode_outcome']
                                       )
                res.append(ttt_e)
        except Exception as e:
            raise RuntimeError(
                "Get session failed with exception [{}]".format(str(e)))

        # Always return sorted in uuid & step order.
        res = sorted(res, key=lambda x: "{}{}".format(x.episode_uuid, x.episode_step))
        return res
