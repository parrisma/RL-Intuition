from typing import List
from datetime import datetime
from elasticsearch import Elasticsearch
from src.reflrn.interface.state import State
from src.lib.elastic.esutil import ESUtil


class TicTacToeEventStream:
    _es: Elasticsearch
    _session_uuid: str
    _jflds: List

    _jflds = ["timestamp",
              "session_uuid",
              "episode_uuid",
              "episode_step",
              "state",
              "action",
              "reward",
              "episode_end"]

    def __init__(self,
                 es: Elasticsearch,
                 es_index: str,
                 session_uuid: str):
        self._es = es
        if ESUtil.index_exists(es=es, idx_name=es_index):
            self._es_index = es_index
        else:
            raise RuntimeError("Cannot create TicTacToeEventStream as index {} does not exist".format(es_index))
        self._session_uuid = session_uuid
        self._fmt = '{{{{"{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}"}}}}'
        self._fmt = self._fmt.format(*self._jflds)
        self._date_formatter = ESUtil.DefaultElasticDateFormatter()
        return

    def record_event(self,
                     episode_uuid: str,
                     episode_step: int,
                     state: State,
                     action: str,
                     reward: float,
                     episode_end: bool) -> None:
        """
        Persist the given event in the ttt_event index. Assume the given elastic instance has the index already
        created.
        :param episode_uuid: the UUID of the episode
        :param episode_step: the step number of the episode
        :param state: the current state of the environment
        :param action: the action taken that gave rise to the given state
        :param reward: the reward given to the agent on entering this state
        :param episode_end: true if the given state is a terminal state for the episode
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
                                        ESUtil.bool_as_es_value(episode_end))
            ESUtil.write_doc_to_index(es=self._es,
                                      idx_name=self._es_index,
                                      document_as_json=json_msg)
        except Exception as e:
            raise RuntimeError(
                "Record event failed to index [{}] with exception [{}]".format(self._es_index, str(e)))

        return
