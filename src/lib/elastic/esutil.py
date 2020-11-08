from abc import ABC, abstractmethod
from typing import Dict, Callable, IO, List
from datetime import datetime
import pytz
import re
from elasticsearch import Elasticsearch


class ESUtil:
    # Annotations
    _es: Dict[str, Elasticsearch]

    _es = dict()

    _ALL = 10000
    _SCROLL = '1m'
    _COUNT = 0
    MATCH_ALL = '{"query":{"match_all": {}}}}'

    class ElasticDateFormatter(ABC):

        @abstractmethod
        def format(self, dtm) -> str:
            pass

    class DefaultElasticDateFormatter(ElasticDateFormatter):

        def format(self, dtm) -> str:
            if isinstance(dtm, float):
                dtm = datetime.fromtimestamp(dtm)
            elif not isinstance(dtm, datetime):
                raise ValueError(
                    "Log created date must be supplied as float (timestamp) or datetime not {}".format(str(type(dtm))))
            return self._elastic_time_format(pytz.utc.localize(dtm))

        @staticmethod
        def _elastic_time_format(dt: datetime) -> str:
            return dt.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    @classmethod
    def get_connection(cls,
                       hostname: str,
                       port_id: str) -> Elasticsearch:
        """
        Get the default Elastic host and port from the environment elastic settings Yaml and make an
        Elasticsearch connection to that return host and port
        :return: The Elasticsearch connection object for default host and port
        """
        connection_str = "{}:{}".format(hostname, port_id)
        if cls._es.get(connection_str, None) is None:
            try:
                cls._es[connection_str] = Elasticsearch([connection_str])
            except Exception as e:
                raise RuntimeError("Failed to open Elastic search connection {}:{}".format(hostname, port_id))
        return cls._es[connection_str]

    @staticmethod
    def load_json(json_stream: Callable[[], IO[str]]) -> str:
        """
        Read the given JSON stream and return contents as string
        :param json_stream: A callable that returns an open stream to the YAML source
        :return: The contents of the Json file as string.
        """
        res = str()
        try:
            _stream = json_stream()
            for _line in _stream:
                res += _line.decode("utf-8")
        except Exception as e:
            raise RuntimeError("Unable to read json from given stream")
        return res

    @staticmethod
    def json_insert_args(json_source: str,
                         **kwargs) -> str:
        """
        Replace all parameters in kwargs with name that matches arg<999> with the value of the parameter
        where the marker in the source json is of the form <0> = arg0, <1> = arg1 etc
        :param json_source: The Json source to do parameter insertion on
        :param kwargs: arbitrary arguments to insert that match pattern arg0, arg1, arg2, ...
        :return: Source json with the arguments substituted
        """
        arg_pattern = re.compile("^arg[0-9]+$")
        for k, v in kwargs.items():
            if arg_pattern.search(k):
                repl_re = "(<{}>)".format(k)
                json_source = re.sub(repl_re, v, json_source)
        return json_source

    @staticmethod
    def datetime_in_elastic_time_format(dt: datetime) -> str:
        """
        Return a datetime in the format to be written to elastic as timezone aware
        :param dt:
        :return:
        """
        return pytz.utc.localize(dt).strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    @staticmethod
    def create_index_from_json(es: Elasticsearch,
                               idx_name: str,
                               json_stream: Callable[[], IO[str]]) -> bool:
        """

        :param es: An open elastic search connection
        :param idx_name: The name of the index to create
        :param json_stream: A callable that returns an open stream to the YAML source
        :return: True if created or if index already exists
        """
        try:
            body = ESUtil.load_json(json_stream)

            # Exception will indicate index create error.
            _ = es.indices.create(index=idx_name,
                                  body=body,
                                  wait_for_active_shards=1,
                                  ignore=[400, 404])
        except Exception as e:
            raise RuntimeError(
                "Failed to create elastic index [{}] from Json stream".format(idx_name))
        return True

    @staticmethod
    def delete_index(es: Elasticsearch,
                     idx_name: str) -> bool:
        """

        :param es: An open elastic search connection
        :param idx_name: The name of the index to delete
        :return: True if deleted or is not there
        """
        try:
            # Exception will indicate index create error.
            _ = es.indices.delete(index=idx_name,
                                  ignore=[400, 404])
        except Exception as e:
            raise RuntimeError(
                "Failed to delete elastic index [{}]`".format(idx_name))
        return True

    @staticmethod
    def delete_documents(es: Elasticsearch,
                         idx_name: str,
                         json_query: str,
                         **kwargs) -> None:
        """
        Delete all documents on the givenindex that match the parameterised query
        :param es: An open elastic search connection
        :param idx_name: The name of the index to execute delete on
        :param json_query: The Json query to delete by
        :param kwargs: arguments to the json_query of the form arg0='value 0', arg1='value 1' .. argn='value n'
                       where the argument values will be substituted into the json query before it is executed.
                       The raw query { x: { y: <arg0> } } will have <arg0> fully replaced with the corresponding
                       kwargs value supplied for all arg0..argn. Where there are multiple occurrences of any <argn>
                       all occurrences will be replaced.
        """
        try:
            json_query_to_delete = ESUtil.json_insert_args(json_source=json_query, **kwargs)
            # Exception will indicate delete error.
            es.delete_by_query(index=idx_name,
                               body=json_query_to_delete)
        except Exception as e:
            raise RuntimeError(
                "Failed to execute delete by query [{}] on Index [{}]".format(json_query, idx_name))
        return

    @staticmethod
    def run_search(es: Elasticsearch,
                   idx_name: str,
                   json_query: str,
                   **kwargs) -> List[Dict]:
        """

        :param es: An open elastic search connection
        :param idx_name: The name of the index to execute search on
        :param json_query: The Json query to run
        :param kwargs: arguments to the json_query of the form arg0='value 0', arg1='value 1' .. argn='value n'
                       where the argument values will be substituted into the json query before it is executed.
                       The raw query { x: { y: <arg0> } } will have <arg0> fully replaced with the corresponding
                       kwargs value supplied for all arg0..argn. Where there are multiple occurrences of any <argn>
                       all occurrences will be replaced.
        :return: A list of the resulting documents
        """
        try:
            json_query_to_execute = ESUtil.json_insert_args(json_source=json_query, **kwargs)
            # Exception will indicate search error.
            all = list()
            res = es.search(index=idx_name,
                            body=json_query_to_execute,
                            scroll=ESUtil._SCROLL,
                            size=ESUtil._ALL)
            scroll_id = res['_scroll_id']
            scroll_size = len(res['hits']['hits'])
            all.extend(res['hits']['hits'])
            while scroll_size > 0:
                res = es.scroll(scroll_id=scroll_id, scroll=ESUtil._SCROLL)
                scroll_id = res['_scroll_id']
                scroll_size = len(res['hits']['hits'])
                all.extend(res['hits']['hits'])

        except Exception as e:
            raise RuntimeError(
                "Failed to execute query [{}] on Index [{}]".format(json_query, idx_name))
        return all

    @staticmethod
    def run_count(es: Elasticsearch,
                  idx_name: str,
                  json_query: str,
                  **kwargs) -> int:
        """
        Get the number of records that match the query on the given index
        :param es: An open elastic search connection
        :param idx_name: The name of the index to execute count on
        :param json_query: The Json query to run
        :param kwargs: arguments to the json_query of the form arg0='value 0', arg1='value 1' .. argn='value n'
                       where the argument values will be substituted into the json query before it is executed.
                       The raw query { x: { y: <arg0> } } will have <arg0> fully replaced with the corresponding
                       kwargs value supplied for all arg0..argn. Where there are multiple occurrences of any <argn>
                       all occurrences will be replaced.
        :return: The number of matching documents
        """
        try:
            json_query_to_execute = ESUtil.json_insert_args(json_source=json_query, **kwargs)
            # Exception will indicate search error.
            res = es.count(index=idx_name,
                           body=json_query_to_execute)
        except Exception as e:
            raise RuntimeError(
                "Failed to execute query [{}] on Index [{}]".format(json_query, idx_name))
        return int(res['count'])

    @staticmethod
    def write_doc_to_index(es: Elasticsearch,
                           idx_name: str,
                           document_as_json: str) -> None:
        """
        Apply the associated formatter to the given LogRecord and persist it to Elastic
        :param es: An open elastic search connection
        :param idx_name: The name of the index to add the document to
        :param document_as_json: The document (as json) to add to the index
        """
        try:
            res = es.index(index=idx_name,
                           body=document_as_json)
            if res.get('result', None) != 'created':
                raise RuntimeError(
                    "Index [{}] bad elastic return status when adding document[{}]".format(idx_name, str(res)))
        except Exception as e:
            raise RuntimeError("Elastic failed to document to index [{}] with exception [{}]".format(idx_name, str(e)))
        return

    @staticmethod
    def index_exists(es: Elasticsearch,
                     idx_name: str) -> bool:
        """
        Return true if teh given index exists
        :param es: An open elastic search connection
        :param idx_name: The name of the index to add the document to
        :return True if given index exists in the given Elasticsearch instance
        """
        try:
            res = es.indices.exists(index=idx_name)
        except Exception as e:
            raise RuntimeError("Elastic failed to document to index [{}] with exception [{}]".format(idx_name, str(e)))
        return res

    @staticmethod
    def bool_as_es_value(b: bool) -> str:
        return str(b).lower()
