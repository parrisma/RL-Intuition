from elasticsearch import Elasticsearch
from src.lib.elastic.esutil import ESUtil
from src.lib.rltrace.notificationformatter import NotificationFormatter


class TraceReport:
    # Annotation
    _es: Elasticsearch
    _nf: NotificationFormatter

    def __init__(self,
                 es: Elasticsearch,
                 trace_log_index_name: str,
                 notification_log_index_name: str):
        self._es = es
        self._nf = NotificationFormatter()
        self._trace_log_index_name = trace_log_index_name
        self._notification_log_index_name = notification_log_index_name
        return

    def _index_count(self,
                     index_name: str,
                     field_name: str = "session_uuid",
                     pattern: str = None) -> int:
        """
        Count the number of records in the given index that match the supplied pattern
        :param index_name: The name of the index in which to count documents
        :param field_name: The field name to pattern search on - must be type text
        :param pattern: The Elastic regular expression pattern - default is * => Match all
        """
        if pattern is None:
            pattern = "*"
        res = ESUtil.run_count(es=self._es,
                               idx_name=index_name,
                               json_query=self.search_wildcard,
                               arg0=field_name,
                               arg1=pattern)
        return res

    def trace_log_count(self,
                        field_name: str,
                        session_uuid_pattern: str) -> int:
        """
        Return the number of documents from trace_log where the given field (type text) matches the given pattern
        :param field_name: A field in the trace_log document (text type) to match on
        :param session_uuid_pattern: The pattern to match for
        :return: The number of matching documents.
        """
        return self._index_count(index_name=self._trace_log_index_name,
                                 field_name=field_name,
                                 pattern=session_uuid_pattern)

    def log_notification(self,
                         notification_type_uuid: str,
                         work_ref_uuid: str,
                         session_uuid: str,
                         sink_uuid: str,
                         src_uuid: str) -> None:
        """
        Create an entry for a notification (src to sink message) Event
        :param notification_type_uuid: The Notification (message) type
        :param work_ref_uuid: The work reference that was the subject of the notification
        :param session_uuid: The session id
        :param sink_uuid: The sink (rx'er) UUID
        :param src_uuid: The source (sender) UUID
        """
        notification_as_json = self._nf.format(notification_type_uuid=notification_type_uuid,
                                               work_ref_uuid=work_ref_uuid,
                                               session_uuid=session_uuid,
                                               sink_uuid=sink_uuid,
                                               src_uuid=src_uuid)
        ESUtil.write_doc_to_index(es=self._es,
                                  idx_name=self._notification_log_index_name,
                                  document_as_json=notification_as_json)
        pass

    def notification_log_count(self,
                               field_name: str,
                               session_uuid_pattern: str) -> int:
        """
        Return the number of documents from notification_log where the given field (type text) matches the given pattern
        :param field_name: A field in the notification_log document (text type) to match on
        :param session_uuid_pattern: The pattern to match for
        :return: The number of matching documents.
        """
        return self._index_count(index_name=self._notification_log_index_name,
                                 field_name=field_name,
                                 pattern=session_uuid_pattern)

    @property
    def search_wildcard(self) -> str:
        return """{
                    "query": {
                        "wildcard": {
                            "<arg0>": "<arg1>"
                        }
                    }
                }"""
