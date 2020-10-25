from datetime import datetime
from src.lib.elastic.esutil import ESUtil


class NotificationFormatter:
    _jflds = ["notification_type_uuid",
              "work_ref_uuid",
              "session_uuid",
              "sink_uuid",
              "src_uuid",
              "timestamp"]

    def __init__(self):
        """
        Formatter for a notification log message
        """
        self._fmt = '{{{{"{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}","{}":"{{}}"}}}}'
        self._fmt = self._fmt.format(*self._jflds)
        self._date_formatter = ESUtil.DefaultElasticDateFormatter()
        return

    def format(self,
               notification_type_uuid: str,
               work_ref_uuid: str,
               session_uuid: str,
               sink_uuid: str,
               src_uuid: str,
               timestamp: datetime = None) -> str:
        """
        Format an Elastic Json document to make a notification_log entry
        :param notification_type_uuid: The UUID of the Notification Message Type
        :param work_ref_uuid: The payload UUID or WorkReference UUID
        :param session_uuid: The session UUID
        :param sink_uuid: The Sink (receiver) UUID
        :param src_uuid: The Src (sender) UUID
        :param timestamp: The timestamp of the event (time zone aware)
        :return: The notification log entry as JSON Document string
        """
        if timestamp is None:
            now_timestamp = self._date_formatter.format(datetime.now())
        else:
            now_timestamp = self._date_formatter.format(timestamp)

        json_msg = self._fmt.format(notification_type_uuid,
                                    work_ref_uuid,
                                    session_uuid,
                                    sink_uuid,
                                    src_uuid,
                                    now_timestamp)
        return json_msg
