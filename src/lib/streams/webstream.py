"""
Render URL as stream
"""
import requests
import io


class WebStream:
    """
    Render a file at a given URI as a stream
    """
    url: str

    def __init__(self,
                 url: str):
        self._url = url
        return

    def __call__(self, *args, **kwargs):
        """
        Open the url set at __init__ and render as a stream
        :param args: not used
        :param kwargs: not used
        :return: a stream object for the given URI
        """
        try:
            url_stream = requests.get(self._url, stream=True)
            if url_stream.encoding is None:
                url_stream.encoding = 'utf-8'
            res_stream = io.BytesIO(url_stream.content)
            url_stream.close()
        except Exception as e:
            raise ValueError("Web Stream - unable read URL {} with error {}".format(self._url, str(e)))
        return res_stream
