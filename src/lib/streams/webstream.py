"""
Render URL as stream
"""
import requests
import io


class WebStream:
    def __init__(self,
                 url: str):
        self._url = url
        return

    def __call__(self, *args, **kwargs):
        try:
            url_stream = requests.get(self._url, stream=True)
            if url_stream.encoding is None:
                url_stream.encoding = 'utf-8'
            res_stream = io.BytesIO(url_stream.content)
            url_stream.close()
        except Exception as e:
            raise ValueError("Web Stream - unable read URL {} with error {}".format(self._url, str(e)))
        return res_stream
