"""
Render file as stream
"""


class FileStream:
    def __init__(self,
                 filename: str):
        self._filename = filename
        return

    def __call__(self, *args, **kwargs):
        try:
            file_stream = open(self._filename, 'r')
        except Exception as e:
            raise ValueError("File Stream - unable to open {} with error {}".format(self._filename, str(e)))
        return file_stream
