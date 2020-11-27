"""
Render file as stream
"""


class FileStream:
    """
    Render a file on local disk as a stream
    """
    filename: str

    def __init__(self,
                 filename: str):
        self._filename = filename
        return

    def __call__(self, *args, **kwargs):
        """
        Open the filename set at __init__ and render as a stream
        :param args: not used
        :param kwargs: not used
        :return: a stream object for the given file
        """
        try:
            file_stream = open(self._filename, 'r')
        except Exception as e:
            raise ValueError("File Stream - unable to open {} with error {}".format(self._filename, str(e)))
        return file_stream
