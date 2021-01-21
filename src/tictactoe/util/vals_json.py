import json


class ValsJson:
    """
    Serialise and de-serialise values to/from JSON when passed as simple python data structures
    """

    @staticmethod
    def vals_to_json(vals) -> str:
        """
        Take the given Values and return as JSON String
        :param vals:
        :return:
        """
        return json.dumps(vals,
                          allow_nan=True)

    @staticmethod
    def json_to_vals(vals_as_json: str):
        """
        Convert the given Json string into values structure
        :param vals_as_json:
        :return: Q Values Dictionary
        """
        return json.loads(vals_as_json)

    @staticmethod
    def save_values_as_json(vals,
                            filename: str):
        """
        Convert the given Values to JSON and persist to the given file name.

        If the given file name exists - overwrite it

        :param vals: values structure to save
        :param filename: The full path and filename to save the Q Values as
        :return: None - but raise exception on error
        """
        fp = None
        try:
            fp = open(filename, "w")
            json.dump(obj=vals,
                      fp=fp,
                      allow_nan=True)
        except Exception as e:
            raise UserWarning("Failed to save values structure to file [{}] with error [{}]".format(
                filename,
                str(e)))
        finally:
            if fp is not None:
                fp.close()
        return

    @staticmethod
    def load_values_from_json(filename: str):

        """
        Load the JSON from the given filename and return as values structure

        If the given file name exists - overwrite it

        :param filename: The full path and filename to save the Q Values as
        :return: values structure or an raise exception on error
        """
        fp = None
        vals = None
        try:
            fp = open(filename, "r")
            vals = json.load(fp=fp)
        except Exception as e:
            raise UserWarning("Failed to load values structure from file [{}] with error [{}]".format(
                filename,
                str(e)))
        finally:
            if fp is not None:
                fp.close()
        return vals
