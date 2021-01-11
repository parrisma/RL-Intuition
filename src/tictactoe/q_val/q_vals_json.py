from typing import Dict, List
import json


class QValsJson:
    """
    Serialise and de-serialise calculated q values to/from JSON

    Dict[Player(X|O), Dict[AgentPerspective(X|O), Dict[StateAsStr('000000000'), List[of nine q values as float]]]]

    """

    @staticmethod
    def q_vals_to_json(q_vals: Dict[str, Dict[str, Dict[str, List[float]]]]) -> str:
        """
        Take the given Q Values and return as JSON String
        :param q_vals:
        :return:
        """
        return json.dumps(q_vals,
                          allow_nan=True)

    @staticmethod
    def json_to_q_vals(q_vals_as_json: str) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """
        Convert the given Json string into Q Values
        :param q_vals_as_json:
        :return: Q Values Dictionary
        """
        return json.loads(q_vals_as_json)

    @staticmethod
    def save_q_values_as_json(q_vals: Dict[str, Dict[str, Dict[str, List[float]]]],
                              filename: str):
        """
        Convert the given Q Values to JSON and persist to the given file name.

        If the given file name exists - overwrite it

        :param q_vals: Q Values to save
        :param filename: The full path and filename to save the Q Values as
        :return: None - but raise exception on error
        """
        fp = None
        try:
            fp = open(filename, "w")
            json.dump(obj=q_vals,
                      fp=fp,
                      allow_nan=True)
        except Exception as e:
            raise UserWarning("Failed to save Q Values to file [{}] with error [{}]".format(
                filename,
                str(e)))
        finally:
            if fp is not None:
                fp.close()
        return

    @staticmethod
    def load_q_values_from_json(filename: str) -> Dict[str, Dict[str, Dict[str, List[float]]]]:

        """
        Load the JSON from the given filename and return as Q Values

        If the given file name exists - overwrite it

        :param filename: The full path and filename to save the Q Values as
        :return: Q Values or an raise exception on error
        """
        fp = None
        q_vals = None
        try:
            fp = open(filename, "r")
            q_vals = json.load(fp=fp,
                               allow_nan=True)
        except Exception as e:
            raise UserWarning("Failed to load Q Values from file [{}] with error [{}]".format(
                filename,
                str(e)))
        finally:
            if fp is not None:
                fp.close()
        return q_vals
