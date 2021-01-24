from typing import List, Dict, Any
from json import JSONEncoder
from src.tictactoe.net.hyper_param import HyperParam
from src.tictactoe.util.vals_json import ValsJson


class HyperParams:
    """
    Container for all hyper parameters

    We do this as these 'magic' numbers often get lost in code as constants but are critical to the
    performance of the training and prediction. As such we group them here to highlight there significance.

    We also track changes to hyper parameters
    """

    class ParamJsonEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__

    _params: Dict[str, List[HyperParam]]

    def __init__(self):
        self._params = dict()

    def set(self,
            param_name: str,
            param_value: Any,
            param_comment: str = "") -> None:
        """
        Set the current value of the given Hyper-Parameter
        :param param_name: The name of the Hyper Parameter
        :param param_value: The value of the Hyper Parameter
        :param param_comment: An (optional) comment on the parameter use/value
        :return:
        """
        if param_name not in self._params:
            self._params[param_name] = list()
        self._params[param_name].append(HyperParam(param_name, param_value, param_comment))
        return

    def get(self,
            param_name) -> Any:
        """
        Get the current value of the given Hyper-Parameter
        :param param_name: Name of exiting (already set) hyper parameter to get
        :return: The value of the requested hyper parameter or None if not set
        """
        value = None
        if param_name in self._params:
            value = self._params[param_name][-1].value
        return value

    def save_to_json(self,
                     filename: str) -> None:
        """
        Dump the Hyper Parameters and fully history to JSON
        :param filename: The full path and filename to dump summary to as JSON
        """
        ValsJson.save_values_as_json(vals=self._params,
                                     filename=filename,
                                     encoder=self.ParamJsonEncoder)
        return
