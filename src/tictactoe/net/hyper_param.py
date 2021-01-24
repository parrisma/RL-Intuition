from typing import Any


class HyperParam:
    """
    A single Hyper Parameter
    """
    _name: str
    _value: Any
    _comment: str

    def __init__(self,
                 name: str,
                 value: Any,
                 comment: str):
        self._name = name
        self._value = value
        self._comment = comment
        return

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Any:
        return self._value

    @property
    def comment(self) -> str:
        return self._comment
