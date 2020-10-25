import re
from typing import Callable


class Transformer:
    class Transform:
        def __init__(self,
                     regular_expression: str,
                     transform: Callable[[str], str]):
            if regular_expression is None or len(regular_expression) == 0:
                raise ValueError("Transform regular expression must be non empty string")
            if transform is None or not callable(transform):
                raise ValueError("Transform transformation must be callable")
            self._pattern = re.compile(regular_expression)
            self._transform = transform
            return

        def __call__(self, *args, **kwargs) -> str:
            transformed = None
            string_to_transform = kwargs.get('string_to_transform', None)
            if string_to_transform is not None:
                if self._pattern.search(string_to_transform):
                    transformed = self._transform(string_to_transform)
                else:
                    transformed = string_to_transform
            return transformed

    def __init__(self):
        self._transforms = list()
        return

    def add_transform(self,
                      transform: Transform) -> None:
        """
        Add the given transform to the list of transforms for this Transformer
        :param transform: The transform to add
        """
        if transform is None:
            raise ValueError("Transformer cannot add None transform")
        self._transforms.append(transform)
        return

    def transform(self,
                  string_to_transform: str) -> str:
        """
        Apply all registered transforms to the given string
        :param string_to_transform: The string the transform
        :return: the transformed string
        """
        transformed = string_to_transform
        if transformed is not None and len(transformed) > 0:
            for transform in self._transforms:
                transformed = transform(string_to_transform=transformed)
        return transformed
