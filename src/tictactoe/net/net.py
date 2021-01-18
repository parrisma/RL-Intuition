import abc


class NetI(metaclass=abc.ABCMeta):
    """
    Signature of all classes that can build Neural Nets
    """

    @abc.abstractmethod
    def build(self,
              *args,
              **kwargs) -> None:
        """
        Build a Neural Network based on the given arguments.
        :params args: The arguments to parse for net build parameters
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def compile(self,
                *args,
                **kwargs) -> None:
        """
        Compile the Neural Network based on the given arguments.
        :params args: The arguments to parse for net compile parameters
        """
        raise NotImplementedError()
