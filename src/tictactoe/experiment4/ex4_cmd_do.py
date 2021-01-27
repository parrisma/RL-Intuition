import sys
import os
from typing import Tuple, Dict
from src.tictactoe.experiment4.ex4_cmd import Ex4Cmd
from src.tictactoe.experiment.cmd_options import CmdOptions
from src.lib.rltrace.trace import Trace
from src.tictactoe.ttt.tictactoe import TicTacToe
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.net.neural_net import NeuralNet
from src.tictactoe.net.hello_world_net import HelloWorldNet
from src.tictactoe.net.q_net import QNet


class Ex4CmdDo(Ex4Cmd):
    """
    Actions for NN Agent training
    """
    _ttt: TicTacToe
    _trace: Trace
    _ttt_event_stream: TicTacToeEventStream
    _dir_to_use: str
    _last_net: str
    _idx: int
    _neural_net: NeuralNet
    _build_command_options: CmdOptions

    def __init__(self,
                 ttt: TicTacToe,
                 ttt_event_stream: TicTacToeEventStream,
                 trace: Trace,
                 dir_to_use: str):
        self._dir_to_use = os.path.abspath(dir_to_use)
        self._trace = trace
        self._ttt_event_stream = ttt_event_stream
        self._ttt = ttt
        self._last_net = None  # noqa
        self._idx = 0
        self._neural_net = None  # noqa
        self._build_command_options = CmdOptions(trace, self._dir_to_use)
        self._build_command_options.add_option(aliases=['H'],
                                               function=self._hello_world,
                                               description="Hello World Network")
        self._build_command_options.add_option(aliases=['Q'],
                                               function=self._q_val,
                                               description="Q Value Network")

        self._ls_command_options = CmdOptions(trace, self._dir_to_use)
        self._ls_command_options.add_option(aliases=['H'],
                                            function=self._ls_command_options.ls,
                                            description="Hello World XY Data",
                                            settings={CmdOptions.Settings.pattern: '*_XY.json'})
        self._ls_command_options.add_option(aliases=['Q'],
                                            function=self._ls_command_options.ls,
                                            description="Q Value XY Data",
                                            settings={CmdOptions.Settings.pattern: 'q_vals_*.json'})

        self._train_command_options = CmdOptions(trace, self._dir_to_use)
        self._train_command_options.add_option(aliases=['H'],
                                               function=self._train_hello_world,
                                               description="Load data & Train Hello World network",
                                               settings={CmdOptions.Settings.pattern: '*{}*_XY.json'})
        self._train_command_options.add_option(aliases=['Q'],
                                               function=self._train_q,
                                               description="Load data & Train Q Value network",
                                               settings={CmdOptions.Settings.pattern: 'q_vals_{}*.json'})

        self._predict_command_options = CmdOptions(trace, self._dir_to_use)
        self._predict_command_options.add_option(aliases=['H'],
                                                 function=self._predict_hello_world,
                                                 description="Make a prediction using the Hello World network",
                                                 settings={})
        return

    def do_build(self,
                 args) -> str:
        """
        Create a Neural Network of the requested type
        :param args: The parameters required to create / define the network
        """
        self._neural_net = self._build_command_options.do_option(args)  # noqa
        return self._prompt()

    def _hello_world(self,
                     _: Dict,
                     __: Tuple[str]) -> NeuralNet:
        """
        Create a HelloWorld Neural Network to cover the intuition of training
        :param __: Optional settings to pass to the NN
        :param _: Optional arguments to pass to the NN
        :return: A simple hello world NN
        """
        sn = HelloWorldNet(self._trace, self._dir_to_use)
        sn.build()
        self._trace.log().info("{}".format(sn))
        return sn

    def _q_val(self,
               _: Dict,
               __: Tuple[str]) -> NeuralNet:
        """
        Create a Q Value Neural Network to 'learn' TicTacToe state to Q Value prediction
        :param __: Optional settings to pass to the NN
        :param _: Optional arguments to pass to the NN
        :return: A simple hello world NN
        """
        sn = QNet(self._trace, self._dir_to_use)
        sn.build()
        self._trace.log().info("{}".format(sn))
        return sn

    def do_exit(self,
                args) -> None:
        """
        Terminate the session
        """
        sys.exit(0)

    def do_train(self,
                 args) -> str:
        """
        Train the neural network created by the net command
        :param args: The parameters required to Train the network
        """
        self._train_command_options.do_option(args, [])
        return self._prompt()

    def _train(self,
               net_type: type,
               settings: Dict,
               args: Tuple[str]) -> None:
        """
        Load teat/training data and train the neural network created by the build command
        :param args: The parameters required to Load and Train the network
        """
        if args is not None and len(args) > 0:
            data_file = str(settings[CmdOptions.Settings.pattern]).format(args[0])
            data_file = self._train_command_options.pattern_to_fully_qualified_filename(data_file)
            if data_file is not None:
                if self._neural_net is not None:
                    if isinstance(self._neural_net, net_type):
                        self._neural_net.load_and_train_from_json(data_file)  # noqa (confused by type check)
                    else:
                        self._trace.log().info(
                            "Current model is type [{}] not expected [{}]".format(type(self._neural_net).__name__,
                                                                                  net_type.__name__))
                else:
                    self._trace.log().info("Must create a model before training")
            else:
                self._trace.log().info("Failed to load data and train network")
        else:
            self._trace.log().info("missing X Y data file name required to load and train network")
        return

    def _train_hello_world(self,
                           settings: Dict,
                           args: Tuple[str]) -> None:
        """
        Load hello world data and train the neural network created by the net command
        :param args: The parameters required to Load and Train the network
        """
        self._train(HelloWorldNet, settings, args)
        return

    def _train_q(self,
                 settings: Dict,
                 args: Tuple[str]) -> None:
        """
        Load hello Q Value data and train the neural network created by the net command
        :param args: The parameters required to Load and Train the network
        """
        self._train(QNet, settings, args)
        return

    def do_predict(self,
                   args) -> str:
        """
        Make a prediction using the built and trained hello world Neural Network
        :param args: The parameters required to Test the network
        """
        self._predict_command_options.do_option(args)
        return self._prompt()

    def _predict_hello_world(self,
                             _: Dict,
                             args: Tuple[str]) -> None:
        """
        Make a prediction based on a built and trained HelloWorld NN
        :param _: Settings Optional settings to specialise the prediction
        :param args: The parameters required to Load and Train the network
        """
        if args is not None and len(args) > 0:
            x_value = args[0]
            try:
                x_value = float(x_value)  # will throw if str not float
                if self._neural_net is not None:
                    if isinstance(self._neural_net, HelloWorldNet):
                        y_actual, y_expected = self._neural_net.predict(float(x_value))
                        self._trace.log().info(
                            "For [{:10.6f}] y predicted = [{:10.6f}] y expected = [{:10.6f}]".format(x_value,
                                                                                                     y_actual,
                                                                                                     y_expected))
                        pass
                    else:
                        self._trace.log().info(
                            "Current model is type [{}] not expected [{}]".format(type(self._neural_net).__name__,
                                                                                  HelloWorldNet.__name__))
                else:
                    self._trace.log().info("Must create a model before training")
            except Exception as e:
                self._trace.log().info("Failed to predict for value [{:12.6f}]".format(x_value))
        else:
            self._trace.log().info("missing X value to predict")
        return

    def do_list(self,
                args) -> str:
        """
        List local files that match the requested type
        :param args: Optional arguments to narrow the list
        :return: updated prompt
        """
        matching_files = self._ls_command_options.do_option(args, [])
        for file_name in matching_files:  # noqa
            self._trace.log().info(file_name)
        return self._prompt()

    def _prompt(self) -> str:
        """
        The navigation prompt
        :return: The dynamic net prompt
        """
        prompt = '(net)'
        if self._neural_net is not None:
            prompt = "({}:{})".format(self._neural_net.build_context_name(), self._neural_net.network_architecture())
        return prompt
