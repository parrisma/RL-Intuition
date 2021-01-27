from typing import List, Dict, Callable, Tuple
from os import path, mkdir
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from src.lib.namegen.namegen import NameGen
from src.lib.rltrace.trace import Trace
from src.tictactoe.net.neural_net import NeuralNet
from src.tictactoe.util.vals_json import ValsJson
from src.tictactoe.net.hyper_params import HyperParams
from src.tictactoe.net.names import Names
from src.tictactoe.net.lr_decay import LRDecay


class QNet(NeuralNet):
    """
    Build a fully connected NN for prediction of Q Values given a 9 feature TicTacToe state
    """
    _model: tf.keras.models.Sequential
    _summary: Dict[str, List[float]]
    _train_context_name: str
    _trace: Trace
    _base_dir_to_use: str
    _dir_to_use: str
    _actual_func: Callable[[float], float]
    _hyper_params: HyperParams
    #  Dict[Player(X|O), Dict[AgentPerspective(X|O), Dict[StateAsStr('000000000'), List[of nine q values as float]]]]
    _q_values: Dict[str, Dict[str, Dict[str, List[float]]]]

    class TestCallback(tf.keras.callbacks.Callback):
        """
        Call back to dump interim test details while training
        """

        def __init__(self,
                     trace: Trace,
                     x_test: np.ndarray,
                     y_test: np.ndarray,
                     num_epoch: int,
                     summary: Dict[str, List[float]],
                     summary_file_root: str):
            super().__init__()
            self._trace = trace
            self._x_test = x_test
            self._y_test = y_test
            self._interim_steps = NeuralNet.exp_steps(num_epoch)
            self._summary = summary
            self.summary_file_root = summary_file_root
            return

        def on_epoch_end(self, epoch, logs=None):
            """
            Run and save predictions if epoch in the list of interim epochs to report on
            :param epoch: The current epoch
            :param logs: n/a
            """
            if epoch in self._interim_steps:
                self._trace.log().info("Learning rate {:12.11f}".format(tf.keras.backend.eval(self.model.optimizer.lr)))
                self._trace.log().info("Run interim prediction at epoch {}".format(epoch))
                predictions = self.model.predict(self._x_test)
                predictions = predictions.reshape(len(predictions))
                self._summary["{}_{:04d}".format(self.summary_file_root, epoch)] = predictions.tolist()
                return

    def __init__(self,
                 trace: Trace,
                 dir_to_use: str):
        self._trace = trace
        self._base_dir_to_use = dir_to_use
        self._dir_to_use = None  # noqa
        self._train_context_name = None  # noqa
        self._hyper_params = HyperParams()
        self._set_hyper_params()
        self._model = None  # noqa
        self._summary = None  # noqa
        self._q_values = None  # noqa
        self._reset()
        return

    def _new_context(self) -> Tuple[str, str]:
        """
        Create a new context name and save area for mode build/run/test
        :return: the name of the new context and its directory 'to use'
        """
        context_name = NameGen.generate_random_name()
        context_dir = "{}//{}".format(self._base_dir_to_use, context_name)
        while path.exists(context_dir):
            context_name = NameGen.generate_random_name()
            context_dir = "{}//{}".format(self._base_dir_to_use, context_name)
        mkdir(context_dir)
        return context_dir, context_name

    def _reset(self) -> None:
        """
        Clear model status after rebuild
        """
        self._model = None  # noqa
        self._summary = dict()
        self._actual_func = np.sin
        self._dir_to_use, self._train_context_name = self._new_context()
        return

    def _set_hyper_params(self) -> None:
        """
        Set the Hyper Parameters for the Hello World Net
        """
        self._hyper_params.set(Names.learning_rate, 0.001, "Adam Optimizer learning rate")
        self._hyper_params.set(Names.num_epoch, 1000, "Number of training epochs")
        self._hyper_params.set(Names.batch_size, 32, "Number of samples per training batch")
        return

    def build_context_name(self,
                           *args,
                           **kwargs) -> str:
        """
        A unique name given to the network each time a build net is executed
        :params args: The unique name of the build context
        """
        if self._train_context_name is None:
            self._new_context()
        return self._train_context_name

    def network_architecture(self,
                             *args,
                             **kwargs) -> str:
        """
        The architecture of the Neural Network
        :params args: The architecture name
        """
        return self.__class__.__name__

    def build(self,
              *args,
              **kwargs) -> None:
        """
        Build a Neural Network based on the given arguments.
        :params args: The arguments to parse for net build parameters
        """
        self._reset()

        self._model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(input_shape=(9,), units=1, name='input'),
                tf.keras.layers.Dense(100, activation=tf.nn.relu, name='dense1'),
                tf.keras.layers.Dropout(.1, name='dropout-1-10pct'),
                tf.keras.layers.Dense(500, activation=tf.nn.relu, name='dense2'),
                tf.keras.layers.Dropout(.1, name='dropout-2-10pct'),
                tf.keras.layers.Dense(1500, activation=tf.nn.relu, name='dense3'),
                tf.keras.layers.Dropout(.1, name='dropout-3-10pct'),
                tf.keras.layers.Dense(300, activation=tf.nn.relu, name='dense4'),
                tf.keras.layers.Dropout(.1, name='dropout-4-10pct'),
                tf.keras.layers.Dense(50, activation=tf.nn.relu, name='dense5'),
                tf.keras.layers.Dense(9, name='output')
            ]
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self._hyper_params.get(Names.learning_rate))
        self._model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.mean_squared_error)

        self._hyper_params.save_to_json(self.hyper_params_file)
        # ToDo save model weights to allow reset on new training

        return

    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_test: np.ndarray,
              y_test: np.ndarray,
              *args,
              **kwargs) -> None:
        """
        :param x_train: The X Training data : shape (n,1)
        :param y_train: The Y Training data : shape (n,1)
        :param x_test: The X Test data : shape (n,1)
        :param y_test: The Y Test (actual) data : shape (n,1)
        :params args: The arguments to parse for net compile parameters
        :param args:
        :param kwargs:
        :return:
        """
        if self._model is None:
            self._trace.log().info("Build model before training")

        self._trace.log().info("Start training: context [{}]".format(self._train_context_name))
        num_epochs = self._hyper_params.get(Names.num_epoch)
        history = self._model.fit(x_train, y_train,
                                  epochs=num_epochs,
                                  batch_size=self._hyper_params.get(Names.batch_size),
                                  verbose=2,
                                  callbacks=[self.TestCallback(trace=self._trace,
                                                               x_test=x_test,
                                                               y_test=y_test,
                                                               num_epoch=num_epochs,
                                                               summary=self._summary,
                                                               summary_file_root=Names.predictions_interim),
                                             tf.keras.callbacks.LearningRateScheduler(
                                                 LRDecay(num_epochs).lr_10step_decay)
                                             ]
                                  )
        self._test(x_test, y_test)
        self._summary[Names.loss] = history.history['loss']
        self._summary[Names.x_train] = x_train.tolist()
        self._summary[Names.y_train] = y_train.tolist()
        self._trace.log().info("End training: context [{}]".format(self._train_context_name))
        return

    def _test(self,
              x_test: np.ndarray,
              y_test: np.ndarray,
              *args,
              **kwargs) -> float:
        """
        :param x_test: The X Test data : shape (n,1)
        :param y_test: The Y Test (actual) data : shape (n,1)
        :param args: The arguments to parse for net compile parameters
        :param kwargs:
        :return: overall mean squared error between test and predictions
        """
        if self._model is None:
            self._trace.log().info("Build model before training")

        predictions = self._model.predict(x_test)
        predictions = predictions.reshape(len(predictions))
        self._summary[Names.x_test] = x_test.tolist()
        self._summary[Names.y_test] = y_test.tolist()
        self._summary[Names.predictions] = predictions.tolist()
        mse = (np.square(y_test - predictions)).mean(axis=0)
        self._trace.log().info("Mean Squared Error on test set [{}]".format(mse))

        return mse

    def _dump_summary_to_json(self,
                              filename: str) -> None:
        """
        Dump the summary of model build and test to given file
        :param filename: The full path and filename to dump summary to as JSON
        """
        ValsJson.save_values_as_json(vals=self._summary,
                                     filename=filename)
        return

    def _load_from_json(self,
                        filename: str,
                        test_split: float = 0.2,
                        shuffle: bool = True) -> List[np.ndarray]:
        """
        Load XY training data from given JSON file
        :param filename: The JSON file name with the training data in
        :param test_split: The % (0.0 tp 1.0) of the training data to use as test data, default = 20% (0.2)
        :param shuffle: If True shuffle the data before splitting, default = True
        :return: x_train, y_train, x_test, y_test

        Q Value data is supplied for each of players X and O, and then for each player the perspective of the
        board state and Q Values for both X and O.

        So this load function creates an X feature vector of the form
            <Player X|O><Perspective X|O><Board State>
        By default X is represented by -1
                   O is represented by 1
        This the full feature X feature vector is 11 digits

        The Y Value is the nine Q Values for each of the actions possible in a given state so the Y value is
        nine digits. NaN in the input means the Q Value training saw no data for that action. So we replace NaN
        with zero. This NaN substitution presents a risk as we could break the greedy evaluation of the state
        as zero may suddenly look like the best action if the 'real' Q Values for given state are all negative.
        This cannot be avoided and the upshot is that a Q Value data set with NaN is weak data and needs more games
        and exploration to plug these gaps with actual Q Value update events.
        """
        if test_split < 0.1 or test_split > 1.0:
            test_split = 1.0
        x = list()
        y = list()
        res = list()
        data_dict = ValsJson().load_values_from_json(filename)
        if data_dict is not None:
            if type(data_dict):
                for player, plid in [['X', -1], ['O', 1]]:
                    if player in data_dict:
                        for perspective, psid in [['X', -1], ['O', 1]]:
                            if perspective in data_dict[player]:
                                for state, qvals in data_dict[player][perspective].items():
                                    x_as_feature_str = "{}{}{}".format(plid, psid, state)
                                    x_as_feature_vec = np.zeros((11))
                                    i = 0
                                    j = 0
                                    while j < len(x_as_feature_str):
                                        if x_as_feature_str[j] == '-':
                                            x_as_feature_vec[i] = -1
                                            j += 2
                                        elif x_as_feature_str[j] == '1' or x_as_feature_str[j] == '0':
                                            x_as_feature_vec[i] = int(x_as_feature_str[j])
                                            j += 1
                                        else:
                                            raise ValueError(
                                                "Bad value in Q Val state got [{}], expected only 0, 1 or -1".format(
                                                    x_as_feature_str[j]))
                                        i += 1
                                    y_val = np.nan_to_num(qvals, nan=-np.inf)
                                    x.append(x_as_feature_vec)
                                    y.append(y_val)
                x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                    test_size=test_split,
                                                                    random_state=42,
                                                                    shuffle=shuffle)
                res = [x_train, x_test, y_train, y_test]
            else:
                self._trace.log().info("No X and Y data found in [{}]".format(filename))
        else:
            self._trace.log().info("Failed to load HelloWorld training data from [{}]".format(filename))
        return res

    def load_and_train_from_json(self,
                                 filename: str) -> None:
        """
        Load the X,Y data from the given json file and train the modle
        :param filename: The file containing the X, Y data
        """
        if self._model is not None:
            try:
                x_train, x_test, y_train, y_test = self._load_from_json(filename)
                self._trace.log().info("Training Starts [{}]".format(self._train_context_name))
                self.train(x_train, y_train, x_test, y_test)
                self._trace.log().info("Training Ends [{}]".format(self._train_context_name))
                self._trace.log().info("Testing Starts [{}]".format(self._train_context_name))
                self._test(x_test, y_test)
                self._trace.log().info("Testing Ends [{}]".format(self._train_context_name))
                self._dump_summary_to_json(self.summary_file)
                self._trace.log().info("Saved Train & Test results as json [{}]".format(self.summary_file))
            except Exception as e:
                self._trace.log().info("Failed to load data to train model [{}]".format(str(e)))
        else:
            self._trace.log().info("Model not built, cannot load and train")
        return

    def predict(self,
                x_value: np.ndarray,
                *args,
                **kwargs) -> Tuple[np.float, np.float]:
        """
        :param x_value: The X feature vector to predict
        :params args: The arguments to parse for net compile parameters
        :param args:
        :param kwargs:
        :return: predicted Y and expected Y (based on actual function)
        """
        if self._model is None:
            self._trace.log().info("Build model before predicting")
            return 0, 0

        if self._summary is None:
            self._trace.log().info("Train model before predicting")
            return 0, 0

        self._trace.log().info(
            "Y expected based on assumption source function is [{}]".format(self._actual_func.__name__))

        x = np.zeros((1))
        x[0] = np.float(x_value)
        predictions = self._model.predict(x)
        y_actual = predictions[0]
        y_expected = self._actual_func(x[0])

        return y_actual[0], y_expected

    @property
    def summary_file(self) -> str:
        """
        The name of the training summary file.
        :return: The full name and path of the training summary file.
        """
        return "{}//summary.json".format(self._dir_to_use)

    @property
    def hyper_params_file(self) -> str:
        """
        The name of the hyper parameters file
        :return: The full name and path of the hyper parameters file.
        """
        return "{}//hyper-parameters.json".format(self._dir_to_use)

    @property
    def model_checkpoint_file(self) -> str:
        """
        The file pattern to use for saving model checkpoints from 'fit' Callback
        :return: The file pattern to use for saving model checkpoints from 'fit' Callback
        """
        return '{}//weights.{}.hdf5'.format(self._dir_to_use, '{epoch:02d}')

    def __str__(self) -> str:
        """
        Capture model summary (structure) as string
        :return: Model summary as string
        """
        res = None
        if self._model is not None:
            summary = list()
            self._model.summary(print_fn=lambda s: summary.append(s))
            res = "\n{}".format("\n".join(summary))
        else:
            res = "Model not built"
        return res
