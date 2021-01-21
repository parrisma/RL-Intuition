import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.tictactoe.experiment4.ex4_cmd import Ex4Cmd
from src.lib.rltrace.trace import Trace
from src.tictactoe.ttt.tictactoe import TicTacToe
from src.tictactoe.event.TicTacToeEventStream import TicTacToeEventStream
from src.tictactoe.net.simple_net import SimpleNet
import logging


class Ex4CmdDo(Ex4Cmd):
    """
    Actions for NN Agent training
    """
    _ttt: TicTacToe
    _trace: Trace
    _ttt_event_stream: TicTacToeEventStream
    _dir_to_use: str

    def __init__(self,
                 ttt: TicTacToe,
                 ttt_event_stream: TicTacToeEventStream,
                 trace: Trace,
                 dir_to_use: str):
        self._dir_to_use = os.path.abspath(dir_to_use)
        self._trace = trace
        self._ttt_event_stream = ttt_event_stream
        self._ttt = ttt
        return

    def do_net(self,
               args) -> str:
        """Create a Neural Network of given type"""
        sn = SimpleNet()
        sn.build()
        self._trace.log().info("{}".format(sn))
        sz = 250
        x = np.zeros((sz, 1))
        y = np.zeros((sz, 1))
        for i in range(0, sz):
            x[i] = i * ((2 * np.pi) / sz)
            y[i] = np.sin(x[i][0])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
        # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        sn.train(x_train, y_train)
        sn.test(x_test, y_test)
        sn.dump_to_json("{}//{}".format(self._dir_to_use, 'test.json'))

        return "(net)"

    def do_exit(self,
                args) -> None:
        """
        Terminate the session
        """
        sys.exit(0)
