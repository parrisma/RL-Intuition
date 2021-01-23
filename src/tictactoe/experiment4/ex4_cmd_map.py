import cmd
from src.tictactoe.experiment4.ex4_cmd import Ex4Cmd


class Ex4CmdMap(cmd.Cmd):
    """
    Commands to manage training a Neural Net Agent
    """
    prompt_default = '(net)'
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = prompt_default

    _nav: Ex4Cmd

    def __init__(self,
                 nav: Ex4Cmd):
        super().__init__()
        self._nav = nav
        self.intro = "\n\n\nType help for all commands"
        return

    def do_build(self, arg):
        """Build a Neural Network of the given type"""
        self.prompt = Ex4Cmd.Ex4Action.build.do(self._nav, arg)
        return

    def do_train(self, arg):
        """Train and Test the Neural Network created by net command"""
        self.prompt = Ex4Cmd.Ex4Action.train.do(self._nav, arg)
        return

    def do_list(self, arg):
        """List data files that match the given pattern"""
        self.prompt = Ex4Cmd.Ex4Action.list.do(self._nav, arg)
        return

    def do_predict(self,
                   arg):
        """Make a prediction using the currently built and trained Neural Network"""
        self.prompt = Ex4Cmd.Ex4Action.predict.do(self._nav, arg)

    def do_exit(self, arg):
        """Terminate the session"""
        self.prompt = Ex4Cmd.Ex4Action.exit.do(self._nav, arg)
        return True

    do_EOF = do_exit  # Ctrl+D = Exit
