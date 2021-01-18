import cmd
from src.tictactoe.experiment4.ex4_cmd import Ex4Cmd


class Ex4CmdMap(cmd.Cmd):
    """
    Commands to manage training a Neural Net Agent
    """
    prompt_default = '(setup)'
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = prompt_default

    _nav: Ex4Cmd

    def __init__(self,
                 nav: Ex4Cmd):
        super().__init__()
        self._nav = nav
        self.intro = "\n\n\nType help for all commands"
        return

    def do_exit(self, arg):
        """Terminate the session"""
        self.prompt = Ex4Cmd.Ex4Action.exit.do(self._nav, arg)
        return
