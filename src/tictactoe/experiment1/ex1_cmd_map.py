import sys
import cmd
from src.tictactoe.experiment1.ex1_cmd import Ex1Cmd


class Ex1CmdMap(cmd.Cmd):
    """
    Run TTT Games
    """
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = '(nav) '

    _nav: Ex1Cmd

    def __init__(self,
                 nav: Ex1Cmd):
        super().__init__()
        self._nav = nav
        self.intro = "\n\n\nType help for all commands or use [list] to see session id's you can then [explore]"
        return

    def do_run(self, arg):
        """Run the given number of TT Games"""
        Ex1Cmd.Ex1Actions.run.do(self._nav, args=arg)
        return

    def do_list(self, arg):
        """Run the given number of TT Games"""
        Ex1Cmd.Ex1Actions.list.do(self._nav)
        return

    def do_head(self, arg):
        """Show 10 events for the given session_uuid"""
        Ex1Cmd.Ex1Actions.head.do(self._nav, args=arg)
        return

    def do_set(self, arg):
        """Run a set piece game of specific moves"""
        Ex1Cmd.Ex1Actions.set.do(self._nav, args=arg)
        return

    def do_exit(self, arg):
        """End navigation session"""
        Ex1Cmd.Ex1Actions.exit.do(self._nav, args=arg)
        return
