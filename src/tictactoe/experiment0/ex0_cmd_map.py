import cmd
from src.tictactoe.experiment0.ex0_cmd import Ex0Cmd


class Ex0CmdMap(cmd.Cmd):
    """
    Action [0 to 8] centric navigation of game data
    """
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = '(nav) '

    _nav: Ex0Cmd

    def __init__(self,
                 nav: Ex0Cmd):
        super().__init__()
        self._nav = nav
        self.intro = "\n\n\nType help for all commands or use [list] to see session id's you can then [load]"
        return

    def do_0(self, _):
        """Navigate to state level 0"""
        Ex0Cmd.Ex0Actions.level_0.do(self._nav)
        return

    def do_1(self, _):
        """Navigate to state level 1"""
        Ex0Cmd.Ex0Actions.level_1.do(self._nav)
        return

    def do_2(self, _):
        """Navigate to state level 2"""
        Ex0Cmd.Ex0Actions.level_2.do(self._nav)
        return

    def do_3(self, _):
        """Navigate to state level 3"""
        Ex0Cmd.Ex0Actions.level_3.do(self._nav)
        return

    def do_4(self, _):
        """Navigate to state level 4"""
        Ex0Cmd.Ex0Actions.level_4.do(self._nav)
        return

    def do_5(self, _):
        """Navigate to state level 5"""
        Ex0Cmd.Ex0Actions.level_5.do(self._nav)
        return

    def do_6(self, _):
        """Navigate to state level 6"""
        Ex0Cmd.Ex0Actions.level_6.do(self._nav)
        return

    def do_7(self, _):
        """Navigate to state level 7"""
        Ex0Cmd.Ex0Actions.level_7.do(self._nav)
        return

    def do_8(self, _):
        """Navigate to state level 8"""
        Ex0Cmd.Ex0Actions.level_8.do(self._nav)
        return

    def do_9(self, _):
        """Navigate to state level 9"""
        Ex0Cmd.Ex0Actions.level_9.do(self._nav)
        return

    def do_load(self, arg):
        """Load the given session"""
        Ex0Cmd.Ex0Actions.load.do(self._nav, args=arg)
        return

    def do_list(self, arg):
        """List the id's of the sessions that can be loaded"""
        Ex0Cmd.Ex0Actions.list.do(self._nav, args=arg)
        return

    def do_summary(self, _):
        """Summary of all loaded visits or graph"""
        Ex0Cmd.Ex0Actions.summary.do(self._nav)
        return

    def do_explore(self, arg):
        """Run a simulation of all TicTacToe games and save as State List & Action Graph"""
        Ex0Cmd.Ex0Actions.explore.do(self._nav, args=arg)
        return

    def do_exit(self, arg):
        """Terminate session"""
        Ex0Cmd.Ex0Actions.exit.do(self._nav, args=arg)
        return
