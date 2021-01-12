import cmd
from src.tictactoe.experiment0.statenav import StateNav


class StateNavCmd(cmd.Cmd):
    """
    Action [0 to 8] centric navigation of game data
    """
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = '(nav) '

    _nav: StateNav

    def __init__(self,
                 nav: StateNav):
        super().__init__()
        self._nav = nav
        self.intro = "\n\n\nType help for all commands or use [list] to see session id's you can then [load]"
        return

    def do_0(self, arg):
        """Navigate to state level 0"""
        StateNav.Action.level_0.do(self._nav)
        return

    def do_1(self, arg):
        """Navigate to state level 1"""
        StateNav.Action.level_1.do(self._nav)
        return

    def do_2(self, arg):
        """Navigate to state level 2"""
        StateNav.Action.level_2.do(self._nav)
        return

    def do_3(self, arg):
        """Navigate to state level 3"""
        StateNav.Action.level_3.do(self._nav)
        return

    def do_4(self, arg):
        """Navigate to state level 4"""
        StateNav.Action.level_4.do(self._nav)
        return

    def do_5(self, arg):
        """Navigate to state level 5"""
        StateNav.Action.level_5.do(self._nav)
        return

    def do_6(self, arg):
        """Navigate to state level 6"""
        StateNav.Action.level_6.do(self._nav)
        return

    def do_7(self, arg):
        """Navigate to state level 7"""
        StateNav.Action.level_7.do(self._nav)
        return

    def do_8(self, arg):
        """Navigate to state level 8"""
        StateNav.Action.level_8.do(self._nav)
        return

    def do_9(self, arg):
        """Navigate to state level 9"""
        StateNav.Action.level_9.do(self._nav)
        return

    def do_load(self, arg):
        """Load the given session"""
        StateNav.Action.load.do(self._nav, args=arg)
        return

    def do_list(self, arg):
        """List the id's of the sessions that can be loaded"""
        StateNav.Action.list.do(self._nav, args=arg)
        return

    def do_summary(self, arg):
        """Summary of all loaded visits or graph"""
        StateNav.Action.summary.do(self._nav)
        return

    def do_explore(self, arg):
        """Run a simulation of all TicTacToe games and save as State List & Action Graph"""
        StateNav.Action.explore.do(self._nav, args=arg)
        return

    def do_bye(self, arg):
        """End navigation session"""
        return True
