import cmd
from src.tictactoe.experiment2.ex2_cmd import Ex2Cmd


class Ex2CmdMap(cmd.Cmd):
    """
    Action [0 to 8] centric navigation of game data
    """
    prompt_default = '(nav)'
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = prompt_default

    _nav: Ex2Cmd

    def __init__(self,
                 nav: Ex2Cmd):
        super().__init__()
        self._nav = nav
        self.intro = "\n\n\nType help for all commands or use [list] to see session id's you can then [load]"
        return

    def do_0(self, _):
        """Navigate to state as if taking action 0"""
        self.prompt = Ex2Cmd.Ex2Actions.action_0.do(self._nav)
        return

    def do_1(self, _):
        """Navigate to state as if taking action 1"""
        self.prompt = Ex2Cmd.Ex2Actions.action_1.do(self._nav)
        return

    def do_2(self, _):
        """Navigate to state as if taking action 2"""
        self.prompt = Ex2Cmd.Ex2Actions.action_2.do(self._nav)
        return

    def do_3(self, _):
        """Navigate to state as if taking action 3"""
        self.prompt = Ex2Cmd.Ex2Actions.action_3.do(self._nav)
        return

    def do_4(self, _):
        """Navigate to state as if taking action 4"""
        self.prompt = Ex2Cmd.Ex2Actions.action_4.do(self._nav)
        return

    def do_5(self, _):
        """Navigate to state as if taking action 5"""
        self.prompt = Ex2Cmd.Ex2Actions.action_5.do(self._nav)
        return

    def do_6(self, _):
        """Navigate to state as if taking action 6"""
        self.prompt = Ex2Cmd.Ex2Actions.action_6.do(self._nav)
        return

    def do_7(self, _):
        """Navigate to state as if taking action 7"""
        self.prompt = Ex2Cmd.Ex2Actions.action_7.do(self._nav)
        return

    def do_8(self, _):
        """Navigate to state as if taking action 8"""
        self.prompt = Ex2Cmd.Ex2Actions.action_8.do(self._nav)
        return

    def do_9(self, _):
        """Navigate to state as if taking action 9"""
        self.prompt = Ex2Cmd.Ex2Actions.action_9.do(self._nav)
        return

    def do_hist(self, arg):
        """Show history of Q Values for action in current state"""
        self.prompt = Ex2Cmd.Ex2Actions.hist.do(self._nav, args=arg)
        return

    def do_back(self, _):
        """Navigate to back to previous state"""
        self.prompt = Ex2Cmd.Ex2Actions.back.do(self._nav)
        return

    def do_home(self, _):
        """Navigate to back to initial state"""
        self.prompt = Ex2Cmd.Ex2Actions.home.do(self._nav)
        return

    def do_load(self, arg):
        """Load the given session UUID"""
        self.prompt = Ex2Cmd.Ex2Actions.load.do(self._nav, args=arg)
        return

    def do_switch(self, _):
        """Switch player perspective to take next action on board"""
        self.prompt = Ex2Cmd.Ex2Actions.switch.do(self._nav)
        return

    def do_list(self, _):
        """List the UUIDs of the sessions that can be loaded"""
        Ex2Cmd.Ex2Actions.list.do(self._nav)
        self.prompt = self.prompt_default
        return

    def do_show(self, _):
        """Show (or re-show) the details of the current game position"""
        Ex2Cmd.Ex2Actions.show.do(self._nav)
        self.prompt = self.prompt_default
        return

    def do_dump(self, arg):
        """Dump the nominated structure as local JSON file"""
        self.prompt = Ex2Cmd.Ex2Actions.dump.do(self._nav, args=arg)
        return

    @staticmethod
    def do_bye(self, _):
        """End navigation session"""
        quit(0)
