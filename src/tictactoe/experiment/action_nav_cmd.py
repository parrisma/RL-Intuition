import cmd
from src.tictactoe.interface.actionnav import ActionNav


class ActionNavCmd(cmd.Cmd):
    """
    Action [0 to 8] centric navigation of game data
    """
    prompt_default = '(nav)'
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = prompt_default

    _nav: ActionNav

    def __init__(self,
                 nav: ActionNav):
        super().__init__()
        self._nav = nav
        self.intro = "\n\n\nType help for all commands or use [list] to see session id's you can then [load]"
        return

    def do_0(self, arg):
        """Navigate to state as if taking action 0"""
        self.prompt = ActionNav.Action.action_0.do(self._nav)
        return

    def do_1(self, arg):
        """Navigate to state as if taking action 1"""
        self.prompt = ActionNav.Action.action_1.do(self._nav)
        return

    def do_2(self, arg):
        """Navigate to state as if taking action 2"""
        self.prompt = ActionNav.Action.action_2.do(self._nav)
        return

    def do_3(self, arg):
        """Navigate to state as if taking action 3"""
        self.prompt = ActionNav.Action.action_3.do(self._nav)
        return

    def do_4(self, arg):
        """Navigate to state as if taking action 4"""
        self.prompt = ActionNav.Action.action_4.do(self._nav)
        return

    def do_5(self, arg):
        """Navigate to state as if taking action 5"""
        self.prompt = ActionNav.Action.action_5.do(self._nav)
        return

    def do_6(self, arg):
        """Navigate to state as if taking action 6"""
        self.prompt = ActionNav.Action.action_6.do(self._nav)
        return

    def do_7(self, arg):
        """Navigate to state as if taking action 7"""
        self.prompt = ActionNav.Action.action_7.do(self._nav)
        return

    def do_8(self, arg):
        """Navigate to state as if taking action 8"""
        self.prompt = ActionNav.Action.action_8.do(self._nav)
        return

    def do_9(self, arg):
        """Navigate to state as if taking action 9"""
        self.prompt = ActionNav.Action.action_9.do(self._nav)
        return

    def do_hist(self, arg):
        """Show history of Q Values for action in current state"""
        self.prompt = ActionNav.Action.hist.do(self._nav, args=arg)
        return

    def do_back(self, arg):
        """Navigate to back to previous state"""
        self.prompt = ActionNav.Action.back.do(self._nav)
        return

    def do_home(self, arg):
        """Navigate to back to initial state"""
        self.prompt = ActionNav.Action.home.do(self._nav)
        return

    def do_load(self, arg):
        """Load the given session UUID"""
        self.prompt = ActionNav.Action.load.do(self._nav, args=arg)
        return

    def do_switch(self, arg):
        """Switch player perspective to take next action on board"""
        self.prompt = ActionNav.Action.switch.do(self._nav)
        return

    def do_list(self, arg):
        """List the UUIDs of the sessions that can be loaded"""
        ActionNav.Action.list.do(self._nav)
        self.prompt = self.prompt_default
        return

    def do_show(self, arg):
        """Show (or re-show) the details of the current game position"""
        ActionNav.Action.show.do(self._nav)
        self.prompt = self.prompt_default
        return

    @staticmethod
    def do_bye(self, arg):
        """End navigation session"""
        quit()
