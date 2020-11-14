import cmd
from src.tictactoe.interface.nav import Nav


class NavCmd(cmd.Cmd):
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = '(nav) '

    _nam: Nav

    def __init__(self,
                 nav: Nav):
        super().__init__()
        self._nav = nav
        self.intro = "\n\n\nType help for all commands or use [list] to see session id's you can then [load]"
        return

    def do_0(self, arg):
        """Navigate to state as if taking action 0"""
        Nav.Action.action_0.do(self._nav)
        return

    def do_1(self, arg):
        """Navigate to state as if taking action 1"""
        Nav.Action.action_1.do(self._nav)
        return

    def do_2(self, arg):
        """Navigate to state as if taking action 2"""
        Nav.Action.action_2.do(self._nav)
        return

    def do_3(self, arg):
        """Navigate to state as if taking action 3"""
        Nav.Action.action_3.do(self._nav)
        return

    def do_4(self, arg):
        """Navigate to state as if taking action 4"""
        Nav.Action.action_4.do(self._nav)
        return

    def do_5(self, arg):
        """Navigate to state as if taking action 5"""
        Nav.Action.action_5.do(self._nav)
        return

    def do_6(self, arg):
        """Navigate to state as if taking action 6"""
        Nav.Action.action_6.do(self._nav)
        return

    def do_7(self, arg):
        """Navigate to state as if taking action 7"""
        Nav.Action.action_7.do(self._nav)
        return

    def do_8(self, arg):
        """Navigate to state as if taking action 8"""
        Nav.Action.action_8.do(self._nav)
        return

    def do_9(self, arg):
        """Navigate to state as if taking action 9"""
        Nav.Action.action_9.do(self._nav)
        return

    def do_back(self, arg):
        """Navigate to back to previous state"""
        Nav.Action.action_back.do(self._nav)
        return

    def do_home(self, arg):
        """Navigate to back to initial state"""
        Nav.Action.action_home.do(self._nav)
        return

    def do_load(self, arg):
        """Load the given session UUID"""
        Nav.Action.action_load.do(self._nav, arg=arg)
        return

    def do_switch(self, arg):
        """Switch player perspective to take next action on board"""
        Nav.Action.action_switch.do(self._nav)
        return

    def do_list(self, arg):
        """List the UUIDs of the sessions that can be loaded"""
        Nav.Action.action_list.do(self._nav)
        return

    def do_bye(self, arg):
        """End navigation session"""
        return True
