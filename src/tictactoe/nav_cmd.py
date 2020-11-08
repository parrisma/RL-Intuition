import cmd
from src.tictactoe.nav import Nav


class NavCmd(cmd.Cmd):
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = '(nav) '

    _nam: Nav

    def __init__(self,
                 nav: Nav):
        super().__init__()
        self._nav = nav
        return

    def do_0(self, arg):
        'Navigate to state as if taking action 0'
        Nav.Action.action_0.do(self._nav)
        return

    def do_1(self, arg):
        'Navigate to state as if taking action 1'
        Nav.Action.action_1.do(self._nav)
        return

    def do_2(self, arg):
        'Navigate to state as if taking action 2'
        Nav.Action.action_2.do(self._nav)
        return

    def do_3(self, arg):
        'Navigate to state as if taking action 3'
        Nav.Action.action_3.do(self._nav)
        return

    def do_4(self, arg):
        'Navigate to state as if taking action 4'
        Nav.Action.action_4.do(self._nav)
        return

    def do_5(self, arg):
        'Navigate to state as if taking action 5'
        Nav.Action.action_5.do(self._nav)
        return

    def do_6(self, arg):
        'Navigate to state as if taking action 6'
        Nav.Action.action_6.do(self._nav)
        return

    def do_7(self, arg):
        'Navigate to state as if taking action 7'
        Nav.Action.action_7.do(self._nav)
        return

    def do_8(self, arg):
        'Navigate to state as if taking action 8'
        Nav.Action.action_8.do(self._nav)
        return

    def do_9(self, arg):
        'Navigate to state as if taking action 9'
        Nav.Action.action_9.do(self._nav)
        return

    def do_back(self, arg):
        'Navigate to back to previous state'
        Nav.Action.action_back.do(self._nav)
        return

    def do_bye(self, arg):
        'End navigation session'
        return True
