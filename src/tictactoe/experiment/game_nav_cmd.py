import cmd
from src.tictactoe.interface.gamenav import GameNav


class GameNavCmd(cmd.Cmd):
    """
    Run TTT Games
    """
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = '(nav) '

    _nav: GameNav

    def __init__(self,
                 nav: GameNav):
        super().__init__()
        self._nav = nav
        self.intro = "\n\n\nType help for all commands or use [list] to see session id's you can then [explore]"
        return

    def do_run(self, arg):
        """Run the given number of TT Games"""
        GameNav.Action.run.do(self._nav, args=arg)
        return

    def do_list(self, arg):
        """Run the given number of TT Games"""
        GameNav.Action.list.do(self._nav)
        return

    def do_head(self, arg):
        """Show 10 events for the given session_uuid"""
        GameNav.Action.head.do(self._nav, args=arg)
        return

    def do_bye(self, arg):
        """End navigation session"""
        return True
