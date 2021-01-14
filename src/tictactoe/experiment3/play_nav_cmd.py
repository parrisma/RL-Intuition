import cmd
from src.tictactoe.experiment3.playnav import PlayNav


class PlayNavCmd(cmd.Cmd):
    """
    Commands to manage play between two AI Agents
    """
    prompt_default = '(setup)'
    intro = 'Welcome. Type help or ? to list commands.\n'
    prompt = prompt_default

    _nav: PlayNav

    def __init__(self,
                 nav: PlayNav):
        super().__init__()
        self._nav = nav
        self.intro = "\n\n\nType help for all commands or use [x], [o] to create agents and then [play]"
        return

    def do_x(self, arg):
        """Create an Agent to play as X"""
        self.prompt = PlayNav.Action.x.do(self._nav, arg)
        return

    def do_o(self, arg):
        """Create an Agent to play as O"""
        self.prompt = PlayNav.Action.o.do(self._nav, arg)
        return

    def do_play(self, arg):
        """Play X and O against each other"""
        self.prompt = PlayNav.Action.play.do(self._nav, arg)
        return

    def do_bye(self, arg):
        """Terminate the session"""
        self.prompt = PlayNav.Action.bye.do(self._nav, arg)
        return

    def do_list(self, arg):
        """List the data types that can be loaded into different agents"""
        self.prompt = PlayNav.Action.list.do(self._nav, arg)
        return
