from src.tictactoe.event.tictacttoe_event import TicTacToeEvent


class TicTacToeLinkedEvent:
    event: TicTacToeEvent
    prev_event: TicTacToeEvent
    next_event: TicTacToeEvent

    def __init__(self,
                 event: TicTacToeEvent):
        self.event = event
        self.prev_event = None
        self.next_event = None
        return

    def __str__(self):
        prv = ""
        nxt = ""
        if self.prev_event is not None:
            prv = "{}-{}".format(self.prev_event.episode_uuid, self.prev_event.episode_step)
        if self.next_event is not None:
            nxt = "{}-{}".format(self.next_event.episode_uuid, self.next_event.episode_step)

        return "[{}] -> [{}-{}] -> [{}]".format(prv,
                                                self.event.episode_uuid,
                                                self.event.episode_step,
                                                nxt)

    def __repr__(self):
        return self.__str__()
