from typing import Dict, List

from src.tictactoe.event.tictacttoe_event import TicTacToeEvent
from src.tictactoe.event.tictactoe_linked_event import TicTacToeLinkedEvent


class SessionStats:
    # -- Inner Class: Episode Stats --
    #
    class EpisodeStats:
        episodes: Dict
        terminal_episodes: Dict

        def __init__(self):
            self.episodes = dict()
            return

    # -- Inner Class: State Stats --
    #
    class StateStats:
        states: Dict

        def __init__(self):
            self.states = dict()
            self.terminal_states = dict()
            pass

    # -- Class: Session Stats --
    #
    """
    Analyse a given set of episode events for a session and report statistics on that session
    """
    _episode_stats: EpisodeStats  # Episode specific stats
    _state_stats: StateStats
    _session_events: List[TicTacToeLinkedEvent]

    def __init__(self,
                 session_events: List['TicTacToeEvent']):
        """
        Load all events for the given session and run analysis to create stats
        :param session_events: A list of session events sorted in Episode, Step order
        """
        evt = 0
        self._session_events = list()
        while evt < len(session_events):
            lnk_evt = TicTacToeLinkedEvent(session_events[evt])
            if lnk_evt.event.episode_step > 1:
                lnk_evt.prev_event = session_events[evt - 1]
            if not lnk_evt.event.episode_end:
                lnk_evt.next_event = session_events[evt + 1]
            evt += 1
            self._session_events.append(lnk_evt)
        self._episode_stats = SessionStats.EpisodeStats()
        self._state_stats = SessionStats.StateStats()
        self._analyse()
        return

    def _analyse_episode(self,
                         lnk_event: TicTacToeLinkedEvent) -> None:
        """
        Include given event in Episode analysis
        :param lnk_event: The event to include in the Episode analysis
        """
        if lnk_event.event.episode_uuid not in self._episode_stats.episodes:
            self._episode_stats.episodes[lnk_event.event.episode_uuid] = 0
        self._episode_stats.episodes[lnk_event.event.episode_uuid] += 1
        return

    def _analyse_state(self,
                       lnk_event: TicTacToeLinkedEvent) -> None:
        """
        Include given event in State analysis
        :param lnk_event: The event to include in the State analysis
        """
        state_as_str = lnk_event.event.state.state_as_string()
        if state_as_str not in self._state_stats.states:
            self._state_stats.states[state_as_str] = 1
        self._state_stats.states[state_as_str] += 1

        if lnk_event.event.episode_end:
            if state_as_str not in self._state_stats.terminal_states:
                self._state_stats.terminal_states[state_as_str] = 0
            self._state_stats.terminal_states[state_as_str] += 1

        return

    def _analyse(self) -> None:
        """
        Iterate over all events and create stats by episode by state
        """
        for event in self._session_events:
            self._analyse_episode(lnk_event=event)
            self._analyse_state(lnk_event=event)
        return
