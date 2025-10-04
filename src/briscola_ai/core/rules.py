from __future__ import annotations
from typing import List, Tuple, Optional
from .cards import Card, Suit, trick_winner, trick_points

class TrickState:
    """Trick corrente: carta di uscita (lead) e risposta (follow), più info utility."""
    def __init__(self, briscola: Suit):
        self.briscola = briscola
        self.lead: Optional[Card] = None
        self.follow: Optional[Card] = None

    def play_lead(self, c: Card):
        if self.lead is not None:
            raise ValueError("Lead già giocata")
        self.lead = c

    def play_follow(self, c: Card):
        if self.lead is None:
            raise ValueError("Manca la lead")
        if self.follow is not None:
            raise ValueError("Follow già giocata")
        self.follow = c

    def is_complete(self) -> bool:
        return self.lead is not None and self.follow is not None

    def winner_idx(self) -> int:
        if not self.is_complete():
            raise ValueError("Trick incompleto")
        return trick_winner(self.lead, self.follow, self.briscola)

    def points(self) -> int:
        if not self.is_complete():
            return 0
        return trick_points((self.lead, self.follow))
