from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random

from .cards import Card, Suit, shuffled_deck, trick_winner, trick_points

@dataclass
class Observation:
    player: int
    hand: List[Card]
    briscola: Suit
    lead_card: Optional[Card]       # carta sul tavolo se stai seguendo, altrimenti None
    deck_count: int                 # carte ancora da pescare (esclusa la briscola scoperta)
    trick_points_so_far: int        # punti nel trick corrente (0, 2 carte sommate quando completo)
    played: List[Card]              # TUTTE le carte già uscite (visibili)
    opponent_count: int             # numero carte in mano all'avversario (informazione pubblica)

class BriscolaEnv:
    """
    Env fedele a Briscola a 2. Observe() restituisce solo info lecite.
    Aggiunge:
      - played: lista carte già giocate (per determinizzazioni corrette)
      - clone(): copia profonda per simulazioni MCTS
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        deck = shuffled_deck(self.rng)
        self.trump_card = deck[-1]
        self.briscola: Suit = self.trump_card.suit
        self.deck: List[Card] = deck[:-1]
        self.hands: List[List[Card]] = [[], []]
        for _ in range(3):
            self.hands[0].append(self.deck.pop(0))
            self.hands[1].append(self.deck.pop(0))
        self.points = [0, 0]
        self.leader = 0
        self.turn_player = self.leader
        self.lead_card: Optional[Card] = None
        self.follow_card: Optional[Card] = None
        self.played: List[Card] = []
        self.done = False
        return self.observe(self.turn_player)

    def legal_moves(self, player: int) -> List[int]:
        return list(range(len(self.hands[player])))

    def observe(self, player: int) -> Observation:
        trick_pts = 0
        if self.lead_card:
            trick_pts += self.lead_card.points()
        if self.follow_card:
            trick_pts += self.follow_card.points()
        return Observation(
            player=player,
            hand=list(self.hands[player]),
            briscola=self.briscola,
            lead_card=self.lead_card if player == (1 - self.leader) else None,
            deck_count=len(self.deck),
            trick_points_so_far=trick_pts,
            played=list(self.played),
            opponent_count=len(self.hands[1 - player]),
        )

    def step(self, player: int, action_idx: int):
        assert not self.done and player == self.turn_player
        card = self.hands[player].pop(action_idx)
        if self.turn_player == self.leader:
            self.lead_card = card
            self.turn_player = 1 - player
        else:
            self.follow_card = card
            winner_rel = trick_winner(self.lead_card, self.follow_card, self.briscola)
            winner = self.leader if winner_rel == 0 else (1 - self.leader)
            gained = trick_points((self.lead_card, self.follow_card))
            self.points[winner] += gained
            # segna carte giocate
            self.played.append(self.lead_card)
            self.played.append(self.follow_card)
            # pesca: winner poi loser
            for who in [winner, 1 - winner]:
                if self.deck:
                    self.hands[who].append(self.deck.pop(0))
                elif self.trump_card is not None:
                    self.hands[who].append(self.trump_card)
                    self.trump_card = None
            # nuova mano
            self.leader = winner
            self.turn_player = self.leader
            self.lead_card = None
            self.follow_card = None
            # fine partita?
            if not self.hands[0] and not self.hands[1] and self.trump_card is None and not self.deck:
                self.done = True
        return self.observe(self.turn_player), self.done

    def play_game(self, agent0, agent1, verbose: bool = False):
        self.reset()
        agents = [agent0, agent1]
        while not self.done:
            p = self.turn_player
            obs = self.observe(p)
            move = agents[p].act(self, obs)
            self.step(p, move)
        if self.points[0] > self.points[1]:
            return 0, self.points
        if self.points[1] > self.points[0]:
            return 1, self.points
        return -1, self.points

    # -------- utilità per MCTS --------
    def clone(self) -> "BriscolaEnv":
        new = object.__new__(BriscolaEnv)
        new.rng = random.Random(self.rng.random())
        new.trump_card = self.trump_card
        new.briscola = self.briscola
        new.deck = list(self.deck)
        new.hands = [list(self.hands[0]), list(self.hands[1])]
        new.points = [self.points[0], self.points[1]]
        new.leader = self.leader
        new.turn_player = self.turn_player
        new.lead_card = self.lead_card
        new.follow_card = self.follow_card
        new.played = list(self.played)
        new.done = self.done
        return new
