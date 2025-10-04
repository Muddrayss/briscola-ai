from __future__ import annotations
import random
from typing import List, Optional
from .base import Agent
from ..core.cards import Card, Suit, trick_winner

def _card_value(c: Card) -> int:
    return c.points()

def _lowest_card_idx(hand: List[Card]) -> int:
    # tra pari punteggio, scegli quella meno “forte” nell'ordine di presa
    order = {"A":9,"3":8,"K":7,"C":6,"F":5,"7":4,"6":3,"5":2,"4":1,"2":0}
    best = None
    for i,c in enumerate(hand):
        key = (c.points(), order[c.rank])
        if best is None or key < best[0]:
            best = ((c.points(), order[c.rank]), i)
    return best[1]

class RandomAgent(Agent):
    """5.1 Random Player: sceglie una carta a caso dalla mano. :contentReference[oaicite:4]{index=4}"""
    def act(self, env, obs) -> int:
        return random.randrange(len(obs.hand))

class HighestFirstAgent(Agent):
    """5.2 Highest First: gioca la carta col maggior valore di punti (greedy). :contentReference[oaicite:5]{index=5}"""
    def act(self, env, obs) -> int:
        vals = [_card_value(c) for c in obs.hand]
        # se pari, tieni quella “meno forte” in ordine presa per preservare A/3
        order = {"A":9,"3":8,"K":7,"C":6,"F":5,"7":4,"6":3,"5":2,"4":1,"2":0}
        best_i = max(range(len(obs.hand)), key=lambda i: (vals[i], order[obs.hand[i].rank]))
        return best_i

class BestChoiceFirstAgent(Agent):
    """
    5.3 Best Choice First (riassunto fedele alla tesi):
    1) Se guidi il trick: gioca la carta col valore più alto.
    2) Se segui e la prima carta non vale punti: gioca la carta più bassa.
    3) Se la prima carta vale punti e puoi vincere senza usare briscola (stesso seme e più alta): gioca la prima tale carta.
    4) Altrimenti, se puoi vincere con briscola: gioca la prima briscola.
    5) Altrimenti (trick “non profittevole” o non vincibile): gioca la carta più bassa. 
    """
    def act(self, env, obs) -> int:
        hand = obs.hand
        # 1) Se guido
        if obs.lead_card is None:
            # “carta di valore più alto” = max per punti
            return HighestFirstAgent().act(env, obs)

        lead = obs.lead_card
        briscola = obs.briscola

        # 2) Se la prima carta non porta punti, scarta la più bassa
        if lead.points() == 0:
            return _lowest_card_idx(hand)

        # helper: trova prima carta che vince SENZA usare briscola
        same_suit_idxs = [i for i,c in enumerate(hand) if c.suit == lead.suit]
        for i in same_suit_idxs:
            if trick_winner(lead, hand[i], briscola) == 1:
                return i

        # 4) se posso vincere colla briscola, gioca la prima briscola
        for i,c in enumerate(hand):
            if c.suit == briscola and trick_winner(lead, c, briscola) == 1:
                return i

        # 5) altrimenti la più bassa
        return _lowest_card_idx(hand)
